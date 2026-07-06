"""Dataset hierarchy support: a parent dataset that aggregates several child
wsds datasets of the same kind into one logical dataset.

A `WSMetaDataset` is a thin *runtime-aggregation* layer over N child
`WSDataset` objects. It does NOT rebuild or duplicate any index — each child
keeps its own `index.sqlite3` (which itself already spans many partitions), and
the meta dataset only adds an offset-routing layer on top:

    global index ──bisect──▶ (child, local index) ──▶ child[local index]

This is the only design that scales to the segmented children here (individual
datasets reach >1e9 samples, so a merged SQLite `files` table is infeasible).

Children are referenced explicitly via a small JSON manifest (`*.wsds-meta`):

    {
        "wsds_meta_version": 1,
        "kind": "source",
        "children": [
            {"name": "en", "path": "../data-en/indices/source"},
            {"name": "de", "path": "../data-de/indices/source"}
        ]
    }

`path` is resolved relative to the manifest file's directory. `WSDataset(p)`
transparently returns a `WSMetaDataset` when `p` points at (or contains) a
manifest, so callers use the same entry point for flat and hierarchical
datasets.

Because the children are schema-heterogeneous (they share only `__key__`), the
meta dataset takes the *union* of their fields. `sql_select` runs per child and
diagonally concatenates the results, filling columns absent from a child with
nulls; children that don't expose a queried column at all are skipped.
"""

import bisect
import json
import random
from pathlib import Path

from .utils import WSShardMissingError, format_duration


def default_child_name(path) -> str:
    """Derive a short, stable child name from a dataset root path.

    For the canonical `.../<dataset>/indices/<kind>` layout this returns the
    `<dataset>` component (e.g. `data-en`), dropping the noisy `indices`/`<kind>`
    tail. Otherwise it falls back to the leaf directory name.
    """
    parts = Path(path).parts
    if len(parts) >= 3 and parts[-2] == "indices":
        return parts[-3]
    return Path(path).name


def find_meta_manifest(path) -> Path | None:
    """Return the manifest Path if `path` is/contains a `.wsds-meta` file, else None.

    Accepts a direct path to a `*.wsds-meta` file, or a directory containing a
    `meta.wsds-meta` (or exactly one `*.wsds-meta`).
    """
    p = Path(path)
    if p.suffix == ".wsds-meta" and p.is_file():
        return p
    if p.is_dir():
        cand = p / "meta.wsds-meta"
        if cand.is_file():
            return cand
        metas = sorted(p.glob("*.wsds-meta"))
        if len(metas) == 1:
            return metas[0]
    return None


class WSMetaDataset:
    """A parent dataset aggregating several child `WSDataset`s of the same kind.

    Duck-types the public surface of `WSDataset` (`__len__`, `__getitem__`,
    `__iter__`, `random_sample`, `sequential_from`, `sql_select`, `sql_filter`,
    `filtered`, `fields`, `segmented`, `close`) so it can be dropped into the
    same dataloader code paths.
    """

    def __init__(self, children, names=None, rng: random.Random | int | None = None, kind: str | None = None):
        """
        Args:
            children: list of child dataset roots (str/Path) or already-built `WSDataset`s.
            names: optional per-child short names (default: derived from path).
            rng: seed or `random.Random` for sampling.
            kind: optional informational tag ("source" / "filtered_vad").
        """
        from .ws_dataset import WSDataset

        if isinstance(rng, int):
            self.rng = random.Random(rng)
        elif rng is not None:
            self.rng = rng
        else:
            self.rng = random

        self.kind = kind
        self.children: list = []
        self.child_names: list[str] = []
        for i, child in enumerate(children):
            ds = child if isinstance(child, WSDataset) else WSDataset(child, rng=self.rng)
            if ds.index is None:
                raise ValueError(f"Child dataset {ds.dataset_root} has no index; meta datasets require indexed children")
            self.children.append(ds)
            if names is not None:
                self.child_names.append(names[i])
            else:
                self.child_names.append(default_child_name(ds.dataset_root))

        if not self.children:
            raise ValueError("WSMetaDataset requires at least one child dataset")

        # All children must agree on segmentation (mixing source + segmented is meaningless).
        segmenteds = {ds.segmented for ds in self.children}
        if len(segmenteds) > 1:
            raise ValueError(
                f"Children disagree on `segmented` ({segmenteds}); a meta dataset must aggregate one kind. "
                f"Per child: {[(n, ds.segmented) for n, ds in zip(self.child_names, self.children)]}"
            )
        self.segmented = next(iter(segmenteds))

        # Cumulative global-offset map for index routing: child c owns
        # [self._starts[c], self._starts[c] + len(child)).
        self._starts: list[int] = []
        total = 0
        for ds in self.children:
            self._starts.append(total)
            total += len(ds)
        self._total = total

        # Union of child fields (children share only __key__). First child to
        # expose a field wins for the provenance value; `_field_children` records
        # every child that has it (used by sql_select to skip children lacking a column).
        self.fields: dict = {}
        self._field_children: dict[str, list[int]] = {}
        for ci, ds in enumerate(self.children):
            for k, v in ds.fields.items():
                self.fields.setdefault(k, v)
                self._field_children.setdefault(k, []).append(ci)

        self.computed_columns = {}  # meta routes through children; no own computed columns

    @classmethod
    def from_manifest(cls, manifest_path, rng: random.Random | int | None = None):
        manifest_path = Path(manifest_path)
        spec = json.loads(manifest_path.read_text())
        base = manifest_path.parent
        children = []
        names = []
        for entry in spec["children"]:
            if isinstance(entry, str):
                entry = {"path": entry}
            path = Path(entry["path"])
            if not path.is_absolute():
                path = (base / path).resolve()
            children.append(path)
            names.append(entry.get("name") or default_child_name(path))
        meta = cls(children, names=names, rng=rng, kind=spec.get("kind"))
        meta.manifest_path = manifest_path
        return meta

    #
    # Routing helpers
    #
    def _child_of_index(self, index: int) -> tuple[int, int]:
        """Map a global index to (child_idx, local_index)."""
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(f"index {index} out of range for meta dataset of length {self._total}")
        ci = bisect.bisect_right(self._starts, index) - 1
        return ci, index - self._starts[ci]

    #
    # Access (mirrors WSDataset)
    #
    def __len__(self):
        return self._total

    def __getitem__(self, key_or_index):
        if isinstance(key_or_index, int):
            ci, local = self._child_of_index(key_or_index)
            return self.children[ci][local]
        if isinstance(key_or_index, str):
            # Optional explicit routing: "child_name::key".
            if "::" in key_or_index:
                name, raw = key_or_index.split("::", 1)
                ci = self.child_names.index(name)
                return self.children[ci][raw]
            # Otherwise search children in order (keys may collide across children).
            for ds in self.children:
                sample = ds[key_or_index]
                if sample is not None:
                    return sample
            return None
        raise TypeError(f"Invalid key type: {type(key_or_index)}")

    def random_sample(self):
        return self[self.rng.randrange(self._total)]

    def random_samples(self, N: int = 1):
        for _ in range(N):
            yield self.random_sample()

    def sequential_from(self, sample, max_N=None):
        # The sample carries its owning child dataset; delegate so iteration
        # stays within one child (and never crosses a child boundary).
        yield from sample.dataset.sequential_from(sample, max_N=max_N)

    def __iter__(self):
        while True:
            yield from self.sequential_from(self.random_sample())

    def random_chunks(self, max_N: int):
        while True:
            yield from self.sequential_from(self.random_sample(), max_N=max_N)

    #
    # SQL (per-child, diagonally concatenated to tolerate heterogeneous schemas)
    #
    def _queried_columns(self, queries):
        import polars as pl

        cols = set()
        for q in queries:
            try:
                cols.update(pl.sql_expr(q).meta.root_names())
            except Exception:
                pass
        return cols

    def sql_select(
        self,
        *queries,
        return_as_lazyframe=False,
        shard_subsample=None,
        rng=42,
        shard_pipe=None,
        with_dataset_col=False,
    ):
        """Run the query on every child and diagonally concat the results.

        Children missing a queried column are skipped (with a note). Columns
        present in some children but not others are null-filled by the diagonal
        concat. Pass `with_dataset_col=True` to add a `__dataset__` column
        identifying each row's source child.
        """
        import polars as pl

        wanted = self._queried_columns(queries) - {"__key__", "__shard_path__", "__shard_offset__"}
        lazy_frames = []
        skipped = []
        for name, ds in zip(self.child_names, self.children):
            missing = [c for c in wanted if c not in ds.fields]
            if missing:
                skipped.append((name, missing))
                continue
            try:
                lf = ds.sql_select(
                    *queries,
                    return_as_lazyframe=True,
                    shard_subsample=shard_subsample,
                    rng=rng,
                    shard_pipe=shard_pipe,
                )
            except WSShardMissingError:
                continue
            if with_dataset_col:
                lf = lf.with_columns(pl.lit(name).alias("__dataset__"))
            lazy_frames.append(lf)

        if skipped:
            print(f"NOTE: skipped {len(skipped)} child(ren) lacking queried columns: {skipped[:5]}")
        if not lazy_frames:
            raise WSShardMissingError("No child dataset could satisfy the query")

        out = pl.concat(lazy_frames, how="diagonal_relaxed")
        return out if return_as_lazyframe else out.collect()

    def sql_filter(self, query, shard_subsample=None, rng=42):
        """Return namespaced keys ("child_name::key") matching the boolean query."""
        keys = []
        for name, ds in zip(self.child_names, self.children):
            if any(c not in ds.fields for c in self._queried_columns([query]) - {"__key__"}):
                continue
            try:
                child_keys = ds.sql_filter(query, shard_subsample=shard_subsample, rng=rng)
            except WSShardMissingError:
                continue
            keys.extend(f"{name}::{k}" for k in child_keys)
        return keys

    def filtered(self, query, infinite=False, shuffle=True, N=None, seed=None, shard_subsample=None, rng=42):
        keys = self.sql_filter(query, shard_subsample=shard_subsample, rng=rng)
        self.last_query_n_samples = len(keys)
        shuffler = random.Random(seed)
        i = 0
        while True:
            order = list(keys)
            if shuffle:
                shuffler.shuffle(order)
            for key in order:
                yield self[key]
                i += 1
                if N is not None and i >= N:
                    return
            if not infinite:
                break

    #
    # Misc
    #
    def close(self):
        for ds in self.children:
            ds.close()

    @property
    def audio_duration(self):
        return sum(ds.index.audio_duration or 0 for ds in self.children)

    @property
    def speech_duration(self):
        return sum(ds.index.speech_duration or 0 for ds in self.children)

    @property
    def n_shards(self):
        return sum(ds.index.n_shards for ds in self.children)

    def __repr__(self):
        return f"WSMetaDataset(n_children={len(self.children)}, segmented={self.segmented}, kind={self.kind!r})"

    def __str__(self):
        out = repr(self) + "\n"
        out += f"     Audio duration: {format_duration(self.audio_duration)}\n"
        if self.segmented:
            out += f"    Speech duration: {format_duration(self.speech_duration)}\n"
        out += f"   Number of shards: {self.n_shards}\n"
        out += f"  Number of samples: {format(len(self), ',d').replace(',', ' ')}\n"
        out += f"          Children: {len(self.children)}\n"
        for name, ds in zip(self.child_names, self.children):
            out += f"            - {name}: {format(len(ds), ',d').replace(',', ' ')} samples\n"
        return out
