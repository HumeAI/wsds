"""Dataset hierarchy + subsetting.

A `WSMetaDataset` aggregates several `WSDataset`s of the same kind into one
logical dataset. It has two modes:

  * base (aggregation): stacks the full parents; access routes by per-child
    offset ranges, `sql_select` concats across children.

  * subset (offset-index): backed by an explicit **selection table** — one row
    per selected sample, `(parent, partition, shard, offset[, __key__])`,
    pointing DIRECTLY into the parent shards. Random access is a direct
    `(parent, shard, offset)` lookup (no scan, no per-row query re-evaluation);
    `sql_select` scans the touched shards once per parent and restricts by a
    global row index.

Subsets are produced by composable operators that resolve to the offset table:
  `meta.filter(where)`  — rows matching a boolean SQL predicate
  `meta.sample(f, by=)` — a random `f`; by="row" (hash of __key__) or by="shard"
                          (whole shards, every offset enumerated)
  `meta.select_keys(k)` — an explicit __key__ set (located via each parent index)

Offset resolution is fully vectorized: each parent is scanned once (multi-file
fast path) with a global row index attached, and the matching row indices are
mapped back to (partition, shard, offset) via a join_asof against the
cumulative shard sizes from the parent's index. Per-parent scans are collected
in parallel (`pl.collect_all`). Nothing is materialized on the data side —
only the small offset table (saved as one `selection.feather`).
"""

import bisect
import json
import os
import random
from pathlib import Path

import polars as pl

from .utils import WSShardMissingError, abort_if_dataloader_worker, format_duration

_HASH_MOD = 1 << 32
_SPECIAL_COLS = ("__key__", "__shard_path__", "__shard_offset__")
# canonical in-memory selection schema: partition/shard are Categorical so a
# many-million-row table stores 4-byte codes per row instead of repeated strings
_SEL_SCHEMA = {
    "parent": pl.UInt32,
    "partition": pl.Categorical,
    "shard": pl.Categorical,
    "offset": pl.UInt32,
    "__key__": pl.Utf8,
}
# schema used while *building* selection frames (plain strings; cast on entry)
_SEL_BUILD_SCHEMA = {
    "parent": pl.UInt32,
    "partition": pl.Utf8,
    "shard": pl.Utf8,
    "offset": pl.UInt32,
    "__key__": pl.Utf8,
}


def default_child_name(path) -> str:
    """Derive a short, stable child name from a dataset root path.

    For the canonical `.../<dataset>/indices/<kind>` layout this returns the
    `<dataset>` component (e.g. `data-en`), dropping the `indices`/`<kind>` tail.
    """
    parts = Path(path).parts
    if len(parts) >= 3 and parts[-2] == "indices":
        return parts[-3]
    return Path(path).name


def find_meta_manifest(path) -> Path | None:
    """Return the manifest Path if `path` is/contains a `.wsds-meta` file, else None."""
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


def _hash_sample_expr(fraction, seed):
    """Boolean expr selecting ~`fraction` of rows uniformly by hash(__key__)."""
    return (pl.col("__key__").hash(seed=seed) % _HASH_MOD) < int(fraction * _HASH_MOD)


def _locate_root(root):
    """Return `root` if it exists; otherwise try to relocate it under each
    directory of WSDS_DATASET_SEARCH_PATH by matching trailing path components,
    so a subset saved on one machine loads where the data is mounted at a
    different prefix."""
    p = Path(root)
    if p.exists():
        return p
    parts = p.parts[1:] if p.is_absolute() else p.parts
    for d in [s for s in os.environ.get("WSDS_DATASET_SEARCH_PATH", "").split(":") if s]:
        for k in range(min(len(parts), 5), 0, -1):
            cand = Path(d, *parts[-k:])
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"Parent dataset root not found: {root}. If the data is mounted elsewhere on this "
        f"machine, set WSDS_DATASET_SEARCH_PATH to the directory containing the datasets."
    )


class WSMetaDataset:
    """Aggregates several `WSDataset`s; optionally backed by a direct offset index
    (a subset). Duck-types the `WSDataset` access surface for dataloaders."""

    def __init__(self, children, names=None, rng: random.Random | int | None = None,
                 kind: str | None = None, selection: pl.DataFrame | None = None):
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
            ds = WSDataset(child, rng=self.rng) if isinstance(child, (str, Path)) else child
            if ds.index is None:
                raise ValueError(f"Child dataset {ds.dataset_root} has no index; meta datasets require an index")
            self.children.append(ds)
            self.child_names.append(names[i] if names is not None else default_child_name(ds.dataset_root))

        if not self.children:
            raise ValueError("WSMetaDataset requires at least one child dataset")

        segmenteds = {ds.segmented for ds in self.children}
        if len(segmenteds) > 1:
            raise ValueError(f"Children disagree on `segmented` ({segmenteds}); a meta must aggregate one kind.")
        self.segmented = next(iter(segmenteds))

        # subset mode: the selection table (offset index). None => base aggregation.
        if selection is not None:
            selection = selection.select(list(_SEL_SCHEMA)).cast(_SEL_SCHEMA)
        self._sel = selection
        # base-mode cumulative offset map, built lazily (calls len(child)).
        self._starts: list[int] | None = None
        self._total: int | None = None
        self._keymap = None  # sorted (__key__, row) frame for subset key lookups

        # union of child fields (children share only __key__)
        self.fields: dict = {}
        self._field_children: dict[str, list[int]] = {}
        for ci, ds in enumerate(self.children):
            for k, v in ds.fields.items():
                self.fields.setdefault(k, v)
                self._field_children.setdefault(k, []).append(ci)
        self.computed_columns = {}

    @classmethod
    def from_manifest(cls, manifest_path, rng=None):
        manifest_path = Path(manifest_path)
        spec = json.loads(manifest_path.read_text())
        base = manifest_path.parent
        children, names = [], []
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

    @property
    def is_subset(self):
        return self._sel is not None

    # ------------------------------------------------------------------------
    # Subset operators -> a new offset-backed WSMetaDataset
    # ------------------------------------------------------------------------
    def _with_selection(self, sel):
        if sel is None or sel.is_empty():
            sel = pl.DataFrame(schema=_SEL_SCHEMA)
        return WSMetaDataset(self.children, names=list(self.child_names),
                             kind=self.kind, rng=self.rng, selection=sel)

    def filter(self, where):
        """Rows matching a boolean SQL predicate, as a direct offset index.

        Children that don't expose every column referenced by the predicate are
        skipped (they cannot match), mirroring `sql_select`."""
        expr = pl.sql_expr(where)
        needed = set(expr.meta.root_names()) - set(_SPECIAL_COLS)
        plans, skipped = [], []
        if self._sel is None:
            for pi, p in enumerate(self.children):
                if any(c not in p.fields for c in needed):
                    skipped.append(self.child_names[pi])
                    continue
                plans.append((pi, *self._predicate_plan(pi, [expr])))
        else:
            for pi, sub in self._sel_by_parent():
                if any(c not in self.children[pi].fields for c in needed):
                    skipped.append(self.child_names[pi])
                    continue
                shards, _, keep_g = self._sub_scan(pi, sub)
                plans.append((pi, *self._predicate_plan(pi, [expr], shards=shards, keep_g=keep_g)))
        if skipped:
            print(f"NOTE: skipped {len(skipped)} child(ren) lacking queried columns: {skipped[:5]}")
        frames = self._collect_selections(plans)
        return self._with_selection(pl.concat(frames) if frames else None)

    def sample(self, fraction=None, n=None, by="row", seed=0):
        """A random subset. by="row" (default): uniform per-row via hash(__key__);
        by="shard": whole shards, every offset enumerated.

        With `n=`, row-mode subsets contain exactly `n` rows (or all rows if the
        dataset is smaller); shard-mode treats `n` as a target and stays
        approximate, since whole shards are picked."""
        if fraction is None and n is None:
            raise ValueError("give fraction= or n=")
        total = len(self)
        frac = fraction if fraction is not None else min(1.0, n / max(1, total))
        if self._sel is not None:  # subsample the existing table
            if by == "shard":
                shards = self._sel.select("parent", "partition", "shard").unique(maintain_order=True)
                keep = shards.sample(fraction=frac, seed=seed)
                return self._with_selection(self._sel.join(keep, on=["parent", "partition", "shard"], how="semi"))
            if n is not None:
                return self._with_selection(self._sel.sample(n=min(n, total), seed=seed))
            return self._with_selection(self._sel.sample(fraction=frac, seed=seed))
        if by == "shard":
            frames = []
            for pi, p in enumerate(self.children):
                shards = p.get_shard_list()
                k = max(1, min(len(shards), round(frac * len(shards))))
                picked = random.Random(seed + pi).sample(list(shards), k)
                frames.append(self._enumerate_shards(pi, picked))
            return self._with_selection(pl.concat(frames))
        if by == "row":
            # oversample slightly when n= is given, then trim to exactly n below
            f = frac if n is None else min(1.0, (n * 1.05 + 256) / max(1, total))
            plans = [(pi, *self._predicate_plan(pi, [_hash_sample_expr(f, seed + pi)]))
                     for pi in range(len(self.children))]
            frames = self._collect_selections(plans)
            sel = pl.concat(frames) if frames else None
            if n is not None and sel is not None and sel.height > n:
                # exact-n trim; re-sort to keep the shard-clustered row order
                sel = sel.sample(n=n, seed=seed).sort("parent", "partition", "shard", "offset")
            return self._with_selection(sel)
        raise ValueError("by must be 'row' or 'shard'")

    def select_keys(self, keys):
        """An explicit __key__ set, located via each parent's index (batched
        lookups, one IN query per chunk instead of one query per key).

        Keys may be `name::key`-prefixed (as returned by `sql_filter`) to target
        a specific child. Unprefixed keys are resolved against the children in
        order — the first child containing the key wins (no duplicates)."""
        keys = list(keys)
        resolved: dict[str, tuple] = {}
        for pi, (name, p) in enumerate(zip(self.child_names, self.children)):
            todo = {}
            for kk in keys:
                if kk in resolved:
                    continue
                raw = kk
                if "::" in kk:
                    prefix, rest = kk.split("::", 1)
                    if prefix in self.child_names:
                        if prefix != name:
                            continue
                        raw = rest
                todo.setdefault(kk, (raw, *p.parse_key(raw)))
            if not todo:
                continue
            found = p.index.lookup_by_keys({fname for _, fname, _ in todo.values()})
            for kk, (raw, fname, koff) in todo.items():
                r = found.get(fname)
                if r is not None:
                    resolved[kk] = (pi, r[0], r[1], int(r[2]) + koff, raw)
        rows = [resolved[kk] for kk in keys if kk in resolved]
        if not rows:
            return self._with_selection(None)
        sel = pl.DataFrame(rows, schema=["parent", "partition", "shard", "offset", "__key__"],
                           orient="row").cast({"parent": pl.UInt32, "offset": pl.UInt32})
        return self._with_selection(sel)

    # ---- offset resolution -------------------------------------------------
    def _shard_sizes(self, pi):
        """{(partition, shard): n_samples} for parent `pi`; cached on the parent
        (children are shared across derived metas, so the cache is too)."""
        ds = self.children[pi]
        cache = getattr(ds, "_shard_sizes_cache", None)
        if cache is None:
            cache = {(part, shard): n for part, shard, n in ds.index.shard_sizes()}
            ds._shard_sizes_cache = cache
        return cache

    def _scan_frame(self, pi, shards):
        """DataFrame(partition, shard, n, start) describing a scan over `shards`
        in order; `start` is each shard's first global row index in that scan."""
        sizes = self._shard_sizes(pi)
        try:
            ns = [sizes[(part, shard)] for part, shard in shards]
        except KeyError as e:
            raise KeyError(
                f"Shard {e.args[0]} not found in the index of parent {self.child_names[pi]!r}; "
                f"the parent dataset likely changed since this selection was built"
            ) from None
        return pl.DataFrame({
            "partition": pl.Series([s[0] for s in shards], dtype=pl.Utf8),
            "shard": pl.Series([s[1] for s in shards], dtype=pl.Utf8),
            "n": pl.Series(ns, dtype=pl.UInt32),
        }).with_columns(start=(pl.col("n").cum_sum() - pl.col("n")).cast(pl.UInt32))

    def _predicate_plan(self, pi, exprs, shards=None, keep_g=None):
        """Lazy plan selecting the global row indices (and keys) of the rows of
        parent `pi` matching `exprs`, plus the scan frame to map them back.
        Uses the multi-file fast path: no per-shard columns are requested; the
        row index is attached to the whole scan."""
        parent = self.children[pi]
        shards = list(shards) if shards is not None else parent.get_shard_list()
        sdf = self._scan_frame(pi, shards)
        needed = set()
        for e in exprs:
            needed |= set(e.meta.root_names())
        extra = sorted(needed - {"__key__"})
        lf = parent.sql_select("`__key__`", *[f"`{c}`" for c in extra],
                               shards=shards, shard_subsample=1, return_as_lazyframe=True)
        lf = lf.with_row_index("__g__")
        if keep_g is not None:
            lf = lf.filter(pl.col("__g__").is_in(keep_g))
        for e in exprs:
            lf = lf.filter(e)
        return lf.select("__g__", "__key__"), sdf

    def _g_to_selection(self, pi, loc, sdf):
        """Map matched global row indices back to (partition, shard, offset)
        rows, vectorized via join_asof on the cumulative shard starts."""
        if loc.height == 0:
            return pl.DataFrame(schema=_SEL_BUILD_SCHEMA)
        hit = (loc.with_columns(pl.col("__g__").set_sorted())
               .join_asof(sdf.select("start", "partition", "shard")
                          .with_columns(pl.col("start").set_sorted()),
                          left_on="__g__", right_on="start", strategy="backward"))
        return hit.select(
            pl.lit(pi, dtype=pl.UInt32).alias("parent"),
            "partition", "shard",
            (pl.col("__g__") - pl.col("start")).cast(pl.UInt32).alias("offset"),
            pl.col("__key__").cast(pl.Utf8),
        )

    def _collect_selections(self, plans):
        """Collect all per-parent scan plans in parallel and map the results to
        selection frames. `plans`: [(pi, lazyframe, scan_frame), ...]."""
        locs = pl.collect_all([lf for _, lf, _ in plans])
        frames = [self._g_to_selection(pi, loc, sdf) for (pi, _, sdf), loc in zip(plans, locs)]
        return [f for f in frames if f.height]

    def _enumerate_shards(self, pi, picked):
        """Every offset of each picked shard (from the index; no scan)."""
        sdf = self._scan_frame(pi, list(picked))
        if sdf.height == 0:
            return pl.DataFrame(schema=_SEL_BUILD_SCHEMA)
        return sdf.select(
            pl.lit(pi, dtype=pl.UInt32).alias("parent"),
            "partition", "shard",
            pl.int_ranges(0, pl.col("n"), dtype=pl.UInt32).alias("offset"),
            pl.lit(None, dtype=pl.Utf8).alias("__key__"),
        ).explode("offset")

    def _sel_by_parent(self):
        yield from _group_by_parent(self._sel)

    def _sub_scan(self, pi, sub):
        """For a parent's selection rows: the touched shards (first-appearance
        order), their scan frame, and the selected rows' global row indices
        within a scan over exactly those shards."""
        plain = sub.select(pl.col("partition").cast(pl.Utf8), pl.col("shard").cast(pl.Utf8), "offset")
        shards = [tuple(r) for r in plain.select("partition", "shard").unique(maintain_order=True).rows()]
        sdf = self._scan_frame(pi, shards)
        g = (plain.join(sdf.select("partition", "shard", "start"), on=["partition", "shard"], how="left")
             .select((pl.col("start") + pl.col("offset")).alias("g"))["g"])
        return shards, sdf, g

    # ------------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------------
    def _ensure_offsets(self):
        if self._starts is not None:
            return
        starts, total = [], 0
        for ds in self.children:
            starts.append(total)
            total += len(ds)
        self._starts, self._total = starts, total

    def __len__(self):
        if self._sel is not None:
            return self._sel.height
        self._ensure_offsets()
        return self._total

    def _child_of_index(self, index):
        self._ensure_offsets()
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(f"index {index} out of range for meta dataset of length {self._total}")
        ci = bisect.bisect_right(self._starts, index) - 1
        return ci, index - self._starts[ci]

    def __getitem__(self, key_or_index):
        from .ws_sample import WSSample
        if isinstance(key_or_index, int):
            if self._sel is not None:
                i = key_or_index + len(self) if key_or_index < 0 else key_or_index
                if not 0 <= i < self._sel.height:
                    raise IndexError(f"index {key_or_index} out of range for subset of length {self._sel.height}")
                r = self._sel.row(i, named=True)
                return WSSample(self.children[r["parent"]], (r["partition"], r["shard"]), int(r["offset"]))
            ci, local = self._child_of_index(key_or_index)
            return self.children[ci][local]
        if isinstance(key_or_index, str):
            if self._sel is not None:
                # sorted-key binary search: no Python dict over millions of keys
                if self._keymap is None:
                    self._keymap = (self._sel.select("__key__").with_row_index("__i__")
                                    .drop_nulls("__key__").sort("__key__"))
                km = self._keymap
                pos = int(km["__key__"].search_sorted(key_or_index, side="left"))
                if pos < km.height and km["__key__"][pos] == key_or_index:
                    return self[int(km["__i__"][pos])]
                return None
            if "::" in key_or_index:
                name, raw = key_or_index.split("::", 1)
                return self.children[self.child_names.index(name)][raw]
            for ds in self.children:
                s = ds[key_or_index]
                if s is not None:
                    return s
            return None
        raise TypeError(f"Invalid key type: {type(key_or_index)}")

    def random_sample(self):
        return self[self.rng.randrange(len(self))]

    def random_samples(self, N=1):
        for _ in range(N):
            yield self.random_sample()

    def sequential_from(self, sample, max_N=None):
        yield from sample.dataset.sequential_from(sample, max_N=max_N)

    def __iter__(self):
        """Base mode: infinite random-walk iteration (matches `WSDataset.__iter__`).
        Subset mode: one finite pass over the selection in table order
        (shard-clustered, so sequential reads stay local)."""
        if self._sel is not None:
            from .ws_sample import WSSample
            for r in self._sel.iter_rows(named=True):
                yield WSSample(self.children[r["parent"]], (r["partition"], r["shard"]), int(r["offset"]))
            return
        while True:
            yield from self.sequential_from(self.random_sample())

    def random_chunks(self, max_N):
        while True:
            yield from self.sequential_from(self.random_sample(), max_N=max_N)

    # ------------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------------
    def _queried_columns(self, queries):
        cols = set()
        for q in queries:
            try:
                cols.update(pl.sql_expr(q).meta.root_names())
            except Exception:
                pass
        return cols

    def _resolve_shard_subsample(self, shard_subsample):
        """Meta-level default subsampling: one decision (and one INFO line)
        across all children instead of each child capping at 150 shards."""
        if shard_subsample is not None:
            return shard_subsample
        abort_if_dataloader_worker()
        total = self.n_shards
        if total < 150:
            return 1
        frac = 150 / total
        if not hasattr(self, "_shown_subsampling_info"):
            print(f"INFO: to speed things up wsds is loading a random {frac * 100:.2f}% subset of the "
                  f"shards across all children, pass shard_subsample=1 to force it to load the whole dataset")
            self._shown_subsampling_info = True
        return frac

    def sql_select(self, *queries, return_as_lazyframe=False, shard_subsample=None,
                   rng=42, shard_pipe=None, with_dataset_col=False):
        if self._sel is not None:
            return self._sql_select_subset(queries, return_as_lazyframe, with_dataset_col,
                                           shard_subsample=shard_subsample, rng=rng, shard_pipe=shard_pipe)

        shard_subsample = self._resolve_shard_subsample(shard_subsample)
        wanted = self._queried_columns(queries) - set(_SPECIAL_COLS)
        lazy_frames, skipped = [], []
        for name, ds in zip(self.child_names, self.children):
            if [c for c in wanted if c not in ds.fields]:
                skipped.append(name)
                continue
            child_frac = shard_subsample
            if isinstance(child_frac, float) and 0 < child_frac < 1:
                # make sure the global fraction still picks >=1 shard per child
                child_frac = max(child_frac, min(1.0, 1.5 / max(1, ds.index.n_shards)))
            try:
                lf = ds.sql_select(*queries, return_as_lazyframe=True, shard_subsample=child_frac,
                                   rng=rng, shard_pipe=shard_pipe)
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

    def _sql_select_subset(self, queries, return_as_lazyframe, with_dataset_col,
                           shard_subsample=None, rng=42, shard_pipe=None):
        """Query the offset-index subset: one scan per touched parent over its
        touched shards, restricted by global row index (the restriction is
        skipped when every row of the touched shards is selected)."""
        if shard_pipe is not None:
            raise ValueError("shard_pipe is not supported when querying a subset meta")
        sel = self._sel
        if shard_subsample not in (None, 1):
            if isinstance(rng, int):
                rng = random.Random(rng)
            shards = sel.select("parent", "partition", "shard").unique(maintain_order=True)
            keep = shards.sample(fraction=shard_subsample, seed=rng.randrange(1 << 30))
            sel = sel.join(keep, on=["parent", "partition", "shard"], how="semi")
        frames = []
        for pi, sub in _group_by_parent(sel):
            parent = self.children[pi]
            shards, sdf, keep_g = self._sub_scan(pi, sub)
            lf = parent.sql_select(*queries, shards=shards, shard_subsample=1, return_as_lazyframe=True)
            if sub.height < int(sdf["n"].sum()):
                lf = lf.with_row_index("__g__").filter(pl.col("__g__").is_in(keep_g)).drop("__g__")
            if with_dataset_col:
                lf = lf.with_columns(pl.lit(self.child_names[pi]).alias("__dataset__"))
            frames.append(lf)
        if not frames:
            return pl.DataFrame().lazy() if return_as_lazyframe else pl.DataFrame()
        out = pl.concat(frames, how="diagonal_relaxed")
        return out if return_as_lazyframe else out.collect()

    def sql_filter(self, query, shard_subsample=None, rng=42):
        keys = []
        for name, ds in zip(self.child_names, self.children):
            if any(c not in ds.fields for c in self._queried_columns([query]) - {"__key__"}):
                continue
            try:
                keys.extend(f"{name}::{k}" for k in ds.sql_filter(query, shard_subsample=shard_subsample, rng=rng))
            except WSShardMissingError:
                continue
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

    # ------------------------------------------------------------------------
    # Persistence: one manifest + (subset) one selection.feather. No data copied.
    # ------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta-view.json").write_text(json.dumps({
            "wsds_metaview_version": 4,
            "kind": self.kind,
            "segmented": self.segmented,
            "is_subset": self._sel is not None,
            # n_shards/n_samples fingerprint each parent so `load` can detect
            # a re-indexed/re-sharded parent (saved offsets would silently
            # dereference to different samples)
            "parents": [{"name": n, "root": str(ds.dataset_root),
                         "n_shards": ds.index.n_shards, "n_samples": ds.index.n_samples}
                        for n, ds in zip(self.child_names, self.children)],
        }, indent=2))
        if self._sel is not None:
            self._sel.write_ipc(path / "selection.feather", compression="zstd")
        return path

    @classmethod
    def load(cls, path, rng=None, check_parents=True):
        """Reopen a saved meta/subset. Parents are located via their saved root,
        falling back to a WSDS_DATASET_SEARCH_PATH suffix search when the root
        doesn't exist on this machine.

        With `check_parents` (default), each parent's shard/sample counts must
        match the fingerprint recorded at save time — a mismatch means the
        saved offsets would point at different samples, so loading refuses.
        Pass check_parents=False to override."""
        from .ws_dataset import WSDataset
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        manifest = path / "meta-view.json" if path.is_dir() else path
        spec = json.loads(manifest.read_text())
        parents, names = [], []
        for p in spec["parents"]:
            ds = WSDataset(str(_locate_root(p["root"])))
            if check_parents and "n_samples" in p:
                if ds.index.n_samples != p["n_samples"] or ds.index.n_shards != p["n_shards"]:
                    raise ValueError(
                        f"Parent {p['name']!r} ({ds.dataset_root}) changed since this meta was saved "
                        f"(saved {p['n_shards']} shards / {p['n_samples']} samples, found "
                        f"{ds.index.n_shards} / {ds.index.n_samples}); the saved offsets may point at "
                        f"different samples. Pass check_parents=False to load anyway.")
            parents.append(ds)
            names.append(p["name"])
        sel = None
        if spec.get("is_subset") and (directory / "selection.feather").exists():
            # memory_map=False: the file is zstd-compressed, mmap can't apply
            sel = pl.read_ipc(directory / "selection.feather", memory_map=False)
        return cls(parents, names=names, kind=spec.get("kind"), rng=rng, selection=sel)

    # ------------------------------------------------------------------------
    # Misc / stats
    # ------------------------------------------------------------------------
    def stats(self):
        if self._sel is None:
            return {"is_subset": False, "n_samples": len(self), "n_shards": self.n_shards,
                    "n_children": len(self.children)}
        shards = self._sel.select("parent", "partition", "shard").unique()
        per_lang = (self._sel.group_by("parent").agg(pl.len().alias("n"))
                    .sort("parent"))
        return {
            "is_subset": True,
            "n_selected": self._sel.height,
            "shards_touched": shards.height,
            "per_child": {self.child_names[p]: int(n) for p, n in per_lang.iter_rows()},
        }

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
        mode = f"subset n_selected={self._sel.height}" if self._sel is not None else "base"
        return f"WSMetaDataset(n_children={len(self.children)}, {mode}, kind={self.kind!r})"

    def __str__(self):
        out = repr(self) + "\n"
        if self._sel is not None:
            out += f"  Selected samples: {format(len(self), ',d').replace(',', ' ')}\n"
            for name, n in self.stats()["per_child"].items():
                out += f"            - {name}: {format(n, ',d').replace(',', ' ')} samples\n"
            return out
        out += f"     Audio duration: {format_duration(self.audio_duration)}\n"
        if self.segmented:
            out += f"    Speech duration: {format_duration(self.speech_duration)}\n"
        out += f"   Number of shards: {self.n_shards}\n"
        out += f"  Number of samples: {format(len(self), ',d').replace(',', ' ')}\n"
        out += f"          Children: {len(self.children)}\n"
        for name, ds in zip(self.child_names, self.children):
            out += f"            - {name}: {format(len(ds), ',d').replace(',', ' ')} samples\n"
        return out


def _group_by_parent(sel):
    for (pi,), sub in sel.group_by("parent", maintain_order=True):
        yield int(pi), sub
