import contextlib
import importlib
import itertools
import json
import os
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import polars as pl

from .utils import (
    WSShardMissingError,
    format_duration,
    list_all_columns,
    list_all_shards,
    parse_key,
    scan_ipc,
    validate_shards,
)
from .ws_index import WSIndex
from .ws_sample import WSSample
from .ws_shard import WSShard


class _OpenShardCache:
    """Process-global LRU of open shard handles, keyed by absolute shard path.

    One OrderedDict per WSDataset is no bound at all for a dataset whose partitions link
    elsewhere: a derived dataset over hundreds of catalog partitions (local20-500k: 646 linked
    datasets) gets 646 x the cap, and each handle retains its reader's forward + tail read
    buffers plus, for a WSShard, the current RecordBatch. Keyed by path there is one LRU for the
    whole process and the hot set is what it should be -- concurrent readers x column dirs --
    however many datasets the reads are spread over.

    Entries remember their owning dataset so WSDataset.close() drops just its own.

    PINS make the bound structural instead of tuned. A reader that walks one shard for many
    consecutive samples needs its handles for exactly that visit: see WSDataset.shard_visit,
    which pins what it opens and releases on exit. A pinned entry is never evicted, and
    releasing a visit CLOSES its handles immediately, so memory comes back when a loader lane
    rotates rather than when something else needs the slot. `cap` (WSDS_OPEN_SHARDS, default
    32) then only bounds the unpinned remainder.
    """

    def __init__(self, cap):
        self._d = OrderedDict()          # path -> [shard, owner_id, pin]
        self.cap = max(2, cap)
        self.evictions = 0
        self.pin_overflow = 0            # evictions skipped because every candidate was pinned

    def __len__(self):
        return len(self._d)

    def n_pinned(self):
        return sum(1 for e in self._d.values() if e[2] is not None)

    def get(self, path, default=None):
        e = self._d.get(path)
        return default if e is None else e[0]

    def claim(self, path, owner_id, pin=None):
        """A cache HIT by `owner_id`: LRU-touch the entry, make this dataset its owner (so its
        close() drops what it last used, even if another dataset opened it) and, when the caller
        is inside a shard_visit, pin it -- a visit must own every handle it reads through, not
        just the ones it opened itself. Returns the shard, or None if absent."""
        e = self._d.get(path)
        if e is None:
            return None
        self._d.move_to_end(path)
        e[1] = owner_id
        if pin is not None and e[2] is None:
            e[2] = pin
        return e[0]

    def touch(self, path):
        if path in self._d:
            self._d.move_to_end(path)

    def put(self, path, shard, owner_id, pin=None):
        e = self._d.get(path)
        if e is not None and pin is None:
            pin = e[2]                   # re-opening inside a visit keeps that visit's pin
        self._d[path] = [shard, owner_id, pin]
        self._d.move_to_end(path)
        self._evict(keep=path)

    def _evict(self, keep=None):
        while len(self._d) > self.cap:
            # never the entry being handed out right now (`keep`): when everything else is
            # pinned it would be the only candidate, and closing it before the caller reads it
            # is a crash, not an eviction.
            victim = next((p for p, e in self._d.items() if e[2] is None and p != keep), None)
            if victim is None:           # everything is pinned: exceeding cap beats closing a
                self.pin_overflow += 1   # handle its reader is using. Means cap < lanes x dirs.
                return
            shard = self._d.pop(victim)[0]
            self.evictions += 1
            _close_shard(shard)

    def pin(self, path, token):
        e = self._d.get(path)
        if e is not None:
            e[2] = token

    def release_pin(self, token):
        """End a visit: close and drop the handles it pinned (see shard_visit)."""
        for path in [p for p, e in self._d.items() if e[2] == token]:
            shard = self._d.pop(path)[0]
            _close_shard(shard)

    def pop_owner(self, owner_id):
        """Close and drop every shard opened by this dataset (its close()/eviction)."""
        for path in [p for p, e in self._d.items() if e[1] == owner_id]:
            shard = self._d.pop(path)[0]
            _close_shard(shard)

    def stats(self):
        return dict(open=len(self._d), pinned=self.n_pinned(), cap=self.cap,
                    evictions=self.evictions, pin_overflow=self.pin_overflow)


def _close_shard(victim):
    try:
        victim.close()
    except Exception:
        pass


_OPEN_SHARDS = None
_OWNER_SEQ = itertools.count()      # owner tokens, not id(): a cached _MissingShard / WSS3Shard
                                    # does not keep its dataset alive, so ids get reused


def open_shard_cache():
    """The process-global open-shard LRU (created on first use so the env is read late)."""
    global _OPEN_SHARDS
    if _OPEN_SHARDS is None:
        _OPEN_SHARDS = _OpenShardCache(int(os.environ.get("WSDS_OPEN_SHARDS", "32")))
    return _OPEN_SHARDS


class _MissingShard:
    """Negative-cache entry for `get_shard`: this shard_ref is known-absent in
    this column dir. Mirrors the `.shard_ref` attribute the cache check uses."""

    __slots__ = ("shard_ref", "fname")

    def __init__(self, shard_ref, fname):
        self.shard_ref = shard_ref
        self.fname = fname

    def close(self):                 # so the open-shard LRU / dataset.close() can call it uniformly
        pass


class WSDataset:
    """A multimodal dataset.

    A dataset works like a table (dataframe) of samples. Samples are split into column directories,
    with each column directory storing a subset of columns. Inside these directories are shards, with
    each shard storing a subset of rows. This enables very efficient parallelization of data
    processing.

    This class offers a straightforward way to access the dataset, both sequentially and randomly
    (by key or index) and returns dict-like `hume_wsds.ws_sample.WSSample` objects which transparently and lazily
    load the requested data.

    Examples:
    >>> dataset = WSDataset("librilight/v3-vad_ws", rng=42)
    >>> sample = dataset["large/5304/the_tinted_venus_1408_librivox_64kb_mp3/tintedvenus_05_anstey_64kb_090"]
    >>> print(repr(sample["transcription_wslang_raw.txt"]))
    ' I will accompany you," she said.'
    >>> sample['audio'].load().shape
    torch.Size([1, 42049])
    """

    dataset_root: Path
    """Path to the dataset root directory."""
    fields: dict
    """List of fields available for each sample."""
    computed_columns: dict
    """List of computed columns (e.g. the source audio or video link). @private"""

    def __init__(
        self,
        dataset_root: str | Path,
        include_in_progress: bool = True,
        key_folder: str | None = None,
        ignore_index: bool = False,
        rng: random.Random | int | None = None,
    ):
        self.dataset_root = self._resolve_path(dataset_root)
        # Cached so get_shard_path can build paths via f-string in one shot
        # instead of chaining pathlib `/` operations (10–60× speedup on hot calls).
        self._dataset_root_str = str(self.dataset_root)

        if isinstance(rng, int):
            self.rng = random.Random(rng)
        elif rng is not None:
            self.rng = rng
        else:
            self.rng = random

        if include_in_progress is not True:
            print("NOTE: include_in_progress is deprecated and all subdirs are included by default")
        if key_folder is not None:
            print("NOTE: key_folder is deprecated and key folder is selected automatically")

        self.index = None
        self.segmented = False
        index_file = self.dataset_root / "index.sqlite3"
        if not ignore_index and index_file.exists():
            self.index = WSIndex(index_file)
            meta = self.index.metadata
            self.segmented = meta.get("segmented", False)
        else:
            meta = {}

        if "fields" in meta:
            self.fields = meta["fields"]
        else:
            partition, shard_name = next(self.index.shards()) if self.index else ("", None)
            self.fields = list_all_columns(self.dataset_root / partition, shard_name)

        if "computed_columns" in meta:
            self.computed_columns = meta["computed_columns"]
        else:
            self.computed_columns = {}

        # look for additional columns that are not in the index (like a wsds-link to S3 storage)
        self.fields.update(list_all_columns(self.dataset_root))

        # Normalize old-style single-tuple fields to list-of-tuples
        for k, v in self.fields.items():
            if v and isinstance(v[0], str):
                self.fields[k] = [v]

        self._filter_dfs = None  # mapping of "filter name" -> polars dataframe representing the filter

        # Open-shard cache: a PROCESS-GLOBAL LRU keyed by absolute shard path (see
        # _OpenShardCache). Keyed by path so concurrent reads of several shards in one column
        # dir coexist; global so a dataset that links out to hundreds of catalog datasets cannot
        # multiply the bound. Size it >= active readers x column_dirs via WSDS_OPEN_SHARDS.
        self._open_shards = open_shard_cache()
        self._cache_owner = next(_OWNER_SEQ)
        self._active_visits = {}         # shard_ref -> live visit count (see shard_visit)
        # Linked datasets (one per catalog dir a partition link points at): ~5 KB and one sqlite
        # fd each, ~30 ms to construct, so keeping them all is the right trade for hundreds of
        # them and this LRU is only a safety valve (WSDS_LINKED_DATASETS) for a catalog with
        # orders of magnitude more partitions. Evicting one closes its sqlite index and drops
        # its shards from the global LRU.
        self._linked_datasets = OrderedDict()
        self._max_linked = max(2, int(os.environ.get("WSDS_LINKED_DATASETS", "1024")))
        self._partition_links = {}
        # column-dirs tuple -> set of shard refs that already passed validate_shards
        self._validated_shards: dict[tuple[str, ...], set] = {}

        self._register_wsds_links()

    def close(self):
        """Close all cached shard file handles and linked datasets."""
        self._open_shards.pop_owner(self._cache_owner)
        for ds in self._linked_datasets.values():
            ds.close()
        self._linked_datasets.clear()
        self._validated_shards.clear()

    def enable_filter(self, filter_name: str, filter_df: pl.DataFrame):
        """
        Enabling a filter adds extra columns to the dataset, each column representing a filter.
        """
        assert self._filter_dfs is None or filter_name not in self._filter_dfs, "Filter already enabled"
        assert len(filter_df.columns) == 1, f"Filter must have exactly one column, got {len(filter_df.columns)}"
        assert filter_df.dtypes[0] == pl.Boolean, f"Filter must have a boolean column, got {filter_df.dtypes[0]}"

        if self._filter_dfs is None:
            self._filter_dfs = dict()

        self._filter_dfs[filter_name] = filter_df

        rows_satisfying_filter = filter_df.sum().item()
        print(
            f"Filter enabled on dataset {repr(self)}. Rows satisfying the filter: {rows_satisfying_filter} / {len(filter_df)}"
        )

    #
    # Accessing samples randomly and sequentially
    #
    def random_sample(self):
        """Returns one random sample.

        Example:
        >>> dataset = WSDataset('librilight/v3-vad_ws')
        >>> sample = dataset.random_sample()
        >>> 'transcription_wslang_raw.txt' in sample
        True
        """
        assert self.index is not None, "Random access is only supported for indexed datasets"
        return self[self.rng.randrange(self.index.n_samples)]

    def __iter__(self):
        """Starts at a random position in the dataset and yields samples sequentially.
        Once it reaches the end of a shard it will jump to a new random position.

        @public
        """
        while True:
            yield from self.sequential_from(self.random_sample())

    def random_samples(self, N: int = 1):
        """Yields N random samples (not sequential)."""
        for _ in range(N):
            yield self.random_sample()

    def random_chunks(self, max_N: int):
        """Like `__iter__`, but jumps to a random position after yielding `max_N` samples."""
        while True:
            yield from self.sequential_from(self.random_sample(), max_N=max_N)

    def __getitem__(self, key_or_index: str | int):
        """Returns a sample with the given __key__ or sample index."""
        if isinstance(key_or_index, int):
            r = self.index.lookup_by_index(key_or_index)
            if not r:
                return None
            partition, shard_name, local_offset = r
            global_offset = key_or_index
        elif isinstance(key_or_index, str):
            file_name, offset_of_key_wrt_file = self.parse_key(key_or_index)
            r = self.index.lookup_by_key(file_name, offset_of_key_wrt_file)
            if not r:
                return None
            partition, shard_name, local_offset, global_offset = r
        else:
            raise TypeError(f"Invalid key type: {type(key_or_index)}")

        overrides = dict()
        if self._filter_dfs is not None:
            overrides.update(
                {filter_name: filter_df.row(global_offset)[0] for filter_name, filter_df in self._filter_dfs.items()}
            )
        return WSSample(self, (partition, shard_name), local_offset, overrides=overrides)

    def sequential_from(self, sample, max_N=None):
        """Yields samples sequentially from the given `sample`, stopping after `max_N` samples."""
        shard_ref, i = sample.shard_ref, sample.offset
        max_N = min(i + (max_N or sys.maxsize), self._shard_n_samples(shard_ref))
        # without an index, we still return the sample but you'll get an error on first field access

        shard_global_offset = None
        if self._filter_dfs is not None:
            # We need to know the global shard offset to know what filter values to use for the sample
            shard_global_offset = self.index.shard_global_offset(shard_ref)

        while i < max_N:
            sample = WSSample(self, shard_ref, i)
            if self.index is None:
                # if we don't have an index we have to try loading
                # the sample to check if it exists
                try:
                    sample["__key__"]
                except IndexError:
                    return
            if self._filter_dfs is not None:
                # TODO: treat this as just another (unsharded) column
                for filter_name, filter_df in self._filter_dfs.items():
                    sample[filter_name] = filter_df.row(shard_global_offset + i)[0]
            yield sample
            i += 1

    @contextlib.contextmanager
    def shard_visit(self, shard_ref):
        """Own this shard's open handles for the duration of the block, then release them.

        A reader that walks one shard for many consecutive samples (`sequential_from`) hits the
        same handles over and over, then never comes back. Inside this block every shard opened
        for `shard_ref` is PINNED: never evicted however much else the process opens, and closed
        on exit instead of lingering until some later open needs the slot. The resident handle
        count is then readers x column dirs by construction.

            with ds.shard_visit(sref):
                for sample in ds.sequential_from(WSSample(ds, sref, off)):
                    ...

        Re-entrant across concurrent readers: two visits to the same shard_ref refcount, and the
        handles are released when the last one exits. Outside a visit nothing changes -- handles
        are cached and evicted LRU exactly as before.
        """
        key = (self._cache_owner, shard_ref)
        self._active_visits[shard_ref] = self._active_visits.get(shard_ref, 0) + 1
        try:
            yield self
        finally:
            n = self._active_visits.get(shard_ref, 0) - 1
            if n > 0:
                self._active_visits[shard_ref] = n
            else:
                self._active_visits.pop(shard_ref, None)
                self._open_shards.release_pin(key)

    def _visit_token(self, shard_ref):
        return (self._cache_owner, shard_ref) if shard_ref in self._active_visits else None

    def _shard_n_samples(self, shard_ref: (str, str)) -> int:
        if not self.index:
            return sys.maxsize
        return self.index.shard_n_samples(shard_ref)

    def iter_shard(self, shard_ref):
        partition, shard_name = shard_ref
        if shard_name.endswith(".wsds"):
            shard_name = shard_name[:-5]
        return self.sequential_from(WSSample(self, (partition, shard_name), 0))

    def __len__(self):
        """Returns the number of samples in the dataset.

        @public"""
        assert self.index is not None, "Length is only known for indexed datasets"
        return self.index.n_samples

    #
    # SQL support, using Polars
    #
    def _parse_sql_queries_polars(
        self, *queries, shard_subsample=1, rng=None, shard_pipe=None, key_column=None, shard_filter=None
    ):
        """Parses SQL queries via Polars to:
        - extract the Polars expressions for each query
        - use the expressions to build a list of column dirs to load shards from

        `key_column` anchors `__key__`/`__shard_path__`/`__shard_offset__` extraction
        (and shard validation) to the column dir containing that column, without
        reading the column itself. `shard_filter` restricts the scan to shards for
        which `shard_filter((partition, shard_name))` is true."""

        column_dirs = defaultdict(list)
        exprs = []
        needed_special_columns = []
        for query in queries:
            if "." in query and query in self.fields:
                print(f"TIP: You seem to have passes a column name ({query}) which has dots in it.")
                query = f"`{query}`"
                print(
                    f"We expect to get SQL expressions which requires quoting such names, in this cases it should likely be: {query}"
                )
                print(
                    "I fixed it for you in this simple case but am not smart enough to do it in real SQL expressions."
                )

            expr = pl.sql_expr(query)
            for col in expr.meta.root_names():
                if col == "__key__" or col == "__shard_path__" or col == "__shard_offset__":
                    # __key__ exists in all shards
                    needed_special_columns.append(col)
                    continue
                column_dir, field = self.fields[col][0]
                # Check if this is a computed/remote column (e.g., source-linked or S3-backed field)
                if column_dir in self.computed_columns:
                    raise ValueError(
                        f"Column '{col}' is a computed/remote column and cannot be used in SQL queries. "
                        f"Use sample['{col}'] to access it instead."
                    )
                assert col == field, "renamed fields are not supported in SQL queries yet"
                column_dirs[column_dir].append(field)
            exprs.append(expr)

        # If only __key__ is in the query, we need to load shards from at least one column_dir
        if key_column is not None:
            (key_column_dir, _column) = self.fields[key_column][0]
        else:
            (key_column_dir, _column) = self.fields["__key__"][0]
            if needed_special_columns and column_dirs:
                key_column_dir = list(column_dirs.keys())[0]
        if needed_special_columns:
            column_dirs[key_column_dir] += needed_special_columns

        if rng is None:
            rng = self.rng
        shard_list = self.get_shard_list()
        if shard_filter is not None:
            shard_list = [s for s in shard_list if shard_filter(s)]
        if shard_subsample != 1:
            shard_list = rng.sample(shard_list, int(len(shard_list) * shard_subsample))

        # Prefetch shard tails concurrently to warm up the filesystem cache
        verified_shard_list = validate_shards(self, shard_list, list(column_dirs.keys()))

        # Fast path: when no shard_pipe, no __shard_offset__, and every shard is
        # valid in every col_dir, we can collapse the per-shard scan loop into
        # one multi-file `pl.scan_ipc(paths_list)` per col_dir. The plan goes
        # from ~N_shards × N_col_dirs scan nodes to N_col_dirs — the polars
        # optimizer scales super-linearly with plan size, so this is ~2× faster
        # on wide workloads. Falls back to the per-shard path below when any
        # condition fails.
        if (
            shard_pipe is None
            and "__shard_offset__" not in needed_special_columns
            and verified_shard_list
            and all(ok for _, ok in verified_shard_list)
        ):
            return exprs, self._build_multifile_plan(
                [s for s, _ in verified_shard_list],
                column_dirs,
                exprs,
                key_column_dir,
                "__shard_path__" in needed_special_columns,
            )

        row_merge = []
        column_dir_samples = {}
        missing = defaultdict(list)
        for shard_ref, shard_ok in verified_shard_list:
            col_merge = []
            for column_dir, fields in column_dirs.items():
                shard_path = self.get_shard_path(column_dir, shard_ref)
                if shard_ok:
                    df = scan_ipc(
                        shard_path,
                        glob=False,
                        include_file_paths="__shard_path__" if column_dir == key_column_dir else None,
                        row_index_name="__shard_offset__" if column_dir == key_column_dir else None,
                    ).select(fields)
                    if column_dir not in column_dir_samples:
                        column_dir_samples[column_dir] = df.clear().collect()
                else:
                    # create a fake dataframe with all NULL rows and matching schema
                    if self.index:
                        n_samples = self.index.shard_n_samples(shard_ref)
                        df = pl.defer(
                            lambda column_dir=column_dir, n_samples=n_samples: column_dir_samples[column_dir].clear(
                                n=n_samples
                            ),
                            schema=lambda column_dir=column_dir: column_dir_samples[column_dir].schema,
                        )
                    else:
                        df = None
                    missing[column_dir].append(shard_ref)
                if df is not None:
                    col_merge.append(df)
            if col_merge:
                merged = pl.concat(col_merge, how="horizontal").select(exprs)
                if shard_pipe:
                    merged = merged.pipe(shard_pipe)
                row_merge.append(merged)

        if missing:
            filled = " (filled them with NULLs)" if self.index else " (skipped them)"
            print(f"WARNING: You are missing or invalid shards for some of the columns{filled}:")
            for column_dir, shards in missing.items():
                msg = f"{column_dir}: {shards[:10]}"
                if len(shards) > 10:
                    msg += f" ... ({len(shards) - 10} more)"
                print(msg)
            if not row_merge:
                raise WSShardMissingError(
                    f"No usable shards found (columns: {', '.join(column_dirs)}) for dataset in: {str(self.dataset_root)}"
                )

        return exprs, pl.concat(row_merge)

    def _build_multifile_plan(self, shards, column_dirs, exprs, key_column_dir, needs_shard_path):
        """Fast path for `_parse_sql_queries_polars`: one `pl.scan_ipc(paths_list)`
        per col_dir, horizontally concatenated. See the comment in the caller
        for why this is faster than the per-shard scan loop.

        Pre-conditions enforced by the caller:
        - all shards are valid in every col_dir (no `pl.defer` NULL-fill path)
        - no per-shard `shard_pipe` (semantics would be lost across the union)
        - `__shard_offset__` is not requested (multi-file scan's row index is
          global rather than per-file)
        """
        per_dir_frames = []
        for column_dir, fields in column_dirs.items():
            paths = [self.get_shard_path(column_dir, s) for s in shards]
            is_key_dir = column_dir == key_column_dir
            # `__shard_path__` is synthesized by polars via `include_file_paths`;
            # strip it (and any stray `__shard_offset__`) from the file-column
            # select list and re-add `__shard_path__` after the scan if needed.
            select_fields = [f for f in fields if f not in ("__shard_path__", "__shard_offset__")]
            df = pl.scan_ipc(
                paths,
                include_file_paths="__shard_path__" if (is_key_dir and needs_shard_path) else None,
            )
            if is_key_dir and needs_shard_path:
                select_fields = select_fields + ["__shard_path__"]
            per_dir_frames.append(df.select(select_fields))
        return pl.concat(per_dir_frames, how="horizontal").select(exprs)

    def _check_for_subsampling(self, shard_subsample):
        if shard_subsample is None:
            # Check if we're running inside a PyTorch DataLoader worker
            try:
                import torch.utils.data as torch_data

                worker_info = torch_data.get_worker_info()
                if worker_info is not None:
                    print("\n" + "=" * 80)
                    print("WARNING: wsds is running in subsampling modee inside a PyTorch DataLoader!")
                    print("Each worker will only load the same small subset of shards by default!")
                    print("This is probably not what you want, so we abort.")
                    print("")
                    print("To fix this, explicitly pass shard_subsample=1 to the WSDataset constructor.")
                    print("=" * 80 + "\n")
                    raise ValueError("WSDataset was used in a dataloader without an explicit subsampling config")
            except ImportError:
                pass  # torch not installed

            if not self.index or self.index.n_shards < 150:
                shard_subsample = 1
            else:
                shard_subsample = 150 / self.index.n_shards
                if not hasattr(self, "_shown_subsampling_info"):
                    print(
                        f"INFO: to speed things up wsds is loading a random {shard_subsample * 100:.2f}% subset of the shards, pass shard_subsample=1 to force it to load the whole dataset"
                    )
                    self._shown_subsampling_info = True
        return shard_subsample

    def sql_select(
        self,
        *queries,
        return_as_lazyframe=False,
        shard_subsample=None,
        rng=42,
        shard_pipe=None,
        key_column=None,
        shard_filter=None,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Given a list of SQL expressions, returns a Polars DataFrame/ LazyFrame with the results.

        `key_column` anchors `__key__` (and shard validation) to the column dir holding
        that column — pass a column from a known-complete dir when others are in-progress.
        `shard_filter((partition, shard_name)) -> bool` restricts which shards are scanned."""
        if isinstance(rng, int):
            rng = random.Random(rng)
        exprs, df = self._parse_sql_queries_polars(
            *queries,
            shard_subsample=self._check_for_subsampling(shard_subsample),
            rng=rng,
            shard_pipe=shard_pipe,
            key_column=key_column,
            shard_filter=shard_filter,
        )

        if return_as_lazyframe:
            return df

        return df.collect()

    def sql_filter(self, query, shard_subsample=None, rng=42):
        """Given a boolean SQL expression, returns a list of keys for samples that match the query."""
        if isinstance(rng, int):
            rng = random.Random(rng)

        exprs, df = self._parse_sql_queries_polars(
            query, "__key__", shard_subsample=self._check_for_subsampling(shard_subsample), rng=rng
        )
        return df.filter(pl.first()).select("__key__").filter(pl.col("__key__").is_not_null()).collect()["__key__"]

    def filtered(
        self,
        query,
        infinite: bool = False,  # keep yielding samples indefinitely (restarting from the beginning)
        shuffle: bool = True,  # shuffle the sample order (otherwise it will return them as they appear in the dataset)
        N: int = None,  # optional maximum number of samples to yield (otherwise it will yield all matching samples)
        seed: int = None,  # optional random seed used shuffling
        shard_subsample=None,
        rng=42,
    ):
        """Given an boolean SQL expression, returns an iterator which yields random samples
        that match the query.

        Examples:
        >>> dataset = WSDataset("librilight/v3-vad_ws")
        >>> next(dataset.filtered('pq < 3', shuffle=False, shard_subsample=1))['__key__']  # first low-quality sample
        'large/6454/over_plum_pudding_1305_librivox_64kb_mp3/plumpudding_09_bangs_64kb_072'
        >>> next(dataset.filtered("CAST(`transcription_wslang_raw.txt` AS string) ILIKE '%between New Orleans%'", shuffle=False, shard_subsample=1))['__key__']
        'large/10244/carpentersna_1612_librivox_64kb_mp3/geographicalreaderna_40_carpenter_64kb_034'
        """
        import polars as pl

        i = 0
        keys = self.sql_filter(query, shard_subsample=shard_subsample, rng=rng)
        self.last_query_n_samples = len(keys)
        while True:
            if N is None:
                if shuffle:
                    keys = keys.sample(fraction=1, shuffle=shuffle, seed=seed)
            else:
                keys = keys.sample(n=pl.len().clip(0, N), shuffle=shuffle, seed=seed)
            for key in keys:
                yield self[key]
                i += 1
                if N is not None and i >= N:
                    return
            if not infinite:
                break

    #
    # Helper and internal API
    #
    def _resolve_path(self, path_str: str) -> Path:
        """If the 'path' is relative and does not exist, we search for it using 'WSDS_DATASET_SEARCH_PATH' env var.
        WSDS_DATASET_SEARCH_PATH is a colon-separated list of directories where datasets are stored.

        Example:
            WSDS_DATASET_SEARCH_PATH=/path/to/datasets:/another/path/to/datasets"""

        path = Path(path_str)
        if path.is_absolute() or path.exists():
            return path

        for base_path_str in os.environ.get("WSDS_DATASET_SEARCH_PATH", "").split(":"):
            base_path = Path(base_path_str)
            if (base_path / path).exists():
                return base_path / path

        raise ValueError(f"Dataset {repr(str(path))} not found.")

    def get_shard_list(self, ignore_index=False):
        if not ignore_index and self.index:
            return list(self.index.shards())
        else:
            return list_all_shards(self.dataset_root)

    def get_shard_path(self, column_dir, shard_ref):
        partition, shard_name = shard_ref
        # Strip any existing extension, matching the old `.with_suffix(".wsds")` behavior.
        if "." in shard_name:
            shard_name = shard_name.rsplit(".", 1)[0]
        return f"{self._dataset_root_str}/{partition}/{column_dir}/{shard_name}.wsds"

    def _get_loader_class(self, spec: dict):
        """Get the loader class from a link spec."""
        loader_class = spec["loader"]
        if isinstance(loader_class, list):
            loader_mod, loader_name = loader_class
            if loader_mod.startswith("hume_wsds."):
                loader_mod = "wsds." + loader_mod[len("hume_wsds.") :]
            loader_module = importlib.import_module(loader_mod)
            return getattr(loader_module, loader_name)
        return loader_class

    def _register_wsds_links(self):
        # Collect links first to avoid modifying dict during iteration
        links_to_register = []
        for value in self.fields.values():
            (column_dir, _column) = value[0]
            if column_dir.endswith(".wsds-link"):
                spec = json.loads((self.dataset_root / column_dir).read_text())
                self.computed_columns[column_dir] = spec
                links_to_register.append((column_dir, spec))

        # Ask each loader class what columns it provides
        for link_file, spec in links_to_register:
            loader_class = self._get_loader_class(spec)
            columns = loader_class.get_columns(spec, self)

            if columns:
                # Loader provides multiple columns - register them all
                for col_name in columns:
                    self.fields[col_name] = [(link_file, col_name)]

    def add_computed(self, name, **link):
        column_dir = name + ".wsds-computed"
        self.computed_columns[column_dir] = link
        self.fields[name] = [(column_dir, name)]

    def _partition_link(self, column_dir, shard_ref):
        """A `<column>.wsds-link` placed INSIDE a partition folder overrides the dataset-root
        link for the shards of that partition. Lets each partition of a derived dataset
        name where its own column comes from (e.g. the matching partition of a catalog of
        original deliveries) without a dataset-wide merged index. `dataset_dir` in the spec
        is relative to the partition folder and is resolved here."""
        partition = shard_ref[0] if shard_ref else ""
        if not partition or not column_dir.endswith(".wsds-link"):
            return None
        key = (partition, column_dir)
        if key not in self._partition_links:
            path = self.dataset_root / partition / column_dir
            spec = None
            if path.exists():
                spec = json.loads(path.read_text())
                if "dataset_dir" in spec:
                    spec["dataset_dir"] = str((path.parent / spec["dataset_dir"]).resolve())
            self._partition_links[key] = spec
        return self._partition_links[key]

    def get_linked_dataset(self, relative_path):
        linked_root = self.dataset_root / relative_path
        ds = self._linked_datasets.get(linked_root)
        if ds is None:
            ds = self._linked_datasets[linked_root] = WSDataset(linked_root)
            while len(self._linked_datasets) > self._max_linked:
                _root, victim = self._linked_datasets.popitem(last=False)   # least-recently-used
                try:
                    victim.close()          # closes its sqlite index + drops its shards from the
                except Exception:           # global LRU; a hot shard is re-opened on next touch
                    pass
        else:
            self._linked_datasets.move_to_end(linked_root)
        return ds

    def get_linked_shard(self, link, shard_ref):
        loader_class = self._get_loader_class(link)
        return loader_class.from_link(link, self, shard_ref)

    def get_shard(self, column_dir, shard_ref):
        shard_path = self.get_shard_path(column_dir, shard_ref)   # unique per shard_ref+column_dir

        shard = self._open_shards.claim(shard_path, self._cache_owner, self._visit_token(shard_ref))
        if shard is not None:
            if isinstance(shard, _MissingShard):
                raise WSShardMissingError(shard.fname)
            return shard

        if column_dir in self.computed_columns:
            spec = self._partition_link(column_dir, shard_ref) or self.computed_columns[column_dir]
            shard = self.get_linked_shard(spec, shard_ref)
        else:
            try:
                shard = WSShard(self, shard_path, shard_ref=shard_ref)
            except WSShardMissingError:
                # Negative cache: without it, every sample access whose field
                # has an alternative in a missing/partial column dir (e.g.
                # `__key__` living in every dir, incl. *.in-progress ones)
                # retries the open -- an ENOENT path-resolution RPC per access
                # on network filesystems.
                self._open_shards.put(shard_path, _MissingShard(shard_ref, shard_path),
                                      self._cache_owner, self._visit_token(shard_ref))
                raise

        self._open_shards.put(shard_path, shard, self._cache_owner, self._visit_token(shard_ref))
        return shard

    def get_sample(self, shard_ref, field, offset, raw=False):
        alternatives = self.fields[field]
        if len(alternatives) > 1:
            # A field replicated across column dirs (e.g. __key__ lives in every
            # one) reads the same value wherever it comes from, so prefer a dir
            # whose shard is already open for this shard_ref. The alternatives
            # list is sorted smallest-shard-first, which today means half-built
            # *.in-progress dirs sort to the front -- walking it in order opens
            # an extra column dir per shard visit for no reason.
            for column_dir, column in alternatives:
                path = self.get_shard_path(column_dir, shard_ref)
                shard = self._open_shards.get(path)
                if shard is not None and not isinstance(shard, _MissingShard):
                    self._open_shards.claim(path, self._cache_owner, self._visit_token(shard_ref))
                    try:
                        return shard.get_sample(column, offset, raw=raw)
                    except (WSShardMissingError, KeyError):
                        break  # fall back to the ordered walk below
        last_err = None
        for column_dir, column in alternatives:
            try:
                return self.get_shard(column_dir, shard_ref).get_sample(column, offset, raw=raw)
            except WSShardMissingError as e:
                last_err = e
                continue
        raise last_err

    def parse_key(self, key):
        if self.segmented:
            return parse_key(key)
        else:
            return key, 0

    def __str__(self):
        out = ""
        out += repr(self) + "\n"
        if self.index is None:
            return out
        out += f"     Audio duration: {format_duration(self.index.audio_duration)}\n"
        if self.segmented:
            out += f"    Speech duration: {format_duration(self.index.speech_duration)}\n"
        out += f"   Number of shards: {self.index.n_shards}\n"
        out += f"  Number of samples: {format(len(self), ',d').replace(',', ' ')}\n"
        return out

    def __repr__(self):
        if self.index is None:
            return f"WSDataset({repr(str(self.dataset_root))}, segmented={self.segmented}, index=None)"
        return f"WSDataset({repr(str(self.dataset_root))}, segmented={self.segmented})"

    def _display_(self):
        import marimo

        if self.index is None:
            return marimo.md(f"```python\n{self.__str__()}\n```\n")

        return marimo.vstack(
            [
                marimo.md(f"```python\n{self.__str__()}\n```\n### One sample:\n"),
                self.random_sample()._display_(),
            ]
        )

    def _ipython_display_(self):
        from .utils import is_notebook

        if not is_notebook():
            print(str(self))
            return

        from IPython.display import Markdown, display

        if self.index is None:
            display(Markdown(f"```python\n{self.__str__()}\n```"))
            return

        display(Markdown(f"```python\n{self.__str__()}\n```\n### One sample:"))
        self.random_sample()._ipython_display_()
