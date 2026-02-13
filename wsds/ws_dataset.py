import importlib
import json
import os
import random
import sys
from collections import defaultdict
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
    >>> dataset = WSDataset("librilight/v3-vad_ws")
    >>> sample = dataset["large/5304/the_tinted_venus_1408_librivox_64kb_mp3/tintedvenus_05_anstey_64kb_090"]
    >>> print(repr(sample["transcription_wslang_raw.txt"]))
    ' I will accompany you," she said.'
    >>> sample['audio']
    WSAudio(audio_reader=AudioReader(src=<class 'pyarrow.lib.BinaryScalar'>, sample_rate=None), tstart=1040.2133, tend=1042.8413)
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
        disable_memory_map: bool = False,
        ignore_index: bool = False,
        rng: random.Random | int | None = None,
    ):
        self.dataset_root = self._resolve_path(dataset_root)

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
        self.disable_memory_map = disable_memory_map
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

        self._open_shards = {}
        self._linked_datasets = {}

        self._register_wsds_links()

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
    def _parse_sql_queries_polars(self, *queries, shard_subsample=1, rng=None, shard_pipe=None):
        """Parses SQL queries via Polars to:
        - extract the Polars expressions for each query
        - use the expressions to build a list of column dirs to load shards from"""

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
        (key_column_dir, _column) = self.fields["__key__"][0]
        if needed_special_columns:
            if column_dirs:
                key_column_dir = list(column_dirs.keys())[0]
            column_dirs[key_column_dir] += needed_special_columns

        if rng is None:
            rng = self.rng
        shard_list = self.get_shard_list()
        if shard_subsample != 1:
            shard_list = rng.sample(shard_list, int(len(shard_list) * shard_subsample))

        # Prefetch shard tails concurrently to warm up the filesystem cache
        verified_shard_list = validate_shards(self, shard_list, list(column_dirs.keys()))

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
    ) -> pl.DataFrame | pl.LazyFrame:
        """Given a list of SQL expressions, returns a Polars DataFrame/ LazyFrame with the results."""
        if isinstance(rng, int):
            rng = random.Random(rng)
        exprs, df = self._parse_sql_queries_polars(
            *queries,
            shard_subsample=self._check_for_subsampling(shard_subsample),
            rng=rng,
            shard_pipe=shard_pipe,
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
        return df.filter(exprs[0]).select("__key__").filter(pl.col("__key__").is_not_null()).collect()["__key__"]

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
        >>> next(dataset.filtered('pq < 3', shuffle=False))['__key__']  # first low-quality sample
        'large/6454/over_plum_pudding_1305_librivox_64kb_mp3/plumpudding_09_bangs_64kb_072'
        >>> next(dataset.filtered("CAST(`transcription_wslang_raw.txt` AS string) ILIKE '%between New Orleans and St. Louis%'", shuffle=False))['__key__']
        'large/107/oldtimes_jg_librivox_64kb_mp3/oldtimesonthemississippi_07_twain_64kb_032'
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
        dir = self.dataset_root / partition / column_dir
        return (Path(dir) / shard_name).with_suffix(".wsds")

    def _get_loader_class(self, spec: dict):
        """Get the loader class from a link spec."""
        loader_class = spec["loader"]
        if isinstance(loader_class, list):
            loader_mod, loader_name = loader_class
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

    def get_linked_dataset(self, relative_path):
        linked_root = self.dataset_root / relative_path
        if linked_root not in self._linked_datasets:
            self._linked_datasets[linked_root] = WSDataset(linked_root)
        return self._linked_datasets[linked_root]

    def get_linked_shard(self, link, shard_ref):
        loader_class = self._get_loader_class(link)
        return loader_class.from_link(link, self, shard_ref)

    def get_shard(self, column_dir, shard_ref):
        shard_path = self.get_shard_path(column_dir, shard_ref)

        shard = self._open_shards.get(shard_path.parent, None)
        if shard is not None and shard.shard_ref == shard_ref:
            return shard

        if column_dir in self.computed_columns:
            shard = self.get_linked_shard(self.computed_columns[column_dir], shard_ref)
        else:
            shard = WSShard(self, shard_path, shard_ref=shard_ref)

        self._open_shards[shard_path.parent] = shard
        return shard

    def get_sample(self, shard_ref, field, offset):
        alternatives = self.fields[field]
        last_err = None
        for column_dir, column in alternatives:
            try:
                return self.get_shard(column_dir, shard_ref).get_sample(column, offset)
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
