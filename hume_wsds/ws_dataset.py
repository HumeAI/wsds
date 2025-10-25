import importlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

from hume_wsds.utils import list_all_columns, list_all_shards, parse_key, scan_ipc
from hume_wsds.ws_index import WSIndex
from hume_wsds.ws_sample import WSSample
from hume_wsds.ws_shard import WSShard


class WSDataset:
    """A multimodal dataset.

    A dataset works like a table (dataframe) of samples. Samples are split into directories,
    with each directory storing a subset of columns. Inside these directories are shards, with
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

    dataset_dir: Path
    """Path to the dataset directory."""
    fields: dict
    """List of fields available for each sample."""
    computed_columns: dict
    """List of computed columns (e.g. the source audio or video link). @private"""

    # FIXME: this should be overridable with metadata in index.sqlite3
    _audio_file_keys = ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma", "opus", "audio"]

    def __init__(self, dataset_dir: str):
        self.dataset_dir = self._resolve_path(dataset_dir)

        self.index = None
        self.segmented = False
        index_file = self.dataset_dir / "index.sqlite3"
        if index_file.exists():
            self.index = WSIndex(index_file)
            self.segmented = self.index.metadata.get("segmented", False)

        self.fields = list_all_columns(self.dataset_dir, next(self.index.shards()) if self.index else None)
        self.computed_columns = {}

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
        return self[random.randrange(self.index.n_samples)]

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
        # Figure out the shard name, local offset (wrt shard) and global offset for the given key or index
        shard_name, local_offset, global_offset = None, None, None

        if isinstance(key_or_index, int):
            r = self.index.query(
                "SELECT s.shard, global_offset FROM shards AS s WHERE s.global_offset <= ? ORDER BY s.global_offset DESC LIMIT 1",
                key_or_index,
            ).fetchone()
            if not r:
                return None

            shard_name, shard_global_offset = r
            global_offset = key_or_index
            local_offset = global_offset - shard_global_offset
        elif isinstance(key_or_index, str):
            # FIXME: push `parse_key` to the index class
            file_name, offset_of_key_wrt_file = self.parse_key(key_or_index)
            r = self.index.query(
                "SELECT s.shard, s.global_offset, f.offset FROM files AS f, shards AS s WHERE f.name = ? AND s.shard_id == f.shard_id",
                file_name,
            ).fetchone()
            if not r:
                return None

            shard_name, shard_global_offset, file_offset_in_shard = r
            local_offset = file_offset_in_shard + offset_of_key_wrt_file
            global_offset = shard_global_offset + local_offset
        else:
            raise TypeError(f"Invalid key type: {type(key_or_index)}")

        overrides = dict()
        if self._filter_dfs is not None:
            overrides.update(
                {filter_name: filter_df.row(global_offset)[0] for filter_name, filter_df in self._filter_dfs.items()}
            )
        return WSSample(self, shard_name, local_offset, overrides=overrides)

    def sequential_from(self, sample, max_N=None):
        """Yields samples sequentially from the given `sample`, stopping after `max_N` samples."""
        shard_name, i = sample.shard_name, sample.offset
        max_N = min(i + (max_N or sys.maxsize), self._shard_n_samples(shard_name))
        # without an index, we still return the sample but you'll get an error on first field access

        shard_global_offset = None
        if self._filter_dfs is not None:
            # We need to know the global shard offset to know what filter values to use for the sample
            shard_global_offset = self.index.query(
                "SELECT global_offset FROM shards WHERE shard = ?", shard_name
            ).fetchone()[0]

        while i < max_N:
            overrides = dict()
            if self._filter_dfs is not None:
                overrides.update(
                    {
                        filter_name: filter_df.row(shard_global_offset + i)[0]
                        for filter_name, filter_df in self._filter_dfs.items()
                    }
                )
            yield WSSample(self, shard_name, i, overrides=overrides)
            i += 1

    def _shard_n_samples(self, shard_name: str) -> int:
        if not self.index:
            return sys.maxsize
        return self.index.query("SELECT n_samples FROM shards WHERE shard = ?", shard_name).fetchone()[0]

    #
    # SQL support, using Polars
    #
    def _parse_sql_queries_polars(self, *queries):
        """Parses SQL queries via Polars to:
        - extract the Polars expressions for each query
        - use the expressions to build a list of subdirs to load shards from"""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for SQL queries")
        subdirs = set()
        exprs = []
        for query in queries:
            expr = pl.sql_expr(query)
            for col in expr.meta.root_names():
                if col == "__key__":
                    # __key__ exists in all shards
                    continue
                subdir, field = self.fields[col]
                assert col == field, "renamed fields are not supported in SQL queries yet"
                subdirs.add(subdir)
            # If only __key__ is in the query, we need to load shards from at least one subdir
            if not subdirs and "__key__" in expr.meta.root_names():
                subdirs.add(self.fields["__key__"][0])
            exprs.append(expr)

        row_merge = []
        subdir_samples = {}
        missing = defaultdict(list)
        for shard in self.get_shard_list():
            col_merge = []
            for subdir in subdirs:
                try:
                    df = scan_ipc(self.get_shard(subdir, shard).fname, glob=False)
                    if len(col_merge) > 0:
                        df = df.drop("__key__")  # ensure only one __key__ column
                    if subdir not in subdir_samples:
                        subdir_samples[subdir] = df.clear().collect()
                except FileNotFoundError:
                    # create a fake dataframe with all NULL rows and matching schema
                    n_samples = self.index.query("SELECT n_samples FROM shards WHERE shard=?", shard).fetchone()[0]
                    df = pl.defer(
                        lambda subdir=subdir, n_samples=n_samples: subdir_samples[subdir].clear(n=n_samples),
                        schema=lambda subdir=subdir: subdir_samples[subdir].schema,
                    )
                    missing[subdir].append(shard)
                col_merge.append(df)
            if col_merge:
                row_merge.append(pl.concat(col_merge, how="horizontal"))

        if missing:
            print("WARNING: You are missing shards for some of the columns (filled them with NULLs):")
            for subdir, shards in missing.items():
                print(f"{subdir}: {shards}")
            if not row_merge:
                raise FileNotFoundError(
                    f"No usable shards found (columns: {', '.join(subdirs)}) for dataset in: {str(self.dataset_dir)}"
                )

        return exprs, pl.concat(row_merge)

    def sql_select(self, *queries):
        """Given a list of SQL expressions, returns a Polars DataFrame with the results."""
        exprs, df = self._parse_sql_queries_polars(*queries)
        return df.select(exprs).collect()

    def sql_filter(self, query):
        """Given a boolean SQL expression, returns a list of keys for samples that match the query."""
        exprs, df = self._parse_sql_queries_polars(query)
        return df.filter(exprs[0]).select("__key__").filter(pl.col("__key__").is_not_null()).collect()["__key__"]

    def filtered(
        self,
        query,
        infinite: bool = False,  # keep yielding samples indefinitely (restarting from the beginning)
        shuffle: bool = True,  # shuffle the sample order (otherwise it will return them as they appear in the dataset)
        N: int = None,  # optional maximum number of samples to yield (otherwise it will yield all matching samples)
        seed: int = None,  # optional random seed used shuffling
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
        keys = self.sql_filter(query)
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
        """If the 'path' is relative and does not exist, we search for it using 'WSDS_DATASET_PATH' env var.
        WSDS_DATASET_PATH is a colon-separated list of directories where datasets are stored.

        Example:
            WSDS_DATASET_PATH=/path/to/datasets:/another/path/to/datasets"""
        path = Path(path_str)
        if path.is_absolute() or path.exists():
            return path

        for base_path_str in os.environ.get("WSDS_DATASET_PATH", "").split(":"):
            base_path = Path(base_path_str)
            if (base_path / path).exists():
                return base_path / path

        raise ValueError(f"Dataset {repr(str(path))} not found.")

    def get_shard_list(self):
        if self.index:
            return list(self.index.shards())
        else:
            return list_all_shards(self.dataset_dir)

    def _register_wsds_links(self):
        for subdir, _ in self.fields.values():
            if subdir.endswith(".wsds-link"):
                spec = json.loads((self.dataset_dir / subdir).read_text())
                self.computed_columns[subdir] = spec

    def add_computed(self, name, **link):
        subdir = name + ".wsds-computed"
        self.computed_columns[subdir] = link
        self.fields[name] = (subdir, name)

    def get_linked_dataset(self, dataset_dir):
        if dataset_dir not in self._linked_datasets:
            self._linked_datasets[dataset_dir] = WSDataset(dataset_dir)
        return self._linked_datasets[dataset_dir]

    def get_linked_shard(self, link, shard_name):
        loader_class = link["loader"]
        if isinstance(loader_class, list):
            loader_mod, loader = loader_class
            loader_module = importlib.import_module(loader_mod)
            loader_class = getattr(loader_module, loader)

        return loader_class.from_link(
            link, self.get_linked_dataset(self.dataset_dir / link["dataset_dir"]), self, shard_name
        )

    def get_shard(self, subdir, shard_name):
        dir = self.dataset_dir / subdir

        shard = self._open_shards.get(dir, None)
        if shard is not None and shard.shard_name == shard_name:
            return shard

        if subdir in self.computed_columns:
            shard = self.get_linked_shard(self.computed_columns[subdir], shard_name)
        else:
            shard = WSShard(self, f"{dir}/{shard_name}.wsds", shard_name=shard_name)

        self._open_shards[dir] = shard
        return shard

    def get_sample(self, shard_name, field, offset):
        subdir, column = self.fields[field]
        return self.get_shard(subdir, shard_name).get_sample(column, offset)

    def parse_key(self, key):
        if self.segmented:
            return parse_key(key)
        else:
            return key, 0

    def __str__(self):
        out = ""
        out += repr(self) + "\n"
        out += f"     Audio duration: {format_duration(self.index.audio_duration)}\n"
        if self.segmented:
            out += f"    Speech duration: {format_duration(self.index.speech_duration)}\n"
        out += f"   Number of shards: {self.index.n_shards}\n"
        out += f"Number of samples: {format(self.index.n_samples, ',d').replace(',', ' ')}\n"
        return out

    def __repr__(self):
        return f"WSDataset({repr(str(self.dataset_dir))}, segmented={self.segmented})"


def format_duration(duration):
    """Formats a duration in seconds as hours (or kilo-hours)."""
    hours = duration // 3600
    if hours > 1000:
        return f"{hours / 1000:.2f} k hours"
    else:
        return f"{hours:.2f} hours"
