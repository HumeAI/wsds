import importlib
import json
import random
import sys
from pathlib import Path

from hume_wsds.utils import list_all_columns, list_all_shards, make_key, parse_key
from hume_wsds.ws_index import WSIndex
from hume_wsds.ws_sample import WSSample
from hume_wsds.ws_shard import WSShard


class WSDataset:
    # FIXME: this should be overridable with metadata in index.sqlite3
    _audio_file_keys = ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma", "opus"]

    def __init__(self, dir, segmented=None):
        self.dir = dir
        index_file = f"{self.dir}/index.sqlite3"
        if Path(index_file).exists():
            self.index = WSIndex(index_file)
            self.segmented = self.index.metadata.get('segmented', False)
        else:
            self.index = None
            self.segmented = False
        self.fields = list_all_columns(self.dir, next(self.index.shards()) if self.index else None)
        self.computed_columns = {}

        self._open_shards = {}
        self._linked_datasets = {}

        self._register_wsds_links()

    #
    # Accessing samples randomly and sequentially
    #
    def random_sample(self):
        """Returns one random sample."""
        assert self.index is not None, "Random access is only supported for indexed datasets"
        # FIXME: randomize the global sample index once (this has a bias towards smaller shards)
        # we'll need to modify the index to store the cumulative number of samples in each shard
        shard_name, n_samples = self.index.query(
            "SELECT s.shard, s.n_samples FROM shards AS s WHERE s.rowid = ?",
            random.randrange(self.index.n_shards) + 1, # sqlite starts indexing at 1
        ).fetchone()
        return WSSample(self, shard_name, random.randrange(n_samples))

    def __iter__(self):
        """Starts at a random position in the dataset and yields samples sequentially.
        Once it reaches the end of a shard it will jump to a new random position.
        """
        while True:
            yield from self.sequential_from(self.random_sample())

    def random_samples(self, N:int = 1):
        """Yields N random samples."""
        for _ in range(N):
            yield self.random_sample()

    def random_chunks(self, max_N:int):
        """Like `__iter__`, but jumps to a random position after yielding `max_N` samples."""
        while True:
            yield from self.sequential_from(self.random_sample(), end = start + max_N)

    def __getitem__(self, key:str):
        """Returns a sample with the given key."""
        # FIXME: push `parse_key` to the index class
        file_name, offset = self.parse_key(key)
        r = self.index.query(
            "SELECT s.shard, offset FROM files AS f, shards AS s WHERE f.name = ? AND s.shard_id == f.shard_id",
            file_name,
        ).fetchone()
        if not r: return None
        shard_name, file_offset = r
        return WSSample(self, shard_name, file_offset + offset)

    def sequential_from(self, sample, max_N = None):
        """Yields samples sequentially from the given `sample`, stopping after `max_N` samples."""
        shard_name, i = sample.shard_name, sample.offset
        max_N = min(max_N or sys.maxsize, self._shard_n_samples(shard_name)
        # without an index, we still return the sample but you'll get an error on first field access
        while i < max_N:
            yield WSSample(self, shard_name, i)
            i += 1

    def _shard_n_samples(self, shard_name:str) -> int:
        if not self.index: return sys.maxsize
        return self.index.query(
            "SELECT n_samples FROM shards WHERE shard = ?", shard_name
        ).fetchone()[0]

    #
    # SQL support, using Polars
    #
    def _sql(self, *queries):
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Polars is required for SQL queries")
        subdirs = set()
        exprs = []
        for query in queries:
            expr = pl.sql_expr(query)
            for col in expr.meta.root_names():
                if col == '__key__':
                    # __key__ exists in all shards
                    continue
                subdir, field = self.fields[col]
                assert col == field, "renamed fields are not supported in SQL queries yet"
                subdirs.add(subdir)
            exprs.append(expr)
        row_merge = []
        missing = defaultdict(list)
        for shard in self.get_shard_list():
            col_merge = []
            for subdir in subdirs:
                try:
                    df = pl.scan_ipc(self.get_shard(subdir, shard).fname)
                    if len(col_merge) > 0: df = df.drop('__key__') # ensure only one __key__ column
                    col_merge.append(df)
                except FileNotFoundError:
                    missing[subdir].append(shard)
                    # if any of the subdirs are missing this shard, skip it completely
                    col_merge = []
                    break
            if col_merge:
                row_merge.append(pl.concat(col_merge, how = 'horizontal'))
        if missing:
            print("WARNING: You are missing shards for some of the columns:")
            for subdir, shards in missing.items():
                print(f"{subdir}: {shards}")
            if not row_merge:
                raise FileNotFoundError(f"No usable shards found (columns: {', '.join(subdirs)}) for dataset in: {self.dir}")
        return exprs, pl.concat(row_merge)

    def sql(self, *queries):
        exprs, df = self._sql(*queries)
        return df.select(exprs).collect()

    def sql_filter(self, query):
        exprs, df = self._sql(query)
        return df.filter(exprs[0]).select('__key__').collect()['__key__']

    def filtered(self,
            query,
            infinite:bool = False, # keep yielding samples indefinitely (restarting from the beginning)
            shuffle:bool = True, # shuffle the sample order (otherwise it will return them as they appear in the dataset)
            N:int = None, # optional maximum number of samples to yield (otherwise it will yield all matching samples)
            seed:int = None, # optional random seed used shuffling
        ):
        """Given an boolean SQL expression, returns an iterator which yields random samples
        that match the query.

        Examples:
        >>> dataset.filtered('pq < 3', N=10) # ten low-quality samples
        >>> dataset.filtered('pq > 3 AND CAST(txt AS string) ILIKE '%hello%' AND pq < 3', infinite=True)
        >>> dataset.filtered('CAST(`transcription_wslang_raw.txt` AS string) ILIKE "%between New Orleans and St. Louis%"')
        """
        import polars as pl
        i = 0
        keys = self.sql_filter(query)
        while True:
            if N is None:
                keys = keys.sample(frac=1, shuffle=shuffle, seed=seed)
            else:
                keys = keys.sample(n=pl.all().len().clip(0, N), shuffle=shuffle, seed=seed)
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
    def get_shard_list(self):
        if self.index:
            return list(self.index.shards())
        else:
            return list_all_shards(self.dir)

    def _register_wsds_links(self):
        for subdir, _ in self.fields.values():
            if subdir.endswith('.wsds-link'):
                spec = json.loads(Path(f"{self.dir}/{subdir}").read_text())
                self.computed_columns[subdir] = spec

    def add_computed(self, name, **link):
        subdir = name+'.wsds-computed'
        self.computed_columns[subdir] = link
        self.fields[name] = (subdir, name)

    def get_key(self, shard_name, offset):
        file_name, start_offset = self.index.query(
            """
        SELECT f.name, f.offset FROM files AS f, shards AS s
        WHERE s.shard = ? AND f.offset <= ? AND s.shard_id == f.shard_id
        ORDER BY f.offset DESC
        LIMIT 1""",
            shard_name,
            offset,
        ).fetchone()
        if self.segmented:
            return make_key(file_name, offset - start_offset)
        else:
            return file_name

    def get_linked_dataset(self, dataset_dir):
        if dataset_dir not in self._linked_datasets:
            self._linked_datasets[dataset_dir] = WSDataset(dataset_dir)
        return self._linked_datasets[dataset_dir]

    def get_linked_shard(self, link, shard_name):
        loader_class = link['loader']
        if isinstance(loader_class, list):
            loader_mod, loader = loader_class
            loader_module = importlib.import_module(loader_mod)
            loader_class = getattr(loader_module, loader)

        return loader_class.from_link(link, self.get_linked_dataset(f"{self.dir}/{link['dataset_dir']}"), self, shard_name)

    def get_shard(self, subdir, shard_name):
        dir = f"{self.dir}/{subdir}"

        shard = self._open_shards.get(dir, None)
        if shard is not None and shard.shard_name == shard_name: return shard

        if subdir in self.computed_columns:
            shard = self.get_linked_shard(self.computed_columns[subdir], shard_name)
        else:
            shard = WSShard(f"{dir}/{shard_name}.wsds", shard_name=shard_name)

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
        out += f"   Audio duration: {format_duration(self.index.audio_duration)}\n"
        if self.segmented:
            out += f"  Speech duration: {format_duration(self.index.speech_duration)}\n"
        out += f" Number of shards: {self.index.n_shards}\n"
        out += f"Number of samples: {format(self.index.n_samples, ',d').replace(',', ' ')}\n"
        return out

    def __repr__(self):
        return f"WSDataset({repr(self.dir)}, segmented={self.segmented})"

def format_duration(duration):
    """Formats a duration in seconds as a string."""
    hours = duration // 3600
    if hours > 1000:
        return f"{hours/1000:.2f} k hours"
    else:
        return f"{hours:.2f} hours"

def thousand_separator(num):
    """Formats a number with a thousand separator."""
    return f"{num:,}"
