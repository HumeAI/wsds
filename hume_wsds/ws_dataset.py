import importlib
import json
import random
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
        self._open_shards = {}
        self._linked_datasets = {}

        self.computed_columns = {}

    def get_shard_list(self):
        if self.index:
            return list(self.index.shards())
        else:
            return list_all_shards(self.dir)

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
        elif subdir.endswith(".wsds-link"): # not a directory, but a JSON file
            self.computed_columns[subdir] = json.loads(Path(dir).read_text())
            shard = self.get_linked_shard(self.computed_columns[subdir], shard_name)
        else:
            shard = WSShard(f"{dir}/{shard_name}.wsds", shard_name=shard_name)

        self._open_shards[dir] = shard
        return shard

    def get_sample(self, shard_name, field, offset):
        subdir, column = self.fields[field]
        return self.get_shard(subdir, shard_name).get_sample(column, offset)

    def sequential_from(self, shard_name, start, end=None):
        if end is None and self.index:
            end = self.index.query(
                "SELECT n_samples FROM shards WHERE shard = ?", shard_name
            ).fetchone()[0]
        i = start
        # without an index, we still return the sample but you'll get an error on first field access
        while i < end if end is not None else True:
            yield WSSample(self, shard_name, i)
            i += 1

    def parse_key(self, key):
        if self.segmented:
            return parse_key(key)
        else:
            return key, 0

    def get_position(self, key):
        file_name, offset = self.parse_key(key)
        shard_name, file_offset = self.index.query(
            "SELECT s.shard, offset FROM files AS f, shards AS s WHERE f.name = ? AND s.shard_id == f.shard_id",
            file_name,
        ).fetchone()
        return shard_name, file_offset + offset

    def random_position(self):
        # FIXME: randomize the global sample index once (this has a bias towards smaller shards)
        # we'll need to modify the index to store the cumulative number of samples in each shard
        shard_name, n_samples = self.index.query(
            "SELECT s.shard, s.n_samples FROM shards AS s WHERE s.rowid = ?",
            random.randrange(self.index.n_shards) + 1, # sqlite starts indexing at 1
        ).fetchone()
        return shard_name, random.randrange(n_samples)

    def __iter__(self, max_per_shard=None):
        while True:
            shard_name, start = self.random_position()
            end = start + max_per_shard if max_per_shard else None
            yield from self.sequential_from(shard_name, start, end)

    def __repr__(self):
        return f"WSDataset({repr(self.dir)}, segmented={self.segmented})"
