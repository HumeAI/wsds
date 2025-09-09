import importlib
import json
import random
from pathlib import Path

from hume_wsds.utils import list_all_columns, make_key, parse_key
from hume_wsds.ws_index import WSIndex
from hume_wsds.ws_sample import WSSample
from hume_wsds.ws_shard import WSShard


class WSDataset:
    def __init__(self, dir, segmented=None):
        self.dir = dir
        self.fields = list_all_columns(self.dir)
        self.index = WSIndex(f"{self.dir}/index.sqlite3")
        self.segmented = (
            (Path(self.dir) / "segmented").exists() if segmented is None else segmented
        )
        self._open_shards = {}
        self._linked_datasets = {}

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
        return make_key(file_name, offset - start_offset)

    def get_linked_dataset(self, dataset_dir):
        if dataset_dir not in self._linked_datasets:
            self._linked_datasets[dataset_dir] = WSDataset(dataset_dir)
        return self._linked_datasets[dataset_dir]

    def get_linked_shard(self, subdir, shard_name):
        link = json.loads(Path(subdir).read_text())

        loader_mod, loader = link['loader']
        loader_module = importlib.import_module(loader_mod)
        loader_class = getattr(loader_module, loader)

        return loader_class.from_link(link, self.get_linked_dataset(f"{self.dir}/{link['dataset_dir']}"), self, shard_name)

    def get_shard(self, subdir, shard_name):
        dir = f"{self.dir}/{subdir}"

        shard = self._open_shards.get(dir, None)
        if shard is not None and shard.shard_name == shard_name: return shard

        if subdir.endswith(".wsds-link"): # not a directory, but a JSON file
            shard = self.get_linked_shard(dir, shard_name)
        else:
            shard = WSShard(f"{dir}/{shard_name}.wsds", shard_name=shard_name)

        self._open_shards[dir] = shard
        return shard

    def get_sample(self, shard_name, field, offset):
        subdir, column = self.fields[field]
        return self.get_shard(subdir, shard_name).get_sample(column, offset)

    def sequential_from(self, shard_name, start, end=None):
        if end is None:
            end = self.index.query(
                "SELECT n_samples FROM shards WHERE shard = ?", shard_name
            ).fetchone()[0]
        i = start
        while i < end:
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
        shard_name, offset = self.index.query(
            "SELECT s.shard, offset FROM files AS f, shards AS s WHERE f.rowid = ? AND s.shard_id == f.shard_id",
            random.randrange(self.index.n_files),
        ).fetchone()
        return shard_name, offset

    def __iter__(self, max_per_shard=None):
        while True:
            shard_name, start = self.random_position()
            end = start + max_per_shard if max_per_shard else None
            yield from self.sequential_from(shard_name, start, end)

    def __repr__(self):
        return f"WSDataset({repr(self.dir)}, segmented={self.segmented})"
