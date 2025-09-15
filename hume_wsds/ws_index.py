import functools
import gzip
import sqlite3
import tarfile
from pathlib import Path


def list_keys_tarfile(input_shard):
    last_key = None
    if input_shard.endswith("gz"):
        o = gzip.open
    else:
        o = open
    with o(input_shard, "rb") as f:
        for name in tarfile.TarFile(fileobj=f).getnames():
            path, name = name.rsplit("/", 1)
            key = name.split(".", 1)[0]
            if key != last_key:
                yield f"{path}/{key}"
            last_key = key


def shard_base_name(input_shard):
    name = input_shard.rsplit("/", 1)[1]
    if name.endswith(".gz"):
        return name.rsplit(".", 3)[0]
    else:
        return name.rsplit(".", 2)[0]


# TODO:
# - add a function to generate based on mvad shards instead of downstream shards
def extract_index(input_shard, segmented=False):
    last_key = None
    index = []
    for i, key in enumerate(list_keys_tarfile(input_shard)):
        if segmented:
            key = key.rsplit("_", 1)[0]
        if key != last_key:
            index.append((key, i))
        last_key = key
    return {
        "shard_name": shard_base_name(input_shard),
        "index": index,
        "n_samples": i + 1,
    }


# TODO:
# - add comulative segment duration to the index
# - add support for dataset splits (split on file-level or segment-level?)
class WSDSIndexWriter:
    def __init__(self, fname):
        self.fname = Path(fname)

    def __enter__(self):
        self.fname.unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.fname)
        print("opening", self.fname)

        self.conn.execute("""
        CREATE TABLE files (
            name TEXT PRIMARY KEY NOT NULL,
            shard_id INTEGER NOT NULL,
            offset INTEGER NOT NULL
        );""")
        self.conn.execute("""
        CREATE UNIQUE INDEX files_name ON files (name);
        """)
        self.conn.execute("""
        CREATE UNIQUE INDEX files_shard_id_offset ON files (shard_id, offset);
        """)
        self.conn.execute("""
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY,
            shard TEXT NOT NULL,
            n_samples INTEGER NOT NULL
        );""")
        self.conn.execute("""
        CREATE UNIQUE INDEX shard_name ON shards (shard);
        """)

        return self

    def append(self, s):
        shard_id = self.conn.execute(
            "INSERT INTO shards (shard, n_samples) VALUES (?, ?);",
            (s["shard_name"], s["n_samples"]),
        ).lastrowid
        for name, offset in s["index"]:
            try:
                self.conn.execute(
                    "INSERT INTO files (name, shard_id, offset) VALUES (?, ?, ?);",
                    (name, shard_id, offset),
                )
            except sqlite3.IntegrityError as err:
                if err.args[0] == "UNIQUE constraint failed: files.name":
                    old_shard = self.conn.execute(
                        "SELECT s.shard FROM shards AS s, files AS f WHERE f.name == ?",
                        (name,),
                    ).fetchone()[0]
                    raise ValueError(
                        f"Detected duplicate file name: {repr(name)} in shard \n{repr(s['shard_name'])}, previously seen in {repr(old_shard)}"
                    )
                raise

    def __exit__(self, exc_type, exc_value, traceback):
        print("exiting:", exc_type, exc_value, traceback)
        if exc_type is None:
            print("closing", self.fname)
            self.conn.commit()
            self.conn.close()


class WSIndex:
    def __init__(self, fname):
        self.fname = fname
        if not Path(fname).exists():
            raise ValueError(f"WSIndex not found: {fname}")
        self.conn = sqlite3.connect(f"file:{fname}?immutable=1,ro=True", uri=True)

    @functools.cached_property
    def n_shards(self):
        return self.conn.execute("SELECT COUNT(n_samples) FROM shards;").fetchone()[0]

    @functools.cached_property
    def n_files(self):
        return self.conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]

    @functools.cached_property
    def n_samples(self):
        return self.conn.execute("SELECT SUM(n_samples) FROM shards;").fetchone()[0]

    def shards(self):
        return (shard for shard, in self.conn.execute("SELECT shard FROM shards;"))

    def query(self, query, *args):
        return self.conn.execute(query, args)

    def __repr__(self):
        return f"WSIndex({repr(self.fname)})"
