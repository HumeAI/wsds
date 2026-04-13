import functools
import json
import sqlite3
from pathlib import Path

from . import utils


# TODO:
# - add support for dataset splits (split on file-level or segment-level?)
class WSDSIndexWriter:
    def __init__(self, fname):
        self.fname = Path(fname)
        self.global_offset = 0

    def __enter__(self):
        self.fname.unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.fname)

        self.conn.execute("PRAGMA journal_mode = OFF;")
        self.conn.execute("PRAGMA synchronous = 0;")
        self.conn.execute("PRAGMA locking_mode = EXCLUSIVE;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")

        self.conn.execute("""
        CREATE TABLE files (
            name TEXT PRIMARY KEY NOT NULL,
            shard_id INTEGER NOT NULL,
            offset INTEGER NOT NULL,
            audio_duration REAL NOT NULL,
            speech_duration REAL NOT NULL
        ) WITHOUT ROWID;""")
        self.conn.execute("""
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY,
            shard TEXT NOT NULL,
            n_samples INTEGER NOT NULL,
            global_offset INTEGER NOT NULL,
            partition TEXT NULL
        );""")
        self.conn.execute("""
        CREATE UNIQUE INDEX shard_name ON shards (shard, partition);
        """)
        self.conn.execute("""
        CREATE UNIQUE INDEX shard_global_offset ON shards (global_offset);
        """)
        self.conn.execute("""
        CREATE TABLE metadata (
            value TEXT NOT NULL
        );""")

        return self

    def append_metadata(self, metadata):
        self.conn.execute(
            "INSERT INTO metadata (value) VALUES (?);",
            (json.dumps(metadata),),
        )

    def append(self, s):
        # we ensure plain Python types for everything passed in, otherwise sqlite will silently save invalid data
        shard_id = self.conn.execute(
            "INSERT INTO shards (shard, n_samples, global_offset, partition) VALUES (?, ?, ?, ?);",
            (str(s["shard_name"]), int(s["n_samples"]), self.global_offset, str(s["partition"])),
        ).lastrowid
        for name, offset, audio_duration, speech_duration in s["index"]:
            try:
                self.conn.execute(
                    "INSERT INTO files (name, shard_id, offset, audio_duration, speech_duration) VALUES (?, ?, ?, ?, ?);",
                    (str(name), shard_id, int(offset), float(audio_duration), float(speech_duration)),
                )
            except sqlite3.IntegrityError as err:
                if err.args[0] == "UNIQUE constraint failed: files.name":
                    old_shard, old_duration = self.conn.execute(
                        "SELECT s.shard, f.audio_duration FROM shards AS s, files AS f WHERE f.name == ?",
                        (name,),
                    ).fetchone()
                    if audio_duration - old_duration < 10e-3:
                        print(
                            f"Skipping duplicate episode: {repr(name)} ({utils.format_duration(audio_duration)} long)"
                        )
                        continue
                    raise ValueError(
                        f"Detected duplicate file name: {repr(name)} in shard \n{repr(s['shard_name'])}, previously seen in {repr(old_shard)}"
                    )
                else:
                    print("Error inserting file:", name, offset, shard_id)
                    for row in self.conn.execute("SELECT * FROM files WHERE shard_id == ?", (shard_id,)):
                        print(row)
                    raise
        self.global_offset += s["n_samples"]

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.conn.commit()
        self.conn.close()


class WSIndex:
    def __init__(self, fname: str):
        self.fname = fname
        if not Path(fname).exists():
            raise ValueError(f"WSIndex not found: {fname}")
        # immutable=1,ro=True greatly speeds up all queries when the database is on a remote/cluster file system
        self.conn = sqlite3.connect(f"file:{fname}?immutable=1,ro=True", uri=True)

        # Detect index format: new indexes have a 'partition' column, old indexes have 'dataset_path'
        columns = {
            row[0]
            for row in self.conn.execute("SELECT name FROM pragma_table_info('shards')").fetchall()
        }
        self.has_partition = "partition" in columns
        self.has_dataset_path = "dataset_path" in columns

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @functools.cached_property
    def n_shards(self):
        return self.conn.execute("SELECT COUNT(*) FROM shards;").fetchone()[0]

    @functools.cached_property
    def n_files(self):
        return self.conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]

    @functools.cached_property
    def n_samples(self):
        return self.conn.execute("SELECT SUM(n_samples) FROM shards;").fetchone()[0]

    @functools.cached_property
    def audio_duration(self):
        if "audio_duration" in self.metadata:
            return self.metadata["audio_duration"]
        return self.conn.execute("SELECT SUM(audio_duration) FROM files;").fetchone()[0]

    @functools.cached_property
    def speech_duration(self):
        if "speech_duration" in self.metadata:
            return self.metadata["speech_duration"]
        return self.conn.execute("SELECT SUM(speech_duration) FROM files;").fetchone()[0]

    @functools.cached_property
    def _partition_col(self):
        """SQL expression for the partition column, adapting to old/new index formats."""
        if self.has_partition:
            return "s.partition"
        if self.has_dataset_path:
            return "s.dataset_path"
        return "''"

    def shards(self):
        return self.conn.execute(f"SELECT {self._partition_col}, shard FROM shards AS s ORDER BY rowid;")

    def lookup_by_index(self, index: int):
        """Look up a sample by global index. Returns (partition, shard_name, local_offset) or None."""
        r = self.conn.execute(
            f"SELECT s.shard, s.global_offset, {self._partition_col} FROM shards AS s"
            f" WHERE s.global_offset <= ? ORDER BY s.global_offset DESC LIMIT 1",
            (index,),
        ).fetchone()
        if not r:
            return None
        shard_name, shard_global_offset, partition = r
        local_offset = index - shard_global_offset
        return partition, shard_name, local_offset

    def lookup_by_key(self, file_name: str, offset_of_key_wrt_file: int):
        """Look up a sample by file name and offset within file.
        Returns (partition, shard_name, local_offset, global_offset) or None."""
        r = self.conn.execute(
            f"SELECT s.shard, s.global_offset, f.offset, {self._partition_col}"
            f" FROM files AS f, shards AS s WHERE f.name = ? AND s.shard_id == f.shard_id",
            (file_name,),
        ).fetchone()
        if not r:
            return None
        shard_name, shard_global_offset, file_offset_in_shard, partition = r
        local_offset = file_offset_in_shard + offset_of_key_wrt_file
        global_offset = shard_global_offset + local_offset
        return partition, shard_name, local_offset, global_offset

    def _query_shard(self, columns, shard_ref):
        """Query shard table columns for a given (partition, shard_name) ref."""
        partition, shard_name = shard_ref
        if self.has_partition or self.has_dataset_path:
            return self.conn.execute(
                f"SELECT {columns} FROM shards AS s WHERE {self._partition_col} = ? AND s.shard = ?",
                (partition, shard_name),
            ).fetchone()
        else:
            return self.conn.execute(
                f"SELECT {columns} FROM shards WHERE shard = ?", (shard_name,)
            ).fetchone()

    def shard_n_samples(self, shard_ref):
        """Return the number of samples in a shard, given a (partition, shard_name) ref."""
        r = self._query_shard("n_samples", shard_ref)
        if r is None:
            raise IndexError(f"Shard not found: {shard_ref}")
        return r[0]

    def shard_global_offset(self, shard_ref):
        """Return the global offset of a shard, given a (partition, shard_name) ref."""
        r = self._query_shard("global_offset", shard_ref)
        if r is None:
            raise IndexError(f"Shard not found: {shard_ref}")
        return r[0]

    def dataframe(self):
        import polars as pl

        df = pl.read_database_uri(
            """
            SELECT f.name, audio_duration, speech_duration, s.shard, s.n_samples
            FROM files as f, shards as s
            WHERE f.shard_id == s.shard_id""",
            f"sqlite://{self.fname}",
        )
        return df

    @functools.cached_property
    def metadata(self):
        metadata = {}
        try:
            for (metadata_chunk,) in self.conn.execute("SELECT value FROM metadata;"):
                metadata.update(json.loads(metadata_chunk))
        except sqlite3.OperationalError as err:
            if err.args[0] != "no such table: metadata":
                raise
        return metadata

    def query(self, query, *args):
        return self.conn.execute(query, args)

    def __repr__(self):
        return f"WSIndex({repr(self.fname)})"
