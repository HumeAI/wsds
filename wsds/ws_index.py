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

        self.conn.execute('PRAGMA journal_mode = OFF;')
        self.conn.execute('PRAGMA synchronous = 0;')
        self.conn.execute('PRAGMA locking_mode = EXCLUSIVE;')
        self.conn.execute('PRAGMA temp_store = MEMORY;')

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
            dataset_path TEXT NULL
        );""")
        self.conn.execute("""
        CREATE UNIQUE INDEX shard_name ON shards (shard, dataset_path);
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
            "INSERT INTO shards (shard, n_samples, global_offset, dataset_path) VALUES (?, ?, ?, ?);",
            (str(s["shard_name"]), int(s["n_samples"]), self.global_offset, str(s["dataset_path"])),
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
                        print(f"Skipping duplicate episode: {repr(name)} ({utils.format_duration(audio_duration)} long)")
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
    """SQLite-based index for fast random access to samples in a wsds dataset.

    The index stores:
    - `shards` table: shard names, sample counts, and global offsets
    - `files` table: source file names, their shard, offset within shard, and duration info
    - `metadata` table: JSON-encoded dataset metadata (e.g., segmented flag, fields)

    This enables O(1) lookups by global sample index or by file name.
    """

    def __init__(self, fname: str):
        self.fname = fname
        if not Path(fname).exists():
            raise ValueError(f"WSIndex not found: {fname}")
        # immutable=1,ro=True greatly speeds up all queries when the database is on a remote/cluster file system
        self.conn = sqlite3.connect(f"file:{fname}?immutable=1,ro=True", uri=True)
        self.has_dataset_path = self.conn.execute("SELECT COUNT(*) FROM pragma_table_info('shards') WHERE name='dataset_path'").fetchone()[0]

    #
    # Aggregate properties
    #

    @functools.cached_property
    def n_shards(self) -> int:
        """Total number of shards in the dataset."""
        return self.conn.execute("SELECT COUNT(*) FROM shards;").fetchone()[0]

    @functools.cached_property
    def n_files(self) -> int:
        """Total number of source files in the dataset."""
        return self.conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]

    @functools.cached_property
    def n_samples(self) -> int:
        """Total number of samples across all shards."""
        return self.conn.execute("SELECT SUM(n_samples) FROM shards;").fetchone()[0]

    @functools.cached_property
    def audio_duration(self) -> float:
        """Total audio duration in seconds across all files."""
        return self.conn.execute("SELECT SUM(audio_duration) FROM files;").fetchone()[0]

    @functools.cached_property
    def speech_duration(self) -> float:
        """Total speech duration in seconds (for segmented datasets)."""
        return self.conn.execute("SELECT SUM(speech_duration) FROM files;").fetchone()[0]

    @functools.cached_property
    def metadata(self) -> dict:
        """Dataset metadata dictionary (merged from all metadata rows)."""
        metadata = {}
        try:
            for (metadata_chunk,) in self.conn.execute("SELECT value FROM metadata;"):
                metadata.update(json.loads(metadata_chunk))
        except sqlite3.OperationalError as err:
            if err.args[0] != "no such table: metadata":
                raise
        return metadata

    #
    # Shard iteration
    #

    def shards(self):
        """Iterate over all shards as (dataset_path, shard_name) tuples.

        Yields tuples in the order shards were added to the index.
        """
        dataset_path = 'dataset_path' if self.has_dataset_path else "''"
        return self.conn.execute(f"SELECT {dataset_path}, shard FROM shards ORDER BY rowid;")

    #
    # DataFrame export
    #

    def dataframe(self):
        """Export the index as a Polars DataFrame.

        Returns:
            DataFrame with columns: name, audio_duration, speech_duration, shard, n_samples.
        """
        import polars as pl
        df = pl.read_database_uri("""
            SELECT f.name, audio_duration, speech_duration, s.shard, s.n_samples
            FROM files as f, shards as s
            WHERE f.shard_id == s.shard_id""", f"sqlite://{self.fname}"
        )
        return df

    #
    # Low-level query access (prefer using specific methods above)
    #

    def query(self, query, *args):
        """Execute a raw SQL query on the index database.

        Prefer using the specific lookup methods above when possible.
        """
        return self.conn.execute(query, args)

    def __repr__(self):
        return f"WSIndex({repr(self.fname)})"
