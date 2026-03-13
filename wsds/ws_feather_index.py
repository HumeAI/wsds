import functools
import json
from pathlib import Path

import polars as pl


class WSFeatherIndex:
    """Feather/Arrow-based index for fast random access to samples in a wsds dataset.

    Uses feather files:
    - `shard-index.feather`: shard metadata with columns:
        shard_id, dataset_path, shard_name, n_samples, segment_id (global offset),
        audio_duration, speech_duration
    - `episode-index.feather`: episode/file info sorted by segment_id, with columns:
        segment_id, shard_id, episode_id, audio_duration, speech_duration
    - `episode-name-index.feather` (optional): name to episode_id mapping sorted by name,
        with columns: name, shard_id, episode_id

    This enables O(log n) lookups by global sample index or by file name using search_sorted.
    """

    def __init__(self, index_dir: str | Path):
        """Initialize the feather index from a directory containing the index files.

        Args:
            index_dir: Path to directory containing shard-index.feather, episode-index.feather,
                      and optionally episode-name-index.feather files.
        """
        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            raise ValueError(f"WSFeatherIndex directory not found: {index_dir}")

        shard_path = self.index_dir / "shard-index.feather"
        episode_path = self.index_dir / "episode-index.feather"
        name_path = self.index_dir / "episode-name-index.feather"

        # Required files
        for p in [shard_path, episode_path]:
            if not p.exists():
                raise ValueError(f"Required index file not found: {p}")

        # Load shard index - segment_id column already contains global offsets
        self._shard_df = pl.read_ipc(shard_path)

        # Load episode index (sorted by segment_id for binary search)
        self._episode_df = pl.read_ipc(episode_path)

        # Load name index if available (sorted by name for binary search)
        if name_path.exists():
            self._name_df = pl.read_ipc(name_path)
        else:
            self._name_df = None

    #
    # Aggregate properties
    #

    @functools.cached_property
    def n_shards(self) -> int:
        """Total number of shards in the dataset."""
        return len(self._shard_df)

    @functools.cached_property
    def n_files(self) -> int:
        """Total number of source files (episodes) in the dataset."""
        return len(self._episode_df)

    @functools.cached_property
    def n_samples(self) -> int:
        """Total number of samples across all shards."""
        return int(self._shard_df["n_samples"].sum())

    @functools.cached_property
    def audio_duration(self) -> float:
        """Total audio duration in seconds across all files."""
        return float(self._shard_df["audio_duration"].sum())

    @functools.cached_property
    def speech_duration(self) -> float:
        """Total speech duration in seconds (for segmented datasets)."""
        return float(self._shard_df["speech_duration"].sum())

    @functools.cached_property
    def metadata(self) -> dict:
        """Dataset metadata dictionary.

        Reads from metadata.json if present, otherwise returns empty dict.
        """
        metadata_path = self.index_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    #
    # Shard iteration
    #

    def shards(self):
        """Iterate over all shards as (dataset_path, shard_name) tuples.

        Yields tuples in the order shards were added to the index.
        """
        for row in self._shard_df.iter_rows(named=True):
            yield (row["dataset_path"], row["shard_name"])

    #
    # Shard lookups
    #

    def get_shard_by_global_index(self, global_index: int) -> tuple[str, int, str] | None:
        """Find the shard containing a given global sample index.

        Args:
            global_index: The global sample index (0-based across the entire dataset).

        Returns:
            Tuple of (shard_name, shard_global_offset, dataset_path) or None if not found.
            The local offset within the shard is: global_index - shard_global_offset.
        """
        if global_index < 0 or global_index >= self.n_samples:
            return None

        # Binary search for the shard containing this index
        # search_sorted with side="right" returns index where global_index would be inserted
        # We want the shard where global_offset <= global_index, so subtract 1
        idx = self._shard_df.select(pl.col('segment_id').search_sorted(global_index, side='right')).item()

        if idx < 0:
            return None

        row = self._shard_df.row(idx, named=True)
        return (row["shard_name"], int(row["segment_id"]), row["dataset_path"])

    def get_shard_by_file_name(self, file_name: str) -> tuple[str, int, int, str] | None:
        """Find the shard containing a given source file.

        Args:
            file_name: The source file name (without segment suffix).

        Returns:
            Tuple of (shard_name, shard_global_offset, file_offset_in_shard, dataset_path)
            or None if not found.
        """
        if self._name_df is None:
            raise RuntimeError("episode-name-index.feather is required to search by episode name")

        # Binary search in sorted name series
        idx = self._name_df.select(pl.col('name').search_sorted(file_name, side='right')).item()

        if idx >= len(self._names) or self._names[idx] != file_name:
            return None

        name_row = self._name_df.row(idx, named=True)
        shard_id = name_row["shard_id"]
        episode_id = name_row["episode_id"]

        # Get shard info by shard_id (shard_id is the row index)
        shard_row = self._shard_df.row(shard_id, named=True)
        shard_global_offset = int(shard_row["segment_id"])

        # Get episode info to find segment_id
        # Use search_sorted on episode_id column for efficient lookup
        episode_row = self._episode_df.filter(pl.col("episode_id") == episode_id).row(0, named=True)
        segment_id = int(episode_row["segment_id"])

        # file_offset_in_shard = segment_id - shard_global_offset
        file_offset = segment_id - shard_global_offset

        return (
            shard_row["shard_name"],
            shard_global_offset,
            file_offset,
            shard_row["dataset_path"],
        )

    def get_shard_global_offset(self, shard_name: str) -> int | None:
        """Get the global sample offset for a shard.

        Args:
            shard_name: The shard name (without .wsds extension).

        Returns:
            The global offset (first sample index in this shard), or None if not found.
        """
        filtered = self._shard_df.filter(pl.col("shard_name") == shard_name)
        if len(filtered) == 0:
            return None
        return int(filtered.row(0, named=True)["segment_id"])

    def get_shard_n_samples(self, shard: tuple[str, str]) -> int | None:
        """Get the number of samples in a shard.

        Args:
            shard: Tuple of (dataset_path, shard_name).

        Returns:
            The number of samples in the shard, or None if not found.
        """
        dataset_path, shard_name = shard
        if dataset_path:
            filtered = self._shard_df.filter(
                (pl.col("dataset_path") == dataset_path) & (pl.col("shard_name") == shard_name)
            )
        else:
            filtered = self._shard_df.filter(pl.col("shard_name") == shard_name)

        if len(filtered) == 0:
            return None
        return int(filtered.row(0, named=True)["n_samples"])

    def get_shard_info(self, shard: tuple[str, str]) -> tuple[int, int] | None:
        """Get n_samples and shard_id for a shard.

        Args:
            shard: Tuple of (dataset_path, shard_name).

        Returns:
            Tuple of (n_samples, shard_id) or None if not found.
        """
        dataset_path, shard_name = shard
        if dataset_path:
            filtered = self._shard_df.filter(
                (pl.col("dataset_path") == dataset_path) & (pl.col("shard_name") == shard_name)
            )
        else:
            filtered = self._shard_df.filter(pl.col("shard_name") == shard_name)

        if len(filtered) == 0:
            return None
        row = filtered.row(0, named=True)
        return (int(row["n_samples"]), int(row["shard_id"]))

    #
    # File lookups
    #

    def get_files_for_shard(self, shard_id: int) -> list[tuple[str, int]]:
        """Get all files in a shard with their offsets.

        Args:
            shard_id: The internal shard ID.

        Returns:
            List of (file_name, offset) tuples.
        """
        if not self._has_name_index:
            raise RuntimeError("episode-name-index.feather is required for get_files_for_shard")

        # Get shard global offset
        shard_row = self._shard_df.row(shard_id, named=True)
        shard_global_offset = int(shard_row["segment_id"])

        # Get all episodes for this shard
        episodes = self._episode_df.filter(pl.col("shard_id") == shard_id)

        # Get names for these episodes
        episode_ids = set(episodes["episode_id"].to_list())
        names = self._name_df.filter(pl.col("episode_id").is_in(episode_ids))

        # Join to get name with segment_id
        joined = names.join(episodes.select(["episode_id", "segment_id"]), on="episode_id")

        result = []
        for row in joined.iter_rows(named=True):
            offset = int(row["segment_id"]) - shard_global_offset
            result.append((row["name"], offset))

        return result

    def iter_files(self):
        """Iterate over all files with their shard and offset info.

        Yields tuples of (file_name, shard_name, offset) ordered by file name.
        """
        if not self._has_name_index:
            raise RuntimeError("episode-name-index.feather is required for iter_files")

        # Join all three dataframes
        # name_df has: name, shard_id, episode_id
        # episode_df has: segment_id, shard_id, episode_id, ...
        # shard_df has: shard_id, shard_name, segment_id (global_offset), ...

        joined = (
            self._name_df.join(self._episode_df.select(["episode_id", "segment_id"]), on="episode_id")
            .join(
                self._shard_df.select(["shard_id", "shard_name", pl.col("segment_id").alias("global_offset")]),
                on="shard_id",
            )
            .with_columns([(pl.col("segment_id") - pl.col("global_offset")).cast(pl.Int64).alias("offset")])
            .sort("name")
        )

        for row in joined.iter_rows(named=True):
            yield (row["name"], row["shard_name"], row["offset"])

    #
    # DataFrame export
    #

    def dataframe(self):
        """Export the index as a Polars DataFrame.

        Returns:
            DataFrame with columns: name, audio_duration, speech_duration, shard, n_samples.
        """
        if not self._has_name_index:
            raise RuntimeError("episode-name-index.feather is required for dataframe")

        # Join name_df with episode_df to get durations, then with shard_df
        df = (
            self._name_df.join(
                self._episode_df.select(["episode_id", "audio_duration", "speech_duration"]), on="episode_id"
            )
            .join(self._shard_df.select(["shard_id", "shard_name", "n_samples"]), on="shard_id")
            .select(["name", "audio_duration", "speech_duration", pl.col("shard_name").alias("shard"), "n_samples"])
        )
        return df

    def __repr__(self):
        return f"WSFeatherIndex({repr(str(self.index_dir))})"
