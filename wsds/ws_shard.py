from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import pyarrow as pa

from .utils import WSShardMissingError
from .ws_audio import AudioReader, WSAudio
from .ws_decode import decode_sample
from .ws_sample import WSSample

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


class WSShardInterface:
    shard_ref: Optional[tuple[str, str]]
    """Used by WSDataset to invalidate cached shards."""

    @classmethod
    def get_columns(cls, link: dict[str, Any], dataset: "WSDataset") -> Optional[dict[str, str]]:
        """Return columns this link provides: {column_name: column_name}.

        Override this to provide multiple columns from a single link.
        Return None to use the default behavior (link file stem as single column).
        """
        return None

    def get_sample(self, column: str, offset: int) -> object:
        raise NotImplementedError


class WSShard(WSShardInterface):
    """Represents a single open data shard (`.wsds` file).

    Caches one batch worth of data for efficient sequential access to samples."""

    def __init__(self, dataset: WSDataset, fname: str | Path, shard_ref: Optional[tuple[str, str]] = None) -> None:
        self.dataset = dataset
        self.shard_ref = shard_ref
        self.fname = fname

        try:
            if dataset.disable_memory_map:
                self.reader = pa.RecordBatchFileReader(pa.OSFile(str(fname)))
            else:
                self.reader = pa.RecordBatchFileReader(pa.memory_map(str(fname)))
        except FileNotFoundError:
            raise WSShardMissingError(str(fname)) from None

        self.batch_size = int(self.reader.schema.metadata[b"batch_size"])

        # cache
        self._start: Optional[int] = None
        self._end: Optional[int] = None
        self._data: Optional[pa.RecordBatch] = None

    def get_sample(self, column: str, offset: int) -> object:
        if self._data is None or self._start is None or self._end is None or offset < self._start or offset >= self._end:
            i = offset // self.batch_size
            if i >= self.reader.num_record_batches:
                raise IndexError(f"{offset} is out of range for shard {self.fname}")
            self._data = self.reader.get_batch(i)
            if i < self.reader.num_record_batches - 1:
                if self._data.num_rows < self.batch_size:
                    raise ValueError(
                        f"Batch {i} in shard {self.fname} is incomplete (has only {self._data.num_rows} rows instead of {self.batch_size})"
                    )
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= len(self._data):
            raise IndexError(f"{offset} is out of range for shard {self.fname}")
        if self._data.schema.get_field_index(column) == -1:
            raise KeyError(f"column {column} not found in shard {self.fname}")
        data = self._data[column][j]
        col_type = self._data.schema.field(column).type
        try:
            if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
                return decode_sample(column, io.BytesIO(data.as_buffer()))
            return data.as_py(maps_as_pydicts="strict")
        except Exception as e:
            raise ValueError(f"Failed to decode column {column} in shard {self.fname} (offset {offset}): {e}")

    def __repr__(self) -> str:
        r = f"WSShard({repr(self.fname)})"
        if self._data:
            r += f" # cached_region = [{self._start, self._end}]"
        return r


@dataclass(slots=True)
class WSSourceAudioShard(WSShardInterface):
    """A proxy shard class (does not correspond to an actual `.wsds` file) to access audio data from a source dataset.

    It is used via the `WSDataset.add_computed` method or the `.wsds-link` file mechanism."""

    shard_ref: tuple[str, str]
    source_dataset: WSDataset
    derived_dataset: WSDataset
    vad_column: str

    # cache
    _source_file_name: Optional[str] = None
    _source_sample: Optional[WSSample] = None
    _source_reader: Optional[AudioReader] = None

    @classmethod
    def from_link(cls, link: dict[str, Any], dataset: WSDataset, shard_ref: tuple[str, str]) -> WSSourceAudioShard:
        source_dataset = dataset.get_linked_dataset(link["dataset_dir"])
        return cls(shard_ref, source_dataset, dataset, link["vad_column"])

    def get_timestamps(self, segment_offset: int) -> object:
        assert self._source_sample is not None
        timestamps: Any = self._source_sample[self.vad_column]
        return timestamps[segment_offset]

    def get_sample(self, _column: str, offset: int) -> object:
        key = cast(str, WSSample(self.derived_dataset, self.shard_ref, offset)["__key__"])
        file_name, segment_offset = self.derived_dataset.parse_key(key)

        if self._source_file_name != file_name:
            self._source_sample = self.source_dataset[file_name]
            assert self._source_sample is not None
            try:
                self._source_reader = self._source_sample.get_audio()  # type: ignore[assignment]
            except KeyError:
                raise WSShardMissingError("no audio shards found")
            self._source_file_name = file_name

        assert self._source_reader is not None
        timestamps: Any = self.get_timestamps(segment_offset)
        tstart, tend = timestamps
        return WSAudio(self._source_reader, tstart, tend)


class WSYoutubeVideoShard(WSSourceAudioShard):
    re_pattern: re.Pattern[str]

    @classmethod
    def from_link(cls, link: dict[str, Any], dataset: WSDataset, shard_ref: tuple[str, str]) -> WSYoutubeVideoShard:
        self = cast(WSYoutubeVideoShard, super().from_link(link, dataset, shard_ref))
        self.re_pattern = re.compile(link["youtube_id_regexp"])
        return self

    def get_sample(self, _column: str, offset: int) -> object:
        sample = super().get_sample(_column, offset)
        assert isinstance(sample, WSAudio)
        assert self._source_file_name is not None
        match = self.re_pattern.search(self._source_file_name)
        if not match:
            raise ValueError(
                f"No Youtube ID found in file name: {self._source_file_name} (using pattern: {self.re_pattern.pattern})"
            )
        return WSYouTubeVideo(match[1], sample.tstart)


@dataclass(slots=True)
class WSSourceLink(WSShardInterface):
    """A proxy shard class to access all fields from a linked source dataset.

    It is used via the `.wsds-link` file mechanism with a `key_prefix` to expose
    all source dataset fields with a prefix (e.g., `source.audio`, `source.vad`).

    The link file format:
    {"dataset_dir": "../source", "loader": ["wsds.ws_shard", "WSSourceLink"], "key_prefix": "source."}
    """

    shard_ref: tuple[str, str]
    source_dataset: WSDataset
    derived_dataset: WSDataset
    key_prefix: str

    # cache
    _source_file_name: Optional[str] = None
    _source_sample: Optional[WSSample] = None

    @classmethod
    def get_columns(cls, link: dict[str, Any], dataset: WSDataset) -> dict[str, str]:
        """Return all source dataset fields with the configured prefix."""
        source_dataset = dataset.get_linked_dataset(link["dataset_dir"])
        key_prefix = link.get("key_prefix", "source.")
        columns = {}
        for field_name in source_dataset.fields:
            if field_name == "__key__":
                continue
            prefixed = f"{key_prefix}{field_name}"
            columns[prefixed] = prefixed
        return columns

    @classmethod
    def from_link(cls, link: dict[str, Any], dataset: WSDataset, shard_ref: tuple[str, str]) -> WSSourceLink:
        source_dataset = dataset.get_linked_dataset(link["dataset_dir"])
        key_prefix = link.get("key_prefix", "source.")
        return cls(shard_ref, source_dataset, dataset, key_prefix)

    def get_sample(self, column: str, offset: int) -> object:
        # Parse the derived dataset's key to get the source file name
        derived_key = cast(str, WSSample(self.derived_dataset, self.shard_ref, offset)["__key__"])
        file_name, _segment_offset = self.derived_dataset.parse_key(derived_key)

        if self._source_file_name != file_name:
            self._source_sample = self.source_dataset[file_name]
            self._source_file_name = file_name

        # Strip prefix to get the actual source field name
        if column.startswith(self.key_prefix):
            source_field = column[len(self.key_prefix) :]
        else:
            source_field = column

        assert self._source_sample is not None
        return self._source_sample[source_field]


@dataclass
class WSYouTubeVideo:
    id: str
    tstart: float

    def get_url(self) -> str:
        return f"https://www.youtube.com/embed/{self.id}?start={int(self.tstart)}"

    def _repr_html_(self) -> str:
        return f'<iframe width="560" height="315" src="{self.get_url()}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'

    def __repr__(self) -> str:
        return f'WSYouTubeVideo(video_url="{self.get_url()}")'
