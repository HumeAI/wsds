import io
import re
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from .utils import WSShardMissingError
from .ws_audio import WSAudioEpisode, WSAudioSegment
from .ws_decode import decode_sample, decode_arr, AUDIO_FILE_KEYS
from .ws_sample import WSSample

import struct


def _zero_copy_blob_reader(arr, j):
    """Zero-copy seekable file-like over element ``j`` of a (large_)binary array.

    Reads only ``offsets[j:j+2]`` and slices the values buffer, WITHOUT
    materializing the element into a pyarrow scalar. ``col[j]`` copies/faults the
    ENTIRE blob (measured 100-300 ms on cold 1.8 GB mmap'd shards) even when the
    decoder only seeks to a small region; this hands the decoder an mmap-backed
    view so it faults just the pages it actually reads. Returns None if null."""
    vbuf, offbuf, valbuf = arr.buffers()   # [validity, offsets, values]
    idx = arr.offset + j
    if vbuf is not None and not (memoryview(vbuf)[idx >> 3] & (1 << (idx & 7))):
        return None
    w, fmt = (8, "<q") if pa.types.is_large_binary(arr.type) else (4, "<i")
    ob = memoryview(offbuf)
    o1 = struct.unpack_from(fmt, ob, idx * w)[0]
    o2 = struct.unpack_from(fmt, ob, (idx + 1) * w)[0]
    return pa.BufferReader(valbuf.slice(o1, o2 - o1))

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


class WSShardInterface:
    shard_ref: (str, str)
    """Used by WSDataset to invalidate cached shards."""

    @classmethod
    def get_columns(cls, link: dict, dataset: "WSDataset") -> dict[str, str] | None:
        """Return columns this link provides: {column_name: column_name}.

        Override this to provide multiple columns from a single link.
        Return None to use the default behavior (link file stem as single column).
        """
        return None

    def get_sample(self, column: str, offset: int) -> typing.Any:
        raise NotImplementedError


class WSShard(WSShardInterface):
    """Represents a single open data shard (`.wsds` file).

    Caches one batch worth of data for efficient sequential access to samples."""

    fname: str
    reader: pa.RecordBatchFileReader
    batch_size: int
    dataset: "WSDataset"

    def __init__(self, dataset, fname, shard_ref=None):
        self.dataset = dataset
        self.shard_ref = shard_ref
        self.fname = fname

        try:
            if dataset.disable_memory_map:
                self._source_file = pa.OSFile(str(fname))
            else:
                self._source_file = pa.memory_map(str(fname))
            self.reader = pa.RecordBatchFileReader(self._source_file)
        except FileNotFoundError:
            raise WSShardMissingError(fname) from None

        self.batch_size = int(self.reader.schema.metadata[b"batch_size"])

        # cache
        self._start = None
        self._end = None
        self._data = None

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._data is None or offset < self._start or offset >= self._end:
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
        col_type = self._data.schema.field(column).type
        ext = column.rsplit(".", 1)[-1] if "." in column else column
        # Audio columns: hand the decoder a ZERO-COPY reader over just this
        # element's bytes (mmap-backed) instead of materializing the whole blob
        # via col[j]. Only audio benefits (it seeks; npy/pyd read fully anyway).
        if ext in AUDIO_FILE_KEYS and (pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type)):
            fd = _zero_copy_blob_reader(self._data.column(column), j)
            return None if fd is None else WSAudioEpisode(fd)
        data = self._data[column][j]
        if not data.is_valid:
            return None # Return None for any null pyarrow scalars
        try:
            if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
                return decode_sample(column, data)
            if ext == "arr":
                return decode_arr(data, col_type)   # native variable-length array
            return data.as_py(maps_as_pydicts="strict")
        except Exception as e:
            raise ValueError(f"Failed to decode column {column} in shard {self.fname} (offset {offset}): {e}")

    def close(self):
        """Close the underlying pyarrow file handle."""
        self.reader = None
        self._data = None
        if hasattr(self, "_source_file") and self._source_file is not None:
            try:
                self._source_file.close()
            except Exception:
                pass
            self._source_file = None

    def __repr__(self):
        r = f"WSShard({repr(self.fname)})"
        if self._data:
            r += f" # cached_region = [{self._start, self._end}]"
        return r


@dataclass(slots=True)
class WSSourceAudioShard(WSShardInterface):
    """A proxy shard class (does not correspond to an actual `.wsds` file) to access audio data from a source dataset.

    It is used via the `WSDataset.add_computed` method or the `.wsds-link` file mechanism."""

    shard_ref: (str, str)
    source_dataset: "WSDataset"  # noqa: F821
    derived_dataset: "WSDataset"  # noqa: F821
    vad_column: str

    # cache
    _source_file_name: str = None
    _source_sample: WSSample = None
    _source_reader: WSAudioEpisode = None

    @classmethod
    def from_link(cls, link, dataset, shard_ref):
        source_dataset = dataset.get_linked_dataset(link["dataset_dir"])
        return cls(shard_ref, source_dataset, dataset, link["vad_column"])

    def get_timestamps(self, segment_offset):
        return self._source_sample[self.vad_column][segment_offset]

    def get_sample(self, _column, offset):
        file_name, segment_offset = self.derived_dataset.parse_key(
            WSSample(self.derived_dataset, self.shard_ref, offset)["__key__"]
        )

        if self._source_file_name != file_name:
            self._source_sample = self.source_dataset[file_name]
            try:
                self._source_reader = self._source_sample.get_audio()
            except KeyError:
                raise WSShardMissingError("no audio shards found")
            self._source_file_name = file_name

        tstart, tend = self.get_timestamps(segment_offset)
        return WSAudioSegment(self._source_reader, tstart, tend)


class WSYoutubeVideoShard(WSSourceAudioShard):
    re_pattern: re.Pattern[str]

    @classmethod
    def from_link(cls, link, dataset, shard_ref):
        self = super().from_link(link, dataset, shard_ref)
        self.re_pattern = re.compile(link["youtube_id_regexp"])
        return self

    def get_sample(self, _column, offset):
        sample = super().get_sample(_column, offset)
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

    shard_ref: (str, str)
    source_dataset: "WSDataset"
    derived_dataset: "WSDataset"
    key_prefix: str

    # cache
    _source_file_name: str = None
    _source_sample: WSSample = None

    @classmethod
    def get_columns(cls, link, dataset):
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
    def from_link(cls, link, dataset, shard_ref):
        source_dataset = dataset.get_linked_dataset(link["dataset_dir"])
        key_prefix = link.get("key_prefix", "source.")
        return cls(shard_ref, source_dataset, dataset, key_prefix)

    def get_sample(self, column: str, offset: int):
        # Parse the derived dataset's key to get the source file name
        derived_key = WSSample(self.derived_dataset, self.shard_ref, offset)["__key__"]
        file_name, _segment_offset = self.derived_dataset.parse_key(derived_key)

        if self._source_file_name != file_name:
            self._source_sample = self.source_dataset[file_name]
            self._source_file_name = file_name

        # Strip prefix to get the actual source field name
        if column.startswith(self.key_prefix):
            source_field = column[len(self.key_prefix) :]
        else:
            source_field = column

        return self._source_sample[source_field]


@dataclass
class WSYouTubeVideo:
    id: str
    tstart: float

    def get_url(self):
        return f"https://www.youtube.com/embed/{self.id}?start={int(self.tstart)}"

    def _repr_html_(self):
        return f'<iframe width="560" height="315" src="{self.get_url()}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'

    def __repr__(self):
        return f'WSYouTubeVideo(video_url="{self.get_url()}")'
