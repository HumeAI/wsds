import bisect
import re
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from .pupyarrow.file_reader import FileReader, LocalFileReader
from .utils import WSShardMissingError
from .ws_audio import WSAudioEpisode, WSAudioSegment
from .ws_decode import decode_sample
from .ws_sample import WSSample

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

    #
    # Shared batch location for readers of .wsds files (local, S3, Modal).
    # Subclasses provide _num_batches/_get_batch/_shard_name and call
    # _locate_batch(offset) from get_sample.
    #
    _start = None
    _end = None
    _row_offsets = None  # cumulative per-batch row offsets, built when batch_size metadata is unreliable
    _batches_verified = 0  # batches [0, _batches_verified) confirmed to hold exactly batch_size rows

    def _num_batches(self) -> int:
        raise NotImplementedError

    def _get_batch(self, index: int):
        raise NotImplementedError

    def _shard_name(self) -> str:
        raise NotImplementedError

    def _batch_row_counts(self) -> list[int]:
        return [self._get_batch(i).num_rows for i in range(self._num_batches())]

    def _locate_batch(self, offset: int):
        """Return the record batch containing row `offset`, setting self._start/_end
        to the batch's row range.

        The `offset // batch_size` arithmetic is only sound when every batch BEFORE
        the target holds exactly `batch_size` rows — some shards have wrong
        batch_size metadata or irregular batch sizes, where it would silently
        return the wrong row. So the fast path is only trusted for batch prefixes
        this shard object has already verified (sequential reads verify as they
        go, at no extra cost); anything else falls back to true cumulative row
        offsets derived from the batch headers themselves.
        """
        if self._row_offsets is None:
            i = offset // self.batch_size
            n = self._num_batches()
            if 0 <= i < n and i <= self._batches_verified:
                batch = self._get_batch(i)
                if batch.num_rows == self.batch_size or i == n - 1:
                    if batch.num_rows == self.batch_size:
                        self._batches_verified = max(self._batches_verified, i + 1)
                    self._start = i * self.batch_size
                    self._end = self._start + batch.num_rows
                    return batch
            offsets = [0]
            for num_rows in self._batch_row_counts():
                offsets.append(offsets[-1] + num_rows)
            self._row_offsets = offsets
        if not 0 <= offset < self._row_offsets[-1]:
            raise IndexError(f"{offset} is out of range for shard {self._shard_name()}")
        i = bisect.bisect_right(self._row_offsets, offset) - 1
        self._start = self._row_offsets[i]
        self._end = self._row_offsets[i + 1]
        return self._get_batch(i)

    def get_reader(self) -> FileReader:
        """Return a pupyarrow FileReader for the underlying shard file."""
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
        self._data = None

    def _num_batches(self) -> int:
        return self.reader.num_record_batches

    def _get_batch(self, index: int):
        return self.reader.get_batch(index)

    def _shard_name(self) -> str:
        return str(self.fname)

    def _batch_row_counts(self) -> list[int]:
        # use a fresh memory map for the scan: it only faults in batch-header pages,
        # while the OSFile reader (disable_memory_map) would read whole batches
        with pa.memory_map(str(self.fname)) as source:
            reader = pa.RecordBatchFileReader(source)
            return [reader.get_batch(i).num_rows for i in range(reader.num_record_batches)]

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._data is None or offset < self._start or offset >= self._end:
            self._data = self._locate_batch(offset)

        j = offset - self._start
        if j >= len(self._data):
            raise IndexError(f"{offset} is out of range for shard {self.fname}")
        if self._data.schema.get_field_index(column) == -1:
            raise KeyError(f"column {column} not found in shard {self.fname}")
        data = self._data[column][j]
        if not data.is_valid:
            return None # Return None for any null pyarrow scalars
        col_type = self._data.schema.field(column).type
        try:
            if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
                return decode_sample(column, data)
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

    def get_reader(self):
        return LocalFileReader(self.fname)

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
