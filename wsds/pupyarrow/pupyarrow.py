"""
Pure-Python API for parsing PyArrow Feather (IPC) files using FlatBuffers.

This module provides a clean, Pythonic interface for reading Arrow IPC files
without requiring PyArrow at runtime (only the flatbuffers package is needed).

Features:
- Lazy data loading with numpy arrays
- Coalesced small reads for network efficiency
- Eager metadata loading (offsets, lengths)
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from wsds.pupyarrow.file_reader import FileReader, LocalFileReader

# Import generated flatbuffer classes
from wsds.pupyarrow.flatbuf import Footer as FBFooter
from wsds.pupyarrow.flatbuf import Message as FBMessage
from wsds.pupyarrow.flatbuf import RecordBatch as FBRecordBatch

# Arrow magic bytes
ARROW_MAGIC = b"ARROW1"

# Minimum size for individual reads - smaller reads are coalesced
MIN_READ_SIZE = 128


class MetadataVersion(IntEnum):
    """Arrow metadata version."""

    V1 = 0  # 0.1.0
    V2 = 1  # 0.2.0
    V3 = 2  # 0.3.0 -> 0.7.1
    V4 = 3  # >= 0.8.0
    V5 = 4  # >= 1.0.0


class ArrowType(IntEnum):
    """Arrow logical type identifiers."""

    NONE = 0
    Null = 1
    Int = 2
    FloatingPoint = 3
    Binary = 4
    Utf8 = 5
    Bool = 6
    Decimal = 7
    Date = 8
    Time = 9
    Timestamp = 10
    Interval = 11
    List = 12
    Struct = 13
    Union = 14
    FixedSizeBinary = 15
    FixedSizeList = 16
    Map = 17
    Duration = 18
    LargeBinary = 19
    LargeUtf8 = 20
    LargeList = 21
    RunEndEncoded = 22
    BinaryView = 23
    Utf8View = 24
    ListView = 25
    LargeListView = 26


class MessageType(IntEnum):
    """Arrow message header type."""

    NONE = 0
    Schema = 1
    DictionaryBatch = 2
    RecordBatch = 3
    Tensor = 4
    SparseTensor = 5


@dataclass
class Buffer:
    """Describes a contiguous memory segment in the record batch body."""

    offset: int
    length: int


@dataclass
class FieldNode:
    """Metadata about a field at some level of a nested type tree."""

    length: int
    null_count: int


@dataclass
class Field:
    """A named column in a record batch or child of a nested type."""

    name: str | None
    nullable: bool
    type_id: ArrowType
    type_metadata: dict[str, Any]
    children: list[Field]
    custom_metadata: dict[str, str]

    @classmethod
    def from_flatbuf(cls, fb_field) -> Field:
        """Create a Field from a flatbuffer Field object."""
        name = fb_field.Name()
        if name is not None:
            name = name.decode("utf-8")

        type_id = ArrowType(fb_field.TypeType())
        type_metadata = cls._parse_type_metadata(fb_field, type_id)

        children = []
        for i in range(fb_field.ChildrenLength()):
            children.append(cls.from_flatbuf(fb_field.Children(i)))

        custom_metadata = {}
        for i in range(fb_field.CustomMetadataLength()):
            kv = fb_field.CustomMetadata(i)
            key = kv.Key().decode("utf-8") if kv.Key() else ""
            value = kv.Value().decode("utf-8") if kv.Value() else ""
            custom_metadata[key] = value

        return cls(
            name=name,
            nullable=fb_field.Nullable(),
            type_id=type_id,
            type_metadata=type_metadata,
            children=children,
            custom_metadata=custom_metadata,
        )

    @staticmethod
    def _parse_type_metadata(fb_field, type_id: ArrowType) -> dict[str, Any]:
        """Parse type-specific metadata from the flatbuffer type union."""
        metadata: dict[str, Any] = {}
        fb_type = fb_field.Type()
        if fb_type is None:
            return metadata

        if type_id == ArrowType.Int:
            from wsds.pupyarrow.flatbuf.Int import Int

            int_type = Int()
            int_type.Init(fb_type.Bytes, fb_type.Pos)
            metadata["bit_width"] = int_type.BitWidth()
            metadata["is_signed"] = int_type.IsSigned()

        elif type_id == ArrowType.FloatingPoint:
            from wsds.pupyarrow.flatbuf.FloatingPoint import FloatingPoint

            fp_type = FloatingPoint()
            fp_type.Init(fb_type.Bytes, fb_type.Pos)
            precision_map = {0: "half", 1: "single", 2: "double"}
            metadata["precision"] = precision_map.get(fp_type.Precision(), "unknown")

        elif type_id == ArrowType.FixedSizeBinary:
            from wsds.pupyarrow.flatbuf.FixedSizeBinary import FixedSizeBinary

            fsb_type = FixedSizeBinary()
            fsb_type.Init(fb_type.Bytes, fb_type.Pos)
            metadata["byte_width"] = fsb_type.ByteWidth()

        elif type_id == ArrowType.FixedSizeList:
            from wsds.pupyarrow.flatbuf.FixedSizeList import FixedSizeList

            fsl_type = FixedSizeList()
            fsl_type.Init(fb_type.Bytes, fb_type.Pos)
            metadata["list_size"] = fsl_type.ListSize()

        elif type_id == ArrowType.Timestamp:
            from wsds.pupyarrow.flatbuf.Timestamp import Timestamp

            ts_type = Timestamp()
            ts_type.Init(fb_type.Bytes, fb_type.Pos)
            unit_map = {0: "s", 1: "ms", 2: "us", 3: "ns"}
            metadata["unit"] = unit_map.get(ts_type.Unit(), "unknown")
            tz = ts_type.Timezone()
            metadata["timezone"] = tz.decode("utf-8") if tz else None

        elif type_id == ArrowType.Date:
            from org.apache.arrow.flatbuf.Date import Date

            date_type = Date()
            date_type.Init(fb_type.Bytes, fb_type.Pos)
            unit_map = {0: "day", 1: "ms"}
            metadata["unit"] = unit_map.get(date_type.Unit(), "unknown")

        elif type_id == ArrowType.Time:
            from wsds.pupyarrow.flatbuf.Time import Time

            time_type = Time()
            time_type.Init(fb_type.Bytes, fb_type.Pos)
            unit_map = {0: "s", 1: "ms", 2: "us", 3: "ns"}
            metadata["unit"] = unit_map.get(time_type.Unit(), "unknown")
            metadata["bit_width"] = time_type.BitWidth()

        elif type_id == ArrowType.Decimal:
            from wsds.pupyarrow.flatbuf.Decimal import Decimal

            dec_type = Decimal()
            dec_type.Init(fb_type.Bytes, fb_type.Pos)
            metadata["precision"] = dec_type.Precision()
            metadata["scale"] = dec_type.Scale()
            metadata["bit_width"] = dec_type.BitWidth()

        elif type_id == ArrowType.Duration:
            from wsds.pupyarrow.flatbuf.Duration import Duration

            dur_type = Duration()
            dur_type.Init(fb_type.Bytes, fb_type.Pos)
            unit_map = {0: "s", 1: "ms", 2: "us", 3: "ns"}
            metadata["unit"] = unit_map.get(dur_type.Unit(), "unknown")

        elif type_id == ArrowType.Interval:
            from wsds.pupyarrow.flatbuf.Interval import Interval

            int_type = Interval()
            int_type.Init(fb_type.Bytes, fb_type.Pos)
            unit_map = {0: "year_month", 1: "day_time", 2: "month_day_nano"}
            metadata["unit"] = unit_map.get(int_type.Unit(), "unknown")

        elif type_id == ArrowType.Map:
            from wsds.pupyarrow.flatbuf.Map import Map

            map_type = Map()
            map_type.Init(fb_type.Bytes, fb_type.Pos)
            metadata["keys_sorted"] = map_type.KeysSorted()

        elif type_id == ArrowType.Union:
            from wsds.pupyarrow.flatbuf.Union import Union

            union_type = Union()
            union_type.Init(fb_type.Bytes, fb_type.Pos)
            mode_map = {0: "sparse", 1: "dense"}
            metadata["mode"] = mode_map.get(union_type.Mode(), "unknown")
            type_ids = []
            for i in range(union_type.TypeIdsLength()):
                type_ids.append(union_type.TypeIds(i))
            metadata["type_ids"] = type_ids

        return metadata

    def __repr__(self) -> str:
        type_str = self.type_id.name
        if self.type_metadata:
            type_str += f"({self.type_metadata})"
        nullable_str = "?" if self.nullable else ""
        return f"Field({self.name!r}: {type_str}{nullable_str})"


@dataclass
class Schema:
    """Describes the columns in a record batch."""

    fields: list[Field]
    custom_metadata: dict[str, str]
    endianness: str  # "little" or "big"

    @classmethod
    def from_flatbuf(cls, fb_schema) -> Schema:
        """Create a Schema from a flatbuffer Schema object."""
        fields = []
        for i in range(fb_schema.FieldsLength()):
            fields.append(Field.from_flatbuf(fb_schema.Fields(i)))

        custom_metadata = {}
        for i in range(fb_schema.CustomMetadataLength()):
            kv = fb_schema.CustomMetadata(i)
            key = kv.Key().decode("utf-8") if kv.Key() else ""
            value = kv.Value().decode("utf-8") if kv.Value() else ""
            custom_metadata[key] = value

        endianness = "little" if fb_schema.Endianness() == 0 else "big"

        return cls(
            fields=fields,
            custom_metadata=custom_metadata,
            endianness=endianness,
        )

    @property
    def names(self) -> list[str | None]:
        """Return the list of field names."""
        return [f.name for f in self.fields]

    def field(self, name: str) -> Field | None:
        """Get a field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, key: int | str) -> Field:
        if isinstance(key, int):
            return self.fields[key]
        for f in self.fields:
            if f.name == key:
                return f
        raise KeyError(f"Field {key!r} not found")


@dataclass
class BlockInfo:
    """Information about a record batch block in the file."""

    offset: int
    metadata_length: int
    body_length: int


@dataclass
class RecordBatchInfo:
    """Parsed record batch metadata (without the actual data)."""

    length: int  # Number of rows
    nodes: list[FieldNode]
    buffers: list[Buffer]
    compression: str | None
    variadic_buffer_counts: list[int]


def _decompress_buffer(raw_data: bytes, compression: str | None) -> bytes:
    """
    Decompress a buffer if compression is enabled.

    Arrow IPC compression format: first 8 bytes are uncompressed length as int64 LE,
    followed by compressed data. If uncompressed length is -1, data is not compressed.
    """
    if compression is None or len(raw_data) == 0:
        return raw_data

    # First 8 bytes are the uncompressed length as int64 little-endian
    uncompressed_length = struct.unpack("<q", raw_data[:8])[0]
    compressed_data = raw_data[8:]

    if uncompressed_length == -1:
        # Data is not compressed
        return compressed_data

    # Decompress based on codec
    if compression == "zstd":
        import zstd

        return zstd.decompress(compressed_data)
    elif compression == "lz4_frame":
        import lz4.frame

        return lz4.frame.decompress(compressed_data)
    else:
        raise ValueError(f"Unknown compression codec: {compression}")


class LazyBuffer:
    """
    A lazy buffer that reads data on demand and implements the file-like interface.

    Stores offset/length metadata eagerly, loads actual data lazily.
    Supports zstd and lz4_frame compression.

    Supports the Python file-like interface (read, seek, tell) so it can be
    passed directly to consumers expecting a binary file object (e.g. np.load,
    pickle.load, torchaudio).

    Use ``slice(start, end)`` to create a lightweight sub-buffer backed by the
    same reader with adjusted offset/length.  For uncompressed buffers the
    slice goes directly to the reader; for compressed buffers the parent data
    is decompressed once and the slice is pre-populated.
    """

    # __slots__ = ("_reader", "_offset", "_length", "_data", "_compression", "_pos")

    def __init__(self, reader: FileReader, offset: int, length: int, compression: str | None = None):
        self._reader = reader
        self._offset = offset
        self._length = length
        self._data: bytes | None = None
        self._compression = compression
        self._pos = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def length(self) -> int:
        return self._length

    def _read_all(self) -> bytes:
        """Read and cache the full buffer data, decompressing if necessary."""
        if self._data is None:
            raw_data = self._reader.read(self._offset, self._length)
            self._data = _decompress_buffer(raw_data, self._compression)
        return self._data

    def read(self, size: int = -1) -> bytes:
        """Read up to *size* bytes from the current position.

        With no argument or size=-1, reads through the end of the buffer.
        Advances the position by the number of bytes returned.
        """
        remaining = self._length - self._pos
        if size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        if size <= 0:
            return b""
        data = self.read_range(self._pos, self._pos + size)
        self._pos += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the read position.

        whence: 0 = from start, 1 = from current, 2 = from end.
        """
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._length + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._length))
        return self._pos

    def tell(self) -> int:
        """Return the current read position."""
        return self._pos

    def read_range(self, start: int, end: int) -> bytes:
        """Read a byte range from the (decompressed) buffer.

        If the buffer is already cached, slices it. If uncompressed,
        reads directly from the range without reading the whole buffer.
        If compressed, falls back to a full read and slices.
        """
        if self._data is not None:
            return self._data[start:end]
        if self._compression is None:
            return self._reader.read(self._offset + start, end - start)
        return self._read_all()[start:end]

    def slice(self, start: int, end: int) -> "LazyBuffer":
        """Create a sub-buffer over the byte range [start, end).

        For uncompressed data the returned buffer points directly at the
        reader with an adjusted offset — no data is copied or loaded.
        For compressed data the parent is decompressed once and the slice
        is pre-populated with the requested range.
        """
        child = LazyBuffer(self._reader, self._offset + start, end - start, compression=None)
        if self._data is not None:
            child._data = self._data[start:end]
        elif self._compression is not None:
            child._data = self._read_all()[start:end]
        return child

    def as_numpy(self, dtype: np.dtype) -> np.ndarray:
        """Read the buffer as a numpy array with the given dtype."""
        data = self._read_all()
        return np.frombuffer(data, dtype=dtype)

    # -- Async API (IO plan aware) ------------------------------------------

    async def async_read_all(self) -> bytes:
        """Async read via reader (uses cache/plan mode if active)."""
        if self._data is not None:
            return self._data
        raw = await self._reader.async_read(self._offset, self._length)
        self._data = _decompress_buffer(raw, self._compression)
        return self._data

    async def async_as_numpy(self, dtype: np.dtype) -> np.ndarray:
        """Async version of as_numpy."""
        data = await self.async_read_all()
        return np.frombuffer(data, dtype=dtype)

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        loaded = "loaded" if self._data is not None else "not loaded"
        return f"LazyBuffer(offset={self._offset}, length={self._length}, {loaded})"


class LazyArray:
    """
    Base class for lazy Arrow arrays.

    Provides numpy array access with lazy loading.
    Metadata (offsets, lengths) is loaded eagerly.
    """

    def __init__(
        self,
        field: Field,
        node: FieldNode,
        buffers: list[LazyBuffer],
    ):
        self.field = field
        self.node = node
        self._buffers = buffers
        self._validity: np.ndarray | None = None

    @property
    def length(self) -> int:
        return self.node.length

    @property
    def null_count(self) -> int:
        return self.node.null_count

    def __len__(self) -> int:
        return self.node.length

    def validity_mask(self) -> np.ndarray | None:
        """
        Return a boolean mask indicating valid (non-null) values.

        Returns None if there are no nulls (all values valid).
        """
        if self.null_count == 0:
            return None

        if self._validity is None:
            validity_buf = self._buffers[0]
            if validity_buf.length == 0:
                return None
            validity_bytes = validity_buf._read_all()
            # Unpack bits into boolean array
            packed = np.frombuffer(validity_bytes, dtype=np.uint8)
            self._validity = np.unpackbits(packed, bitorder="little")[: self.length].astype(bool)

        return self._validity

    async def async_resolve(self) -> None:
        """Prefetch all buffers for this array (coalesced if IO plan active)."""
        futs = [buf.async_read_all() for buf in self._buffers if buf._length > 0]
        if futs:
            await asyncio.gather(*futs)

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not support to_numpy()")

    def to_masked_array(self) -> np.ma.MaskedArray:
        """Return values as a masked array with nulls masked."""
        values = self.to_numpy()
        mask = self.validity_mask()
        if mask is None:
            return np.ma.array(values, mask=False)
        return np.ma.array(values, mask=~mask)

    def __getitem__(self, idx: int | slice) -> Any:
        return self.to_numpy()[idx]

    async def async_to_numpy(self) -> np.ndarray:
        await self.async_resolve()
        return self.to_numpy()

    async_to_py = async_to_numpy


class LazyIntArray(LazyArray):
    """Lazy integer array with numpy access."""

    _DTYPE_MAP = {
        (8, True): np.int8,
        (8, False): np.uint8,
        (16, True): np.int16,
        (16, False): np.uint16,
        (32, True): np.int32,
        (32, False): np.uint32,
        (64, True): np.int64,
        (64, False): np.uint64,
    }

    def __init__(self, field: Field, node: FieldNode, buffers: list[LazyBuffer]):
        super().__init__(field, node, buffers)
        bit_width = field.type_metadata.get("bit_width", 64)
        is_signed = field.type_metadata.get("is_signed", True)
        self._dtype = self._DTYPE_MAP[(bit_width, is_signed)]
        self._values: np.ndarray | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dtype)

    def to_numpy(self) -> np.ndarray:
        """Return values as a numpy array (without null handling)."""
        if self._values is None:
            self._values = self._buffers[1].as_numpy(self._dtype)[: self.length]
        return self._values

    def __repr__(self) -> str:
        return f"LazyIntArray(dtype={self._dtype.__name__}, length={self.length}, nulls={self.null_count})"


class LazyFloatArray(LazyArray):
    """Lazy floating point array with numpy access."""

    _DTYPE_MAP = {"half": np.float16, "single": np.float32, "double": np.float64}

    def __init__(self, field: Field, node: FieldNode, buffers: list[LazyBuffer]):
        super().__init__(field, node, buffers)
        precision = field.type_metadata.get("precision", "double")
        self._dtype = self._DTYPE_MAP[precision]
        self._values: np.ndarray | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._dtype)

    def to_numpy(self) -> np.ndarray:
        """Return values as a numpy array (without null handling)."""
        if self._values is None:
            self._values = self._buffers[1].as_numpy(self._dtype)[: self.length]
        return self._values

    def __repr__(self) -> str:
        return f"LazyFloatArray(dtype={self._dtype.__name__}, length={self.length}, nulls={self.null_count})"


class LazyBoolArray(LazyArray):
    """Lazy boolean array with numpy access."""

    def __init__(self, field: Field, node: FieldNode, buffers: list[LazyBuffer]):
        super().__init__(field, node, buffers)
        self._values: np.ndarray | None = None

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(bool)

    def to_numpy(self) -> np.ndarray:
        """Return values as a numpy boolean array."""
        if self._values is None:
            data = self._buffers[1]._read_all()
            packed = np.frombuffer(data, dtype=np.uint8)
            self._values = np.unpackbits(packed, bitorder="little")[: self.length].astype(bool)
        return self._values

    def __repr__(self) -> str:
        return f"LazyBoolArray(length={self.length}, nulls={self.null_count})"


class LazyBinaryArray(LazyArray):
    """
    Lazy binary (Binary/LargeBinary) array.

    Offsets and binary data are both loaded lazily on first access.
    """

    def __init__(
        self,
        field: Field,
        node: FieldNode,
        buffers: list[LazyBuffer],
        large: bool = False,
    ):
        super().__init__(field, node, buffers)
        self._large = large
        self._offsets: np.ndarray | None = None
        self._data_buffer = buffers[2]
        self._data: bytes | None = None

    def _ensure_offsets(self) -> np.ndarray:
        """Load offset array on first access."""
        if self._offsets is None:
            offset_dtype = np.int64 if self._large else np.int32
            self._offsets = self._buffers[1].as_numpy(offset_dtype)[: self.length + 1]
        return self._offsets

    @property
    def offsets(self) -> np.ndarray:
        """Return the offset array (loaded on first access)."""
        return self._ensure_offsets()

    def _ensure_data(self) -> bytes:
        """Lazily load binary data buffer."""
        if self._data is None:
            self._data = self._data_buffer._read_all()
        return self._data

    def __getitem__(self, idx: int | slice):
        """Get element(s) by index or slice."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.length))
            return [self._get_single(i) for i in indices]

        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for array of length {self.length}")

        return self._get_single(idx)

    def _get_single(self, idx: int) -> LazyBuffer | None:
        """Get a single binary value as a file-like LazyBuffer by index."""
        mask = self.validity_mask()
        if mask is not None and not mask[idx]:
            return None

        offsets = self._ensure_offsets()
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        return self._data_buffer.slice(start, end)

    def read_range(self, idx: int, start: int, end: int) -> bytes:
        """
        Read a byte range from a specific element without loading all data.

        Useful for large binary blobs where you only need a portion.
        """
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range")

        offsets = self._ensure_offsets()
        elem_start = int(offsets[idx])
        elem_end = int(offsets[idx + 1])

        # Clamp range to element bounds
        read_start = elem_start + max(0, start)
        read_end = elem_start + min(end, elem_end - elem_start)

        if read_start >= read_end:
            return b""

        return self._data_buffer.read_range(read_start, read_end)

    def to_list(self) -> list[LazyBuffer | None]:
        """Convert to a Python list of file-like LazyBuffers."""
        offsets = self._ensure_offsets()
        mask = self.validity_mask()
        result: list[LazyBuffer | None] = []

        for i in range(self.length):
            if mask is not None and not mask[i]:
                result.append(None)
            else:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                result.append(self._data_buffer.slice(start, end))

        return result

    def byte_sizes(self) -> np.ndarray:
        """Return array of byte sizes for each element (without loading data)."""
        return np.diff(self._ensure_offsets())

    async def async_to_bytes_list(self) -> list[bytes | None]:
        await self.async_resolve()
        data = self._ensure_data()
        offsets = self.offsets
        mask = self.validity_mask()
        return [
            None if (mask is not None and not mask[i]) else data[int(offsets[i]) : int(offsets[i + 1])]
            for i in range(self.length)
        ]

    async_to_py = async_to_bytes_list

    def __repr__(self) -> str:
        type_name = "LargeBinary" if self._large else "Binary"
        data_loaded = "loaded" if self._data is not None else "not loaded"
        return f"LazyBinaryArray({type_name}, length={self.length}, nulls={self.null_count}, data={data_loaded})"


class LazyStringArray(LazyBinaryArray):
    """
    Lazy string (Utf8/LargeUtf8) array.

    Subclass of LazyBinaryArray that decodes values as UTF-8 strings.
    """

    def _get_single(self, idx: int) -> str | None:
        """Get a single string by index, decoding the LazyBuffer from super()."""
        buf = super()._get_single(idx)
        if buf is None:
            return None
        return buf._read_all().decode("utf-8")

    def to_list(self) -> list[str | None]:
        """Convert to a Python list of strings."""
        offsets = self._ensure_offsets()
        data = self._ensure_data()
        mask = self.validity_mask()
        result: list[str | None] = []

        for i in range(self.length):
            if mask is not None and not mask[i]:
                result.append(None)
            else:
                start = int(offsets[i])
                end = int(offsets[i + 1])
                result.append(data[start:end].decode("utf-8"))

        return result

    def to_numpy(self) -> np.ndarray:
        """Return as numpy object array of strings."""
        return np.array(self.to_list(), dtype=object)

    async def async_to_list(self) -> list[str | None]:
        await self.async_resolve()
        return self.to_list()

    async_to_py = async_to_list

    def __repr__(self) -> str:
        type_name = "LargeUtf8" if self._large else "Utf8"
        data_loaded = "loaded" if self._data is not None else "not loaded"
        return f"LazyStringArray({type_name}, length={self.length}, nulls={self.null_count}, data={data_loaded})"


class LazyFixedSizeBinaryArray(LazyArray):
    """Lazy fixed-size binary array with numpy access."""

    def __init__(self, field: Field, node: FieldNode, buffers: list[LazyBuffer]):
        super().__init__(field, node, buffers)
        self._byte_width = field.type_metadata.get("byte_width", 1)
        self._values: np.ndarray | None = None

    @property
    def byte_width(self) -> int:
        return self._byte_width

    def to_numpy(self) -> np.ndarray:
        """Return as 2D numpy array of shape (length, byte_width)."""
        if self._values is None:
            data = self._buffers[1]._read_all()
            self._values = np.frombuffer(data, dtype=np.uint8).reshape(-1, self._byte_width)[: self.length]
        return self._values

    def __repr__(self) -> str:
        return f"LazyFixedSizeBinaryArray(byte_width={self._byte_width}, length={self.length}, nulls={self.null_count})"


# Type alias for any lazy array
LazyArrayType = (
    LazyIntArray
    | LazyFloatArray
    | LazyBoolArray
    | LazyStringArray
    | LazyBinaryArray
    | LazyFixedSizeBinaryArray
    | LazyArray
)


class RecordBatch:
    """
    A record batch with lazy data access.

    Metadata (schema, buffer offsets, lengths) is loaded eagerly.
    Actual data is loaded lazily when accessed.
    """

    def __init__(
        self,
        info: RecordBatchInfo,
        schema: Schema,
        reader: FileReader,
        body_offset: int,
    ):
        self.info = info
        self.schema = schema
        self._reader = reader
        self._body_offset = body_offset

        # Cache for lazy arrays
        self._columns: dict[int, LazyArrayType] = {}

        # Pre-compute buffer indices for each column
        self._column_buffer_indices = self._compute_buffer_indices()

    def _compute_buffer_indices(self) -> list[tuple[int, int]]:
        """Compute (start_buffer_idx, num_buffers) for each column."""
        indices = []
        buf_idx = 0
        for field in self.schema.fields:
            num_buffers = self._get_num_buffers_for_type(field.type_id)
            indices.append((buf_idx, num_buffers))
            buf_idx += num_buffers
        return indices

    @staticmethod
    def _get_num_buffers_for_type(type_id: ArrowType) -> int:
        """Return the number of buffers used by a type."""
        if type_id == ArrowType.Null:
            return 0
        if type_id in (ArrowType.Utf8, ArrowType.Binary):
            return 3
        if type_id in (ArrowType.LargeUtf8, ArrowType.LargeBinary):
            return 3
        if type_id in (ArrowType.List, ArrowType.Map):
            return 2
        if type_id == ArrowType.LargeList:
            return 2
        return 2

    @property
    def num_rows(self) -> int:
        return self.info.length

    @property
    def num_columns(self) -> int:
        return len(self.schema.fields)

    def _get_lazy_buffers(self, column_index: int) -> list[LazyBuffer]:
        """Get LazyBuffer objects for a column."""
        start_idx, num_buffers = self._column_buffer_indices[column_index]

        lazy_buffers = []
        for i in range(num_buffers):
            buf = self.info.buffers[start_idx + i]
            lazy_buffers.append(
                LazyBuffer(
                    self._reader,
                    self._body_offset + buf.offset,
                    buf.length,
                    self.info.compression,
                )
            )
        return lazy_buffers

    def column(self, column: int | str) -> LazyArrayType:
        """
        Get a lazy array for a column by index or name.

        Returns a type-specific lazy array that loads data on demand.
        """
        if isinstance(column, str):
            for i, f in enumerate(self.schema.fields):
                if f.name == column:
                    column = i
                    break
            else:
                raise KeyError(f"Column {column!r} not found")

        if column in self._columns:
            return self._columns[column]

        field = self.schema.fields[column]
        node = self.info.nodes[column]
        buffers = self._get_lazy_buffers(column)

        # Create appropriate lazy array type
        type_id = field.type_id

        if type_id == ArrowType.Int:
            arr = LazyIntArray(field, node, buffers)
        elif type_id == ArrowType.FloatingPoint:
            arr = LazyFloatArray(field, node, buffers)
        elif type_id == ArrowType.Bool:
            arr = LazyBoolArray(field, node, buffers)
        elif type_id == ArrowType.Utf8:
            arr = LazyStringArray(field, node, buffers, large=False)
        elif type_id == ArrowType.LargeUtf8:
            arr = LazyStringArray(field, node, buffers, large=True)
        elif type_id == ArrowType.Binary:
            arr = LazyBinaryArray(field, node, buffers, large=False)
        elif type_id == ArrowType.LargeBinary:
            arr = LazyBinaryArray(field, node, buffers, large=True)
        elif type_id == ArrowType.FixedSizeBinary:
            arr = LazyFixedSizeBinaryArray(field, node, buffers)
        else:
            # Generic lazy array for unsupported types
            arr = LazyArray(field, node, buffers)

        self._columns[column] = arr
        return arr

    def columns(self) -> list[LazyArrayType]:
        """Get all columns as lazy arrays."""
        return [self.column(i) for i in range(self.num_columns)]

    def __repr__(self) -> str:
        return f"RecordBatch(rows={self.num_rows}, columns={self.num_columns})"


def _parse_record_batch_message(message_bytes: bytes) -> RecordBatchInfo:
    """Parse a record batch message into RecordBatchInfo (no IO)."""
    continuation = struct.unpack("<i", message_bytes[:4])[0]
    metadata_offset = 8 if continuation == -1 else 0

    message = FBMessage.Message.GetRootAsMessage(message_bytes, metadata_offset)

    if message.HeaderType() != MessageType.RecordBatch:
        raise ValueError(f"Expected RecordBatch, got {MessageType(message.HeaderType()).name}")

    rb = FBRecordBatch.RecordBatch()
    rb.Init(message.Header().Bytes, message.Header().Pos)

    nodes = []
    for i in range(rb.NodesLength()):
        node = rb.Nodes(i)
        nodes.append(FieldNode(length=node.Length(), null_count=node.NullCount()))

    buffers = []
    for i in range(rb.BuffersLength()):
        buf = rb.Buffers(i)
        buffers.append(Buffer(offset=buf.Offset(), length=buf.Length()))

    compression = None
    if rb.Compression() is not None:
        comp = rb.Compression()
        codec_map = {0: "lz4_frame", 1: "zstd"}
        compression = codec_map.get(comp.Codec(), "unknown")

    variadic_counts = []
    for i in range(rb.VariadicBufferCountsLength()):
        variadic_counts.append(rb.VariadicBufferCounts(i))

    return RecordBatchInfo(
        length=rb.Length(),
        nodes=nodes,
        buffers=buffers,
        compression=compression,
        variadic_buffer_counts=variadic_counts,
    )


class FeatherFile:
    """
    A pure-Python reader for Arrow IPC (Feather v2) files.

    Features:
    - Lazy data loading with numpy arrays
    - Eager metadata loading

    Example usage:
        with FeatherFile("data.feather") as f:
            print(f.schema)
            for batch in f.record_batches():
                # Get lazy array - offsets loaded, data not yet
                col = batch.column("name")
                # Data loaded on access
                values = col.to_numpy()
    """

    def __init__(self, path_or_file: str | Path | FileReader):
        """
        Open a Feather file.

        Args:
            path_or_file: Path to the file or a FileReader.
        """
        if hasattr(path_or_file, "read_end"):
            self._reader = path_or_file
        elif isinstance(path_or_file, (str, Path)):
            self._reader = LocalFileReader(path_or_file)
        else:
            raise TypeError(
                f"Unsupported type for path_or_file: {type(path_or_file)} (expected str, Path, or FileReader)"
            )

        self._footer = self._read_footer()
        self._schema = Schema.from_flatbuf(self._footer.Schema())
        self._record_batch_blocks = self._read_record_batch_blocks()

        # Eagerly parse record batch metadata
        self._record_batch_infos: list[RecordBatchInfo | None] = [None] * len(self._record_batch_blocks)

    def _read_footer(self):
        """Read and parse the footer from the end of the file."""
        # don't check start magic to reduce I/O latency
        end_magic = self._reader.read_end(-6, 6)

        if end_magic != ARROW_MAGIC:
            raise ValueError("Not a valid Arrow IPC file (magic bytes mismatch)")

        footer_size = struct.unpack("<I", self._reader.read_end(-10, 4))[0]
        footer_bytes = self._reader.read_end(-(10 + footer_size), footer_size)

        return FBFooter.Footer.GetRootAsFooter(footer_bytes, 0)

    def _read_record_batch_blocks(self) -> list[BlockInfo]:
        """Extract record batch block information from the footer."""
        blocks = []
        for i in range(self._footer.RecordBatchesLength()):
            block = self._footer.RecordBatches(i)
            blocks.append(
                BlockInfo(
                    offset=block.Offset(),
                    metadata_length=block.MetaDataLength(),
                    body_length=block.BodyLength(),
                )
            )
        return blocks

    def _parse_record_batch_info(self, index: int) -> RecordBatchInfo:
        """Parse record batch metadata (nodes, buffers, etc.)."""
        if self._record_batch_infos[index] is not None:
            return self._record_batch_infos[index]  # type: ignore

        block = self._record_batch_blocks[index]
        message_bytes = self._reader.read(block.offset, block.metadata_length)
        info = _parse_record_batch_message(message_bytes)

        self._record_batch_infos[index] = info
        return info

    @property
    def schema(self) -> Schema:
        """The schema describing all columns."""
        return self._schema

    @property
    def version(self) -> MetadataVersion:
        """The Arrow metadata version."""
        return MetadataVersion(self._footer.Version())

    @property
    def num_record_batches(self) -> int:
        """Number of record batches in the file."""
        return len(self._record_batch_blocks)

    @property
    def custom_metadata(self) -> dict[str, str]:
        """File-level custom metadata."""
        return self._schema.custom_metadata

    def record_batch(self, index: int) -> RecordBatch:
        """
        Get a specific record batch by index.

        Metadata is loaded eagerly, data is lazy.

        Args:
            index: The record batch index (0-based).

        Returns:
            A RecordBatch object with lazy data access.
        """
        if index < 0 or index >= len(self._record_batch_blocks):
            raise IndexError(f"Record batch index {index} out of range")

        block = self._record_batch_blocks[index]
        info = self._parse_record_batch_info(index)
        body_offset = block.offset + block.metadata_length

        return RecordBatch(
            info=info,
            schema=self._schema,
            reader=self._reader,
            body_offset=body_offset,
        )

    def record_batches(self) -> Iterator[RecordBatch]:
        """Iterate over all record batches in the file."""
        for i in range(self.num_record_batches):
            yield self.record_batch(i)

    async def async_record_batch(self, index: int) -> RecordBatch:
        """Async version of record_batch — reads metadata via async_read if not cached."""
        if self._record_batch_infos[index] is None:
            block = self._record_batch_blocks[index]
            raw = await self._reader.async_read(block.offset, block.metadata_length)
            self._record_batch_infos[index] = _parse_record_batch_message(raw)
        return self.record_batch(index)

    def __getitem__(self, key: str | tuple[str, ...] | list[str]) -> np.ndarray | list | dict[str, np.ndarray | list]:
        """
        Read column data across all batches with concurrent IO.

        Single column returns the resolved array/list directly.
        Multiple columns return a dict mapping column name to resolved data.

        Numeric/bool/fixed-size-binary columns return np.ndarray.
        String columns return list[str | None].
        Binary columns return list[bytes | None].

        Uses the reader's plan mode to coalesce reads across all batches.
        Per-batch async tasks submit reads; the execution loop flushes
        them in coalesced rounds.

        Examples:
            values = f["score"]                  # np.ndarray
            texts = f["text"]                    # list[str]
            cols = f["score", "text"]             # dict
            cols = f[["score", "text"]]           # dict (also works)
        """
        if isinstance(key, str):
            names = [key]
        elif isinstance(key, (tuple, list)):
            names = list(key)
        else:
            raise TypeError(f"Key must be str, tuple[str, ...], or list[str], got {type(key).__name__}")

        for name in names:
            if self._schema.field(name) is None:
                raise KeyError(f"Column {name!r} not in schema")

        from wsds.pupyarrow.file_reader import _get_io_loop

        batch_results = _get_io_loop().run(self._async_getitem(names))

        # Concatenate across batches
        output: dict[str, np.ndarray | list] = {}
        for name in names:
            field = self._schema.field(name)
            is_numpy = field.type_id in (
                ArrowType.Int,
                ArrowType.FloatingPoint,
                ArrowType.Bool,
                ArrowType.FixedSizeBinary,
            )
            chunks = [br[name] for br in batch_results]
            if is_numpy:
                output[name] = np.concatenate(chunks) if len(chunks) > 1 else chunks[0] if chunks else np.array([])
            else:
                output[name] = [item for chunk in chunks for item in chunk]

        return output[names[0]] if isinstance(key, str) else output

    async def _async_getitem(self, names: list[str]) -> list[dict[str, np.ndarray | list]]:
        """Async entry point: resolve all batches with coalesced IO."""

        async def resolve_batch(batch_idx: int) -> dict[str, np.ndarray | list]:
            batch = await self.async_record_batch(batch_idx)
            return {name: await batch.column(name).async_to_py() for name in names}

        self._reader._planned.set(True)
        try:
            tasks = [asyncio.ensure_future(resolve_batch(i)) for i in range(self.num_record_batches)]
            while not all(t.done() for t in tasks):
                await asyncio.sleep(0)
                await self._reader.flush()
            return [t.result() for t in tasks]
        finally:
            self._reader._planned.set(False)
            self._reader.clear_cache()

    # -- Context manager & lifecycle -------------------------------------------

    def __enter__(self) -> FeatherFile:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying reader."""
        self._reader.close()

    def __repr__(self) -> str:
        return (
            f"FeatherFile(version={self.version.name}, columns={len(self._schema)}, batches={self.num_record_batches})"
        )
