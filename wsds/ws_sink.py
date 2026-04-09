from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow

if TYPE_CHECKING:
    from collections.abc import Sequence

from .ws_decode import encode_value


def indented(prefix, obj):
    lines = str(obj).split("\n")
    return "\n".join(p + ln for p, ln in zip([prefix] + [" " * len(prefix)] * (len(lines) - 1), lines))


@dataclass(frozen=True)
class SampleFormatChanged(BaseException):
    old_schema: pyarrow.Schema
    new_schema: pyarrow.Schema

    def __str__(self):
        return (
            f"The dataset format changed:\n\n"
            f"{indented('  OLD: ', self.old_schema)}\n\n"
            f"{indented('  NEW: ', self.new_schema)}"
        )


@dataclass
class KeyMismatchError(BaseException):
    fname: str
    offset: int
    expected_key: str | None
    actual_key: str | None

    def __str__(self):
        if self.expected_key is None:
            return (
                f"Too many samples in {self.fname}: "
                f"unexpected sample at offset {self.offset} with key '{self.actual_key}'"
            )
        if self.actual_key is None:
            return (
                f"Sample at offset {self.offset} in {self.fname} is missing '__key__' "
                f"(expected '{self.expected_key}')"
            )
        return (
            f"Key mismatch at offset {self.offset} in {self.fname}: "
            f"expected '{self.expected_key}' but got '{self.actual_key}'"
        )


@dataclass
class SampleCountMismatchError(BaseException):
    fname: str
    expected_count: int
    actual_count: int

    def __str__(self):
        return (
            f"Sample count mismatch in {self.fname}: "
            f"expected {self.expected_count} samples but wrote {self.actual_count}"
        )


def _find_reference_shard(shard_path: Path) -> Path | None:
    """Find the smallest sibling shard to use as __key__ reference.

    Mirrors the read-side logic in list_all_columns() (utils.py) which sorts
    __key__ sources by ascending shard file size to avoid loading heavy artifacts.
    """
    column_dir = shard_path.parent
    dataset_dir = column_dir.parent
    shard_name = shard_path.name

    candidates: list[tuple[int, Path]] = []
    for sibling in dataset_dir.iterdir():
        if sibling == column_dir:
            continue
        if sibling.suffix in (".wsds-link", ".wsds-computed"):
            continue
        if not sibling.is_dir():
            continue
        sibling_shard = sibling / shard_name
        if sibling_shard.exists():
            candidates.append((sibling_shard.stat().st_size, sibling_shard))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _read_shard_keys(shard_path: Path) -> list[str]:
    """Read all __key__ values from a shard, in order."""
    reader = pyarrow.ipc.open_file(pyarrow.memory_map(str(shard_path)))
    return reader.read_all().column("__key__").to_pylist()


def _resolve_reference_keys(shard_path: Path) -> list[str] | None:
    """Find the smallest sibling artifact and read its __key__ column as reference.

    Returns None (with a printed warning) if no sibling artifacts exist.
    """
    ref_shard = _find_reference_shard(shard_path)
    if ref_shard is None:
        print(f"Warning: no sibling artifacts found for {shard_path}, skipping key validation")
        return None
    return _read_shard_keys(ref_shard)


class WSBatchedSink:
    """A helper for writing data to a PyArrow feather file.

    Automatically batches data and infers the schema from the first batch.

    Example:
    >>> with WSBatchedSink('output.feather', throwaway=True) as sink: sink.write({'a': 1, 'b': 'x'})
    """

    def __init__(
        self,
        fname: str,  # final output file name, intermediate output goes into a temporary file
        min_batch_size_bytes: int = 4 * 1024 * 1024,  # minimum size of a batch in bytes (1MB by default)
        compression: str | None = "zstd",
        throwaway=False,  # discard the temp file, useful for testing and benchmarking
        schema: pyarrow.Schema | dict | None = None,  # optional schema to enforce type coercion
        reference_keys: Sequence[str] | None = None,  # expected __key__ values, validated per-sample
    ):
        self.fname = fname
        self.batch_size = 1
        self.min_batch_size_bytes = min_batch_size_bytes
        self.max_batch_size = 16384
        self.compression = compression
        self.throwaway = throwaway
        self.reference_keys = reference_keys

        self._buffer = []
        self._sink = None
        self._offset = 0
        if isinstance(schema, dict):
            schema = pyarrow.schema(list(schema.items()))
        self._sink_schema = schema

    def write(self, x):
        if self.reference_keys is not None:
            actual_key = x.get("__key__") if isinstance(x, dict) else None
            if self._offset >= len(self.reference_keys):
                raise KeyMismatchError(self.fname, self._offset, None, actual_key)
            expected_key = self.reference_keys[self._offset]
            if actual_key is None:
                raise KeyMismatchError(self.fname, self._offset, expected_key, None)
            if actual_key != expected_key:
                raise KeyMismatchError(self.fname, self._offset, expected_key, actual_key)
        self._offset += 1
        self._buffer.append({k: encode_value(k, v) for k, v in x.items()})
        if len(self._buffer) >= self.batch_size:
            self.write_batch(self._buffer)

    # TODO: test writing batches of data straight from a PyTorch batched processing loop
    def write_batch(self, b, flush=False):
        import pyarrow

        try:
            record = pyarrow.RecordBatch.from_pylist(b, self._sink_schema)
        except Exception:
            def _truncate(v, limit=200):
                r = repr(v)
                return r if len(r) <= limit else r[:limit] + f"... ({len(r)} chars)"
            summary = [{k: _truncate(v) for k, v in row.items()} for row in b]
            print(f"Error while serializing: {summary}")
            raise
        if self._sink is None:
            if not flush and self.min_batch_size_bytes:
                if record.nbytes < self.min_batch_size_bytes and self.batch_size < self.max_batch_size:
                    self.batch_size *= 2
                    return
            schema = record.schema.with_metadata({"batch_size": str(len(b))})
            self._sink = pyarrow.RecordBatchFileWriter(
                self.fname, schema, options=pyarrow.ipc.IpcWriteOptions(compression=self.compression)
            )
            self._sink_schema = schema
        if record.schema != self._sink_schema:
            raise SampleFormatChanged(self._sink_schema, record.schema)
        self._sink.write(record)
        self._buffer.clear()

    def close(self):
        if self._buffer:
            self.write_batch(self._buffer, flush=True)  # flush the last batch
        if self.reference_keys is not None and self._offset != len(self.reference_keys):
            raise SampleCountMismatchError(self.fname, len(self.reference_keys), self._offset)
        assert self._sink is not None, "closing a WSSink that was never written to"
        self._sink.close()

    def __enter__(self):
        assert self._sink is None, "WSSink is not re-entrant"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()


class AtomicFile:
    """Context manager to atomically create a file.

    Example:
    ```
    with AtomicFile('output.txt', ephemeral=True) as fname:
        with open(fname, 'w') as f:
            f.write('Hello, World!')
    ```
    """

    def __init__(self, fname, ephemeral=False):
        self.fname = Path(fname)
        self.ephemeral = ephemeral

    def __enter__(self):
        self.fname.parent.mkdir(exist_ok=True, parents=True)
        self._tmp_name = str(self.fname) + (".%010x" % random.getrandbits(40))
        return self._tmp_name

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and not self.ephemeral:
            os.rename(self._tmp_name, self.fname)
        else:
            # print(f"Deleting partial file: {self._tmp_name}")
            if Path(self._tmp_name).exists():
                os.remove(self._tmp_name)


@contextmanager
def WSSink(
    fname: str,  # final output file name, intermediate output goes into a temporary file
    compression: str | None = "zstd",  # pass None to disable compression
    min_batch_size_bytes: int = 4 * 1024 * 1024,  # auto-increase the batch size until it's at least this size in bytes
    ephemeral: bool = False,  # discard the temp file, useful for testing and benchmarking
    schema: pyarrow.Schema | dict | None = None,  # optional schema to enforce type coercion
    validate_keys: bool = False,  # validate __key__ values against the smallest sibling artifact
):
    """Context manager to atomically create a `.wsds` shard.

    Example:
    ```
        with WSSink('output.wsds') as sink:
            sink.write({quality_metric: 5, transcript: "Hello, World!"})
    ```
    """
    reference_keys = _resolve_reference_keys(Path(fname)) if validate_keys else None
    with AtomicFile(fname, ephemeral) as tmp_fname:
        with WSBatchedSink(
            tmp_fname, min_batch_size_bytes=min_batch_size_bytes, compression=compression,
            schema=schema, reference_keys=reference_keys,
        ) as sink:
            yield sink
