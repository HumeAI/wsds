from __future__ import annotations

import itertools
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pyarrow

from .utils import WSShardMissingError
from .ws_dataset import WSDataset
from .ws_decode import encode_value


def indented(prefix, obj):
    lines = str(obj).split("\n")
    return "\n".join(p + ln for p, ln in zip([prefix] + [" " * len(prefix)] * (len(lines) - 1), lines))


@dataclass(frozen=True)
class KeyMismatchError(Exception):
    message: str

    def __str__(self):
        return self.message


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
        key_iter=None,  # optional iterator of expected __key__ strings
    ):
        self.fname = fname
        self.batch_size = 1
        self.min_batch_size_bytes = min_batch_size_bytes
        self.max_batch_size = 16384
        self.compression = compression
        self.throwaway = throwaway

        self._buffer = []
        self._sink = None
        if isinstance(schema, dict):
            schema = pyarrow.schema(list(schema.items()))
        self._sink_schema = schema
        self._key_iter = key_iter
        self._last_key = None

    def write(self, x):
        if self._key_iter is not None:
            expected = next(self._key_iter, None)
            if expected is None:
                raise KeyMismatchError(
                    f"dest dataset has fewer samples than the new data being written"
                    f" (last matching key: {self._last_key!r}, new key: {x.get('__key__')!r})"
                )
            if x.get("__key__") != expected:
                raise KeyMismatchError(f"__key__ mismatch: expected {expected!r}, got {x.get('__key__')!r}")
            self._last_key = expected
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
        if self._key_iter is not None:
            remaining = next(self._key_iter, None)
            if remaining is not None:
                raise KeyMismatchError(
                    f"new data exhausted before dest dataset"
                    f" (last written key: {self._last_key!r}, next expected: {remaining!r})"
                )
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


def _build_key_iter(fname):
    """Return __key__ iterator from an existing sibling column, or None if this is the first column."""
    p = Path(fname)
    parent_dir = p.parent.parent

    try:
        ds = WSDataset(parent_dir, ignore_index=True)
        it = (s["__key__"] for s in ds.iter_shard(("", p.stem)))
        first = next(it)
        return itertools.chain([first], it)
    except (OSError, WSShardMissingError, StopIteration):
        return None


@contextmanager
def WSSink(
    fname: str,  # final output file name, intermediate output goes into a temporary file
    compression: str | None = "zstd",  # pass None to disable compression
    min_batch_size_bytes: int = 4 * 1024 * 1024,  # auto-increase the batch size until it's at least this size in bytes
    ephemeral: bool = False,  # discard the temp file, useful for testing and benchmarking
    schema: pyarrow.Schema | dict | None = None,  # optional schema to enforce type coercion
):
    """Context manager to atomically create a `.wsds` shard.

    Validates `__key__` alignment against existing columns when writing to an existing dataset.

    Example:
    ```
        with WSSink('output.wsds') as sink:
            sink.write({quality_metric: 5, transcript: "Hello, World!"})
    ```
    """
    key_iter = _build_key_iter(fname)
    with AtomicFile(fname, ephemeral) as fname:
        with WSBatchedSink(
            fname, min_batch_size_bytes=min_batch_size_bytes, compression=compression, schema=schema, key_iter=key_iter
        ) as sink:
            yield sink
