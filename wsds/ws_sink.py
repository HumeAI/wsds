from __future__ import annotations

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


def _cast_batch(batch, target_schema):
    """Cast a RecordBatch to target_schema, adding null columns for missing fields."""
    arrays = []
    for field in target_schema:
        if batch.schema.get_field_index(field.name) >= 0:
            arrays.append(batch.column(field.name).cast(field.type))
        else:
            arrays.append(pyarrow.nulls(batch.num_rows, type=field.type))
    return pyarrow.RecordBatch.from_arrays(arrays, schema=target_schema)


class WSBatchedSink:
    """A helper for writing data to a PyArrow feather file.

    Automatically batches data and infers the schema from the first batch.
    If the schema changes (new columns, type promotions, null -> concrete type),
    the file is transparently rewritten with a unified schema.

    Example:
    >>> with WSBatchedSink('output.feather', throwaway=True) as sink: sink.write({'a': 1, 'b': 'x'})

    Schema evolution -- int to float, null to string, new column:
    >>> import tempfile, pyarrow as pa
    >>> f = tempfile.NamedTemporaryFile(suffix='.wsds')
    >>> sink = WSBatchedSink(f.name, min_batch_size_bytes=0); sink.__enter__()  # doctest: +ELLIPSIS
    <...WSBatchedSink...>
    >>> sink.write({'x': 1, 'y': None})
    >>> sink.write({'x': 2.5, 'y': 'hello', 'z': True})
    >>> sink.close()
    >>> r = pa.ipc.open_file(f.name)
    >>> r.get_batch(0).to_pydict()
    {'x': [1.0], 'y': [None], 'z': [None]}
    >>> r.get_batch(1).to_pydict()
    {'x': [2.5], 'y': ['hello'], 'z': [True]}
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
        self._fixed_schema = schema is not None

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

    def _rewrite_with_new_schema(self, new_record):
        """Rewrite the file with a unified schema when a schema conflict is detected.

        Checks schema compatibility first, then streams old batches through an unlinked
        file handle to avoid loading everything into memory at once.
        Raises SampleFormatChanged if unification fails.
        """
        # Check compatibility before doing any I/O
        old_schema_no_meta = self._sink_schema.remove_metadata()
        new_schema_no_meta = new_record.schema.remove_metadata()
        try:
            unified_no_meta = pyarrow.unify_schemas([old_schema_no_meta, new_schema_no_meta], promote_options="permissive")
        except pyarrow.ArrowInvalid:
            raise SampleFormatChanged(self._sink_schema, new_record.schema)
        unified = unified_no_meta.with_metadata(self._sink_schema.metadata)

        # Close writer, open reader on the old file, then unlink it.
        # The open file handle keeps the data accessible while we overwrite the path.
        self._sink.close()
        old_file = pyarrow.OSFile(self.fname)
        reader = pyarrow.RecordBatchFileReader(old_file)
        os.unlink(self.fname)

        self._native_file = pyarrow.output_stream(self.fname)
        self._sink = pyarrow.RecordBatchFileWriter(
            self._native_file, unified, options=pyarrow.ipc.IpcWriteOptions(compression=self.compression)
        )
        self._sink_schema = unified

        for i in range(reader.num_record_batches):
            self._sink.write(_cast_batch(reader.get_batch(i), unified))
        self._sink.write(_cast_batch(new_record, unified))

    # TODO: test writing batches of data straight from a PyTorch batched processing loop
    def write_batch(self, b, flush=False):
        import pyarrow

        try:
            if self._sink is not None and not self._fixed_schema:
                # Subsequent batch: use natural inference to detect schema evolution
                # (from_pylist with an explicit schema silently drops new columns and coerces types)
                record = pyarrow.RecordBatch.from_pylist(b)
                if record.schema != self._sink_schema:
                    self._rewrite_with_new_schema(record)
                    self._buffer.clear()
                    return
                self._sink.write(record)
                self._buffer.clear()
                return
            record = pyarrow.RecordBatch.from_pylist(b, self._sink_schema)
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
            if self._fixed_schema:
                actual = pyarrow.RecordBatch.from_pylist(b).schema
                raise SampleFormatChanged(self._sink_schema.remove_metadata(), actual) from None
            raise
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
            self._native_file = pyarrow.output_stream(self.fname)
            self._sink = pyarrow.RecordBatchFileWriter(
                self._native_file, schema, options=pyarrow.ipc.IpcWriteOptions(compression=self.compression)
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
        self._native_file.close()
        self._sink = None

    def __enter__(self):
        assert self._sink is None, "WSSink is not re-entrant"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        elif self._sink is not None:
            try:
                self._sink.close()
            except Exception:
                pass
            self._sink = None
            import gc; gc.collect()


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
    """Return __key__ iterator from an existing sibling column, or None if this is the first column.

    Uses a lazy closing iterator so that keys are streamed on-demand (avoiding
    upfront materialization of the entire key list) while still guaranteeing
    the underlying WSDataset is closed when the iterator is exhausted or GC'd.
    """
    p = Path(fname)
    parent_dir = p.parent.parent

    ds = None
    try:
        ds = WSDataset(parent_dir, ignore_index=True)
        it = ds.iter_shard(("", p.stem))
        first = next(it)  # peek to verify the shard is non-empty
    except (OSError, WSShardMissingError, StopIteration, KeyError):
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass
        return None

    # Wrap in a generator so ds.close() runs when the iterator finishes
    # (or is garbage-collected), preventing file-descriptor leaks.
    def _closing_iter():
        try:
            yield first["__key__"]
            for s in it:
                yield s["__key__"]
        finally:
            ds.close()

    return _closing_iter()


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
