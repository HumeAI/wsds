from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pyarrow


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
    ):
        self.fname = fname
        self.batch_size = 1
        self.min_batch_size_bytes = min_batch_size_bytes
        self.max_batch_size = 16384
        self.compression = compression
        self.throwaway = throwaway

        self._buffer = []
        self._sink = None
        self.bytes_written = 0  # estimated uncompressed bytes flushed to disk

    def write(self, x):
        self._buffer.append(x)
        if len(self._buffer) >= self.batch_size:
            self.write_batch(self._buffer)

    # TODO: test writing batches of data straight from a PyTorch batched processing loop
    def write_batch(self, b, flush=False):
        import pyarrow

        try:
            record = pyarrow.RecordBatch.from_pylist(b, self._sink_schema if self._sink else None)
        except Exception:
            print(f"Error while serializing: {repr(b)}")
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
        self.bytes_written += record.nbytes
        self._buffer.clear()

    def close(self):
        if self._buffer:
            self.write_batch(self._buffer, flush=True)  # flush the last batch
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
):
    """Context manager to atomically create a `.wsds` shard.

    Example:
    ```
        with WSSink('output.wsds') as sink:
            sink.write({quality_metric: 5, transcript: "Hello, World!"})
    ```
    """
    with AtomicFile(fname, ephemeral) as fname:
        with WSBatchedSink(fname, min_batch_size_bytes, compression) as sink:
            yield sink


DEFAULT_MAX_SHARD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def _estimate_sample_bytes(sample: dict) -> int:
    """Rough estimate of a sample's size in bytes."""
    total = 0
    for v in sample.values():
        if isinstance(v, (bytes, bytearray)):
            total += len(v)
        elif isinstance(v, str):
            total += len(v.encode("utf-8", errors="replace"))
        else:
            total += 8  # scalar
    return total


class ShardedSink:
    """Write samples across multiple shard files, rotating when a shard exceeds a size limit.

    Generates shard filenames as ``{output_dir}/{prefix}-{index:05d}.wsds``.

    Example::

        with ShardedSink("output/audio") as sink:
            for sample in samples:
                sink.write(sample)
        print(sink.shard_paths)  # list of written shard files
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        prefix: str = "audio",
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        max_samples_per_shard: int | None = None,
        compression: str | None = "zstd",
    ):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.max_shard_bytes = max_shard_bytes
        self.max_samples_per_shard = max_samples_per_shard
        self.compression = compression

        self._shard_index = 0
        self._samples_in_shard = 0
        self._sink = None
        self._atomic = None
        self._shard_paths: list[Path] = []

    @property
    def shard_paths(self) -> list[Path]:
        return list(self._shard_paths)

    def _shard_fname(self) -> Path:
        return self.output_dir / f"{self.prefix}-{self._shard_index:05d}.wsds"

    def _open_shard(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fname = self._shard_fname()
        self._atomic = AtomicFile(fname)
        tmp_name = self._atomic.__enter__()
        self._sink = WSBatchedSink(tmp_name, compression=self.compression)
        self._sink.__enter__()
        self._samples_in_shard = 0

    def _close_shard(self):
        if self._sink is not None:
            self._sink.__exit__(None, None, None)
            self._atomic.__exit__(None, None, None)
            self._shard_paths.append(self._shard_fname())
            self._sink = None
            self._atomic = None
            self._shard_index += 1

    def _estimated_shard_bytes(self) -> int:
        if self._sink is None:
            return 0
        # bytes_written tracks flushed batches; also estimate buffered data
        buf_estimate = sum(_estimate_sample_bytes(s) for s in self._sink._buffer)
        return self._sink.bytes_written + buf_estimate

    def _should_rotate(self) -> bool:
        if self.max_samples_per_shard is not None and self._samples_in_shard >= self.max_samples_per_shard:
            return True
        if self._estimated_shard_bytes() >= self.max_shard_bytes:
            return True
        return False

    def write(self, sample: dict):
        if self._sink is None:
            self._open_shard()
        self._sink.write(sample)
        self._samples_in_shard += 1
        if self._should_rotate():
            self._close_shard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._close_shard()
        elif self._sink is not None:
            # Discard partial shard on error
            self._sink.__exit__(exc_type, exc_value, traceback)
            self._atomic.__exit__(exc_type, exc_value, traceback)
