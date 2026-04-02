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


class ShardedSink:
    """Write samples across multiple .wsds shard files, rotating on size or count.

    Generates filenames as ``{output_dir}/{prefix}-{index:05d}.wsds``.

    Example::

        with ShardedSink("output/audio") as sink:
            for sample in samples:
                sink.write(sample)
        print(sink.shard_paths)
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
        self._current_sink = None   # WSBatchedSink
        self._current_atomic = None  # AtomicFile
        self._shard_paths: list[Path] = []

    @property
    def shard_paths(self) -> list[Path]:
        return list(self._shard_paths)

    def _shard_path(self) -> Path:
        return self.output_dir / f"{self.prefix}-{self._shard_index:05d}.wsds"

    def _open_shard(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_atomic = AtomicFile(self._shard_path())
        tmp = self._current_atomic.__enter__()
        self._current_sink = WSBatchedSink(tmp, compression=self.compression)
        self._current_sink.__enter__()
        self._samples_in_shard = 0

    def _close_shard(self):
        if self._current_sink is None:
            return
        try:
            self._current_sink.__exit__(None, None, None)
            self._current_atomic.__exit__(None, None, None)
        except Exception:
            # Best-effort cleanup on failure
            try:
                self._current_atomic.__exit__(*([Exception] * 3))
            except Exception:
                pass
            raise
        self._shard_paths.append(self._shard_path())
        self._current_sink = None
        self._current_atomic = None
        self._shard_index += 1

    def _should_rotate(self) -> bool:
        sink = self._current_sink
        if sink is None:
            return False
        if self.max_samples_per_shard is not None and self._samples_in_shard >= self.max_samples_per_shard:
            return True
        # Check on-disk file size (accurate after batches flush).
        # For large samples (audio), each sample exceeds the 4MB batch threshold
        # so batches flush immediately and file size is accurate.
        try:
            if os.path.getsize(sink.fname) >= self.max_shard_bytes:
                return True
        except OSError:
            pass
        return False

    def write(self, sample: dict):
        if self._current_sink is None:
            self._open_shard()
        self._current_sink.write(sample)
        self._samples_in_shard += 1
        if self._should_rotate():
            self._close_shard()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._close_shard()
        elif self._current_sink is not None:
            self._current_sink.__exit__(exc_type, exc_value, traceback)
            self._current_atomic.__exit__(exc_type, exc_value, traceback)
