from __future__ import annotations

import os
import time
from pathlib import Path
from typing import BinaryIO

BLOCK_SIZE = 16384  # 16kB minimum read size


class FileReader:
    """Base class for reading bytes from a file with two cache slots.

    Forward cache: caches reads from absolute offsets (for sequential access).
    Tail cache: caches reads from the end of the file (for footer parsing).

    Subclasses implement _raw_read and _raw_read_end only.

    IO stats (io_time, io_count, io_bytes, cache_hits) are always tracked.
    Pass verbose=True to print per-request details to stderr.
    """

    def __init__(self, *, verbose: bool = False):
        self._fwd_start: int = 0
        self._fwd_data: bytes = b""
        self._tail_data: bytes = b""
        self._verbose = verbose
        self.io_time: float = 0.0
        self.io_count: int = 0
        self.io_bytes: int = 0
        self.cache_hits: int = 0

    def read(self, offset: int, length: int) -> bytes:
        """Read length bytes at absolute offset, using the forward cache."""
        fwd_end = self._fwd_start + len(self._fwd_data)
        if self._fwd_data and offset >= self._fwd_start and offset + length <= fwd_end:
            self.cache_hits += 1
            start = offset - self._fwd_start
            return self._fwd_data[start : start + length]
        actual_length = max(length, BLOCK_SIZE)
        t0 = time.monotonic()
        self._fwd_data = self._raw_read(offset, actual_length)
        dt = time.monotonic() - t0
        self.io_time += dt
        self.io_count += 1
        self.io_bytes += len(self._fwd_data)
        if self._verbose:
            print(f"[IO] read offset={offset} len={actual_length} got={len(self._fwd_data)} {dt * 1000:.1f}ms")
        self._fwd_start = offset
        return self._fwd_data[:length]

    def read_end(self, offset: int, length: int) -> bytes:
        """Read length bytes relative to end of file.

        offset is negative (e.g. -6 means '6 bytes before EOF').
        """
        needed = -offset
        if self._tail_data and needed <= len(self._tail_data):
            self.cache_hits += 1
            start = len(self._tail_data) + offset
            return self._tail_data[start : start + length]
        actual_n = max(needed, BLOCK_SIZE)
        t0 = time.monotonic()
        print(self, self._raw_read_end)
        self._tail_data = self._raw_read_end(actual_n)
        dt = time.monotonic() - t0
        self.io_time += dt
        self.io_count += 1
        self.io_bytes += len(self._tail_data)
        if self._verbose:
            print(f"[IO] read_end n={actual_n} got={len(self._tail_data)} {dt * 1000:.1f}ms")
        start = len(self._tail_data) + offset
        return self._tail_data[start : start + length]

    def _raw_read(self, offset: int, length: int) -> bytes:
        """Read length bytes at absolute offset. May return fewer near EOF."""
        raise NotImplementedError

    def _raw_read_end(self, n: int) -> bytes:
        """Read the last n bytes of the file. May return fewer if file is smaller."""
        raise NotImplementedError

    def close(self):
        pass


class LocalFileReader(FileReader):
    """FileReader backed by a local file."""

    def __init__(self, path_or_file: str | Path | BinaryIO, *, verbose: bool = False):
        super().__init__(verbose=verbose)
        if isinstance(path_or_file, (str, Path)):
            self._file: BinaryIO = open(path_or_file, "rb")
            self._owns_file = True
        else:
            self._file = path_or_file
            self._owns_file = False

    def _raw_read(self, offset: int, length: int) -> bytes:
        self._file.seek(offset, os.SEEK_SET)
        return self._file.read(length)

    def _raw_read_end(self, n: int) -> bytes:
        self._file.seek(-n, os.SEEK_END)
        return self._file.read(n)

    def close(self):
        if self._owns_file:
            self._file.close()


class S3FileReader(FileReader):
    """FileReader backed by S3 range requests via boto3."""

    def __init__(self, s3_client, bucket: str, key: str, *, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._client = s3_client
        self._bucket = bucket
        self._key = key

    def _raw_read(self, offset: int, length: int) -> bytes:
        range_header = f"bytes={offset}-{offset + length - 1}"
        resp = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=range_header)
        return resp["Body"].read()

    def _raw_read_end(self, n: int) -> bytes:
        range_header = f"bytes=-{n}"
        resp = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=range_header)
        return resp["Body"].read()
