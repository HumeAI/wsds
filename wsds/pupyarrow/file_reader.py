from __future__ import annotations

import os
import threading
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


class _ModalEventLoop:
    """A persistent event loop running on a dedicated daemon thread.

    All Modal gRPC work is dispatched here so the client's channel stays
    bound to a single loop, and callers on the main thread (or Jupyter,
    or another loop) are never blocked by "loop already running" errors."""

    def __init__(self):
        import asyncio

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro):
        """Submit *coro* to the background loop and block until it completes."""
        import asyncio

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


# Module-level singleton — created on first use.
_modal_loop: _ModalEventLoop | None = None
_modal_loop_lock = threading.Lock()


def _get_modal_loop() -> _ModalEventLoop:
    global _modal_loop
    if _modal_loop is None:
        with _modal_loop_lock:
            if _modal_loop is None:
                _modal_loop = _ModalEventLoop()
    return _modal_loop


class ModalFileReader(FileReader):
    """FileReader backed by Modal Volume range requests via gRPC.

    Uses the undocumented ``start``/``len`` fields on ``VolumeGetFile2Request``
    to fetch only the needed byte ranges.  Presigned block URLs returned by the
    gRPC call are downloaded with ``urllib``.

    All async gRPC work runs on a shared daemon-thread event loop (see
    ``_ModalEventLoop``) so it works regardless of whether the caller already
    has a running loop (Jupyter, Modal synchronizer, etc.)."""

    def __init__(self, vol, path: str, *, verbose: bool = False):
        super().__init__(verbose=verbose)
        self._vol = vol
        self._path = path
        self._size: int | None = None
        self._loop = _get_modal_loop()

    @classmethod
    def from_name(cls, volume_name: str, path: str, *, verbose: bool = False) -> "ModalFileReader":
        """Create a reader for *path* inside the named Modal Volume."""
        loop = _get_modal_loop()
        vol = loop.run(cls._hydrate(volume_name))
        reader = cls(vol, path, verbose=verbose)
        return reader

    @staticmethod
    async def _hydrate(volume_name: str):
        from modal.volume import _Volume

        vol = _Volume.from_name(volume_name)
        await vol.hydrate()
        return vol

    async def _get_range(self, start: int, length: int):
        from modal_proto import api_pb2

        req = api_pb2.VolumeGetFile2Request(
            volume_id=self._vol.object_id,
            path=self._path,
            start=start,
            len=length,
        )
        return await self._vol._client.stub.VolumeGetFile2(req)

    def _fetch_urls(self, resp) -> bytes:
        """Download presigned block URLs and concatenate the bytes."""
        import requests

        chunks = []
        for url in resp.get_urls:
            r = requests.get(url)
            r.raise_for_status()
            chunks.append(r.content)
        return b"".join(chunks)

    def _ensure_size(self) -> int:
        """Fetch the total file size (cached after first call)."""
        if self._size is None:
            resp = self._loop.run(self._get_range(0, 1))
            self._size = resp.size
        return self._size

    def _raw_read(self, offset: int, length: int) -> bytes:
        resp = self._loop.run(self._get_range(offset, length))
        if self._size is None:
            self._size = resp.size
        return self._fetch_urls(resp)

    def _raw_read_end(self, n: int) -> bytes:
        size = self._ensure_size()
        offset = max(size - n, 0)
        return self._raw_read(offset, size - offset)
