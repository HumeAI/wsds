from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

BLOCK_SIZE = 8192  # 8kB minimum sync read size
MIN_ASYNC_READ = 4096  # 4kB minimum async read size
VERBOSE = False


@dataclass
class _Region:
    """A coalesced read region covering one or more buffer descriptors."""

    offset: int
    length: int
    members: list[tuple[int, int, int]]  # list of (abs_offset, start_in_region, end_in_region)


def _coalesce_regions(items: list[tuple[int, int]], gap_threshold: int = 64 * 1024) -> list[_Region]:
    """Merge nearby reads into larger contiguous fetches.

    items: list of (absolute_offset, length) pairs.
    Returns _Region objects with members referencing back to the original offsets.
    """
    if not items:
        return []

    sorted_items = sorted(items, key=lambda x: x[0])

    regions: list[_Region] = []
    cur_offset = sorted_items[0][0]
    cur_end = cur_offset + sorted_items[0][1]
    cur_members: list[tuple[int, int, int]] = [(sorted_items[0][0], 0, sorted_items[0][1])]

    for abs_offset, length in sorted_items[1:]:
        item_end = abs_offset + length
        if abs_offset <= cur_end + gap_threshold:
            member_start = abs_offset - cur_offset
            cur_members.append((abs_offset, member_start, member_start + length))
            cur_end = max(cur_end, item_end)
        else:
            regions.append(_Region(offset=cur_offset, length=cur_end - cur_offset, members=cur_members))
            cur_offset = abs_offset
            cur_end = item_end
            cur_members = [(abs_offset, 0, length)]

    regions.append(_Region(offset=cur_offset, length=cur_end - cur_offset, members=cur_members))
    return regions


class FileReader:
    """Base class for reading bytes from a file.

    Sync reads use a two-slot cache (forward + tail).
    Async reads use a range cache with plan mode for coalesced IO.

    Subclasses implement _raw_read, _raw_read_end, and optionally _async_read_impl.

    IO stats (io_time, io_count, io_bytes, cache_hits) are always tracked.
    """

    def __init__(self):
        # Sync caches
        self._fwd_start: int = 0
        self._fwd_data: bytes = b""
        self._tail_data: bytes = b""
        self._verbose = VERBOSE
        self._async_first = False  # subclasses that track IO in _async_read_impl set this
        self.io_time: float = 0.0
        self.io_count: int = 0
        self.io_bytes: int = 0
        self.cache_hits: int = 0

        # Async plan mode (per-task via contextvar)
        self._planned: contextvars.ContextVar[bool] = contextvars.ContextVar("_planned", default=False)
        self._pending: list[tuple[int, int, int, asyncio.Future]] = []  # (offset, actual, length, future)
        self._cache: list[tuple[int, bytes]] = []  # (offset, data) ranges

    def read(self, offset: int, length: int) -> bytes:
        """Read length bytes at absolute offset, using forward cache."""
        fwd_end = self._fwd_start + len(self._fwd_data)
        if self._fwd_data and offset >= self._fwd_start and offset + length <= fwd_end:
            self.cache_hits += 1
            start = offset - self._fwd_start
            return self._fwd_data[start : start + length]
        actual_length = max(length, BLOCK_SIZE)
        t0 = time.monotonic()
        self._fwd_data = self._raw_read(offset, actual_length)
        dt = time.monotonic() - t0
        if not self._async_first:
            self.io_time += dt
            self.io_count += 1
            self.io_bytes += len(self._fwd_data)
        if self._verbose:
            print(
                f"[IO] read offset={offset} reqn={length} len={actual_length} got={len(self._fwd_data)} {dt * 1000:.1f}ms"
            )
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
        self._tail_data = self._raw_read_end(actual_n)
        dt = time.monotonic() - t0
        if not self._async_first:
            self.io_time += dt
            self.io_count += 1
            self.io_bytes += len(self._tail_data)
        if self._verbose:
            print(f"[IO] read_end reqn={length} n={actual_n} got={len(self._tail_data)} {dt * 1000:.1f}ms")
        start = len(self._tail_data) + offset
        return self._tail_data[start : start + length]

    def _raw_read(self, offset: int, length: int) -> bytes:
        """Read length bytes at absolute offset. May return fewer near EOF."""
        raise NotImplementedError

    def _raw_read_end(self, n: int) -> bytes:
        """Read the last n bytes of the file. May return fewer if file is smaller."""
        raise NotImplementedError

    # -- Async IO with range cache and plan mode --------------------------------

    async def async_read(self, offset: int, length: int) -> bytes:
        """Async read with range cache and optional plan mode.

        Every read fetches at least MIN_ASYNC_READ bytes and caches the result.
        In plan mode, reads are deferred and coalesced on flush().
        """
        # Check range cache
        for c_off, c_data in self._cache:
            if offset >= c_off and offset + length <= c_off + len(c_data):
                self.cache_hits += 1
                s = offset - c_off
                return c_data[s : s + length]

        actual = max(length, MIN_ASYNC_READ)

        if not self._planned.get():
            # Eager mode: read directly
            data = await self._async_read_impl(offset, actual)
            self._cache.append((offset, data))
            return data[:length]

        # Plan mode: submit and await future
        fut = asyncio.get_running_loop().create_future()
        self._pending.append((offset, actual, length, fut))
        return await fut

    async def flush(self):
        """Coalesce pending reads, execute, resolve futures."""
        if not self._pending:
            return
        items = [(off, actual) for off, actual, _, _ in self._pending]
        regions = _coalesce_regions(items)
        fetched = await asyncio.gather(*[self._async_read_impl(r.offset, r.length) for r in regions])

        data_map: dict[int, bytes] = {}
        for region, data in zip(regions, fetched):
            self._cache.append((region.offset, data))
            for abs_offset, start, end in region.members:
                data_map[abs_offset] = data[start:end]

        for offset, _actual, length, fut in self._pending:
            fut.set_result(data_map[offset][:length])
        self._pending.clear()

    def clear_cache(self):
        """Clear the async range cache."""
        self._cache.clear()

    @property
    def has_pending(self) -> bool:
        return len(self._pending) > 0

    async def _async_read_impl(self, offset: int, length: int) -> bytes:
        """Actual async IO. Default runs _raw_read in a thread executor.

        Subclasses with native async IO (S3, Modal) override this.
        """
        return await asyncio.get_event_loop().run_in_executor(None, self._raw_read, offset, length)

    def close(self):
        pass


class LocalFileReader(FileReader):
    """FileReader backed by a local file via os.pread."""

    def __init__(self, path: str | Path):
        super().__init__()
        self._fd = os.open(str(path), os.O_RDONLY)

    def _raw_read(self, offset: int, length: int) -> bytes:
        return os.pread(self._fd, length, offset)

    def _raw_read_end(self, n: int) -> bytes:
        size = os.fstat(self._fd).st_size
        return os.pread(self._fd, n, max(size - n, 0))

    def close(self):
        os.close(self._fd)


class S3FileReader(FileReader):
    """FileReader backed by S3 range requests via aiobotocore.

    Async-first: _async_read_impl is the canonical implementation.
    Sync _raw_read calls into _async_read_impl via the background event loop.

    Takes a pre-created aiobotocore S3 client (not a session) so that
    SSL context and connection pool setup is amortized across readers.
    """

    def __init__(self, client, bucket: str, key: str):
        super().__init__()
        self._async_first = True
        self._client = client  # aiobotocore S3 client (already entered)
        self._bucket = bucket
        self._key = key

    async def _async_read_impl(self, offset: int, length: int) -> bytes:
        range_header = f"bytes={offset}-{offset + length - 1}"
        t0 = time.monotonic()
        resp = await self._client.get_object(Bucket=self._bucket, Key=self._key, Range=range_header)
        async with resp["Body"] as stream:
            data = await stream.read()
        dt = time.monotonic() - t0
        self.io_time += dt
        self.io_count += 1
        self.io_bytes += len(data)
        if self._verbose:
            print(f"[S3] async_read offset={offset} req={length} got={len(data)} {dt * 1000:.1f}ms")
        return data

    async def _async_read_end(self, n: int) -> bytes:
        t0 = time.monotonic()
        resp = await self._client.get_object(Bucket=self._bucket, Key=self._key, Range=f"bytes=-{n}")
        async with resp["Body"] as stream:
            data = await stream.read()
        dt = time.monotonic() - t0
        self.io_time += dt
        self.io_count += 1
        self.io_bytes += len(data)
        if self._verbose:
            print(f"[S3] async_read_end req={n} got={len(data)} {dt * 1000:.1f}ms")
        return data

    def _raw_read(self, offset: int, length: int) -> bytes:
        return _get_io_loop().run(self._async_read_impl(offset, length))

    def _raw_read_end(self, n: int) -> bytes:
        return _get_io_loop().run(self._async_read_end(n))


class _IOLoop:
    """A persistent event loop running on a dedicated daemon thread.

    Used for async-first readers (S3, Modal). Callers on the main thread
    (or Jupyter, or another loop) are never blocked by "loop already running"
    errors."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro):
        """Submit *coro* to the background loop and block until it completes."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()


# Module-level singleton — created on first use.
_io_loop: _IOLoop | None = None
_io_loop_lock = threading.Lock()


def _get_io_loop() -> _IOLoop:
    global _io_loop
    if _io_loop is None:
        with _io_loop_lock:
            if _io_loop is None:
                _io_loop = _IOLoop()
    return _io_loop


class ModalFileReader(FileReader):
    """FileReader backed by Modal Volume range requests via gRPC + aiohttp.

    Async-first: _async_read_impl is the canonical implementation using native
    async gRPC for metadata and aiohttp for presigned URL downloads.
    Sync _raw_read calls into _async_read_impl via the background event loop.

    All async work runs on a shared daemon-thread event loop (see
    ``_IOLoop``) so it works regardless of whether the caller already
    has a running loop (Jupyter, Modal synchronizer, etc.)."""

    def __init__(self, vol, path: str):
        super().__init__()
        self._async_first = True
        self._vol = vol
        self._path = path
        self._size: int | None = None
        self._loop = _get_io_loop()
        self._aiohttp_session = None

    @classmethod
    def from_name(cls, volume_name: str, path: str) -> "ModalFileReader":
        """Create a reader for *path* inside the named Modal Volume."""
        loop = _get_io_loop()
        vol = loop.run(cls._hydrate(volume_name))
        reader = cls(vol, path)
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

    async def _get_aiohttp_session(self):
        if self._aiohttp_session is None:
            import aiohttp

            self._aiohttp_session = aiohttp.ClientSession()
        return self._aiohttp_session

    async def _async_fetch_urls(self, resp) -> bytes:
        """Download presigned block URLs concurrently via aiohttp."""
        session = await self._get_aiohttp_session()
        tasks = [self._fetch_one(session, url) for url in resp.get_urls]
        chunks = await asyncio.gather(*tasks)
        return b"".join(chunks)

    @staticmethod
    async def _fetch_one(session, url: str) -> bytes:
        async with session.get(url) as r:
            r.raise_for_status()
            return await r.read()

    def _ensure_size(self) -> int:
        """Fetch the total file size (cached after first call)."""
        if self._size is None:
            resp = self._loop.run(self._get_range(0, 1))
            self._size = resp.size
        return self._size

    async def _async_read_impl(self, offset: int, length: int) -> bytes:
        """Native async: gRPC for range metadata, aiohttp for URL downloads."""
        resp = await self._get_range(offset, length)
        if self._size is None:
            self._size = resp.size
        return await self._async_fetch_urls(resp)

    def _raw_read(self, offset: int, length: int) -> bytes:
        return self._loop.run(self._async_read_impl(offset, length))

    def _raw_read_end(self, n: int) -> bytes:
        size = self._ensure_size()
        offset = max(size - n, 0)
        return self._raw_read(offset, size - offset)

    async def _async_close(self):
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    def close(self):
        if self._aiohttp_session is not None:
            self._loop.run(self._async_close())
