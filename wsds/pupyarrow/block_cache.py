"""Sparse-mirror block cache on WEKA for S3/B2-backed .wsds shards.

Adapted from weka-locking-bench/mirrorcache.py. Each remote object is cached in one
LOCAL sparse file `<root>/<name><EXT>` (EXT=".sparse", distinct from full `.wsds`
shards) that mirrors it 1:1: block i at offset i*SLOT (128 KiB), sized to the object's
full length but only physically consuming fetched blocks (footprint = working set).

Layout (single fd, sparse tail is ~free):
    [0, nblocks*SLOT)        data blocks, block i at i*SLOT
    [nblocks*SLOT, +nblocks) bitmap: one BYTE/block (present=1)
    [.. , +FOOT_SZ)          footer: magic + object size (self-describing)

Presence-byte reads are LOCK-FREE (single byte can't tear; write-once monotonic; WEKA
preserves cross-node write ordering so bit==1 => the data before it is visible). The
only lock is an fcntl range lock in the DATA region held by the persister to make fills
exactly-once across nodes. Persister (background, group commit): fcntl LOCK_EX|NB the
block range, pwrite, fsync, then set the bitmap byte (fsync-before-byte => bit-durable
implies data-durable; a lost byte is a harmless re-fetch).

read_range() coalesces runs of MISSING blocks into one backend range fetch (so a 2 MB
read is ONE GET on cold, not 16), serves present blocks by pread, and enqueues fetched
blocks for persistence.
"""

import fcntl
import os
import struct
import threading
import time
from queue import Empty, Queue

SLOT = 128 * 1024
FOOTER = struct.Struct("<8sQ")  # magic, object_size
MAGIC = b"WKSPRS1\x00"
FOOT_SZ = FOOTER.size
EXT = os.environ.get("WSDS_S3_CACHE_EXT", ".sparse")


class _Mirror:
    """A cached fd for one shard's sparse mirror. Holds NO authoritative state: the block bitmap is
    read from the file on every test (_bit -> os.pread) and locked with fcntl, so the filesystem is
    the source of truth and a mirror may be dropped and reopened freely. Closing on __del__ means an
    LRU eviction only drops a reference -- an in-flight reader keeps its own alive (2026-09-16)."""

    __slots__ = ("name", "fd", "size", "nblocks", "bm_off")

    def __init__(self, name, fd, size):
        self.name, self.fd, self.size = name, fd, size
        self.nblocks = (size + SLOT - 1) // SLOT if size > 0 else 0
        self.bm_off = self.nblocks * SLOT

    def __del__(self):
        fd = getattr(self, "fd", -1)
        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except Exception:
                pass


class BlockCache:
    def __init__(self, root, logdir=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        # No mirror table. A READER owns the fd for the shard it is reading (open_mirror ->
        # _Mirror, closed by __del__ when the reader is dropped), so the fd count tracks live
        # shards instead of every shard ever touched -- it was 20 mirrors / 68 fds after 768 crops
        # on local20 and still climbing (2026-09-16). The rare WRITER (_persist_batch, only after a
        # genuine miss) reopens by name: the bitmap lives in the FILE, read by pread on every test
        # and locked with fcntl, so nothing has to be shared between handles.
        self._reg = threading.Lock()
        self.hits = self.misses = self.persisted = self.dropped = 0
        # NO per-cache queue or thread. get_block_cache() keys caches by ROOT, and the root of a
        # linked audio column is its own catalog dataset -- so local20-500k reaches 646 caches per
        # worker, and one persister thread each meant 646 threads polling q.get(timeout=0.2) five
        # times a second forever. MEASURED 2026-09-17: 100 idle persister threads burn 8.6% of a
        # core and 658 voluntary context switches/s, so at the plateau that is ~55% of a core per
        # worker (and GIL contention inside a Python-bound loader) for threads with nothing to do.
        # Writes now go to ONE process-wide queue drained by ONE thread, started on the first
        # enqueue -- a worker whose mirrors are already populated never starts it at all.
        # Stage-3 usage logging for LRU eviction (off hot path). Enabled by passing a
        # logdir (the shard consumer passes <cache root>/.access); each block access is
        # aggregated in-memory and periodically flushed to a per-worker Feather snapshot.
        self._log = AccessLogger(logdir) if logdir else None

    # ---- per-object sparse file (create/open, size from footer or size_fn) ----
    def open_mirror(self, name, size_fn):
        """Open (creating if needed) the sparse mirror for `name`; the CALLER owns the returned
        _Mirror and its fd. `size_fn` is consulted only when the mirror does not exist yet."""
        path = os.path.join(self.root, name + EXT)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        st = os.fstat(fd)
        size = None
        if st.st_size >= FOOT_SZ:  # self-describing footer at EOF
            magic, sz = FOOTER.unpack(os.pread(fd, FOOT_SZ, st.st_size - FOOT_SZ))
            if magic == MAGIC:
                size = sz
        if size is None:  # new mirror -> need object size
            size = int(size_fn())
            nblocks = (size + SLOT - 1) // SLOT
            total = nblocks * SLOT + nblocks + FOOT_SZ
            if st.st_size < total:
                os.ftruncate(fd, total)  # sparse: only touched regions cost
            os.pwrite(fd, FOOTER.pack(MAGIC, size), nblocks * SLOT + nblocks)
        return _Mirror(name, fd, size)

    def object_size(self, m):
        return m.size

    # ---- back-compat for the staging tools that predate the 2026-09-17 refactor ----
    # research-experiments/.../octave3-analysis/prefetch.py (and prefetch2.py) call
    # `cache._mirror(name, size_fn)` and poll `cache.q.qsize()`; the refactor renamed the first
    # (the caller now OWNS the returned fd) and replaced the per-cache queue by the process-wide
    # one. Found the hard way: the en-100k prefetch failed on all 898 shards (2026-09-20).
    def _mirror(self, name, size_fn):
        return self.open_mirror(name, size_fn)

    @property
    def q(self):
        return _PERSIST_Q

    def covers(self, m, offset, length):
        """True when every block of [offset, offset+length) is present (no backend needed)."""
        if m.size <= 0 or offset >= m.size or length <= 0:
            return True
        length = min(length, m.size - offset)
        b0, b1 = offset // SLOT, (offset + length - 1) // SLOT
        return all(os.pread(m.fd, b1 - b0 + 1, m.bm_off + b0))

    def _bit(self, m, blk):
        b = os.pread(m.fd, 1, m.bm_off + blk)  # lock-free single byte
        return bool(b) and b[0] != 0

    def _blocklen(self, m, blk):
        return SLOT if blk < m.nblocks - 1 else m.size - blk * SLOT

    def _trylock(self, m, blk):
        try:
            fcntl.lockf(m.fd, fcntl.LOCK_EX | fcntl.LOCK_NB, SLOT, blk * SLOT, os.SEEK_SET)
            return True
        except OSError:
            return False

    def _unlock(self, m, blk):
        fcntl.lockf(m.fd, fcntl.LOCK_UN, SLOT, blk * SLOT, os.SEEK_SET)

    def _reopen_mirror(self, name):
        """Open an EXISTING mirror by name, size from its footer; None if absent/unreadable."""
        path = os.path.join(self.root, name + EXT)
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            return None
        try:
            st = os.fstat(fd)
            if st.st_size >= FOOT_SZ:
                magic, sz = FOOTER.unpack(os.pread(fd, FOOT_SZ, st.st_size - FOOT_SZ))
                if magic == MAGIC:
                    return _Mirror(name, fd, sz)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass
        return None

    # ---- hot path: read [offset, offset+length), filling misses via fetch_range ----
    def _runs(self, m, offset, length):
        """Split [offset, offset+length) into maximal runs of present / missing blocks:
        [(present, first_block, end_block)]. Presence bytes are read lock-free."""
        b0 = offset // SLOT
        b1 = (offset + length - 1) // SLOT
        if self._log is not None:                          # off-hot-path LRU tracking
            for bb in range(b0, b1 + 1):
                self._log.record(m.name, bb)
        runs = []
        blk = b0
        while blk <= b1:
            hit = self._bit(m, blk)
            run0 = blk
            while blk <= b1 and self._bit(m, blk) == hit:
                blk += 1
            runs.append((hit, run0, blk))
        return runs

    def _serve_hit(self, m, run0, run1):
        self.hits += run1 - run0
        start = run0 * SLOT
        return os.pread(m.fd, min(run1 * SLOT, m.size) - start, start)

    def _accept_miss(self, m, run0, run1, data):
        """Account a fetched run and hand its blocks to the persister."""
        self.misses += run1 - run0
        for bb in range(run0, run1):
            off = (bb - run0) * SLOT
            blen = self._blocklen(m, bb)
            if off + blen <= len(data):
                try:
                    _enqueue_write(self, m.name, bb, bytes(data[off:off + blen]))
                except Exception:
                    self.dropped += 1
        return data

    def read_range(self, m, offset, length, fetch_range):
        """Read [offset, offset+length) from the mirror the CALLER owns.
        fetch_range(start_byte, nbytes) -> bytes (one backend ranged GET) fills misses."""
        if m.size <= 0 or offset >= m.size:
            return b""
        length = min(length, m.size - offset)
        buf = bytearray()                                  # bytes starting at b0*SLOT
        for hit, run0, run1 in self._runs(m, offset, length):
            if hit:
                buf += self._serve_hit(m, run0, run1)
            else:
                start = run0 * SLOT
                buf += self._accept_miss(m, run0, run1, fetch_range(start, min(run1 * SLOT, m.size) - start))
        s = offset - (offset // SLOT) * SLOT
        return bytes(buf[s:s + length])

    async def async_read_range(self, m, offset, length, async_fetch_range):
        """read_range for the async reader path: misses are AWAITED (`async_fetch_range(start,
        nbytes)` is a coroutine), so a concurrent batch of reads on the IO loop never parks
        a thread per miss. (Running the sync read_range in the loop's default executor
        deadlocked once a shard's batches were gathered concurrently: every executor worker
        blocked waiting on the loop, and the loop's own DNS lookups needed that executor.)"""
        if m.size <= 0 or offset >= m.size:
            return b""
        length = min(length, m.size - offset)
        buf = bytearray()
        for hit, run0, run1 in self._runs(m, offset, length):
            if hit:
                buf += self._serve_hit(m, run0, run1)
            else:
                start = run0 * SLOT
                data = await async_fetch_range(start, min(run1 * SLOT, m.size) - start)
                buf += self._accept_miss(m, run0, run1, data)
        s = offset - (offset // SLOT) * SLOT
        return bytes(buf[s:s + length])

    # ---- persistence (background, group commit) ----
    def _persist_batch(self, batch):
        written, dirty = [], set()
        reopened = {}
        for name, blk, data in sorted(batch, key=lambda x: (x[0], x[1])):
            # Writing is RARE (only after a miss) and reads never come through here, so the writer
            # simply reopens by name -- memoised per batch. Reads pay nothing for this.
            m = reopened.get(name)
            if m is None:
                m = reopened[name] = self._reopen_mirror(name)
            if m is None or self._bit(m, blk):
                continue
            if not self._trylock(m, blk):
                continue
            if self._bit(m, blk):
                self._unlock(m, blk)
                continue
            os.pwrite(m.fd, data[:SLOT], blk * SLOT)
            written.append((m, blk))
            dirty.add(m.fd)
        if not written:
            return
        for fd in dirty:  # barrier: data durable...
            os.fsync(fd)
        for m, blk in written:  # ...before publishing the byte
            os.pwrite(m.fd, b"\x01", m.bm_off + blk)
            self._unlock(m, blk)
            self.persisted += 1

    def close(self):
        """Stop logging for this root. Queued writes are NOT dropped: they belong to the shared
        persister (see _persist_loop) and only need this object's root, which does not change."""
        if self._log is not None:
            self._log.close()


# ------------------------------------------------------------------------------
# One write queue and one persister thread for the whole process, started lazily.
# ------------------------------------------------------------------------------
_PERSIST_Q = Queue(maxsize=int(os.environ.get("WSDS_BLOCK_PERSIST_QUEUE", "8192")))
_PERSIST_T = None
_PERSIST_LOCK = threading.Lock()
_PERSIST_STOP = threading.Event()
_BATCH_MAX = 64
_BATCH_S = 0.005
_POLL_S = 0.2


def _ensure_persister():
    global _PERSIST_T
    t = _PERSIST_T
    if t is not None and t.is_alive():
        return
    with _PERSIST_LOCK:
        if _PERSIST_T is None or not _PERSIST_T.is_alive():
            _PERSIST_STOP.clear()
            _PERSIST_T = threading.Thread(target=_persist_loop, name="wsds-block-persist", daemon=True)
            _PERSIST_T.start()


def _enqueue_write(cache, name, blk, data):
    """Hand one fetched block to the shared persister. Raises nothing the caller must handle:
    read_range counts a drop if the queue is full (the read already returned the bytes)."""
    _ensure_persister()
    _PERSIST_Q.put_nowait((cache, name, blk, data))


def _drain_batch(timeout=_POLL_S):
    try:
        first = _PERSIST_Q.get(timeout=timeout)
    except Empty:
        return None
    batch = {(id(first[0]), first[1], first[2]): first}
    deadline = time.time() + _BATCH_S
    while len(batch) < _BATCH_MAX:
        rem = deadline - time.time()
        if rem <= 0:
            break
        try:
            item = _PERSIST_Q.get(timeout=rem)
        except Empty:
            break
        batch[(id(item[0]), item[1], item[2])] = item
    return list(batch.values())


def _dispatch(items):
    """Group a mixed batch by cache so each root's _persist_batch still sees one sorted group
    (its fsync barrier and bitmap publish are per mirror fd)."""
    by_cache = {}
    for cache, name, blk, data in items:
        by_cache.setdefault(id(cache), (cache, []))[1].append((name, blk, data))
    for cache, batch in by_cache.values():
        try:
            cache._persist_batch(batch)
        except Exception:
            cache.dropped += len(batch)


def _persist_loop():
    while not _PERSIST_STOP.is_set():
        b = _drain_batch()
        if b:
            _dispatch(b)
    while True:  # drain what is left, then exit
        b = _drain_batch(timeout=0.01)
        if not b:
            break
        _dispatch(b)


def drain_writes(timeout=10.0):
    """Block until the shared queue is empty and its last batch is written (tests, shutdown)."""
    t0 = time.time()
    while not _PERSIST_Q.empty() and time.time() - t0 < timeout:
        time.sleep(0.005)
    time.sleep(_BATCH_S * 2 + 0.02)  # let the in-flight batch finish its fsync


# module-level singleton per cache root (counters + access log only; no thread)
_CACHES = {}
_CACHES_LOCK = threading.Lock()


def get_block_cache(root, logdir=None):
    with _CACHES_LOCK:
        c = _CACHES.get(root)
        if c is None:
            c = BlockCache(root, logdir=logdir)
            _CACHES[root] = c
        return c


# ------------------------------------------------------------------------------
# Stage-3: per-worker access logging (off hot path) + eviction/compaction helpers.
# ------------------------------------------------------------------------------
class AccessLogger:
    """Per-worker in-memory access aggregation -> atomic Feather snapshots.
    (object, block) -> [last_ts, count]; flushed by count/time. No shared state or
    inter-worker locking; snapshots are temp+rename so the evictor globs complete files."""

    def __init__(self, logdir, worker_id=None, flush_every=50000, flush_secs=30.0):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.worker = worker_id or _default_worker_id()
        self.flush_every = flush_every
        self.flush_secs = flush_secs
        self._agg = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._last_flush = time.time()
        self._n_since = 0

    def record(self, name, blk):
        now = time.time()
        with self._lock:
            e = self._agg.get((name, blk))
            if e is None:
                self._agg[(name, blk)] = [now, 1]
            else:
                e[0] = now
                e[1] += 1
            self._n_since += 1
            due = self._n_since >= self.flush_every or now - self._last_flush >= self.flush_secs
        if due:
            self.flush()

    def flush(self):
        with self._lock:
            if not self._agg:
                return
            snap = self._agg
            self._agg = {}
            self._n_since = 0
            self._last_flush = time.time()
            seq = self._seq
            self._seq += 1
        import pyarrow as pa
        import pyarrow.feather as feather

        objs, blks, ts, cnt = [], [], [], []
        for (name, blk), (last_ts, count) in snap.items():
            objs.append(name)
            blks.append(blk)
            ts.append(last_ts)
            cnt.append(count)
        tbl = pa.table(
            {"object": objs, "block": pa.array(blks, pa.int32()), "last_ts": ts, "count": pa.array(cnt, pa.int32())}
        )
        final = os.path.join(self.logdir, f"access-{self.worker}-{seq:06d}.feather")
        tmp = final + ".tmp"
        feather.write_feather(tbl, tmp)
        os.rename(tmp, final)

    def close(self):
        try:
            self.flush()
        except Exception:
            pass


def _default_worker_id():
    import socket

    procid = os.environ.get("SLURM_PROCID")
    tag = f"n{os.environ.get('SLURM_NODEID', '0')}.r{procid}" if procid else socket.gethostname()
    return f"{tag}.{os.getpid()}"


# ---- eviction/compaction (offline; used by wsds.cache_evict) ----
def read_layout(fd):
    sz = os.fstat(fd).st_size
    magic, objsize = FOOTER.unpack(os.pread(fd, FOOT_SZ, sz - FOOT_SZ))
    if magic != MAGIC:
        raise ValueError("not a block-cache .sparse file")
    nblocks = (objsize + SLOT - 1) // SLOT
    return objsize, nblocks, nblocks * SLOT


def present_blocks(fd, nblocks, bm_off):
    bm = os.pread(fd, nblocks, bm_off)
    return {i for i, b in enumerate(bm) if b}


def compact(path, keep_blocks):
    """Offset-preserving rewrite: keep only present blocks in keep_blocks; atomic rename
    (lock-free readers see old-all-blocks or new-evicted-gone, never a torn mix)."""
    import tempfile

    fd = os.open(path, os.O_RDONLY)
    objsize, nblocks, bm_off = read_layout(fd)
    present = present_blocks(fd, nblocks, bm_off)
    keep = present & set(keep_blocks)
    total = nblocks * SLOT + nblocks + FOOT_SZ
    tfd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".compact.")
    try:
        os.ftruncate(tfd, total)
        for blk in sorted(keep):
            n = SLOT if blk < nblocks - 1 else objsize - blk * SLOT
            os.pwrite(tfd, os.pread(fd, n, blk * SLOT), blk * SLOT)
            os.pwrite(tfd, b"\x01", bm_off + blk)
        os.pwrite(tfd, FOOTER.pack(MAGIC, objsize), bm_off + nblocks)
        os.fsync(tfd)
        os.close(tfd)
        tfd = None
        os.rename(tmp, path)
        tmp = None
    finally:
        if tfd is not None:
            os.close(tfd)
        if tmp is not None:
            os.unlink(tmp)
        os.close(fd)
    return len(keep), len(present) - len(keep)
