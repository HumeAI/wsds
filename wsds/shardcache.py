"""Shard cache: block-sparse mirrors on the cluster nodes' local NVMe, read over HTTP.

The block cache (`pupyarrow/block_cache.py`) keeps one sparse mirror per remote shard on the
shared filesystem. This module moves those mirrors onto node-local drives without changing their
format, and reads them back from whichever node holds them:

  * `Ring` places every mirror on one node (or a few) by rendezvous hashing of its name, so any
    process can work out where a shard lives from the node list alone;
  * `ShardCacheReader` is a FileReader that asks those nodes for a byte range and hands the read
    to the reader it wraps -- the shared-filesystem mirror or plain S3 -- whenever no node can
    serve it;
  * `python -m wsds.shardcache populate` runs ON a cache node and copies the mirrors that node
    owns from the shared filesystem onto its drives. Re-running it tops existing copies up.

The nodes run `shardcache-server` (research-infra, hume_iren/files/shardcache). It checks the
mirror's block bitmap on every request, so a reply is either real bytes or an error:

    GET /<name>  Range: bytes=a-b | bytes=-n      (object coordinates)
        206  bytes                 every block of the range is cached
        409  X-Shardcache-Missing  some blocks are not cached
        404                        the node has no mirror of that name

Nothing here writes to a cache node over the network. Blocks that were not cached are fetched by
the wrapped reader, which persists them to the shared-filesystem mirror as before; the next
`populate` run merges them into the node copy. The shared filesystem thus only ever holds what
has not been merged yet (plus whatever the ring marks `keep_source`).

Switched on by `WSDS_SHARDCACHE=<ring.json>`. The ring file:

    {"port": 8765,
     "base": "/mnt/weka",
     "copies": 1,
     "nodes": [{"host": "node-a", "addr": "10.0.0.1"},
               {"host": "node-b", "addr": "10.0.0.2", "keep_source": true},
               {"host": "node-c", "addr": "10.0.0.3", "role": "backup"}],
     "backup_fraction": 0.4}

`base` is the directory mirror names are relative to. `copies` primaries are chosen among the
nodes without a role; a `backup` node additionally takes a second copy of `backup_fraction` of
the mirrors. `keep_source` marks a node that is not trusted to stay up: `populate --consume`
never deletes the shared-filesystem mirror of anything placed there.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http.client
import json
import os
import random
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .pupyarrow.block_cache import EXT, FOOTER, MAGIC, SLOT
from .pupyarrow.file_reader import FileReader

ENV = "WSDS_SHARDCACHE"
FOOT_SZ = FOOTER.size

CONNECT_TIMEOUT = float(os.environ.get("WSDS_SHARDCACHE_CONNECT_TIMEOUT", "1"))
READ_TIMEOUT = float(os.environ.get("WSDS_SHARDCACHE_READ_TIMEOUT", "10"))
DOWN_SECONDS = float(os.environ.get("WSDS_SHARDCACHE_DOWN_SECONDS", "30"))


def _score(a: str, b: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{a}\0{b}".encode(), digest_size=8).digest(), "big")


# ------------------------------------------------------------------------------ placement


class Ring:
    """Which nodes hold a mirror. Pure function of the ring file and the mirror's name.

    >>> ring = Ring({"base": "/data", "nodes": [{"host": h} for h in "abcdefg"]})
    >>> [n["host"] for n in ring.nodes_for("ds/part/audio/shard_0001.wsds")]
    ['b']
    >>> ring.name_for("/data/ds/part/audio/shard_0001.wsds.sparse")
    'ds/part/audio/shard_0001.wsds'
    >>> ring.name_for("/elsewhere/shard.wsds") is None
    True

    Rendezvous hashing: dropping a node moves only the mirrors that node held.

    >>> names = [f"ds/audio/shard_{i:04d}.wsds" for i in range(700)]
    >>> smaller = Ring({"base": "/data", "nodes": [{"host": h} for h in "abcdef"]})
    >>> moved = [n for n in names if ring.nodes_for(n) != smaller.nodes_for(n)]
    >>> all(ring.nodes_for(n)[0]["host"] == "g" for n in moved)
    True
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.port = int(cfg.get("port", 8765))
        self.base = os.path.realpath(cfg.get("base", "/"))
        self.copies = int(cfg.get("copies", 1))
        self.backup_fraction = float(cfg.get("backup_fraction", 0.0))
        self.nodes = [dict(n) for n in cfg["nodes"]]
        for n in self.nodes:
            n.setdefault("addr", n["host"])
        self.primaries = [n for n in self.nodes if n.get("role", "primary") == "primary"]
        self.backups = [n for n in self.nodes if n.get("role") == "backup"]
        if len(self.primaries) < self.copies or self.copies < 1:
            raise ValueError("ring needs at least `copies` primary nodes")
        self._dirs: dict[str, str] = {}

    @classmethod
    def load(cls, path: str) -> "Ring":
        with open(path) as f:
            return cls(json.load(f))

    def nodes_for(self, name: str) -> list[dict]:
        """The nodes holding `name`, in the order a reader should try them."""
        ranked = sorted(self.primaries, key=lambda n: _score(n["host"], name), reverse=True)
        out = ranked[: self.copies]
        if self.backups and _score("backup-fraction", name) / 2**64 < self.backup_fraction:
            out.append(max(self.backups, key=lambda n: _score(n["host"], name)))
        return out

    def name_for(self, path: str) -> str | None:
        """Mirror name for a shared-filesystem path (with or without the .sparse extension):
        the path relative to `base`, with symlinked directories resolved so a shard reached
        through two datasets has one name. None when the path is outside `base`."""
        if path.endswith(EXT):
            path = path[: -len(EXT)]
        d, leaf = os.path.split(os.path.abspath(path))
        real = self._dirs.get(d)
        if real is None:
            real = self._dirs[d] = os.path.realpath(d)
        rel = os.path.relpath(os.path.join(real, leaf), self.base)
        return None if rel.startswith("..") else rel


_ring_lock = threading.Lock()
_ring_cache: dict[str, Ring | None] = {}


def active_ring() -> Ring | None:
    """The ring named by $WSDS_SHARDCACHE, loaded once per process; None when unset or unreadable
    (a broken ring file must never stop a reader from falling back to what worked before)."""
    path = os.environ.get(ENV)
    if not path:
        return None
    with _ring_lock:
        if path not in _ring_cache:
            try:
                _ring_cache[path] = Ring.load(path)
            except (OSError, ValueError, KeyError) as e:
                print(f"[wsds shardcache] ignoring {ENV}={path}: {type(e).__name__}: {e}", file=sys.stderr)
                _ring_cache[path] = None
        return _ring_cache[path]


# ------------------------------------------------------------------------------ client

_local = threading.local()
_down: dict[tuple[str, int], float] = {}  # (addr, port) -> time it may be tried again
STATS = {"hits": 0, "misses": 0, "absent": 0, "errors": 0, "bytes": 0}


def _conn(addr: str, port: int) -> http.client.HTTPConnection:
    """One keep-alive connection per (process, thread, node). A forked DataLoader worker must not
    share its parent's sockets, hence the pid check."""
    if getattr(_local, "pid", None) != os.getpid():
        _local.pid, _local.conns = os.getpid(), {}
    c = _local.conns.get((addr, port))
    if c is None:
        c = _local.conns[addr, port] = http.client.HTTPConnection(addr, port, timeout=CONNECT_TIMEOUT)
    return c


def _drop(addr: str, port: int) -> None:
    c = getattr(_local, "conns", {}).pop((addr, port), None)
    if c is not None:
        c.close()


def fetch(ring: Ring, name: str, range_header: str) -> bytes | None:
    """One range from the first node that has every block of it; None when none does.

    A node that refuses or times out is left alone for DOWN_SECONDS, so a dead node costs one
    connect timeout per process, not one per read.
    """
    for node in ring.nodes_for(name):
        addr, key = node["addr"], (node["addr"], ring.port)
        if _down.get(key, 0) > time.monotonic():
            continue
        for attempt in (0, 1):
            try:
                c = _conn(addr, ring.port)
                if c.sock is None:
                    c.connect()
                    c.sock.settimeout(READ_TIMEOUT)
                c.request("GET", "/" + name, headers={"Range": range_header})
                r = c.getresponse()
                body = r.read()
            except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                _drop(*key)  # a keep-alive connection the server closed: retry once on a fresh one
                if attempt:
                    STATS["errors"] += 1
                    _down[key] = time.monotonic() + DOWN_SECONDS
                continue
            except (OSError, http.client.HTTPException):
                _drop(*key)
                STATS["errors"] += 1
                _down[key] = time.monotonic() + DOWN_SECONDS
                break
            if r.status == 206:
                STATS["hits"] += 1
                STATS["bytes"] += len(body)
                return body
            STATS["misses" if r.status == 409 else "absent" if r.status == 404 else "errors"] += 1
            break
    return None


class ShardCacheReader(FileReader):
    """Reads a shard from the cache nodes; anything they cannot serve goes to `fallback`.

    `fallback_factory` builds the reader used before the shard cache existed (the shared
    filesystem mirror, or plain S3). It is only called on the first read the nodes cannot serve,
    so a fully cached shard opens no S3 client and touches no shared filesystem.
    """

    def __init__(self, ring: Ring, name: str, fallback_factory):
        super().__init__()
        self._ring = ring
        self._name = name
        self._fallback_factory = fallback_factory
        self._fallback = None
        self._lock = threading.Lock()
        self.node_reads = 0
        self.fallback_reads = 0

    @property
    def fallback(self) -> FileReader:
        with self._lock:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            return self._fallback

    def _from_nodes(self, range_header: str) -> bytes | None:
        data = fetch(self._ring, self._name, range_header)
        if data is None:
            self.fallback_reads += 1
        else:
            self.node_reads += 1
        return data

    def _raw_read(self, offset: int, length: int) -> bytes:
        data = self._from_nodes(f"bytes={offset}-{offset + length - 1}")
        return data if data is not None else self.fallback._raw_read(offset, length)

    def _raw_read_end(self, n: int) -> bytes:
        data = self._from_nodes(f"bytes=-{n}")
        return data if data is not None else self.fallback._raw_read_end(n)

    async def _async_read_impl(self, offset: int, length: int) -> bytes:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, self._from_nodes, f"bytes={offset}-{offset + length - 1}")
        if data is not None:
            return data
        # Building the fallback may build an S3 client through the IO loop: never on the loop itself.
        fb = self._fallback or await loop.run_in_executor(None, lambda: self.fallback)
        return await fb._async_read_impl(offset, length)

    def close(self):
        super().close()
        fb, self._fallback = self._fallback, None
        if fb is not None:
            try:
                fb.close()
            except Exception:
                pass


def wrap(mirror_path: str, fallback_factory):
    """`fallback_factory()` behind the shard cache when a ring is active and the mirror path lies
    under its base; otherwise just `fallback_factory()`. The one call site is WSS3Shard."""
    ring = active_ring()
    name = ring.name_for(mirror_path) if ring is not None else None
    if name is None:
        return fallback_factory()
    return ShardCacheReader(ring, name, fallback_factory)


# ------------------------------------------------------------------------------ populate


def read_layout(fd: int) -> tuple[int, int, bytes]:
    """(object size, block count, bitmap) of an open mirror."""
    size = os.fstat(fd).st_size
    if size < FOOT_SZ:
        raise ValueError("too short to be a mirror")
    magic, objsize = FOOTER.unpack(os.pread(fd, FOOT_SZ, size - FOOT_SZ))
    nblocks = (objsize + SLOT - 1) // SLOT
    if magic != MAGIC or size != nblocks * SLOT + nblocks + FOOT_SZ:
        raise ValueError("not a block-cache mirror")
    return objsize, nblocks, os.pread(fd, nblocks, nblocks * SLOT)


def _runs(blocks: list[int], limit: int = 64):
    """Consecutive block indices grouped into (first, count) runs of at most `limit` blocks."""
    i = 0
    while i < len(blocks):
        j = i
        while j + 1 < len(blocks) and blocks[j + 1] == blocks[j] + 1 and j + 1 - i < limit:
            j += 1
        yield blocks[i], j - i + 1
        i = j + 1


def copy_mirror(src: str, dst: str, verify: int = 8) -> dict:
    """Bring the mirror at `dst` up to date with the one at `src`: every block present in the
    source and absent in the destination is copied. Creates `dst` if needed.

    Order matters, because the node may be serving `dst` while this runs: data first, fsync,
    then the bitmap bytes. A new mirror is built under a temporary name and renamed into place.
    `verify` random copied blocks are read back and compared before the result is reported.
    """
    sfd = os.open(src, os.O_RDONLY)
    try:
        objsize, nblocks, sbm = read_layout(sfd)
        bm_off = nblocks * SLOT
        new = not os.path.exists(dst)
        target = dst + ".tmp" if new else dst
        if new:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        dfd = os.open(target, os.O_RDWR | os.O_CREAT | (os.O_TRUNC if new else 0), 0o664)
        try:
            if new:
                os.ftruncate(dfd, bm_off + nblocks + FOOT_SZ)
                dbm = bytes(nblocks)
            else:
                dobj, dnb, dbm = read_layout(dfd)
                if dobj != objsize:
                    raise ValueError(f"object size differs: source {objsize}, destination {dobj}")
            todo = [i for i in range(nblocks) if sbm[i] and not dbm[i]]
            for first, count in _runs(todo):
                n = min((first + count) * SLOT, objsize) - first * SLOT
                data = os.pread(sfd, n, first * SLOT)
                if len(data) != n:
                    raise OSError(f"short read from {src}")
                os.pwrite(dfd, data, first * SLOT)
            if todo or new:
                os.fsync(dfd)  # the data is durable before any bitmap byte says so
                for first, count in _runs(todo, limit=1 << 30):
                    os.pwrite(dfd, b"\x01" * count, bm_off + first)
                if new:
                    os.pwrite(dfd, FOOTER.pack(MAGIC, objsize), bm_off + nblocks)
                os.fsync(dfd)
            for i in random.sample(todo, min(verify, len(todo))):
                n = min(SLOT, objsize - i * SLOT)
                if os.pread(sfd, n, i * SLOT) != os.pread(dfd, n, i * SLOT):
                    raise OSError(f"verification failed for block {i} of {dst}")
            covered = all(d or not s for s, d in zip(sbm, os.pread(dfd, nblocks, bm_off)))
            os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(dfd)
        if new:
            os.rename(target, dst)
    finally:
        os.close(sfd)
    return {"blocks": len(todo), "bytes": len(todo) * SLOT, "new": new, "covered": covered}


def cache_roots(dest: str) -> list[str]:
    """The node's cache drives: every mounted filesystem directly under `dest`, else `dest`."""
    subs = sorted(os.path.join(dest, d) for d in os.listdir(dest))
    return [d for d in subs if os.path.ismount(d)] or [dest]


def pick_root(roots: list[str], name: str, need: int) -> str:
    """Where `name` lives on this node: wherever it already is, else the drive rendezvous hashing
    picks among those with room (so a lost drive only moves its own mirrors)."""
    for r in roots:
        if os.path.exists(os.path.join(r, name + EXT)):
            return r
    ranked = sorted(roots, key=lambda r: _score(os.path.basename(r), name), reverse=True)
    for r in ranked:
        st = os.statvfs(r)
        if st.f_bavail * st.f_frsize > need + (64 << 30):  # keep 64 GB free per drive
            return r
    raise OSError(f"no cache drive has room for {name}")


def scan(roots: list[str], subdir: str | None = "audio"):
    """Mirror files under `roots`. Where a directory has a `subdir` column dir, only that one is
    entered: the other column dirs hold no mirrors and can be very large."""
    for root in roots:
        for d, dirs, files in os.walk(root):
            if subdir and subdir in dirs:
                dirs[:] = [subdir]
            for f in files:
                if f.endswith(EXT):
                    yield os.path.join(d, f)


def populate(ring: Ring, sources, host: str, dest: str, consume: bool = False, dry_run: bool = False,
             threads: int = 8, log=print) -> dict:
    """Copy the mirrors `host` owns from `sources` (shared-filesystem paths) onto its drives.

    With `consume`, a source mirror is deleted once its node copy holds every block it had --
    unless the ring places it on more than one node, or on a `keep_source` node. Blocks a
    training job writes into the source between the copy and the delete are lost, which costs
    one more fetch from the object store later.
    """
    roots = cache_roots(dest)
    tot = {"seen": 0, "mine": 0, "copied": 0, "blocks": 0, "bytes": 0, "consumed": 0, "failed": 0}
    lock = threading.Lock()
    t0 = time.time()

    def one(src: str):
        name = ring.name_for(src)
        owners = ring.nodes_for(name) if name else []
        mine = any(n["host"] == host for n in owners)
        res = None
        if mine and not dry_run:
            try:
                sfd = os.open(src, os.O_RDONLY)
                try:
                    _objsize, _nb, bm = read_layout(sfd)
                finally:
                    os.close(sfd)
                root = pick_root(roots, name, (len(bm) - bm.count(0)) * SLOT)
                res = copy_mirror(src, os.path.join(root, name + EXT))
                may_consume = consume and len(owners) == 1 and not owners[0].get("keep_source")
                if may_consume and res["covered"]:
                    os.unlink(src)
                    res["consumed"] = True
            except (OSError, ValueError) as e:
                res = {"error": f"{type(e).__name__}: {e}"}
                log(f"  FAILED {src}: {res['error']}")
        with lock:
            tot["seen"] += 1
            tot["mine"] += mine
            if res and "error" in res:
                tot["failed"] += 1
            elif res:
                tot["copied"] += bool(res["blocks"])
                tot["blocks"] += res["blocks"]
                tot["bytes"] += res["bytes"]
                tot["consumed"] += bool(res.get("consumed"))
            if tot["seen"] % 1000 == 0:
                dt = time.time() - t0
                log(f"  {tot['seen']} mirrors seen, {tot['mine']} mine, {tot['bytes'] / 1e12:.2f} TB copied, "
                    f"{tot['bytes'] / 1e6 / max(dt, 1e-9):.0f} MB/s")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, sources))
    tot["seconds"] = time.time() - t0
    return tot


def _sources(args):
    if args.list:
        with (sys.stdin if args.list == "-" else open(args.list)) as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    else:
        yield from scan(args.roots, args.subdir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m wsds.shardcache", description="Node-local shard cache tools.")
    ap.add_argument("--ring", default=os.environ.get(ENV), help=f"ring file (default ${ENV})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="print the mirror files under some directories, one per line")
    p.add_argument("roots", nargs="+")
    p.add_argument("--subdir", default="audio", help="column dir that holds mirrors ('' walks everything)")

    p = sub.add_parser("populate", help="copy the mirrors this node owns onto its cache drives")
    p.add_argument("roots", nargs="*", help="directories to scan for mirrors (or use --list)")
    p.add_argument("--list", help="file of mirror paths, one per line ('-' for stdin); skips the scan")
    p.add_argument("--subdir", default="audio")
    p.add_argument("--host", default=socket.gethostname(), help="act as this ring node (default: this machine)")
    p.add_argument("--dest", default="/mnt/shardcache", help="directory whose mounted subdirectories are the drives")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--consume", action="store_true", help="delete each source mirror once its node copy is complete")
    p.add_argument("--dry-run", action="store_true", help="only count what this node would take")

    p = sub.add_parser("where", help="print the nodes that hold a mirror")
    p.add_argument("paths", nargs="+", help="shared-filesystem mirror paths")

    a = ap.parse_args(argv)
    if a.cmd == "scan":
        for path in scan(a.roots, a.subdir or None):
            print(path)
        return
    if not a.ring:
        sys.exit(f"no ring file: pass --ring or set ${ENV}")
    ring = Ring.load(a.ring)
    if a.cmd == "where":
        for path in a.paths:
            name = ring.name_for(path)
            owners = " ".join(n["host"] for n in ring.nodes_for(name)) if name else "(outside the ring's base)"
            print(f"{name or path}: {owners}")
        return
    if not a.list and not a.roots:
        sys.exit("populate: give directories to scan, or --list")
    if a.host not in [n["host"] for n in ring.nodes]:
        sys.exit(f"populate: {a.host} is not a node of {a.ring}")
    print(f"populate: node {a.host}, drives {' '.join(cache_roots(a.dest))}"
          f"{', consuming sources' if a.consume else ''}{' (dry run)' if a.dry_run else ''}")
    tot = populate(ring, _sources(a), a.host, a.dest, consume=a.consume, dry_run=a.dry_run, threads=a.threads)
    print(f"done in {tot['seconds']:.0f}s: {tot['seen']} mirrors seen, {tot['mine']} owned by this node, "
          f"{tot['copied']} updated ({tot['blocks']} blocks, {tot['bytes'] / 1e12:.3f} TB), "
          f"{tot['consumed']} sources deleted, {tot['failed']} failed")
    if tot["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
