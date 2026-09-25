import doctest
import json
import os
import tempfile
import unittest
from pathlib import Path

import wsds
from wsds import audio_codec, ws_audio, ws_dataset, ws_shard, ws_sink
from wsds.ws_sample import WSSample  # noqa: F401


class _FakeObject:
    """An in-memory 'remote object' standing in for S3: counts range reads."""

    def __init__(self, data):
        self.data = data
        self.reads = []          # (offset, length)

    def reader(self):
        from wsds.pupyarrow.file_reader import FileReader

        obj = self

        class _R(FileReader):
            def _raw_read(self, offset, length):
                obj.reads.append((offset, length))
                return obj.data[offset : offset + length]

            def _raw_read_end(self, n):
                obj.reads.append((len(obj.data) - n, n))
                return obj.data[-n:]

        return _R()


class BlockCacheTest(unittest.TestCase):
    """CachedFileReader over a fake backend: what is fetched, what is served locally."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        import numpy as np
        rng = np.random.default_rng(0)
        self.blob = rng.integers(0, 256, 5 * 128 * 1024 + 12345, dtype=np.uint8).tobytes()   # 5.09 blocks

    def tearDown(self):
        self._tmp.cleanup()

    def _reader(self, obj, name="p/audio/x.wsds"):
        from wsds.pupyarrow.file_reader import CachedFileReader

        return CachedFileReader(obj.reader, lambda: len(obj.data), self.root, name)

    def test_cold_read_fetches_one_coalesced_range_then_serves_locally(self):
        from wsds.pupyarrow.block_cache import SLOT, drain_writes

        obj = _FakeObject(self.blob)
        r = self._reader(obj)
        got = r.read(SLOT + 100, 2 * SLOT)                # spans blocks 1..3 -> ONE fetch of 3 blocks
        self.assertEqual(got, self.blob[SLOT + 100 : SLOT + 100 + 2 * SLOT])
        self.assertEqual(r.fetches, 1)
        self.assertEqual(obj.reads, [(SLOT, 3 * SLOT)])
        drain_writes()
        r2 = self._reader(obj)                            # a new reader (another worker/process)
        self.assertEqual(r2.read(SLOT + 100, 2 * SLOT), got)
        self.assertEqual(r2.fetches, 0)                   # served from the mirror
        self.assertEqual(len(obj.reads), 1)
        self.assertEqual(r2.read(0, 10), self.blob[:10])  # block 0 is still a miss
        self.assertEqual(r2.fetches, 1)
        r.close()
        r2.close()

    def test_read_end_and_size_from_footer(self):
        from wsds.pupyarrow.block_cache import drain_writes

        obj = _FakeObject(self.blob)
        r = self._reader(obj)
        self.assertEqual(r.read_end(-6, 6), self.blob[-6:])
        drain_writes()
        sizes = []
        r2 = self._reader(_FakeObject(self.blob))
        r2._size_fn_impl = lambda: sizes.append(1) or len(self.blob)
        self.assertEqual(r2.read_end(-6, 6), self.blob[-6:])
        self.assertEqual(sizes, [])                       # size came from the mirror footer, no HEAD
        self.assertEqual(r2.fetches, 0)
        r.close()
        r2.close()

    def test_inner_reader_is_never_built_when_mirrored(self):
        from wsds.pupyarrow.block_cache import SLOT, drain_writes

        obj = _FakeObject(self.blob)
        r = self._reader(obj)
        r.read(0, SLOT)
        drain_writes()
        r.close()
        built = []
        from wsds.pupyarrow.file_reader import CachedFileReader
        r2 = CachedFileReader(lambda: built.append(1) or obj.reader(), lambda: len(obj.data), self.root, "p/audio/x.wsds")
        self.assertEqual(r2.read(10, 100), self.blob[10:110])
        self.assertEqual(built, [])
        r2.close()

    def test_mirror_layout_and_eviction_scan(self):
        from wsds.cache_evict import scan_cache
        from wsds.pupyarrow.block_cache import EXT, SLOT, drain_writes

        obj = _FakeObject(self.blob)
        r = self._reader(obj, name="delivery/x_batch_0/source/audio/shard_1.wsds")
        r.read(2 * SLOT, 10)
        drain_writes()
        path = os.path.join(self.root, "delivery/x_batch_0/source/audio/shard_1.wsds" + EXT)
        self.assertTrue(os.path.exists(path))
        [(name, p, present)] = scan_cache(self.root)
        self.assertEqual((name, present), ("delivery/x_batch_0/source/audio/shard_1.wsds", {2}))
        r.close()

    def test_async_reads_await_misses(self):
        """The async path (what FeatherFile.async_record_batch / a gathered batch scan uses):
        misses are awaited on the inner reader's async impl, never parked on a thread; hits
        never build the inner reader at all."""
        import asyncio

        from wsds.pupyarrow.block_cache import SLOT, drain_writes
        from wsds.pupyarrow.file_reader import CachedFileReader, FileReader

        obj = _FakeObject(self.blob)
        calls = []

        class _AsyncInner(FileReader):
            async def _async_read_impl(self, offset, length):
                calls.append((offset, length))
                await asyncio.sleep(0)
                return obj.data[offset : offset + length]

            def _raw_read(self, offset, length):          # must not be used by the async path
                raise AssertionError("sync read on the async path")

        built = []
        r = CachedFileReader(lambda: built.append(1) or _AsyncInner(), lambda: len(obj.data), self.root, "p/audio/y.wsds")

        async def scan():                                   # many concurrent misses, like a batch scan
            return await asyncio.gather(*(r.async_read(k * SLOT, 100) for k in range(5)))

        got = asyncio.run(scan())
        self.assertEqual(got, [self.blob[k * SLOT : k * SLOT + 100] for k in range(5)])
        self.assertEqual(len(calls), 5)
        self.assertEqual(built, [1])
        drain_writes()
        r2 = CachedFileReader(lambda: built.append(1) or _AsyncInner(), lambda: len(obj.data), self.root, "p/audio/y.wsds")
        self.assertEqual(asyncio.run(r2.async_read(2 * SLOT + 7, 50)), self.blob[2 * SLOT + 7 : 2 * SLOT + 57])
        self.assertEqual(built, [1])                        # a hit: inner never built
        self.assertEqual(r2.fetches, 0)
        r.close()
        r2.close()

    def test_cached_s3_reader_compat_surface(self):
        from wsds.pupyarrow.file_reader import CachedS3FileReader, mirror_name

        self.assertEqual(mirror_name("b", "/d/x.wsds"), "d/x.wsds")
        self.assertEqual(mirror_name("b", "d/x.wsds", flat=False), "b/d/x.wsds")
        r = CachedS3FileReader(object(), "bucket", "d/x.wsds", self.root, name="p/audio/x.wsds")
        self.assertEqual((r._bucket, r._key, r._name, r.block_cache.root), ("bucket", "d/x.wsds", "p/audio/x.wsds", self.root))
        self.assertIsNone(r._inner)                        # no S3 reader until a miss



def _write_shard(path, keys, _duration=True, **columns):
    """Write one tiny .wsds shard: __key__ + load_duration (needed by the indexer; give it in ONE
    column dir only, a column present in two dirs is exposed as <dir>.<col>) + columns(key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ws_sink.WSSink(str(path)) as sink:
        for k in keys:
            row = {"__key__": k, **{c: f(k) for c, f in columns.items()}}
            if _duration:
                row["load_duration"] = 1.0
            sink.write(row)


def _index(index_dir, partitions=("",)):
    """Index `index_dir` like `wsds init` does, optionally over several partition folders."""
    from wsds.ws_index import WSDSIndexWriter
    from wsds.ws_tools import extract_index_for_shard

    index_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for partition in partitions:   # before the (still empty) index file exists
        root = index_dir / partition
        for shard in wsds.WSDataset(root).get_shard_list(ignore_index=True):
            records.append({**extract_index_for_shard(root, shard), "partition": partition})
    with WSDSIndexWriter(str(index_dir / "index.sqlite3")) as index:
        for r in records:
            index.append(r)
        index.append_metadata({"segmented": False})


def _link(path, dataset_dir, **extra):
    spec = {"dataset_dir": dataset_dir, "loader": ["wsds.ws_shard", "WSKeyedColumnShard"], **extra}
    path.write_text(json.dumps(spec))


class KeyedColumnShardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.keys = [f"k{i}" for i in range(6)]

    def tearDown(self):
        self._tmp.cleanup()

    def _catalog(self, name, tag):
        cat = self.root / name
        _write_shard(cat / "blob" / "c0.wsds", self.keys[:3], payload=lambda k: f"{tag}-{k}", kind=lambda k: "m4a")
        _write_shard(cat / "blob" / "c1.wsds", self.keys[3:], payload=lambda k: f"{tag}-{k}", kind=lambda k: "mp3")
        _index(cat)
        return cat

    def test_lookup_by_key_in_filtered_reordered_dataset(self):
        self._catalog("catalog", "A")
        derived = self.root / "derived"
        # a filtered, reordered, differently sharded view of the catalog + one unknown key
        _write_shard(derived / "meta" / "d0.wsds", ["k4", "k1", "k5", "zz"], note=lambda k: f"note-{k}")
        _link(derived / "blob.wsds-link", "../catalog", columns=["payload", "kind"])
        _index(derived)

        ds = wsds.WSDataset(derived)
        self.assertEqual([ds[i]["payload"] for i in range(3)], ["A-k4", "A-k1", "A-k5"])
        self.assertEqual([ds[i]["kind"] for i in range(3)], ["mp3", "m4a", "mp3"])
        self.assertEqual(ds["k1"]["note"], "note-k1")
        self.assertEqual(ds["k1"]["payload"], "A-k1")
        with self.assertRaises(wsds.utils.WSShardMissingError):
            ds["zz"]["payload"]

    def test_legacy_single_column_link(self):
        self._catalog("catalog", "A")
        derived = self.root / "derived"
        _write_shard(derived / "meta" / "d0.wsds", ["k2"], note=lambda k: k)
        _link(derived / "payload.wsds-link", "../catalog", column="payload")
        _index(derived)
        self.assertEqual(wsds.WSDataset(derived)[0]["payload"], "A-k2")

    def test_partition_link_overrides_root_link(self):
        self._catalog("catalog_a", "A")
        self._catalog("catalog_b", "B")
        base = self.root / "derived"
        _write_shard(base / "p1" / "meta" / "d0.wsds", ["k0", "k3"], note=lambda k: k)
        _write_shard(base / "p2" / "meta" / "d0.wsds", ["k5", "k2"], note=lambda k: k)   # same shard name
        index_dir = base / "index"
        index_dir.mkdir(parents=True)
        _link(index_dir / "blob.wsds-link", "../../catalog_a", columns=["payload"])
        _link(base / "p2" / "blob.wsds-link", "../../catalog_b", columns=["payload"])   # relative to p2/
        _index(index_dir, partitions=("../p1", "../p2"))

        ds = wsds.WSDataset(index_dir)
        self.assertEqual({ds[k]["payload"] for k in ("k0", "k3")}, {"A-k0", "A-k3"})
        self.assertEqual({ds[k]["payload"] for k in ("k5", "k2")}, {"B-k5", "B-k2"})


class OpenShardCacheTest(unittest.TestCase):
    """The process-global open-shard LRU: bound, pins, negative cache, per-dataset close."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = {k: os.environ.get(k) for k in ("WSDS_OPEN_SHARDS", "WSDS_LINKED_DATASETS")}
        self._saved_cache = ws_dataset._OPEN_SHARDS

    def tearDown(self):
        ws_dataset._OPEN_SHARDS = self._saved_cache
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _cache(self, cap):
        os.environ["WSDS_OPEN_SHARDS"] = str(cap)
        ws_dataset._OPEN_SHARDS = None
        return ws_dataset.open_shard_cache()

    def _dataset(self, name="ds", n_shards=6, dirs=("meta", "more")):
        root = self.root / name
        for i in range(n_shards):
            for n, d in enumerate(dirs):
                _write_shard(root / d / f"d{i}.wsds", [f"{name}-k{i}-{j}" for j in range(3)], _duration=(n == 0),
                             **{f"{d}_v": lambda k: k})
        _index(root)
        return wsds.WSDataset(root)

    @staticmethod
    def _at(ds, shard, field):
        """Read `field` of the first row of shard d<shard> (by shard ref, not by index order)."""
        return WSSample(ds, ("", f"d{shard}"), 0)[field]

    def test_eviction_closes_handles(self):
        cache = self._cache(3)
        ds = self._dataset(dirs=("meta",))
        first = ds.get_shard("meta", ("", "d0"))
        self.assertIsNotNone(first.reader)
        for i in range(6):
            self._at(ds, i, "meta_v")
        st = cache.stats()
        self.assertLessEqual(st["open"], 3)
        self.assertGreaterEqual(st["evictions"], 3)
        self.assertIsNone(first.reader)                    # evicted handle was CLOSED, not just dropped
        self.assertIsNone(ds._open_shards.get(ds.get_shard_path("meta", ("", "d0"))))

    def test_shard_visit_pins_then_releases(self):
        cache = self._cache(2)
        ds = self._dataset()
        p0 = [ds.get_shard_path(d, ("", "d0")) for d in ("meta", "more")]
        with ds.shard_visit(("", "d0")):
            self._at(ds, 0, "meta_v")
            self._at(ds, 0, "more_v")
            self.assertEqual(cache.n_pinned(), 2)
            for i in range(1, 6):                          # churn far past the cap
                self._at(ds, i, "meta_v")
            self.assertTrue(all(ds._open_shards.get(p) is not None for p in p0))   # pinned: never evicted
            held = [ds._open_shards.get(p) for p in p0]
        self.assertEqual(cache.n_pinned(), 0)
        self.assertTrue(all(ds._open_shards.get(p) is None for p in p0))          # released on exit...
        self.assertTrue(all(h.reader is None for h in held))                       # ...and closed
        self._at(ds, 0, "meta_v")                                                  # re-opens fine

    def test_visit_is_reentrant(self):
        self._cache(4)
        ds = self._dataset(n_shards=1)
        p = ds.get_shard_path("meta", ("", "d0"))
        with ds.shard_visit(("", "d0")):
            with ds.shard_visit(("", "d0")):
                self._at(ds, 0, "meta_v")
            self.assertIsNotNone(ds._open_shards.get(p))   # inner exit must not close the outer's handles
        self.assertIsNone(ds._open_shards.get(p))

    def test_missing_shard_is_negative_cached(self):
        self._cache(8)
        root = self.root / "neg"
        _write_shard(root / "meta" / "d0.wsds", ["a", "b"], meta_v=lambda k: k)
        _write_shard(root / "meta" / "d1.wsds", ["c", "d"], meta_v=lambda k: k)
        _write_shard(root / "extra" / "d0.wsds", ["a", "b"], _duration=False, extra_v=lambda k: k)   # d1 has no `extra`
        _index(root)
        ds = wsds.WSDataset(root)
        self.assertEqual(ds["a"]["extra_v"], "a")
        with self.assertRaises(wsds.utils.WSShardMissingError):
            ds["c"]["extra_v"]
        entry = ds._open_shards.get(ds.get_shard_path("extra", ("", "d1")))
        self.assertIsInstance(entry, ws_dataset._MissingShard)
        with self.assertRaises(wsds.utils.WSShardMissingError):
            ds["d"]["extra_v"]
        self.assertIs(ds._open_shards.get(ds.get_shard_path("extra", ("", "d1"))), entry)   # no retry/re-open

    def test_close_drops_only_own_shards(self):
        cache = self._cache(16)
        a, b = self._dataset("a", n_shards=2), self._dataset("b", n_shards=2)
        self._at(a, 0, "meta_v")
        self._at(b, 0, "meta_v")
        pa_, pb = a.get_shard_path("meta", ("", "d0")), b.get_shard_path("meta", ("", "d0"))
        a.close()
        self.assertIsNone(cache.get(pa_))
        self.assertIsNotNone(cache.get(pb))
        b.close()
        self.assertIsNone(cache.get(pb))

    def test_linked_datasets_are_bounded(self):
        self._cache(32)
        os.environ["WSDS_LINKED_DATASETS"] = "2"
        cats = []
        for i in range(3):
            cat = self.root / f"cat{i}"
            _write_shard(cat / "blob" / "c.wsds", [f"k{i}"], payload=lambda k, i=i: f"P{i}")
            _index(cat)
            cats.append(cat)
        derived = self.root / "derived"
        for i in range(3):
            _write_shard(derived / f"p{i}" / "meta" / "d.wsds", [f"k{i}"], note=lambda k: k)
            _link(derived / f"p{i}" / "blob.wsds-link", f"../../cat{i}", columns=["payload"])
        index_dir = derived / "index"
        index_dir.mkdir()
        _link(index_dir / "blob.wsds-link", "../../cat0", columns=["payload"])
        _index(index_dir, partitions=("../p0", "../p1", "../p2"))
        ds = wsds.WSDataset(index_dir)
        self.assertEqual([ds[f"k{i}"]["payload"] for i in range(3)], ["P0", "P1", "P2"])
        self.assertEqual(len(ds._linked_datasets), 2)                # the oldest catalog was evicted
        self.assertEqual(ds["k0"]["payload"], "P0")                  # ...and is re-opened on demand


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(wsds))
    tests.addTests(doctest.DocTestSuite(ws_dataset))
    tests.addTests(doctest.DocTestSuite(ws_shard))
    tests.addTests(doctest.DocTestSuite(ws_sink))
    tests.addTests(doctest.DocTestSuite(ws_audio))
    tests.addTests(doctest.DocTestSuite(audio_codec))
    tests.addTests(doctest.DocFileSuite("README.md"))
    return tests


if __name__ == "__main__":
    unittest.main()
