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


def _index_struct(**fields):
    """A one-row pyarrow struct scalar in the seek-index schema (missing fields = older index)."""
    import pyarrow as pa

    types = {"audio_offset": pa.uint64(), "audio_length": pa.uint64(), "file_header": pa.binary(),
             "file_footer": pa.binary(), "moov_offset": pa.uint64(), "moov_size": pa.uint32(),
             "seek_pts": pa.list_(pa.uint32()), "seek_pos": pa.list_(pa.uint64()), "pts_offset": pa.uint32(),
             "flags": pa.uint32(), "edit_media_time": pa.uint32(), "header_bytes": pa.uint32(),
             "footer_bytes": pa.uint32()}
    t = pa.struct([(k, types[k]) for k in fields])
    return pa.array([fields], type=t)[0]


class SeekIndexTest(unittest.TestCase):
    def test_canonical_schema_and_gate(self):
        from wsds.ws_seek_index import FLAG_PROBE_DONE, FLAG_PROBE_FAILED, SeekIndex

        base = dict(audio_offset=1000, audio_length=500_000, file_header=b"", file_footer=b"", moov_offset=1032,
                    moov_size=4000, seek_pts=[0, 10_000, 20_000, 30_000], seek_pos=[1100, 101_100, 201_100, 301_100],
                    pts_offset=25, flags=FLAG_PROBE_DONE, edit_media_time=1024, header_bytes=5032, footer_bytes=0)
        idx = SeekIndex.from_struct(_index_struct(**base))
        self.assertEqual(len(idx), 4)
        self.assertEqual((idx.audio_offset, idx.audio_length, idx.moov_offset, idx.moov_size), (1000, 500_000, 1032, 4000))
        self.assertEqual((idx.header_bytes, idx.edit_media_time), (5032, 1024))
        self.assertAlmostEqual(idx.pts_offset, 0.0025)
        self.assertEqual(idx.pos_at(1), 100_100)                    # blob-relative
        self.assertEqual(idx.pts_at(2), 2.0)
        self.assertEqual(idx.search(1.5), 2)                        # first point PAST 1.5 s
        self.assertTrue(idx.probed and idx.usable)
        bad = SeekIndex.from_struct(_index_struct(**{**base, "flags": FLAG_PROBE_DONE | FLAG_PROBE_FAILED}))
        self.assertTrue(bad.probed)
        self.assertFalse(bad.usable)                                # the accuracy gate

    def test_older_schema_defaults_and_empty(self):
        from wsds.ws_seek_index import SeekIndex

        old = _index_struct(audio_offset=10, audio_length=99, file_header=b"x" * 8192, file_footer=b"",
                            moov_offset=0, moov_size=0, seek_pts=[0, 10_000], seek_pos=[10, 60])
        idx = SeekIndex.from_struct(old)
        self.assertEqual((idx.flags, idx.header_bytes, idx.footer_bytes, idx.pts_offset), (0, 0, 0, None))
        self.assertTrue(idx.usable)
        self.assertIsNone(SeekIndex.from_struct(_index_struct(audio_offset=0, audio_length=0, moov_offset=0,
                                                              moov_size=0, seek_pts=[], seek_pos=[])))
        moov_only = SeekIndex.from_struct(_index_struct(audio_offset=0, audio_length=10, moov_offset=4,
                                                        moov_size=6, seek_pts=[], seek_pos=[]))
        self.assertEqual(len(moov_only), 0)                         # mp4 row with only the moov pointer

    def test_plan_ranges(self):
        from wsds.ws_seek_index import SeekIndex

        one_s = 10_000
        idx = SeekIndex.from_struct(_index_struct(
            audio_offset=1000, audio_length=10_000_000, moov_offset=1000 + 9_000_000, moov_size=50_000,
            seek_pts=[k * one_s for k in range(100)], seek_pos=[1000 + k * 100_000 for k in range(100)],
            header_bytes=4096, footer_bytes=0, flags=0, edit_media_time=0))
        plan = idx.plan_ranges(50.0, 55.0, margin_s=2.0, probe_bytes=1 << 19, pad_bytes=3 << 17)
        self.assertEqual(plan[0], (0, 4096 + (1 << 19)))                       # header + open probe
        self.assertIn((9_000_000, 9_050_000), plan)                             # non-faststart moov
        a, b = [r for r in plan if r[0] not in (0, 9_000_000)][0]
        self.assertEqual(a, 48 * 100_000)                                       # last point at/before t0 - margin
        self.assertEqual(b, 57 * 100_000 + (3 << 17))                           # point after t1 + margin, padded
        self.assertEqual(plan, sorted(plan))


class SeekAccuracyTest(unittest.TestCase):
    """Seeking by index returns the same audio as decoding from the start (bit-identity up to the
    codec's own frame alignment), for the containers we serve; a PROBE_FAILED index is refused."""

    SR, DUR = 16000, 40.0

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch

        from wsds.audio_codec import encode_audio

        t = np.arange(int(cls.SR * cls.DUR)) / cls.SR
        # a chirp plus a slow envelope: every window is distinct, so a misaligned seek shows up
        sig = 0.5 * np.sin(2 * np.pi * (200 + 30 * t) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
        x = torch.tensor(sig, dtype=torch.float32)[None]
        x.sample_rate = cls.SR
        cls.encoded = {}
        for fmt in ("mp3", "ogg", "mp4", "webm"):
            try:
                cls.encoded[fmt] = encode_audio(x, format=fmt)
            except Exception:                    # encoder not available in this build
                pass

    def _episode(self, fmt):
        import io

        from wsds.ws_audio import WSAudioEpisode

        return WSAudioEpisode(io.BytesIO(self.encoded[fmt]))

    def _index_of(self, fmt):
        """Build a SeekIndex the way an indexer does, from humecodec's packet scan."""
        import io

        import numpy as np
        from humecodec import MediaDecoder

        from wsds.ws_seek_index import PTS_UNIT, SeekIndex

        dec = MediaDecoder(io.BytesIO(self.encoded[fmt]))
        entries = dec.build_packet_index(dec.default_audio_stream, 32 * 1024)
        return SeekIndex(np.asarray([e.pos for e in entries], dtype=np.uint64),
                         np.asarray([max(0, round(e.pts_seconds / PTS_UNIT)) for e in entries], dtype=np.uint32),
                         audio_offset=0)

    @staticmethod
    def _lag(ref, test, sr):
        """Lag (samples) of `test` relative to `ref` via cross-correlation of the overlap."""
        import numpy as np

        n = min(len(ref), len(test))
        r, t = ref[:n] - ref[:n].mean(), test[:n] - test[:n].mean()
        c = np.fft.irfft(np.fft.rfft(r, 2 * n) * np.conj(np.fft.rfft(t, 2 * n)))
        k = int(np.argmax(c))
        return k if k <= n else k - 2 * n

    def _check(self, fmt, with_index, t0=21.3, t1=24.1, tol_samples=1):
        import numpy as np

        ep = self._episode(fmt)
        full = ep.read_segment(0, None)                     # ground truth: whole file from the start
        sr = int(full.sample_rate)                          # the codec's rate (opus decodes at 48 kHz)
        ref = full.numpy()[0]
        ep2 = self._episode(fmt)
        if with_index:
            ep2.set_seek_index(self._index_of(fmt))
        seg = ep2.read_segment(t0, t1).numpy()[0]           # a deep seek
        self.assertEqual(len(seg), round((t1 - t0) * sr))
        want = ref[round(t0 * sr):round(t1 * sr)]
        lag = self._lag(want, seg, sr)
        self.assertLessEqual(abs(lag), tol_samples, f"{fmt} index={with_index}: seek landed {lag} samples off")
        if tol_samples <= 1:
            self.assertLess(float(np.abs(want - seg).max()), 0.05, f"{fmt} index={with_index}: content differs")
        return lag, seg

    def test_mp3(self):
        if "mp3" not in self.encoded:
            self.skipTest("no mp3 encoder")
        self._check("mp3", with_index=False)
        self._check("mp3", with_index=True)

    def test_ogg_opus(self):
        if "ogg" not in self.encoded:
            self.skipTest("no ogg encoder")
        self._check("ogg", with_index=False)
        self._check("ogg", with_index=True)

    def test_mp4_aac(self):
        if "mp4" not in self.encoded:
            self.skipTest("no mp4 encoder")
        self._check("mp4", with_index=False)
        self._check("mp4", with_index=True)

    def test_webm_opus(self):
        """matroska carries cues: the index must NOT seed the demuxer (cross-checked on real
        webm/opus: a seeded seek landed 880 ms late), so with-index must equal without.
        (humecodec-MUXED webm/opus seeks a constant 21 ms early on `main` too -- an artefact of
        the synthetic file; real webm episodes seek to <1 ms -- hence the loose absolute bound.)"""
        if "webm" not in self.encoded:
            self.skipTest("no webm encoder")
        import numpy as np

        lag0, seg0 = self._check("webm", with_index=False, tol_samples=1200)
        lag1, seg1 = self._check("webm", with_index=True, tol_samples=1200)
        self.assertEqual(lag0, lag1)
        self.assertTrue(np.array_equal(seg0, seg1))

    def test_failed_probe_is_not_seeded(self):
        if "mp3" not in self.encoded:
            self.skipTest("no mp3 encoder")
        from dataclasses import replace

        from wsds.ws_seek_index import FLAG_PROBE_DONE, FLAG_PROBE_FAILED

        ep = self._episode("mp3")
        ep.set_seek_index(replace(self._index_of("mp3"), flags=FLAG_PROBE_DONE | FLAG_PROBE_FAILED))
        ep.read_segment(21.3, 24.1)
        dec, _ = ep.get_decoder()
        self.assertIsNotNone(dec)
        self.assertFalse(getattr(dec, "_seed_index", None) is not None and dec._seed_index.flags & FLAG_PROBE_FAILED)


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
