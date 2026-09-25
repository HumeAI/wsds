import doctest
import json
import tempfile
import unittest
from pathlib import Path

import wsds
from wsds import audio_codec, ws_audio, ws_dataset, ws_shard, ws_sink
from wsds.ws_sample import WSSample  # noqa: F401


def _write_shard(path, keys, **columns):
    """Write one tiny .wsds shard: __key__ + load_duration (needed by the indexer) + columns(key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ws_sink.WSSink(str(path)) as sink:
        for k in keys:
            sink.write({"__key__": k, "load_duration": 1.0, **{c: f(k) for c, f in columns.items()}})


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
