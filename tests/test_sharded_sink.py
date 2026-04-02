"""Tests for ShardedSink."""

import os
import tempfile
import unittest
from pathlib import Path

import pyarrow

from wsds import ShardedSink
from wsds.ws_sink import _estimate_sample_bytes


def _read_shard_samples(path: str | Path) -> list[dict]:
    """Read all rows from a .wsds shard file as a list of dicts."""
    reader = pyarrow.ipc.open_file(str(path))
    rows = []
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        rows.extend(batch.to_pylist())
    return rows


class TestShardedSinkSampleRotation(unittest.TestCase):
    """Test rotation based on max_samples_per_shard."""

    def test_rotation_by_sample_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out, max_samples_per_shard=3) as sink:
                for i in range(10):
                    sink.write({"id": i, "value": f"sample_{i}"})

            paths = sink.shard_paths
            # 10 samples / 3 per shard = 4 shards (3, 3, 3, 1)
            self.assertEqual(len(paths), 4)

            all_ids = []
            for p in paths:
                rows = _read_shard_samples(p)
                all_ids.extend(r["id"] for r in rows)

            self.assertEqual(sorted(all_ids), list(range(10)))

    def test_exact_multiple(self):
        """When total samples is exact multiple of max, no empty trailing shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out, max_samples_per_shard=5) as sink:
                for i in range(10):
                    sink.write({"id": i})

            self.assertEqual(len(sink.shard_paths), 2)
            total = sum(len(_read_shard_samples(p)) for p in sink.shard_paths)
            self.assertEqual(total, 10)


class TestShardedSinkBytesRotation(unittest.TestCase):
    """Test rotation based on max_shard_bytes."""

    def test_rotation_by_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            # Each sample has ~1024 bytes of incompressible data.
            # Set limit low so we get multiple shards.
            sample_data = os.urandom(1024)
            with ShardedSink(out, max_shard_bytes=3000) as sink:
                for i in range(10):
                    sink.write({"id": i, "data": sample_data})

            paths = sink.shard_paths
            self.assertGreater(len(paths), 1, "Should have rotated into multiple shards")

            all_ids = []
            for p in paths:
                rows = _read_shard_samples(p)
                all_ids.extend(r["id"] for r in rows)
            self.assertEqual(sorted(all_ids), list(range(10)))


class TestShardedSinkBothLimits(unittest.TestCase):
    """Test that whichever limit triggers first wins."""

    def test_sample_limit_triggers_first(self):
        """Small sample_limit with huge byte_limit -- sample limit should trigger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(
                out,
                max_samples_per_shard=2,
                max_shard_bytes=10 * 1024 * 1024,  # 10 MB -- won't trigger
            ) as sink:
                for i in range(6):
                    sink.write({"id": i})

            self.assertEqual(len(sink.shard_paths), 3)

    def test_byte_limit_triggers_first(self):
        """Large sample_limit with small byte_limit -- byte limit should trigger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            sample_data = os.urandom(2048)
            with ShardedSink(
                out,
                max_samples_per_shard=1000,  # won't trigger
                max_shard_bytes=3000,
            ) as sink:
                for i in range(10):
                    sink.write({"id": i, "data": sample_data})

            paths = sink.shard_paths
            self.assertGreater(len(paths), 1)
            # sample limit didn't trigger, so each shard should have < 1000 samples
            for p in paths:
                rows = _read_shard_samples(p)
                self.assertLess(len(rows), 1000)


class TestShardedSinkEdgeCases(unittest.TestCase):

    def test_empty_sink_no_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out) as sink:
                pass  # no writes

            self.assertEqual(sink.shard_paths, [])
            # output dir should not even be created
            self.assertFalse(os.path.exists(out))

    def test_shard_paths_returns_correct_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out, prefix="test", max_samples_per_shard=2) as sink:
                for i in range(5):
                    sink.write({"id": i})

            paths = sink.shard_paths
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(p.exists(), f"Shard file should exist: {p}")
                self.assertTrue(str(p).endswith(".wsds"))
                self.assertIn("test-", str(p.name))

            # Verify naming scheme
            self.assertEqual(paths[0].name, "test-00000.wsds")
            self.assertEqual(paths[1].name, "test-00001.wsds")
            self.assertEqual(paths[2].name, "test-00002.wsds")

    def test_all_samples_accounted_for(self):
        """Write many samples and verify none are lost across shards."""
        n = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out, max_samples_per_shard=7) as sink:
                for i in range(n):
                    sink.write({"id": i, "text": f"hello_{i}"})

            all_rows = []
            for p in sink.shard_paths:
                all_rows.extend(_read_shard_samples(p))

            self.assertEqual(len(all_rows), n)
            ids = [r["id"] for r in all_rows]
            self.assertEqual(sorted(ids), list(range(n)))
            # Verify text values too
            for r in all_rows:
                self.assertEqual(r["text"], f"hello_{r['id']}")

    def test_shard_paths_is_copy(self):
        """shard_paths property returns a copy, not the internal list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "out")
            with ShardedSink(out, max_samples_per_shard=5) as sink:
                for i in range(3):
                    sink.write({"id": i})

            paths = sink.shard_paths
            paths.append(Path("fake"))
            self.assertNotEqual(len(sink.shard_paths), len(paths))


class TestEstimateSampleBytes(unittest.TestCase):

    def test_bytes_field(self):
        data = os.urandom(500)
        self.assertEqual(_estimate_sample_bytes({"data": data}), 500)

    def test_string_field(self):
        self.assertEqual(_estimate_sample_bytes({"text": "hello"}), 5)

    def test_scalar_field(self):
        self.assertEqual(_estimate_sample_bytes({"x": 42}), 8)

    def test_mixed(self):
        sample = {"data": b"abc", "text": "hi", "num": 1}
        self.assertEqual(_estimate_sample_bytes(sample), 3 + 2 + 8)


if __name__ == "__main__":
    unittest.main()
