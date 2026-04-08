import doctest
import tempfile
import unittest
from pathlib import Path

import wsds
from wsds import ws_dataset, ws_shard, ws_sink
from wsds.ws_sink import (
    KeyMismatchError,
    SampleCountMismatchError,
    WSBatchedSink,
    WSSink,
    _find_reference_shard,
    _read_shard_keys,
)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(wsds))
    tests.addTests(doctest.DocTestSuite(ws_dataset))
    tests.addTests(doctest.DocTestSuite(ws_shard))
    tests.addTests(doctest.DocTestSuite(ws_sink))
    tests.addTests(doctest.DocFileSuite("README.md"))
    return tests


def _make_samples(keys: list[str]) -> list[dict]:
    return [{"__key__": k, "value": i} for i, k in enumerate(keys)]


def _write_reference_shard(shard_path: Path, keys: list[str]) -> None:
    """Write a small reference shard with the given __key__ values."""
    with WSSink(str(shard_path)) as sink:
        for sample in _make_samples(keys):
            sink.write(sample)


class TestHelpers(unittest.TestCase):
    """Tests for _find_reference_shard and _read_shard_keys."""

    def test_find_reference_shard_picks_smallest(self):
        """Should pick the sibling shard with the smallest file size."""
        keys = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            # Write a small artifact
            (dataset / "small_artifact").mkdir()
            _write_reference_shard(dataset / "small_artifact" / "shard.wsds", keys)
            # Write a larger artifact (more columns = bigger file)
            (dataset / "large_artifact").mkdir()
            with WSSink(str(dataset / "large_artifact" / "shard.wsds")) as sink:
                for k in keys:
                    sink.write({"__key__": k, "v1": 0, "v2": "x" * 1000, "v3": 1.0})

            target = dataset / "new_artifact" / "shard.wsds"
            (dataset / "new_artifact").mkdir()
            ref = _find_reference_shard(target)
            self.assertIsNotNone(ref)
            self.assertEqual(ref.parent.name, "small_artifact")

    def test_find_reference_shard_skips_current_dir(self):
        keys = ["a", "b"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "artifact_a").mkdir()
            _write_reference_shard(dataset / "artifact_a" / "shard.wsds", keys)

            # Target is in artifact_a itself — should not find itself
            ref = _find_reference_shard(dataset / "artifact_a" / "shard.wsds")
            self.assertIsNone(ref)

    def test_find_reference_shard_skips_link_and_computed(self):
        keys = ["a"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            # Create a .wsds-link file and .wsds-computed dir
            (dataset / "audio.wsds-link").touch()
            (dataset / "audio.wsds-computed").mkdir()
            (dataset / "audio.wsds-computed" / "shard.wsds").touch()

            target = dataset / "new_artifact" / "shard.wsds"
            (dataset / "new_artifact").mkdir()
            ref = _find_reference_shard(target)
            self.assertIsNone(ref)

    def test_find_reference_shard_no_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "lonely_artifact").mkdir()
            ref = _find_reference_shard(dataset / "lonely_artifact" / "shard.wsds")
            self.assertIsNone(ref)

    def test_read_shard_keys(self):
        keys = ["x", "y", "z"]
        with tempfile.TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / "shard.wsds"
            _write_reference_shard(shard_path, keys)
            result = _read_shard_keys(shard_path)
            self.assertEqual(result, keys)


class TestValidateKeys(unittest.TestCase):
    """Tests for validate_keys auto-discovery in WSSink."""

    def _make_dataset(self, tmp: str, keys: list[str]) -> Path:
        """Create a dataset directory with a reference artifact already written."""
        dataset = Path(tmp)
        (dataset / "existing_artifact").mkdir()
        _write_reference_shard(dataset / "existing_artifact" / "shard.wsds", keys)
        (dataset / "new_artifact").mkdir()
        return dataset

    def test_matching_keys_succeeds(self):
        keys = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            with WSSink(fname, validate_keys=True) as sink:
                for s in _make_samples(keys):
                    sink.write(s)
            self.assertTrue(Path(fname).exists())

    def test_key_mismatch_raises(self):
        keys = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, validate_keys=True) as sink:
                    for s in _make_samples(["a", "WRONG", "c"]):
                        sink.write(s)
            self.assertEqual(ctx.exception.offset, 1)
            self.assertEqual(ctx.exception.expected_key, "b")
            self.assertEqual(ctx.exception.actual_key, "WRONG")
            self.assertFalse(Path(fname).exists())

    def test_missing_key_field_raises(self):
        keys = ["a", "b"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, validate_keys=True) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"value": 1})  # missing __key__
            self.assertEqual(ctx.exception.offset, 1)
            self.assertIsNone(ctx.exception.actual_key)

    def test_too_many_samples_raises(self):
        keys = ["a"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, validate_keys=True) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"__key__": "extra", "value": 1})
            self.assertEqual(ctx.exception.offset, 1)
            self.assertIsNone(ctx.exception.expected_key)

    def test_too_few_samples_raises(self):
        keys = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            with self.assertRaises(SampleCountMismatchError) as ctx:
                with WSSink(fname, validate_keys=True) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"__key__": "b", "value": 1})
            self.assertEqual(ctx.exception.expected_count, 3)
            self.assertEqual(ctx.exception.actual_count, 2)

    def test_no_siblings_warns_and_skips(self):
        """When no sibling artifacts exist, prints warning and skips validation."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            (dataset / "only_artifact").mkdir()
            fname = str(dataset / "only_artifact" / "shard.wsds")
            # Should succeed without validation (no siblings to compare against)
            with WSSink(fname, validate_keys=True) as sink:
                sink.write({"__key__": "a", "value": 0})
            self.assertTrue(Path(fname).exists())

    def test_validate_keys_false_no_validation(self):
        """Default behavior: no validation even with siblings present."""
        keys = ["a", "b"]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(tmp, keys)
            fname = str(dataset / "new_artifact" / "shard.wsds")
            # Write mismatched keys — should succeed because validate_keys=False
            with WSSink(fname) as sink:
                for s in _make_samples(["x", "y"]):
                    sink.write(s)
            self.assertTrue(Path(fname).exists())


class TestExceptions(unittest.TestCase):
    """Tests for exception classes."""

    def test_is_base_exception(self):
        self.assertTrue(issubclass(KeyMismatchError, BaseException))
        self.assertFalse(issubclass(KeyMismatchError, Exception))
        self.assertTrue(issubclass(SampleCountMismatchError, BaseException))
        self.assertFalse(issubclass(SampleCountMismatchError, Exception))

    def test_error_messages(self):
        err = KeyMismatchError("shard.wsds", 5, "expected_k", "actual_k")
        self.assertIn("offset 5", str(err))
        self.assertIn("expected_k", str(err))
        self.assertIn("actual_k", str(err))

        err_missing = KeyMismatchError("shard.wsds", 3, "expected_k", None)
        self.assertIn("missing", str(err_missing))

        err_overflow = KeyMismatchError("shard.wsds", 10, None, "extra_k")
        self.assertIn("Too many", str(err_overflow))

        err_count = SampleCountMismatchError("shard.wsds", 5, 3)
        self.assertIn("expected 5", str(err_count))
        self.assertIn("wrote 3", str(err_count))


if __name__ == "__main__":
    unittest.main()
