import doctest
import tempfile
import unittest
from pathlib import Path

import wsds
from wsds import ws_dataset, ws_shard, ws_sink
from wsds.ws_sink import KeyMismatchError, SampleCountMismatchError, WSSink


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(wsds))
    tests.addTests(doctest.DocTestSuite(ws_dataset))
    tests.addTests(doctest.DocTestSuite(ws_shard))
    tests.addTests(doctest.DocTestSuite(ws_sink))
    tests.addTests(doctest.DocFileSuite("README.md"))
    return tests


class TestKeyValidation(unittest.TestCase):
    """Tests for write-time __key__ validation in WSSink."""

    def _make_samples(self, keys: list[str]) -> list[dict]:
        return [{"__key__": k, "value": i} for i, k in enumerate(keys)]

    def test_matching_reference_keys(self):
        keys = ["a", "b", "c"]
        samples = self._make_samples(keys)
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with WSSink(fname, reference_keys=keys) as sink:
                for s in samples:
                    sink.write(s)
            self.assertTrue(Path(fname).exists())

    def test_key_mismatch_raises(self):
        keys = ["a", "b", "c"]
        samples = self._make_samples(["a", "WRONG", "c"])
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, reference_keys=keys) as sink:
                    for s in samples:
                        sink.write(s)
            self.assertEqual(ctx.exception.offset, 1)
            self.assertEqual(ctx.exception.expected_key, "b")
            self.assertEqual(ctx.exception.actual_key, "WRONG")
            self.assertFalse(Path(fname).exists())

    def test_missing_key_raises(self):
        keys = ["a", "b"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, reference_keys=keys) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"value": 1})  # missing __key__
            self.assertEqual(ctx.exception.offset, 1)
            self.assertEqual(ctx.exception.expected_key, "b")
            self.assertIsNone(ctx.exception.actual_key)
            self.assertFalse(Path(fname).exists())

    def test_too_many_samples_raises(self):
        keys = ["a"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with self.assertRaises(KeyMismatchError) as ctx:
                with WSSink(fname, reference_keys=keys) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"__key__": "extra", "value": 1})
            self.assertEqual(ctx.exception.offset, 1)
            self.assertIsNone(ctx.exception.expected_key)
            self.assertEqual(ctx.exception.actual_key, "extra")
            self.assertFalse(Path(fname).exists())

    def test_too_few_samples_raises(self):
        keys = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with self.assertRaises(SampleCountMismatchError) as ctx:
                with WSSink(fname, reference_keys=keys) as sink:
                    sink.write({"__key__": "a", "value": 0})
                    sink.write({"__key__": "b", "value": 1})
                    # missing third sample
            self.assertEqual(ctx.exception.expected_count, 3)
            self.assertEqual(ctx.exception.actual_count, 2)
            self.assertFalse(Path(fname).exists())

    def test_no_reference_keys_backward_compat(self):
        samples = self._make_samples(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with WSSink(fname) as sink:
                for s in samples:
                    sink.write(s)
            self.assertTrue(Path(fname).exists())

    def test_no_reference_keys_no_key_field_ok(self):
        """Without reference_keys, samples without __key__ should still work."""
        with tempfile.TemporaryDirectory() as tmp:
            fname = str(Path(tmp) / "shard.wsds")
            with WSSink(fname) as sink:
                sink.write({"value": 1})
                sink.write({"value": 2})
            self.assertTrue(Path(fname).exists())

    def test_is_base_exception(self):
        """KeyMismatchError and SampleCountMismatchError are BaseException, not Exception."""
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
