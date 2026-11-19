import doctest
import unittest

import wsds
from wsds import ws_dataset, ws_shard, ws_sink, ws_audio, audio_codec


class TestSampleContains(unittest.TestCase):
    """`field in sample` / `sample.get(field)` when a column dir lacks the sample's shard."""

    def test_missing_column_shard(self):
        import os
        import tempfile

        import pyarrow as pa

        from wsds.utils import WSShardMissingError

        def write(path, cols):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            t = pa.table(cols).replace_schema_metadata({"batch_size": "2"})
            with pa.OSFile(path, "wb") as f, pa.RecordBatchFileWriter(f, t.schema) as w:
                w.write_table(t, max_chunksize=2)

        with tempfile.TemporaryDirectory() as d:
            for shard in ("a", "b"):
                write(f"{d}/meta/{shard}.wsds", {"__key__": [f"{shard}0", f"{shard}1"], "x": [1, 2]})
            write(f"{d}/extra/a.wsds", {"__key__": ["a0", "a1"], "y": ["p", "q"]})
            ds = wsds.WSDataset(d)
            a = next(iter(ds.iter_shard(("", "a"))))
            b = next(iter(ds.iter_shard(("", "b"))))
            self.assertIn("y", a)
            self.assertEqual(a.get("y"), "p")
            self.assertIn("x", b)
            self.assertNotIn("y", b)
            self.assertEqual(b.get("y", "default"), "default")
            self.assertNotIn("no_such_field", b)
            with self.assertRaises(WSShardMissingError):
                b["y"]


def load_tests(loader, tests, ignore):
    tests.addTests(loader.loadTestsFromTestCase(TestSampleContains))
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
