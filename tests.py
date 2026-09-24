import doctest
import tempfile
import unittest
from pathlib import Path

import wsds
from wsds import audio_codec, ws_audio, ws_dataset, ws_shard, ws_sink


class ArrColumnTest(unittest.TestCase):
    """".arr" columns: variable-length numpy arrays stored as native pyarrow lists."""

    def test_round_trip(self):
        import numpy as np
        import pyarrow as pa

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"__key__": "a", "vad.arr": np.array([[0.0, 1.5], [2.0, 3.25]], dtype=np.float32), "ids.arr": np.array([1, 2, 3], dtype=np.int64)},
                {"__key__": "b", "vad.arr": np.zeros((0, 2), dtype=np.float32), "ids.arr": np.array([], dtype=np.int64)},
                {"__key__": "c", "vad.arr": np.array([[9.0, 9.5]], dtype=np.float32), "ids.arr": np.array([7], dtype=np.int64)},
            ]
            with ws_sink.WSSink(str(root / "seg" / "s0.wsds")) as sink:
                for r in rows:
                    sink.write(r)
            schema = pa.ipc.open_file(str(root / "seg" / "s0.wsds")).schema
            self.assertEqual(schema.field("vad.arr").type, pa.list_(pa.list_(pa.float32(), 2)))   # native, not a blob
            self.assertEqual(schema.field("ids.arr").type, pa.list_(pa.int64()))
            shard = ws_shard.WSShard(wsds.WSDataset(root, ignore_index=True), str(root / "seg" / "s0.wsds"))
            for i, r in enumerate(rows):
                for col in ("vad.arr", "ids.arr"):
                    got = shard.get_sample(col, i)
                    self.assertIsInstance(got, np.ndarray)
                    self.assertEqual(got.dtype, r[col].dtype, col)
                    self.assertEqual(got.shape, r[col].shape, col)
                    np.testing.assert_array_equal(got, r[col])


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
