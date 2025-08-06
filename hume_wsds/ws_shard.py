import io
import pickle

import numpy as np
import pyarrow as pa


class WSShard:
    def __init__(self, fname, shard_name=None):
        self.shard_name = shard_name
        self.fname = fname
        self.reader = pa.RecordBatchFileReader(pa.memory_map(fname))
        self.batch_size = int(self.reader.schema.metadata[b"batch_size"])

        # cache
        self._start = None
        self._end = None
        self._data = None

    def get_sample(self, column, offset):
        if self._data is None or offset < self._start or offset >= self._end:
            i = offset // self.batch_size
            if i >= self.reader.num_record_batches:
                return
            self._data = self.reader.get_batch(i)
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= len(self._data):
            return
        data = self._data[column][j]
        # FIXME: implement proper encoders and decoders
        if column.endswith("npy"):
            return np.load(io.BytesIO(data.as_buffer()))
        elif column.endswith("pyd"):
            return pickle.load(io.BytesIO(data.as_buffer()))
        elif column.endswith("txt"):
            return bytes(data.as_buffer()).decode("utf-8")
        else:
            return data

    def __repr__(self):
        r = f"WSShard({repr(self.fname)})"
        if self._data:
            r += f" # cached_region = [{self._start, self._end}]"
        return r
