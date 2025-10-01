import io
import pickle
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from hume_wsds.ws_audio import AudioReader, WSAudio
from hume_wsds.ws_sample import WSSample


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
                raise IndexError(f"{offset} is out of range for shard {self.fname}")
            self._data = self.reader.get_batch(i)
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= len(self._data):
            raise IndexError(f"{offset} is out of range for shard {self.fname}")
        data = self._data[column][j]
        # FIXME: implement proper encoders and decoders
        if column.endswith("npy"):
            return np.load(io.BytesIO(data.as_buffer()))
        elif column.endswith("pyd"):
            return pickle.load(io.BytesIO(data.as_buffer()))
        elif column.endswith("txt"):
            return data.as_buffer().to_pybytes().decode("utf-8")
        else:
            # FIXME: we need to handle audio decoding here to avoid copying the entire audio buffer
            return data.as_py(maps_as_pydicts="strict")

    def __repr__(self):
        r = f"WSShard({repr(self.fname)})"
        if self._data:
            r += f" # cached_region = [{self._start, self._end}]"
        return r


@dataclass(slots=True)
class WSSourceAudioShard:
    shard_name: str
    source_dataset: "WSDataset"  # noqa: F821
    derived_dataset: "WSDataset"  # noqa: F821
    vad_column: str

    # cache
    _source_file_name: str = None
    _source_sample: WSSample = None
    _source_reader: AudioReader = None

    @classmethod
    def from_link(cls, link, source_dataset, derived_dataset, shard_name):
        return cls(shard_name, source_dataset, derived_dataset, link["vad_column"])

    def get_timestamps(self, segment_offset):
        return self._source_sample[self.vad_column][segment_offset]

    def get_sample(self, _column, offset):
        file_name, segment_offset = self.derived_dataset.parse_key(
            self.derived_dataset.get_key(self.shard_name, offset)
        )

        if self._source_file_name != file_name:
            self._source_sample = self.source_dataset[file_name]
            self._source_reader = AudioReader(self._source_sample.get_audio())
            self._source_file_name = file_name

        tstart, tend = self.get_timestamps(segment_offset)
        return WSAudio(self._source_reader, tstart, tend)
