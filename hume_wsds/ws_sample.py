from dataclasses import dataclass

from hume_wsds.ws_audio import AudioReader, WSAudio
from hume_wsds.ws_dataset import WSDataset


@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: WSDataset
    shard_name: str
    offset: int

    def get_key(self):
        return self.dataset.get_key(self.shard_name, self.offset)

    def keys(self):
        return self.dataset.fields.keys()

    def items(self):
        yield from ((k, self[k]) for k in self.dataset.fields.keys())

    def values(self):
        yield from (v for k, v in self.items())

    def __getitem__(self, field):
        return self.dataset.get_sample(self.shard_name, field, self.offset)

    def __repr__(self):
        r = f"WSSample({repr(self.dataset)}, shard_name={repr(self.shard_name)}, offset={repr(self.offset)}, fields={'{'}\n"
        for k, v in self.items():
            if hasattr(v, "shape"):
                v = f"array(size={repr(v.shape)}, dtype={v.dtype})"
            else:
                v = repr(v)
                if len(v) > 1000:
                    v = v[:1000] + "…"
            r += f"  {k} = {v},\n"
        r += "})\n"
        return r


@dataclass(frozen=True, slots=True)
class WSSourceShard:
    shard_name: str
    source_dataset: "WSDataset"
    derived_dataset: "WSDataset"

    def get_sample(self, column, offset):
        # FIXME: using the global parse_key here confuses marimo compiler in strange ways
        file_name, segment_offset = self.derived_dataset.parse_key(
            self.derived_dataset.get_key(self.shard_name, offset)
        )
        print(file_name, segment_offset)
        source_sample = WSSample(
            self.source_dataset, *self.source_dataset.get_position(file_name)
        )
        # print(_, offset, self.derived_dataset.get_key(self.shard_name, offset))
        # print(file_name, source_sample['raw.vad.npy'].shape)
        tstart, tend = source_sample["raw.vad.npy"][segment_offset]

        return WSAudio(AudioReader(source_sample["mp3"]), tstart, tend)
