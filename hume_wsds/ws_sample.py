from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: "WSDataset"
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
