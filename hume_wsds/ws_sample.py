from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: "WSDataset"
    shard_name: str
    offset: int
    overrides: dict = field(default_factory=dict)

    def get_key(self):
        return self.dataset.get_key(self.shard_name, self.offset)

    def get_audio(self, audio_columnts=None):
        for col in self.dataset._audio_file_keys:
            if col in self: return self[col]
        raise KeyError(f"No audio column (tried {self.dataset._audio_file_keys}) found among: {self.keys()}")

    def keys(self):
        return self.dataset.fields.keys()

    def items(self):
        yield from ((k, self[k]) for k in self.dataset.fields.keys())

    def values(self):
        yield from (v for k, v in self.items())

    def __getitem__(self, field):
        if field in self.overrides:
            return self.overrides[field]
        return self.dataset.get_sample(self.shard_name, field, self.offset)

    def __setitem__(self, field, value):
        self.overrides[field] = value

    def get(self, field:str, default=None):
        return self[field] if field in self else default

    def __contains__(self, field):
        return field in self.overrides or field in self.dataset.fields.keys()

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
