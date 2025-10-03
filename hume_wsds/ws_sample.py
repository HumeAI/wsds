from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: "WSDataset"  # noqa: F821
    shard_name: str
    offset: int
    overrides: dict = field(default_factory=dict)

    def get_key(self):
        return self.dataset.get_key(self.shard_name, self.offset)

    def get_audio(self, audio_columns=None):
        candidates = audio_columns or self.dataset._audio_file_keys

        # if we normalized into a single 'audio' field
        if "audio" in self:
            return self["audio"]

        r = self.get_one_of(*candidates)
        if not r:
            raise KeyError(f"No audio column (tried {candidates}) found among: {list(self.keys())}")
        return r

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

    def get_one_of(self, *fields, default=None):
        for f in fields:
            if f in self:
                return self[f]
        return default

    def get(self, field: str, default=None):
        return self[field] if field in self else default

    def __contains__(self, field):
        return field in self.overrides or field in self.dataset.fields.keys()

    def __repr_field__(self, field):
        try:
            _, v = field, self[field]
            if hasattr(v, "shape"):
                if v.shape:
                    trunc_repr = " ".join(repr(v).split(" ")[:10])
                    v = f"{trunc_repr}…, shape={repr(v.shape)}, dtype={v.dtype})"
                else:
                    v = repr(v)
            else:
                v = repr(v)
                if len(v) > 1000:
                    v = v[:1000] + "…"
        except FileNotFoundError:
            _, v = field, f"<missing shard: {self.dataset.fields[field][1]}/{self.shard_name}>"
        return v

    def __repr__(self):
        r = f"WSSample({repr(self.dataset)}, shard_name={repr(self.shard_name)}, offset={repr(self.offset)}, fields={'{'}\n"
        for k in self.keys():
            r += f"  {k} = {self.__repr_field__(k)},\n"
        r += "})\n"
        return r
