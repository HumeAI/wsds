from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .utils import WSShardMissingError

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: "WSDataset"
    shard_name: str
    offset: int
    overrides: dict = field(default_factory=dict)

    def get_audio(self, audio_columns=None):
        candidates = audio_columns or self.dataset._audio_file_keys

        r = self.get_one_of(*candidates)

        if not r:
            raise KeyError(f"No audio column (tried {candidates}) found among: {list(self.keys())}")

        return r

    def keys(self):
        return self.dataset.fields.keys() | (self.overrides.keys() if self.overrides else set())

    def items(self):
        yield from ((k, self[k]) for k in self.keys())

    def values(self):
        yield from (v for _, v in self.items())

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

    def __repr_field__(self, field, repr=repr):
        try:
            v = self[field]
            if hasattr(v, "shape"):
                if v.shape:
                    if v.size > 10:
                        trunc_repr = " ".join(repr(v).split(" ")[:10])
                        v = f"{trunc_repr}…], shape={repr(v.shape)}, dtype={v.dtype})"
                    else:
                        v = repr(v)
                else:
                    v = repr(v)
            else:
                v = repr(v)
                if isinstance(v, str) and len(v) > 1000:
                    v = v[:1000] + "…"
            return v
        except WSShardMissingError as err:
            return f"<missing shard: {err.fname}>"
        except KeyError as err:
            return f"<error: {err.args[0]}>"

    def __repr__(self, repr=repr):
        r = [
            f"WSSample({self.dataset.__repr__()}, shard_name={repr(self.shard_name)}, offset={repr(self.offset)}, fields={'{'}"
        ]
        other = []
        txt = []
        arrays = []
        for k in self.keys():
            try:
                v = self[k]
            except WSShardMissingError as err:
                arrays.append(k)
            except KeyError as err:
                other.append(k)
            else:
                if k == '__key__':
                    other.insert(0, k)
                elif hasattr(v, "shape") and v.shape:
                    arrays.append(k)
                elif isinstance(v, (str, bytes)):
                    txt.append(k)
                else:
                    other.append(k)

        def print_keys(keys):
            for k in keys:
                r.append(f"  '{k}': {self.__repr_field__(k, repr=repr)},")

        print_keys(other)
        if txt:
            r.append("# Text:")
            print_keys(txt)
        if arrays:
            r.append("# Arrays:")
            print_keys(arrays)
        r.append("})\n")
        return "\n".join(r)

    def _display_(self):
        import random

        import marimo

        special = {}

        def marimo_repr(x):
            if hasattr(x, "_display_"):
                rand_str = "__%030x__" % random.getrandbits(60)
                special[rand_str] = x._display_().text
                return rand_str
            else:
                return repr(x)

        # print(repr(special))
        html = marimo.md(f"```python\n{self.__repr__(repr=marimo_repr)}\n```").text
        for k, v in special.items():
            html = html.replace(k, v)
        return marimo.Html(html)

    def _ipython_display_(self):
        from IPython.display import display, Markdown
        display(Markdown(f"```python\n{self.__repr__()}\n```"))
