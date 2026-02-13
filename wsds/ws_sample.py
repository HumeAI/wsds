from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .utils import WSShardMissingError, validate_shards
from .ws_decode import get_audio as _get_audio

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


@dataclass(frozen=True)
class WSSample:
    dataset: "WSDataset"
    shard_ref: str
    offset: int
    overrides: dict = field(default_factory=dict)
    # Key verification state (mutable containers to work with frozen dataclass)
    _verified_column_dirs: set = field(default_factory=set, repr=False, compare=False)
    _reference_key: list = field(default_factory=list, repr=False, compare=False)

    def get_audio(self, audio_columns=None):
        return _get_audio(self, audio_columns)

    def keys(self):
        return self.dataset.fields.keys() | (self.overrides.keys() if self.overrides else set())

    def items(self):
        yield from ((k, self[k]) for k in self.keys())

    def values(self):
        yield from (v for _, v in self.items())

    def _verify_key_for_field(self, field: str):
        """Verify __key__ in this field's column_dir matches the reference key."""
        value = self.dataset.fields.get(field)
        if value is None:
            return
        (column_dir, _column) = value[0]

        if column_dir in self._verified_column_dirs:
            return

        # Skip computed columns (they don't have their own __key__)
        if column_dir in self.dataset.computed_columns:
            self._verified_column_dirs.add(column_dir)
            return

        # Get __key__ from this column_dir
        try:
            key = self.dataset.get_shard(column_dir, self.shard_ref).get_sample("__key__", self.offset)
        except (WSShardMissingError, KeyError):
            # Can't verify if shard or key is missing
            self._verified_column_dirs.add(column_dir)
            return

        if not self._reference_key:
            # First column_dir accessed - store as reference
            self._reference_key.append((column_dir, key))
        else:
            ref_column_dir, ref_key = self._reference_key[0]
            if key != ref_key:
                raise ValueError(
                    f"Key mismatch at offset {self.offset} in shard {self.shard_ref}: "
                    f"{ref_column_dir} has '{ref_key}' but {column_dir} has '{key}'"
                )

        self._verified_column_dirs.add(column_dir)

    def __getitem__(self, field):
        if field in self.overrides:
            return self.overrides[field]
        self._verify_key_for_field(field)
        return self.dataset.get_sample(self.shard_ref, field, self.offset)

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
            f"WSSample({self.dataset.__repr__()}, shard_ref={repr(self.shard_ref)}, offset={repr(self.offset)}, fields={'{'}"
        ]
        other = []
        txt = []
        arrays = []

        # Group columns by column directory
        columns_by_dir = {}
        for k in self.keys():
            if k in self.overrides:
                column_dir = "__overrides__"
            elif k in self.dataset.fields:
                value = self.dataset.fields[k]
                (column_dir, _column) = value[0]
            else:
                column_dir = "__unknown__"
            if column_dir not in columns_by_dir:
                columns_by_dir[column_dir] = []
            columns_by_dir[column_dir].append(k)

        # Prefetch shard tails concurrently for all column dirs that will be accessed
        dirs_to_prefetch = [s for s in columns_by_dir.keys() if s not in ("__overrides__", "__unknown__")]
        if dirs_to_prefetch:
            validate_shards(self.dataset, [self.shard_ref], dirs_to_prefetch)

        # Identify large column directories (>10 columns)
        large_dirs = {
            column_dir: cols
            for column_dir, cols in columns_by_dir.items()
            if len(cols) > 10 and column_dir not in ("__overrides__", "__unknown__")
        }

        # Columns in small column directories go through normal classification
        small_dir_keys = set()
        for column_dir, cols in columns_by_dir.items():
            if column_dir not in large_dirs:
                small_dir_keys.update(cols)

        missing = []
        for k in small_dir_keys:
            try:
                v = self[k]
            except WSShardMissingError:
                missing.append(k)
            except KeyError:
                other.append(k)
            else:
                if k == "__key__":
                    other.insert(0, k)
                elif hasattr(v, "shape") and v.shape:
                    arrays.append(k)
                elif isinstance(v, (str, bytes)):
                    txt.append(k)
                else:
                    other.append(k)

        def print_keys(keys, max_keys=None):
            for k in keys[:max_keys] if max_keys else keys:
                r.append(f"  '{k}': {self.__repr_field__(k, repr=repr)},")

        print_keys(other)
        if txt:
            r.append("# Text:")
            print_keys(txt)
        if arrays:
            r.append("# Arrays:")
            print_keys(arrays)

        # Handle large column directories
        for column_dir, cols in sorted(large_dirs.items()):
            r.append(f"# {column_dir} ({len(cols)} columns, showing top 10):")

            # Try to get float values for sorting by highest value
            float_values = []
            for k in cols:
                try:
                    v = self[k]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        float_values.append((k, float(v)))
                    elif hasattr(v, "shape") and not v.shape and hasattr(v, "item"):
                        # scalar numpy array
                        float_values.append((k, float(v.item())))
                except (WSShardMissingError, KeyError, TypeError, ValueError):
                    pass

            if len(float_values) == len(cols):
                # All columns are floats, sort by value descending
                float_values.sort(key=lambda x: x[1], reverse=True)
                top_keys = [k for k, v in float_values[:10]]
            else:
                # Not all floats, just show first 10
                top_keys = sorted(cols)[:10]

            print_keys(top_keys)
            if len(cols) > 10:
                r.append(f"  # ... and {len(cols) - 10} more columns")

        if missing:
            r.append("# Missing shards:")
            print_keys(missing)

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
        from .utils import is_notebook

        if not is_notebook():
            print(repr(self))
            return

        import random

        from IPython.display import HTML, display

        special = {}

        def ipython_repr(x):
            if hasattr(x, "_repr_html_"):
                rand_str = "__%030x__" % random.getrandbits(60)
                special[rand_str] = x._repr_html_()
                return rand_str
            else:
                return repr(x)

        # Jupyter Markdown renders client-side so we cannot use the same trick as Marimo
        html = ('<pre style="font-family: monospace; white-space: pre-wrap;">' +
            self.__repr__(repr=ipython_repr).replace("<", "&lt;").replace(">", "&gt;") +
            '</pre>')

        for k, v in special.items():
            html = html.replace(k, v)

        display(HTML(html))
