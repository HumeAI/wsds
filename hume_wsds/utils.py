from pathlib import Path

import numpy as np
import pyarrow as pa


def get_columns(fname):
    if isinstance(fname, Path):
        fname = str(fname)
    reader = pa.RecordBatchFileReader(pa.memory_map(fname))
    return reader.schema.names


def find_first_shard(path):
    return next(Path(path).iterdir(), None)


def list_all_columns(ds_path):
    dupes = {}
    cols = {}
    for p in Path(ds_path).iterdir():
        if p.suffix == ".wsds-link":
            col = p.with_suffix("").name
            cols[col] = (p.name, col)
            continue
        if not p.is_dir():
            continue
        # FIXME: this is quite expensive (adds up to 100ms on youtube-cc)
        # we should get one shard name from the sqlite index instead
        fname = next(p.iterdir(), None)
        if fname:
            for col in get_columns(fname):
                if col == "__key__":
                    continue
                # seems like we should fix this during the original conversion
                if col in cols or col in dupes:
                    dupes[col] = True
                    if col in cols:
                        dirname = cols[col][0]
                        cols[f"{dirname}.{col}"] = (dirname, col)
                        del cols[col]
                    cols[f"{p.name}.{col}"] = (p.name, col)
                else:
                    cols[col] = (p.name, col)
    return dict(sorted(cols.items()))


def make_key(src_file: str, segment_id: int):
    """Make a composite string key from source file name and sequential segment id.

    >>> make_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 1254)
    '20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_00001254'
    """

    assert isinstance(segment_id, int)
    return f"{src_file}_{segment_id:08d}"


def parse_key(key: str):
    """Parse a composite string key into the source file name and sequential segment id.

    >>> parse_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_00001254')
    ('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 1254)
    """

    src_file, segment_id = key.rsplit("_", 1)
    return src_file, int(segment_id)


def cast_types_for_storage(obj, float_cast=np.float32, int_cast=np.int64):
    if type(obj) is float:
        return float_cast(obj)
    elif type(obj) is int:
        return int_cast(obj)
    elif type(obj) is dict:
        return {
            k: cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast)
            for k, v in obj.items()
        }
    elif type(obj) is list:
        return [
            cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast)
            for v in obj
        ]
    else:
        print("unknown type:", type(obj), obj, type(obj) is float, float)
        return obj


def parse_key_two_parts(key: str):
    """Parse a composite string key into the source file name, segmentation kind and sequential segment id.

    >>> parse_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_rawvad_00001254')
    ('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 'rawvad', 1254)
    """

    src_file, segmentation_kind, segment_id = key.rsplit("_", 2)
    return src_file, segmentation_kind, int(segment_id)
