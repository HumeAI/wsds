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


def list_all_columns(ds_path, shard_name=None):
    """Given a dataset path, return a list of all columns.

    If you also give a shard name it greatly speeds it up
    on network-filesystems where listing folder contents is slow."""
    dupes = {}
    cols = {}
    for p in Path(ds_path).iterdir():
        if p.suffix == ".wsds-link":
            col = p.with_suffix("").name
            cols[col] = (p.name, col)
            continue
        if not p.is_dir():
            continue
        if shard_name is None:
            fname = find_first_shard(p)
        else:
            fname = (p / shard_name).with_suffix(".wsds")
        if fname and fname.exists():
            for col in get_columns(fname):
                if col == "__key__" and "__key__" in cols:
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


def list_all_shards(dataset: str, verbose: bool = False):
    shards = {}
    for subdir in Path(dataset).iterdir():
        if not subdir.is_dir():
            continue
        shards[subdir] = {file.name for file in subdir.iterdir() if file.suffix == ".wsds"}
        if not shards[subdir]:
            print(f"error: empty folder {subdir}")
            del shards[subdir]

    common_shards = {v for shard_values in shards.values() for v in shard_values}
    num_common = len(common_shards)

    errors = False
    for subdir, files in shards.items():
        missing = common_shards - files
        n_missing = len(missing)
        if n_missing == 0:
            status = "[COMPLETE]"
        else:
            status = f"[MISSING {n_missing}]"

        print(f"Path {subdir} has {len(files)}/{num_common} shards {status}")

        if n_missing > 0 and verbose:
            for m in sorted(missing):
                print(f"    {m}")
            errors = True

    if errors:
        print(f"\nFound {num_common} common shards across all dirs.")

    # count len audio
    audio_dir = Path(dataset) / "../source/audio"
    if audio_dir.exists():
        audio_shards = [f for f in audio_dir.iterdir() if f.suffix == ".wsds"]
        print(f"\nAudio dir {audio_dir.resolve()} has {len(audio_shards)} shards.")

    return [x.replace(".wsds", "") for x in common_shards]


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


def cast_types_for_storage(obj, float_cast="float32", int_cast="int32", debug=False):
    """Cast nested JSON-like objects to more resctrictive float and integer types.

    By default PyArrow would cast to float64 and int64, which are not optimal for storage.
    This function casts all numbers to float32/int32 by default.
    """
    if isinstance(float_cast, str):
        float_cast = getattr(np, float_cast)
    if isinstance(int_cast, str):
        int_cast = getattr(np, int_cast)

    def _cast(obj):
        if type(obj) is float:
            return float_cast(obj)
        elif type(obj) is int:
            return int_cast(obj)
        elif type(obj) is dict:
            return {k: _cast(v) for k, v in obj.items()}
        elif type(obj) is list:
            return [_cast(v) for v in obj]
        else:
            if debug:
                print("unknown type:", type(obj), obj, type(obj) is float, float)
            return obj

    return _cast(obj)


def parse_key_two_parts(key: str):
    """Parse a composite string key into the source file name, segmentation kind and sequential segment id.

    >>> parse_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_rawvad_00001254')
    ('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 'rawvad', 1254)
    """

    src_file, segmentation_kind, segment_id = key.rsplit("_", 2)
    return src_file, segmentation_kind, int(segment_id)
