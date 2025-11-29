from dataclasses import dataclass
import re
from pathlib import Path

import numpy as np
import pyarrow as pa


@dataclass
class WSShardMissingError(Exception):
    fname: str


def get_columns(fname):
    if isinstance(fname, Path):
        fname = str(fname)
    reader = pa.RecordBatchFileReader(pa.memory_map(fname))
    return reader.schema.names


def find_first_shard(path):
    for file in Path(path).iterdir():
        if file.suffix == ".wsds":
            return file
    return None


def list_all_columns(ds_path, shard_name=None, include_in_progress=True):
    """Given a dataset path, return a list of all columns.

    If you also give a shard name it greatly speeds it up
    on network-filesystems where listing folder contents is slow."""
    dupes = {}
    cols = {}
    key_col = []
    for p in Path(ds_path).iterdir():
        if p.suffix == ".wsds-link":
            col = p.with_suffix("").name
            cols[col] = (p.name, col)
            continue
        if not p.is_dir():
            continue
        is_in_progress = p.suffix == ".in-progress"
        if is_in_progress and not include_in_progress:
            continue
        if shard_name is None or is_in_progress:
            fname = find_first_shard(p)
        else:
            fname = (p / shard_name).with_suffix(".wsds")
        if fname and fname.exists():
            for col in get_columns(fname):
                if col == "__key__":
                    if not is_in_progress:
                        # We need a subdir that has all shards but we don't wanna list all of them (that's expensive)
                        # so instead we rely on a subdir naming convention (the .in-progress suffix) and never use these
                        key_col.append((fname.stat().st_size, p.name, col))
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
    # use the smallest shards for __key__ (should be the fastest)
    if len(key_col) > 0:
        cols["__key__"] = sorted(key_col)[0][1:]
    return dict(sorted(cols.items()))


def list_all_shards(dataset: str, verbose: bool = False, print_missing: bool = False):
    shards = {}
    for subdir in Path(dataset).iterdir():
        if not subdir.is_dir():
            continue
        shards[subdir] = {file.name for file in subdir.iterdir() if file.suffix == ".wsds"}
        if not shards[subdir]:
            if verbose:
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

        if verbose:
            print(f"Path {subdir} has {len(files)}/{num_common} shards {status}")

        if n_missing > 0 and print_missing:
            for m in sorted(missing):
                print(f"    {m}")
            errors = True

    if errors:
        print(f"\nFound {num_common} common shards across all dirs.")

    # count len audio
    audio_dir = Path(dataset) / "../source/audio"
    if audio_dir.exists():
        audio_shards = [f for f in audio_dir.iterdir() if f.suffix == ".wsds"]
        if verbose:
            print(f"\nAudio dir {audio_dir.resolve()} has {len(audio_shards)} shards.")

    return [("", x.replace(".wsds", "")) for x in common_shards]


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


magic_check = re.compile("([*?[])")
magic_check_bytes = re.compile(b"([*?[])")


def has_magic(s):
    """Does the given input contain any shell globbing characters?"""
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None


def scan_ipc(path: str | Path, *args, glob=True, **kwargs):
    """Like pl.scan_ipc but with a workaround to disable globbing.

    See also: https://github.com/pola-rs/polars/issues/24608"""
    import polars as pl

    path = str(path)
    if glob or not has_magic(path):
        return pl.scan_ipc(path, *args, **kwargs)
    else:
        # we open the file manually since scan_ipc always does globbing which does not work on files with square brackets in their names
        f = open(path, "rb")
        # we will leak the file descriptor in this case but there is not a lot we can do about it (GC will close it eventually)
        return pl.scan_ipc(f)

def format_duration(duration):
    """Formats a duration in seconds as hours (or minutes or kilo-hours)."""
    hours = duration / 3600
    if hours > 1000:
        return f"{hours / 1000:.2f} k hours"
    elif hours < 1:
        return f"{duration / 60:.1f} minutes"
    else:
        return f"{hours:.2f} hours"
