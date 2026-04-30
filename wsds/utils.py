import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


@dataclass
class WSShardMissingError(Exception):
    fname: str

    @classmethod
    def from_s3(cls, s3_client, key, bucket, err):
        return cls(f"{s3_client._endpoint}://{bucket}/{key} [error: {err}]")


@dataclass
class WSShardCorruptedError(Exception):
    fname: str


def get_columns(fname):
    if isinstance(fname, Path):
        fname = str(fname)
    try:
        reader = pa.RecordBatchFileReader(pa.memory_map(fname))
    except pa.ArrowInvalid:
        raise WSShardCorruptedError(fname)
    return reader.schema.names


def find_first_shard(path):
    for file in Path(path).iterdir():
        if file.suffix == ".wsds":
            return file
    return None


def list_all_columns(ds_path, shard_name=None):
    """Given a dataset path, return a list of all columns.

    If you also give a shard name it greatly speeds it up
    on network-filesystems where listing folder contents is slow."""
    dupes = {}
    cols = {}
    key_col = []
    for p in Path(ds_path).iterdir():
        if p.suffix == ".wsds-link":
            col = p.with_suffix("").name
            cols[col] = [(p.name, col)]
            continue
        if not p.is_dir():
            continue
        if shard_name is None:
            fname = find_first_shard(p)
        else:
            fname = (p / shard_name).with_suffix(".wsds")
            if not fname.exists():
                fname = find_first_shard(p)
        if fname and fname.exists():
            try:
                columns = get_columns(fname)
            except WSShardCorruptedError as err:
                print("Got an error listing columns:", repr(err))
                continue
            for col in columns:
                if col == "__key__":
                    # List all potential __key__ columns (they should be in each shard)
                    key_col.append((fname.stat().st_size, p.name, col))
                    continue
                # seems like we should fix this during the original conversion
                if col in cols or col in dupes:
                    dupes[col] = True
                    if col in cols:
                        (dirname, _colname) = cols[col][0]
                        cols[f"{dirname}.{col}"] = [(dirname, col)]
                        del cols[col]
                    cols[f"{p.name}.{col}"] = [(p.name, col)]
                else:
                    cols[col] = [(p.name, col)]
    # use the smallest shards for __key__ (should be the fastest)
    if key_col:
        cols["__key__"] = [x[1:] for x in sorted(key_col)]
    return dict(sorted(cols.items()))


def list_all_shards(dataset: str, verbose: bool = False, print_missing: bool = False):
    shards = {}
    for column_dir in Path(dataset).iterdir():
        if not column_dir.is_dir():
            continue
        shards[column_dir] = {file.name for file in column_dir.iterdir() if file.suffix == ".wsds"}
        if not shards[column_dir]:
            if verbose:
                print(f"error: empty folder {column_dir}")
            del shards[column_dir]

    common_shards = {v for shard_values in shards.values() for v in shard_values}
    num_common = len(common_shards)

    errors = False
    for column_dir, files in shards.items():
        missing = common_shards - files
        n_missing = len(missing)
        if n_missing == 0:
            status = "[COMPLETE]"
        else:
            status = f"[MISSING {n_missing}]"

        if verbose:
            print(f"Path {column_dir} has {len(files)}/{num_common} shards {status}")

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
        return pl.scan_ipc(f, *args, **kwargs)


def is_notebook() -> bool:
    """Detect if running in a Jupyter notebook vs terminal IPython or plain Python."""
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def format_duration(duration):
    """Formats a duration in seconds as hours (or minutes or kilo-hours)."""
    hours = duration / 3600
    if hours > 1000000:
        return f"{hours / 1000000:.2f} M hours"
    elif hours > 1000:
        return f"{hours / 1000:.2f} k hours"
    elif hours < 1:
        return f"{duration / 60:.1f} minutes"
    else:
        return f"{hours:.2f} hours"


_O_NOATIME = getattr(os, "O_NOATIME", 0)  # 0 on non-Linux platforms


def preload_shard(shard_fname):
    """Open a shard file and verify the trailing 6-byte ARROW1 magic.

    Uses raw os.open/lseek/read/close (bypassing Python's BufferedReader) and
    O_NOATIME on Linux (skips a per-open atime write back to the FS server).
    Falls back to plain O_RDONLY on EPERM (file not owned by us).
    """
    fname = str(shard_fname)
    try:
        try:
            fd = os.open(fname, os.O_RDONLY | _O_NOATIME)
        except PermissionError:
            fd = os.open(fname, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError as err:
        traceback.print_exc()
        print(f"OSError while loading {fname}: {err}")
        return False
    try:
        os.lseek(fd, -6, os.SEEK_END)
        if os.read(fd, 6) != b"ARROW1":
            print(f"Invalid file format {fname}: ARROW1 magic not found")
            return False
    except OSError as err:
        traceback.print_exc()
        print(f"OSError while loading {fname}: {err}")
        return False
    finally:
        os.close(fd)
    return True


def _in_debugger() -> bool:
    """True under pdb/debugpy. Used to skip ProcessPoolExecutor — child
    processes aren't traced and can hang on breakpoints."""
    return sys.gettrace() is not None or "debugpy" in sys.modules


def _choose_validation_params(n_ops: int, cpus: int | None = None) -> tuple[int, int]:
    """Pick (n_processes, n_threads) for `validate_shards` based on workload size.

    Calibrated empirically on Weka with ~50µs/op effective latency:
    - debugger:        1 process, threads only (process pool breaks tracing)
    - n_ops <=  512:   1 process, few threads. Threadpool setup itself matters here.
    - n_ops <= 4096:   1 process, many threads. Process fork (~50–100ms each)
                       still dominates over the parallelism gain.
    - n_ops >  4096:   spread across cpu_count processes, 32 threads each.
    """
    cpus = cpus or os.cpu_count() or 8
    if _in_debugger():
        return 1, min(64, max(1, -(-n_ops // 32)))
    if n_ops <= 512:
        return 1, min(32, max(1, -(-n_ops // 16)))
    if n_ops <= 4096:
        return 1, min(128, max(8, -(-n_ops // 32)))
    n_threads = 32
    desired_threads = max(cpus, -(-n_ops // 32))
    n_processes = min(cpus, max(2, -(-desired_threads // n_threads)))
    return n_processes, n_threads


def _validate_shard_chunk(args):
    """Top-level worker (must be picklable for ProcessPoolExecutor).

    Receives (shard_path_chunk, n_threads). Splits the chunk into per-thread
    sub-chunks and runs `preload_shard` on each file sequentially within a
    thread.
    """
    shard_path_chunk, n_threads = args
    sub_size = max(1, -(-len(shard_path_chunk) // n_threads))
    sub_chunks = [shard_path_chunk[i:i + sub_size] for i in range(0, len(shard_path_chunk), sub_size)]

    def check_sub(sub):
        return [(shard, all(preload_shard(p) for p in paths)) for shard, paths in sub]

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(check_sub, sub_chunks))
    return [item for sub in results for item in sub]


def validate_shards(
    dataset: "WSDataset",
    shards: list[tuple[str, str]],
    column_dirs: list[str],
):
    """Prefetch and validate shard files for the given shards and column dirs.

    Verifies each shard's ARROW1 magic across every column dir. Auto-tunes
    its concurrency (single-process threads for small workloads, process pool
    for large ones) via `_choose_validation_params`.

    Already-validated shards on `dataset` (tracked in
    `dataset._validated_shards`) are skipped and returned with `ok=True`.

    Returns a list of `(shard_ref, ok)` for every input shard.
    """
    actual_column_dirs = [c for c in column_dirs if c not in dataset.computed_columns]
    if not actual_column_dirs or not shards:
        return [(s, True) for s in shards]

    cache_key = tuple(sorted(actual_column_dirs))
    cache: set = dataset._validated_shards.setdefault(cache_key, set())
    pending = [s for s in shards if s not in cache]
    if not pending:
        return [(s, True) for s in shards]

    shard_paths = [
        (s, [dataset.get_shard_path(cd, s) for cd in actual_column_dirs])
        for s in pending
    ]
    n_processes, n_threads = _choose_validation_params(len(shard_paths) * len(actual_column_dirs))

    if n_processes <= 1:
        verified = _validate_shard_chunk((shard_paths, n_threads))
    else:
        chunk_size = max(1, -(-len(shard_paths) // n_processes))
        chunks = [shard_paths[i:i + chunk_size] for i in range(0, len(shard_paths), chunk_size)]
        with ProcessPoolExecutor(max_workers=n_processes) as ex:
            chunk_results = list(ex.map(_validate_shard_chunk, [(c, n_threads) for c in chunks]))
        verified = [item for chunk in chunk_results for item in chunk]

    verified_map = dict(verified)
    cache.update(s for s, ok in verified if ok)
    return [(s, True) if s in cache else (s, verified_map.get(s, False)) for s in shards]
