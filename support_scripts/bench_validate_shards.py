#!/usr/bin/env python
#%%
"""bench_validate_shards.py — benchmark wsds shard validation strategies."""

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import wsds
from wsds.utils import preload_shard, validate_shards

# ---- config ----
DATASET_PATH = "/mnt/weka/data-wsds/data-ar/indices/source"   # <-- change me
MAX_SHARDS   = None            # None = all shards
N_RUNS       = 3               # timed runs per strategy
CHUNK_SIZE = 16  # shards per thread for thread_chunked / process_chunked
# hybrid auto-tunes (n_processes, n_threads) from workload size; see _choose_hybrid_params

# %%

ds = wsds.WSDataset(str(DATASET_PATH))
shard_list = ds.get_shard_list()
if MAX_SHARDS:
    shard_list = shard_list[:MAX_SHARDS]

col_dirs = sorted(
    {locs[0][0] for locs in ds.fields.values() if locs} - set(ds.computed_columns)
)

print(f"shards:   {len(shard_list):,}")
print(f"col_dirs: {len(col_dirs)}  {col_dirs[:5]}{'…' if len(col_dirs) > 5 else ''}")

# %%
def validate_stat_only(dataset, shards, column_dirs):
    """Main-thread os.path.exists() — no file I/O, no subprocess."""
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []
    return [
        (shard, all(os.path.exists(dataset.get_shard_path(cd, shard)) for cd in actual))
        for shard in shards
    ]


def validate_thread(dataset, shards, column_dirs, max_workers=64):
    """ThreadPoolExecutor variant — same magic-byte check, avoids fork overhead."""
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []
    files = [dataset.get_shard_path(cd, s) for s in shards for cd in actual]
    with ThreadPoolExecutor(max_workers=min(len(files), max_workers)) as ex:
        results = list(ex.map(preload_shard, files))
    n = len(actual)
    return [(s, all(results[i * n:(i + 1) * n])) for i, s in enumerate(shards)]


def validate_thread_chunked(dataset, shards, column_dirs, max_workers=64, chunk_size=CHUNK_SIZE):
    """ThreadPoolExecutor where each task processes a chunk of shards sequentially.

    Reduces per-task scheduling overhead vs submitting one file at a time.
    """
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []

    def check_chunk(chunk):
        return [
            (s, all(preload_shard(dataset.get_shard_path(cd, s)) for cd in actual))
            for s in chunk
        ]

    chunks = [shards[i:i + chunk_size] for i in range(0, len(shards), chunk_size)]
    with ThreadPoolExecutor(max_workers=min(len(chunks), max_workers)) as ex:
        chunk_results = list(ex.map(check_chunk, chunks))
    return [item for chunk in chunk_results for item in chunk]


def validate_thread_per_chunk(dataset, shards, column_dirs, chunk_size=32):
    """Single ThreadPoolExecutor with one thread per chunk of `chunk_size` shards.

    n_threads = ceil(len(shards) / chunk_size). Each thread takes exactly one
    chunk and processes it sequentially. No process pool — pure threads.
    """
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []

    shard_paths = [
        (s, [dataset.get_shard_path(cd, s) for cd in actual])
        for s in shards
    ]
    chunks = [shard_paths[i:i + chunk_size] for i in range(0, len(shard_paths), chunk_size)]

    def check_chunk(chunk):
        return [(shard, all(preload_shard(p) for p in paths)) for shard, paths in chunk]

    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        results = list(ex.map(check_chunk, chunks))
    return [item for chunk in results for item in chunk]


def _check_shard_chunk_threaded(args):
    """Top-level worker: split a process-chunk into per-thread sub-chunks.

    args = (shard_path_chunk, n_threads)
    shard_path_chunk = list of (shard_ref, [file_paths])
    Each thread receives a contiguous sub-chunk and validates it sequentially.
    """
    shard_path_chunk, n_threads = args
    sub_size = max(1, -(-len(shard_path_chunk) // n_threads))  # ceiling division
    sub_chunks = [shard_path_chunk[i:i + sub_size] for i in range(0, len(shard_path_chunk), sub_size)]

    def check_sub(sub):
        return [(shard, all(preload_shard(p) for p in paths)) for shard, paths in sub]

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(check_sub, sub_chunks))
    return [item for sub in results for item in sub]


def _in_debugger() -> bool:
    """True when running under a debugger (pdb, debugpy, etc.). Process pools
    don't play well with debuggers — child processes aren't traced and can
    hang on breakpoints — so the auto-tune avoids them in that case."""
    import sys
    return sys.gettrace() is not None or "debugpy" in sys.modules


def _choose_hybrid_params(n_ops, cpus=None, verbose=False):
    """Auto-tune (n_processes, n_threads) for hybrid validation.

    Heuristics, calibrated empirically:
    - in a debugger:    1 process, threads only (process pool breaks tracing)
    - n_ops <=  512:    1 process, few threads. Tiny workload — even threadpool setup matters.
    - n_ops <= 4096:    1 process, many threads. Process fork cost (~50–100ms each)
                        still dominates over the parallelism gain.
    - n_ops >  4096:    spread across cpu_count processes, 32 threads each.
    """
    cpus = cpus or os.cpu_count() or 8
    if _in_debugger():
        # Cap threads at 64 — that's already plenty for I/O-bound work and
        # avoids spawning hundreds of threads for huge workloads.
        n_threads = min(64, max(1, -(-n_ops // 32)))
        params = (1, n_threads)
        reason = "debugger"
    elif n_ops <= 512:
        n_threads = min(32, max(1, -(-n_ops // 16)))
        params = (1, n_threads)
        reason = "tiny"
    elif n_ops <= 4096:
        n_threads = min(128, max(8, -(-n_ops // 32)))
        params = (1, n_threads)
        reason = "medium"
    else:
        n_threads = 32
        desired_threads = max(cpus, -(-n_ops // 32))
        n_processes = min(cpus, max(2, -(-desired_threads // n_threads)))
        params = (n_processes, n_threads)
        reason = "large"
    if verbose:
        print(f"  auto-tune ({reason}): n_ops={n_ops} → n_processes={params[0]}, n_threads={params[1]}")
    return params


def validate_hybrid(dataset, shards, column_dirs,
                    n_processes=None, n_threads=None, verbose=False):
    """N processes × K threads. When `n_processes` / `n_threads` are None they
    are auto-picked by `_choose_hybrid_params` based on the workload size."""
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []

    shard_paths = [
        (s, [dataset.get_shard_path(cd, s) for cd in actual])
        for s in shards
    ]
    if n_processes is None or n_threads is None:
        auto_p, auto_t = _choose_hybrid_params(len(shards) * len(actual), verbose=verbose)
        if n_processes is None: n_processes = auto_p
        if n_threads   is None: n_threads   = auto_t

    # Fast path: skip the process pool when one process is enough
    if n_processes <= 1:
        return _check_shard_chunk_threaded((shard_paths, n_threads))

    chunk_size = max(1, -(-len(shard_paths) // n_processes))
    chunks = [shard_paths[i:i + chunk_size] for i in range(0, len(shard_paths), chunk_size)]
    with ProcessPoolExecutor(max_workers=n_processes) as ex:
        chunk_results = list(ex.map(_check_shard_chunk_threaded,
                                    [(c, n_threads) for c in chunks]))
    return [item for chunk in chunk_results for item in chunk]


_O_NOATIME = getattr(os, "O_NOATIME", 0)  # 0 on non-Linux


def _preload_shard_pread(shard_fname):
    """Raw-syscall variant of preload_shard: open + lseek + read + close.

    Same semantics (checks 6-byte ARROW1 magic at file end) but bypasses
    Python's BufferedReader. Adds O_NOATIME on Linux to skip the per-open
    atime write-back (a real cost on networked filesystems).
    """
    fname = str(shard_fname)
    try:
        fd = os.open(fname, os.O_RDONLY | _O_NOATIME)
    except PermissionError:
        # O_NOATIME requires file ownership; fall back without it.
        try:
            fd = os.open(fname, os.O_RDONLY)
        except OSError:
            return False
    except OSError:
        return False
    try:
        os.lseek(fd, -6, os.SEEK_END)
        return os.read(fd, 6) == b"ARROW1"
    except OSError:
        return False
    finally:
        os.close(fd)


def _check_shard_chunk_threaded_pread(args):
    """Same as _check_shard_chunk_threaded but uses _preload_shard_pread."""
    shard_path_chunk, n_threads = args
    sub_size = max(1, -(-len(shard_path_chunk) // n_threads))
    sub_chunks = [shard_path_chunk[i:i + sub_size] for i in range(0, len(shard_path_chunk), sub_size)]

    def check_sub(sub):
        return [(shard, all(_preload_shard_pread(p) for p in paths)) for shard, paths in sub]

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        results = list(ex.map(check_sub, sub_chunks))
    return [item for sub in results for item in sub]


def validate_hybrid_pread(dataset, shards, column_dirs,
                          n_processes=None, n_threads=None, verbose=False):
    """Hybrid (N processes × K threads) using raw os.open/lseek/read.
    Auto-tunes n_processes / n_threads from workload size when not given."""
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []

    shard_paths = [
        (s, [dataset.get_shard_path(cd, s) for cd in actual])
        for s in shards
    ]
    if n_processes is None or n_threads is None:
        auto_p, auto_t = _choose_hybrid_params(len(shards) * len(actual), verbose=verbose)
        if n_processes is None: n_processes = auto_p
        if n_threads   is None: n_threads   = auto_t

    if n_processes <= 1:
        return _check_shard_chunk_threaded_pread((shard_paths, n_threads))

    chunk_size = max(1, -(-len(shard_paths) // n_processes))
    chunks = [shard_paths[i:i + chunk_size] for i in range(0, len(shard_paths), chunk_size)]
    with ProcessPoolExecutor(max_workers=n_processes) as ex:
        chunk_results = list(ex.map(_check_shard_chunk_threaded_pread,
                                    [(c, n_threads) for c in chunks]))
    return [item for chunk in chunk_results for item in chunk]


def _check_shard_chunk(shard_path_chunk):
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Receives a list of (shard_ref, [file_paths]) and returns [(shard_ref, bool)].
    """
    return [(shard, all(preload_shard(p) for p in paths)) for shard, paths in shard_path_chunk]


def validate_process_chunked(dataset, shards, column_dirs, max_workers=64, chunk_size=CHUNK_SIZE):
    """ProcessPoolExecutor where each task validates a chunk of shards.

    Pre-computes file paths in the main process (dataset isn't picklable) then
    dispatches chunks to worker processes.
    """
    actual = [cd for cd in column_dirs if cd not in dataset.computed_columns]
    if not actual or not shards:
        return []

    shard_paths = [
        (s, [dataset.get_shard_path(cd, s) for cd in actual])
        for s in shards
    ]
    chunks = [shard_paths[i:i + chunk_size] for i in range(0, len(shard_paths), chunk_size)]
    with ProcessPoolExecutor(max_workers=min(len(chunks), max_workers)) as ex:
        chunk_results = list(ex.map(_check_shard_chunk, chunks))
    return [item for chunk in chunk_results for item in chunk]


strategies = {
    "original":        lambda sl: validate_shards(ds, sl, col_dirs),
    "thread":          lambda sl: validate_thread(ds, sl, col_dirs),
    "thread_chunked":  lambda sl: validate_thread_chunked(ds, sl, col_dirs),
    "process_chunked": lambda sl: validate_process_chunked(ds, sl, col_dirs),
    "process_chunk64": lambda sl: validate_process_chunked(ds, sl, col_dirs, chunk_size=64),
    "stat_only":        lambda sl: validate_stat_only(ds, sl, col_dirs),
    "thread_per_chunk": lambda sl: validate_thread_per_chunk(ds, sl, col_dirs),
    "hybrid":           lambda sl: validate_hybrid(ds, sl, col_dirs, verbose=True),
    "hybrid_pread":     lambda sl: validate_hybrid_pread(ds, sl, col_dirs, verbose=True),
}

# %%
SHARD_LEVELS = [None, 300, 100, 10]   # None = all shards

results_by_level = {}
times_by_level   = {}

for level in SHARD_LEVELS:
    sl = shard_list if level is None else shard_list[:min(level, len(shard_list))]
    if not sl:
        continue
    print(f"\n=== {len(sl)} shards ===")
    results = {}
    times   = {}
    for name, fn in strategies.items():
        run_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            out = fn(sl)
            run_times.append(time.perf_counter() - t0)
        results[name] = dict(out)
        times[name]   = run_times
        n_ok   = sum(1 for ok in results[name].values() if ok)
        n_bad  = len(results[name]) - n_ok
        best_ms = min(run_times) * 1000
        print(f"  {name:16s}  best={best_ms:.0f}ms  shards={len(results[name])} ({n_ok} ok, {n_bad} missing)  runs={[f'{t*1000:.0f}ms' for t in run_times]}")
    results_by_level[len(sl)] = results
    times_by_level[len(sl)]   = times

# %%
# Verify all strategies agree on which shards are valid (per level)
for n, results in results_by_level.items():
    ref_name = next(iter(results))
    ref = results[ref_name]
    print(f"\n=== {n} shards: vs {ref_name} ===")
    for name, res in results.items():
        match = res == ref
        print(f"  {name:16s}: {'✓ match' if match else '✗ MISMATCH'}")

# %%
# Summary: speedup vs first strategy (per level)
for n, times in times_by_level.items():
    base_name = next(iter(times))
    base = min(times[base_name])
    print(f"\n=== {n} shards: vs {base_name} ===")
    for name, ts in times.items():
        best = min(ts)
        ratio = base / best if best > 0 else float("inf")
        print(f"  {name:16s}  best={best*1000:.0f}ms  ({ratio:.1f}x vs {base_name})")

# %%
