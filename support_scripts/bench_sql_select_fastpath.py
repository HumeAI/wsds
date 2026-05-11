#!/usr/bin/env python
#%%
"""bench_sql_select_fastpath.py — fast vs slow path timing for sql_select().

`_parse_sql_queries_polars` has two code paths:

- fast path: one `pl.scan_ipc(paths_list)` per col_dir, plan = ~N_col_dirs nodes.
- slow path: one `scan_ipc` per (shard, col_dir), plan = N_shards × N_col_dirs.

The slow path is selected when the caller asks for `__shard_offset__`, passes
a `shard_pipe`, or any shard is invalid in some col_dir.

This script just times both, no profiler. N_RUNS each, best-of reported.
"""

import gc
import time

import wsds

# ---- config ----
DATASET_PATH = "/mnt/weka/data-wsds/data-ar/indices/source"   # <-- change me
N_RUNS = 3

QUERIES_FAST = (
    "__key__",
    "__shard_path__",
    "load_duration AS audio_duration",
    "duration",
    "duration_seconds",
    "est_duration",
    "inspected_duration",
    "speech_duration",
)
QUERIES_SLOW = QUERIES_FAST + ("__shard_offset__ AS offset",)

# %%
ds = wsds.WSDataset(str(DATASET_PATH))
print(f"dataset: {DATASET_PATH}")
print(f"shards:  {len(ds.get_shard_list()):,}\n")


def reset():
    if hasattr(ds, "_validated_shards"):
        ds._validated_shards.clear()
    gc.collect()


def time_runs(label, queries):
    runs = []
    rows = None
    for _ in range(N_RUNS):
        reset()
        t0 = time.perf_counter()
        df = ds.sql_select(*queries, shard_subsample=1)
        runs.append((time.perf_counter() - t0) * 1000)
        rows = len(df)
    best = min(runs)
    print(f"  {label:30s} best={best:>7,.0f} ms   rows={rows:,}   runs={[f'{r:.0f}ms' for r in runs]}")
    return best


# %%
fast_ms = time_runs("FAST (multi-file scan)", QUERIES_FAST)
slow_ms = time_runs("SLOW (per-shard scan)",  QUERIES_SLOW)

print()
print(f"  speedup: {slow_ms / fast_ms:.2f}x")

# %%
