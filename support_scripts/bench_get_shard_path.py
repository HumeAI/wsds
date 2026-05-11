#!/usr/bin/env python
#%%
"""bench_get_shard_path.py — micro-benchmark for WSDataset.get_shard_path().

cProfile flagged get_shard_path (~3s cumulative on a 35k-call sql_select) as
a top contributor — but cProfile's instrumentation inflates Python-heavy hot
paths. This script measures the real cost and the speedup from replacing the
pathlib chain with f-string concatenation.

get_shard_path is called twice in a typical sql_select (once in
`validate_shards`, once in `_parse_sql_queries_polars`), so the sql_select-
level savings are roughly 2× the per-pass numbers reported here.
"""

import gc
import time

import polars as pl

import wsds

DATASET_PATH = "/mnt/weka/data-wsds/data-ar/indices/source"  # <-- change me
N_RUNS = 3
QUERIES = ("load_duration", "duration", "duration_seconds",
           "est_duration", "inspected_duration", "speech_duration")

# %%
ds = wsds.WSDataset(str(DATASET_PATH))
shards = ds.get_shard_list()
col_dirs = sorted({ds.fields[f][0][0] for q in QUERIES
                   for f in pl.sql_expr(q).meta.root_names()
                   if not f.startswith("__") and ds.fields[f][0][0] not in ds.computed_columns})
n_calls = len(shards) * len(col_dirs)
print(f"{len(shards):,} shards × {len(col_dirs)} col_dirs = {n_calls:,} calls per pass\n")


# %%
_root = str(ds.dataset_root)

def fn_pathlib(cd, s):
    return ds.get_shard_path(cd, s)

def fn_strcat(cd, s):
    # Matches the shipped implementation in WSDataset.get_shard_path.
    partition, name = s
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return f"{_root}/{partition}/{cd}/{name}.wsds"


def best_ms(fn):
    best = float("inf")
    out = None
    for _ in range(N_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        out = [fn(cd, s) for s in shards for cd in col_dirs]
        best = min(best, time.perf_counter() - t0)
    return best * 1000, out


# %%
pathlib_ms, p_paths = best_ms(fn_pathlib)
strcat_ms,  s_paths = best_ms(fn_strcat)
match = [str(p) for p in p_paths[:200]] == [str(p) for p in s_paths[:200]]

print(f"  pathlib (old impl):    {pathlib_ms:>7,.1f} ms")
print(f"  str-only (SHIPPED):    {strcat_ms:>7,.1f} ms   ({pathlib_ms / strcat_ms:.1f}x faster)")
print(f"  output match:          {'✓' if match else '✗ MISMATCH'}")
print(f"\n  estimated sql_select savings (2 passes): ≈ {2 * (pathlib_ms - strcat_ms):,.0f} ms"
      f"  ({2 * pathlib_ms:,.0f} → {2 * strcat_ms:,.0f} ms)")

# %%
