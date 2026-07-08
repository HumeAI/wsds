"""Full-pipeline test for meta datasets + persisted row subsets:

    meta over 5 FULL language datasets
      -> .sample(2%, by="row")         # random 2% of ALL rows -> new meta (offset index)
      -> .save(dir)                    # persist: meta-view.json + selection.feather
      -> WSMetaDataset.load(dir)       # re-instantiate from disk (fingerprint-checked)
      -> query / index / iterate it    # sql_select, len, sub[i], sub[key] — all timed

The subset references the parents directly (parent, partition, shard, offset);
nothing is materialized or copied — what's saved is only the pointer table.

Run cell-by-cell on the host where the data is mounted (iren).
"""

# %%
import shutil
import time
from pathlib import Path

import polars as pl

from wsds import WSDataset, WSMetaDataset

ROOT = "/mnt/weka/data-wsds/tigran_data_sync_dirs"
LANGS = ["data-en", "data-de", "data-es", "data-it", "data-fr"]  # 5 languages
KIND = "source"          # or "filtered_vad"
FRACTION = 0.02          # 2%
SEED = 0
SAVE_DIR = Path.home() / "wsds_subsets" / f"{KIND}_{int(FRACTION * 100)}pct_seed{SEED}"


class timed:
    """with timed("label"): ...  -> prints wall time of the block."""
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0
        print(f"[{self.dt:8.2f}s] {self.label}")


# %%
# --- 1) meta over the 5 FULL datasets --------------------------------------
with timed("open meta over 5 full parents"):
    meta = WSMetaDataset(
        [WSDataset(f"{ROOT}/{lang}/indices/{KIND}") for lang in LANGS],
        names=LANGS, kind=KIND,
    )
print(meta)

# %%
# --- 2) random 2% of ALL rows -> a NEW meta (direct offset index) ----------
# by="row": uniform per-row via hash(__key__); resolving it costs one parallel
# scan of the parents (the one-time price of a row-level subset — after this,
# access is pure pointer lookups).
with timed(f"build {FRACTION:.0%} row-sample (scan all 5 parents, parallel)"):
    sub = meta.sample(FRACTION, by="row", seed=SEED)
print(sub)
print(f"len(meta)={len(meta):,}   len(sub)={len(sub):,}   (~{len(sub) / len(meta) * 100:.2f}%)")
print(sub.stats())

# %%
# --- 3) persist the subset, then re-instantiate it from disk ---------------
shutil.rmtree(SAVE_DIR, ignore_errors=True)
with timed(f"save -> {SAVE_DIR}"):
    sub.save(SAVE_DIR)
for f in sorted(SAVE_DIR.iterdir()):
    print(f"   {f.name}: {f.stat().st_size / 1e6:.1f} MB")

with timed("load back from disk (reopens parents, fingerprint-checked)"):
    sub = WSMetaDataset.load(SAVE_DIR)
print(sub)

# the loaded subset must be pointer-identical to what we built
sub_fresh = meta.sample(FRACTION, by="row", seed=SEED)
assert len(sub) == len(sub_fresh), "loaded subset size differs from freshly built one"
assert sub._sel.select("parent", "offset", "__key__").equals(
    sub_fresh._sel.select("parent", "offset", "__key__")), "loaded selection differs"
print("loaded selection is pointer-identical to the freshly built one")

# %%
# --- 4) query the LOADED subset (timed) -------------------------------------
def is_queryable(ds, c):
    locs = ds.fields.get(c)
    return bool(locs) and locs[0][0] not in ds.computed_columns


common = set.intersection(*(set(ds.fields) for ds in sub.children))
col = next(c for c in ["est_duration", "duration", "quality_score", "acoustic_noise_score", "snr"]
           if c in common and all(is_queryable(ds, c) for ds in sub.children))
print("querying common column:", col)

# run twice: first pass is cold (page cache), second shows steady-state speed
for attempt in ("cold", "warm"):
    with timed(f"sql_select(__key__, {col}, with_dataset_col) over the 2% subset [{attempt}]"):
        df = sub.sql_select("`__key__`", f"`{col}`", with_dataset_col=True)
print("rows returned:", df.height, " (== len(sub) ==", len(sub), ")")
print(df.head(5))

with timed(f"aggregate: per-language count + mean({col}) on the subset"):
    agg = (sub.sql_select(f"`{col}`", with_dataset_col=True)
           .group_by("__dataset__").agg(count=pl.len(), mean=pl.mean(col))
           .sort("count", descending=True))
print(agg)

assert df.height == len(sub), "query row-count must equal len(sub)"
assert set(df["__dataset__"].unique()) == set(LANGS), "every language should contribute"

# reference point: the query-time knob on the FULL meta (~same data volume,
# no subset object) — how the pointer-restricted scan compares to plain scanning
with timed(f"reference: meta.sql_select(..., shard_subsample={FRACTION}) on the full meta"):
    ref = meta.sql_select("`__key__`", f"`{col}`", shard_subsample=FRACTION)
print("reference rows:", ref.height)

# %%
# --- 5) random access / key lookup / iteration on the loaded subset --------
with timed("1000 random point accesses sub[i] (pointer deref, no scan)"):
    for i in range(0, len(sub), max(1, len(sub) // 1000)):
        _ = sub[i]

some_key = sub._sel["__key__"][len(sub) // 2]
with timed("first key lookup sub[key] (builds sorted keymap once)"):
    s = sub[some_key]
assert s is not None and s["__key__"] == some_key
with timed("1000 warm key lookups"):
    for _ in range(1000):
        _ = sub[some_key]

with timed("iterate first 200 samples (reads sample fields from shards)"):
    for _, s in zip(range(200), sub):
        _ = s["__key__"]

for i in [0, len(sub) // 2, len(sub) - 1]:
    s = sub[i]
    print(f"sub[{i}]: lang={s.dataset.dataset_root.parent.parent.name}  key={s['__key__']!r}  {col}={s[col]}")

# %%
# --- 6) the subset composes further: chain a filter on top of the 2% -------
with timed(f"chained refilter on the subset: {col} > median"):
    med = float(df[col].median())
    sub_hi = sub.filter(f"`{col}` > {med}")
print(f"chained subset: {len(sub_hi):,} of {len(sub):,} rows (~{len(sub_hi) / len(sub) * 100:.1f}%)")
assert 0 < len(sub_hi) < len(sub)

print("\nALL META-SUBSET PIPELINE TESTS PASSED")
# %%
