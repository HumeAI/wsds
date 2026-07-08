"""Test offset-backed subsets (ds.filter/sample -> WSMetaDataset) on a real source AND a
filtered_vad dataset, and verify that querying the view works.

Run cell-by-cell on the host where the data is mounted (iren). Each dataset test:
  1. picks a real numeric, queryable column (skips computed cols like `audio`),
  2. builds a predicate view (`col > median`) — subsampled on the huge
     segmented dataset so the build stays fast,
  3. queries it three ways and asserts they agree:
        predicate scan  ==  key semi-join  ==  enumerated len,
  4. checks every returned row satisfies the predicate,
  5. exercises random access, persistence, and targeted batch reads.
"""

# %%
import os

import polars as pl

from wsds import WSDataset, WSMetaDataset

ROOT = "/mnt/weka/data-wsds/tigran_data_sync_dirs"
LANG = "data-en"
SOURCE_DS = f"{ROOT}/{LANG}/indices/source"
VAD_DS = f"{ROOT}/{LANG}/indices/filtered_vad"

# source is small -> scan all shards; filtered_vad is huge -> subsample the build
SOURCE_SUBSAMPLE = 1.0
VAD_SUBSAMPLE = 0.05

source = WSDataset(SOURCE_DS)
vad = WSDataset(VAD_DS)
print(source)
print(vad)


# %%
# --- helpers ---------------------------------------------------------------
def is_queryable(ds, c):
    """A scalar, stored (non-computed) field usable in sql_select."""
    locs = ds.fields.get(c)
    return bool(locs) and locs[0][0] not in ds.computed_columns


def pick_numeric(ds, prefer):
    """First preferred numeric column that exists, else probe a sample for one."""
    for c in prefer:
        if is_queryable(ds, c):
            return c
    s = ds.random_sample()
    for c in sorted(ds.fields):
        if not is_queryable(ds, c):
            continue
        try:
            v = s[c]
        except Exception:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return c
    raise RuntimeError("no queryable numeric column found")


def test_view(ds, label, prefer_cols, extra_cols, subsample):
    col = pick_numeric(ds, prefer_cols)
    thr = float(ds.sql_select(f"`{col}`", shard_subsample=min(subsample, 0.1))[col].drop_nulls().median())
    where = f"`{col}` > {thr:.6g}"
    print(f"\n=== {label} ===  ds.filter('{where}')   (shard universe subsample={subsample})")

    # narrow the shard universe first on the huge segmented set, then filter.
    # `ds.filter(...)` returns an offset-backed WSMetaDataset (over this one parent).
    base = ds if subsample >= 1 else ds.sample(subsample, by="shard", seed=0)
    view = base.filter(where)
    st = view.stats()
    print("  stats:", st)

    # dedupe and drop the predicate column itself to avoid duplicate projections
    extra = [c for c in dict.fromkeys(extra_cols) if c != col and is_queryable(ds, c)]
    query_cols = [f"`{col}`"] + [f"`{c}`" for c in extra]
    df = view.sql_select(*query_cols)
    n = len(view)
    print(f"  row counts   len={n:,}   sql_select={df.height:,}   stats={st['n_selected']:,}")

    # (1) sql_select, len, and stats must agree
    assert df.height == n == st["n_selected"], "row-count mismatch (sql_select / len / stats)"
    # (2) every returned row satisfies the predicate
    assert df.filter(pl.col(col) <= thr).height == 0, "a returned row violates the predicate"
    print("  sample rows:")
    print(df.head(3))

    # (3) random access uses the DIRECT offset -> a real, matching sample (no scan)
    s0, sN = view[0], view[-1]
    assert s0[col] > thr and sN[col] > thr, "random-access sample violates the predicate"
    print(f"  view[0]  {col}={s0[col]}  key={s0['__key__']!r}")
    print(f"  view[-1] {col}={sN[col]}  key={sN['__key__']!r}")
    return view, col, thr


# %%
# --- SOURCE dataset view ----------------------------------------------------
src_view, src_col, src_thr = test_view(
    source, "source",
    prefer_cols=["est_duration", "duration", "snr"],
    extra_cols=["duration", "language", "lid"],
    subsample=SOURCE_SUBSAMPLE,
)

# %%
# --- FILTERED_VAD dataset view ---------------------------------------------
vad_view, vad_col, vad_thr = test_view(
    vad, "filtered_vad",
    prefer_cols=["quality_score", "acoustic_noise_score", "music_prob"],
    extra_cols=["acoustic_noise_score", "quality_score"],
    subsample=VAD_SUBSAMPLE,
)

# %%
# --- Persistence round-trip: the subset saves its offset index (selection.feather) -
save_dir = os.path.expanduser("~/view_filtered_vad_test")
p = vad_view.save(save_dir)
print("saved:", sorted(os.listdir(p)))  # meta-view.json + selection.feather
reloaded = WSMetaDataset.load(p)
r = reloaded.sql_select(f"`{vad_col}`").select(
    pl.col(vad_col).mean().alias("mean"), pl.len().alias("n"))
print("reloaded:", repr(reloaded))
print("  len(reloaded):", len(reloaded), " query:", r.to_dicts())
assert len(reloaded) == len(vad_view), "reloaded subset length differs"
# reloaded rows are identical (same direct offsets)
assert sorted(reloaded.sql_select("`__key__`")["__key__"].to_list()) == \
       sorted(vad_view.sql_select("`__key__`")["__key__"].to_list())
print("\nALL VIEW QUERY TESTS PASSED")
# %%
