#%%
import os
from wsds import WSMetaDataset  
KIND = "source"  # try "filtered_vad" for the segmented view (billions of rows)

manifest_path = os.path.expanduser(f"~/multilang_{KIND}.wsds-meta")

meta = WSMetaDataset.from_manifest(manifest_path)
print(type(meta).__name__)
print(meta)
print("total samples:", len(meta))
print("union field count:", len(meta.fields))
# %%
# Reference: the BigQuery we want to reproduce over the wsds meta dataset.
#   per-language mean/std  ->  z-score every ep_*/vp_* column  ->  row-wise GREATEST
#
#   WITH lang_stats AS (SELECT language, AVG(ep_anger), STDDEV(ep_anger), ...
#                       FROM segment_filtered_v2 GROUP BY language),
#   zscored AS (SELECT r.*, (r.ep_anger - s.mean_ep_anger)/NULLIF(s.std_ep_anger,0) AS z_ep_anger, ...
#               FROM segment_filtered_v2 r JOIN lang_stats s USING(language))
#   SELECT *, GREATEST(z_ep_*) AS z_ep_max, GREATEST(z_vp_*) AS z_vp_max FROM zscored
#
# FEASIBILITY: yes. In the meta dataset each CHILD IS ONE LANGUAGE, so
# `GROUP BY language` collapses to "per child". Each child computes its own
# per-column mean/std and z-scores its own rows in ONE lazy pass (polars
# broadcasts `pl.col(c).mean()`/`.std()` across the child frame); the cross-
# language self-JOIN in the SQL disappears entirely. The multilingual result is
# just the vertical concat of the per-language frames. `language` is the child
# name; `source_id` falls out of the segment key. Scores are over the SEGMENTED
# (filtered_vad) datasets, not `source`.
import polars as pl  # noqa: E402

SEG_KIND = "filtered_vad"
seg_manifest = os.path.expanduser(f"~/multilang_{SEG_KIND}.wsds-meta")
assert os.path.exists(seg_manifest), (
    "Build the segmented manifest first, e.g.:\n  wsds make_meta "
    f"/mnt/weka/data-wsds/tigran_data_sync_dirs/data-*/indices/{SEG_KIND} "
    f"--out {seg_manifest} --kind {SEG_KIND}\n(remove the -ext children as needed)"
)
seg = WSMetaDataset.from_manifest(seg_manifest)
print(seg)

# 1) INVESTIGATE: the BQ names (ep_*/vp_*) don't exist as wsds columns — the BQ
#    table was flattened from a packed wsds column. Find where the scores live.
common = set.intersection(*(set(ds.fields) for ds in seg.children))
union = set().union(*(set(ds.fields) for ds in seg.children))
print(f"fields: {len(common)} common to all {len(seg.children)} children, {len(union)} in union")

# (a) search by emotion CONTENT, not prefix — tokens straight from the BQ query.
TOKENS = ["anger", "boredom", "yelling", "abrasive", "articulate", "tiredness",
          "calm", "admiration", "amusement", "anxiety", "prosody", "emotion"]
named_hits = sorted(c for c in union if any(t in c.lower() for t in TOKENS))
print(f"\ncolumns whose NAME contains an emotion token ({len(named_hits)}):")
print(named_hits[:40])

# (b) probe a real sample: which fields are dicts/arrays that could PACK the scores?
s = seg.random_sample()
print("\nstructured fields in one sample (dict keys / array shapes):")
suspects = ["speaker_outputs", "audio_phonemes"] + sorted(
    c for c in common if c.endswith((".npy", ".json")) or "output" in c or "score" in c
)
for c in dict.fromkeys(suspects):  # dedup, keep order
    if c not in s.dataset.fields:
        continue
    try:
        v = s[c]
    except Exception as e:
        print(f"  {c}: <err {type(e).__name__}>")
        continue
    if isinstance(v, dict):
        print(f"  {c}: dict keys={list(v.keys())[:16]}")
    elif isinstance(v, (list, tuple)):
        head = v[0] if v else None
        print(f"  {c}: {type(v).__name__}[{len(v)}] first={type(head).__name__} "
              f"{list(head.keys())[:16] if isinstance(head, dict) else ''}")
    elif hasattr(v, "shape"):
        print(f"  {c}: array shape={v.shape} dtype={getattr(v, 'dtype', None)}")
    else:
        print(f"  {c}: {type(v).__name__} = {str(v)[:60]}")

# Once we know the real location, set these so the transform below can run.
# (e.g. if scores are scalar columns, list them here; if packed, we'll unpack.)
ep_cols, vp_cols = [], []
score_cols = ep_cols + vp_cols

# 3) The transform for ONE language (== one child), fused into a single pass.
def zscore_child(ds, subsample):
    cols_q = ["`__key__`"] + [f"`{c}`" for c in score_cols]
    lf = ds.sql_select(*cols_q, return_as_lazyframe=True, shard_subsample=subsample)

    def z(c):  # (x - mean)/std with NULLIF(std,0) -> null
        m, sd = pl.col(c).mean(), pl.col(c).std()
        return ((pl.col(c) - m) / pl.when(sd == 0).then(None).otherwise(sd)).alias(f"z_{c}")

    return lf.with_columns([z(c) for c in score_cols]).with_columns(
        pl.max_horizontal([f"z_{c}" for c in ep_cols]).alias("z_ep_max"),
        pl.max_horizontal([f"z_{c}" for c in vp_cols]).alias("z_vp_max"),
    )

# 4) Prove it on a small subsample of the first language (once score_cols is set).
if score_cols:
    demo = zscore_child(seg.children[0], subsample=0.01).select("__key__", "z_ep_max", "z_vp_max").collect()
    print("language:", seg.child_names[0], " rows:", demo.height)
    print(demo.head())
else:
    print("score_cols not set yet — fill it in from the investigation output above.")

# 5) Full multilingual result = vertical concat of the per-language frames.
def zscore_by_language(meta, subsample=1.0):
    """LazyFrame reproducing parent_scores_v2 across all languages.

    Use subsample=1.0 for exact stats (full scan over the segmented rows — a
    batch job, but embarrassingly parallel per language)."""
    return pl.concat([zscore_child(ds, subsample) for ds in meta.children], how="vertical")

print(
    "\nVERDICT: implementable as a per-language (per-child) lazy pass; the SQL's "
    "GROUP BY + self-JOIN are unnecessary here. Exact stats need shard_subsample=1."
)

# %%
# 1c) RESOLVE which column family is the continuous score, and the FULL set.
#     Scores are bare scalar columns (anger, boredom, ...); find the column_dir
#     that hosts a known emotion and list all its siblings = the full score vector.
from collections import defaultdict  # noqa: E402


def col_dir(ds, c):
    return ds.fields[c][0][0] if c in ds.fields else None


# (a) group emotion-named hits by their column_dir + coverage across children
by_dir = defaultdict(list)
coverage = {}
for c in named_hits:
    cov = sum(1 for ds in seg.children if c in ds.fields)
    coverage[c] = cov
    cd = next((col_dir(ds, c) for ds in seg.children if c in ds.fields), None)
    by_dir[cd].append(c)

print("emotion-named columns grouped by column_dir (n_cols, sample coverage /20):")
for cd, cols in sorted(by_dir.items(), key=lambda kv: -len(kv[1])):
    print(f"  [{cd}]  {len(cols)} cols;  e.g. " + ", ".join(f"{c}={coverage[c]}" for c in cols[:5]))

# (b) full sibling set in the column_dir that hosts a known emotion
PROBE = "anger"
host = next(ds for ds in seg.children if PROBE in ds.fields)
score_dir = col_dir(host, PROBE)
siblings = sorted(c for c, locs in host.fields.items() if locs[0][0] == score_dir)
print(f"\ncolumn_dir hosting {PROBE!r} = {score_dir!r}  (in {host.dataset_root.parent.parent.name})")
print(f"  {len(siblings)} sibling columns:\n{siblings}")

# (c) probe value type for each candidate family (bare / _sparse / tags_ml.*)
def first_nonnull(col, tries=30):
    for _ in range(tries):
        v = seg.random_sample().get(col)
        if v is not None:
            return v
    return None


print("\nvalue types by family:")
for c in ["anger", "anger_sparse", "tags_ml.anger", "tags_ml_unpacked.anger"]:
    v = first_nonnull(c)
    print(f"  {c!r}: {type(v).__name__} = {v}")

# (d) coverage of the score_dir across all children (can we z-score every language?)
have = sum(1 for ds in seg.children if PROBE in ds.fields)
print(f"\n{PROBE!r} present in {have}/{len(seg.children)} children")
for ds in seg.children:
    n = sum(1 for c in siblings if c in ds.fields)
    print(f"  {ds.dataset_root.parent.parent.name}: {n}/{len(siblings)} score cols")

# %%
# ============================================================================
# IMPLEMENTATION (supersedes the placeholder steps 3-5 above).
# Scores = the 604 continuous float columns in the `tags_ml` column dir,
# referenced by bare name. Per language (== per child) z-score + GREATEST.
# Only languages that actually HAVE tags_ml are processed; the rest are flagged.
# ============================================================================
ALL_SCORES = siblings  # 604 continuous columns resolved in cell 1c
scored = [ds for ds in seg.children if PROBE in ds.fields]
missing = [ds.dataset_root.parent.parent.name for ds in seg.children if PROBE not in ds.fields]
print(f"{len(ALL_SCORES)} continuous scores in {score_dir!r}; "
      f"{len(scored)}/{len(seg.children)} languages scored. Missing: {missing}")

# Where (if anywhere) do the missing languages keep their scores?
for ds in seg.children:
    if PROBE in ds.fields:
        continue
    fams = [fam for fam, probe in [("tags_ml", "anger"),
                                   ("tags_ml_unpacked", "tags_ml_unpacked.anger"),
                                   ("tags_ml_u8", "anger_sparse")] if probe in ds.fields]
    print(f"  {ds.dataset_root.parent.parent.name}: score families present -> {fams or 'NONE'}")

# parent_scores_v2 z-scores a CURATED subset, split into ep_* / vp_*. The 604
# tags_ml columns are a superset — drop the exact BQ lists here to match it 1:1.
# Until provided, default to all-scores with a single GREATEST.
EP_COLS = ALL_SCORES  # TODO: the 48 emotion-prosody names used in parent_scores_v2
VP_COLS = []          # TODO: the voice-prosody names used in parent_scores_v2


def lang_of(ds):
    return ds.dataset_root.parent.parent.name


def zscore_child(ds, ep=EP_COLS, vp=VP_COLS, subsample=1.0):
    """Per-language z-score + GREATEST, fused into one lazy pass.

    `pl.col(c).mean()/.std()` broadcast the per-child (== per-language) global
    stats, so this reproduces the BQ GROUP BY + self-JOIN with no join. Columns
    missing in this child are skipped; constant columns (std=0) -> null."""
    cols = [c for c in (list(ep) + list(vp)) if c in ds.fields]

    def z(c):
        m, sd = pl.col(c).mean(), pl.col(c).std()
        return ((pl.col(c) - m) / pl.when(sd == 0).then(None).otherwise(sd)).alias(f"z_{c}")

    lf = ds.sql_select("`__key__`", *[f"`{c}`" for c in cols],
                       return_as_lazyframe=True, shard_subsample=subsample)
    out = lf.with_columns([z(c) for c in cols]).with_columns(pl.lit(lang_of(ds)).alias("language"))
    epp = [f"z_{c}" for c in ep if c in ds.fields]
    vpp = [f"z_{c}" for c in vp if c in ds.fields]
    if epp:
        out = out.with_columns(pl.max_horizontal(epp).alias("z_ep_max"))
    if vpp:
        out = out.with_columns(pl.max_horizontal(vpp).alias("z_vp_max"))
    return out


def zscore_all(subsample=1.0):
    """Full multilingual result = vertical concat of the scored languages."""
    return pl.concat([zscore_child(ds, subsample=subsample) for ds in scored], how="diagonal_relaxed")


# demo: one scored language on a tiny subsample (proves the transform end-to-end)
demo = zscore_child(scored[0], subsample=0.005).select("language", "__key__", "z_ep_max").collect()
print(f"\ndemo language: {lang_of(scored[0])}  rows: {demo.height}")
print(demo.head())

# %%
# ============================================================================
# HUNT: are the missing languages' scores present under a DIFFERENT name/dir?
# Signal: a column_dir whose member names overlap the known 604 score names
# (matched bare, i.e. after stripping any "dir." prefix). A dir with a high
# overlap IS the score set under a different name.
# ============================================================================
KNOWN = set(ALL_SCORES)  # the 604 tags_ml score names


def bare(c):
    return c.split(".", 1)[1] if "." in c else c


def dirs_of(ds):
    d = defaultdict(list)
    for f, locs in ds.fields.items():
        d[locs[0][0]].append(f)
    return d


for ds in seg.children:
    if PROBE in ds.fields:
        continue  # already scored via tags_ml
    lang = lang_of(ds)
    d = dirs_of(ds)
    print(f"\n=== {lang} ===  {len(ds.fields)} fields across {len(d)} column dirs")
    # rank dirs by how many of their (bared) columns are known score names
    ranked = sorted(
        ((cd, cols, sum(1 for c in cols if bare(c) in KNOWN)) for cd, cols in d.items()),
        key=lambda t: -t[2],
    )
    for cd, cols, overlap in ranked[:6]:
        flag = "   <-- SCORES under a different dir/name" if overlap > 50 else ""
        print(f"   [{cd}] {len(cols)} cols, {overlap}/{len(KNOWN)} match known scores{flag}")
    # direct token search + value probe on the best candidate dir
    top_cd, top_cols, top_overlap = ranked[0]
    if top_overlap > 50:
        sample_names = [c for c in top_cols if bare(c) in KNOWN][:5]
        s = ds.random_sample()
        types = {c: type(s.get(c)).__name__ for c in sample_names}
        print(f"   candidate '{top_cd}' value types: {types}")
    else:
        tok = sorted(f for f in ds.fields if any(t in f.lower() for t in TOKENS))
        print(f"   no high-overlap dir; emotion-token fields ({len(tok)}): {tok[:12]}")
# %%
