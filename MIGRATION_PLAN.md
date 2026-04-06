# WSDS Migration Analysis & Plan

## Context

wsds is a ~5,700-line custom multimodal dataset library built on Apache Arrow IPC. It provides segment-level audio seeking, column auto-discovery, batch-cached sequential access, remote range-request reads (S3/Modal), and SQL queries via Polars. The question is whether to migrate to open-source alternatives to reduce maintenance burden.

---

## High-Level Framework Summary

| Component | Lines | Purpose |
|-----------|-------|---------|
| `ws_dataset.py` | 598 | Core API: iteration, indexing, SQL, column routing |
| `ws_sample.py` | 270 | Lazy dict-like sample with key verification |
| `ws_shard.py` + `ws_s3_shard.py` + `ws_modal_shard.py` | ~480 | Batch-cached Arrow IPC readers (local, S3, Modal) |
| `ws_audio.py` + `audio_codec.py` | ~415 | Segment-level audio seeking with MP3 delay compensation |
| `ws_index.py` + `ws_feather_index.py` + `ws_indexer.py` | ~700 | SQLite & Feather indexes for O(1) random access |
| `ws_sink.py` | 147 | Atomic batched Arrow IPC writer |
| `ws_decode.py` | 57 | Column-name-based type dispatch (npy/pyd/txt/audio) |
| `ws_tools.py` | 620+ | CLI: validate, init, inspect, list |
| `pupyarrow/` | ~1,450 | Pure-Python Arrow IPC parser for remote range requests |
| `convplayer.py` | 418 | Interactive HTML audio visualization |
| `utils.py` | 305 | Helpers: shard listing, column discovery, validation |

### Key Features
- **Arrow IPC shards** in column directories -- add a folder, columns appear automatically
- **Segment-level audio seeking** -- WSAudio holds (tstart, tend), AudioReader seeks to keyframe
- **Batch caching** -- `batch_size` in schema metadata enables `offset // batch_size` O(1) lookup
- **Remote range requests** -- pupyarrow fetches only footer + needed batches from S3/Modal
- **Polars SQL** -- `sql_select()` / `sql_filter()` scan IPC lazily across column dirs
- **Linked datasets** -- `.wsds-link` files reference source audio without duplication

### Usage Patterns (from research/ notebooks and scripts)
- PyTorch `IterableDataset` wrapping `ds.iter_shard()`
- Per-worker `WSDataset` instances for distributed training
- `WSSink` for writing new columns from processing pipelines
- `WSAudio.load()` with BytesIO stream reset for multi-segment access
- `pl.scan_ipc()` for metadata exploration

---

## Open-Source Alternatives Evaluated

### HuggingFace Datasets
- **Strengths**: Arrow-backed, huge ecosystem, streaming, PyTorch integration, HF Hub
- **Gaps**: No segment-level audio seeking, no column-directory auto-discovery, downloads whole files (no batch-level range requests), opinionated layout

### Lhotse
- **Strengths**: Purpose-built for speech, segment seeking via CutSet, augmentation (mixing, noise, speed perturbation)
- **Gaps**: JSON manifests + file paths (not Arrow columnar), no remote range requests, no column auto-discovery, fundamentally different storage model

### Lance (LanceDB)
- **Strengths**: Arrow-based, 100x faster random access than Parquet, vector search
- **Gaps**: No audio-specific features, newer ecosystem, no segment seeking

### WebDataset
- **Strengths**: TAR-based streaming, great for large-scale sequential training
- **Gaps**: No random access, no Arrow, no segment seeking

### Others (Polars, DuckDB, TFDS, SpeechBrain)
- Either not dataset libraries or missing too many key features

### Comparison Matrix

| Feature | wsds | HF Datasets | Lhotse | Lance |
|---------|------|-------------|--------|-------|
| Arrow columnar storage | **Yes** | Yes | No | Yes |
| Segment-level audio seeking | **Yes** | No | **Yes** | No |
| Column auto-discovery | **Yes** | No | No | No |
| Batch-level range requests (S3) | **Yes** | No | No | No |
| O(1) random access | **Yes** | Yes | Yes | **Yes** (100x) |
| SQL queries | **Yes** | Partial | No | No |
| PyTorch native | **Yes** | **Yes** | **Yes** | Yes |
| Ecosystem/community | Small | **Huge** | Medium | Growing |

---

## The Three Options

### Option A: Keep wsds, reduce maintenance surface (RECOMMENDED)

**Rationale**: The features that justify wsds's existence (audio seeking into Arrow binary blobs, column auto-discovery, remote range requests, batch caching) are exactly the features no open-source library replicates. The core is ~2,400 lines of stable code.

**Actions**:
1. Add proper test suite (beyond current doctests)
2. Evaluate removing/extracting `convplayer.py` (~418 lines, heavy deps)
3. Evaluate if `pupyarrow/` can be simplified or if pyarrow+fsspec suffices
4. Evaluate if both `ws_index.py` and `ws_feather_index.py` are needed
5. Add type hints to public API, document on-disk format spec
6. Pin dependencies, add CI

**Effort**: 1-2 weeks | **Risk**: None | **Data conversion**: None

### Option B: Hybrid -- Lhotse for audio + wsds for storage

**Actions**: Create adapter wrapping wsds shards as Lhotse Recordings/Supervisions. Use CutSet for augmentation/batching, keep wsds for storage.

**Problem**: Fundamental mismatch -- Lhotse expects file paths, wsds stores binary blobs in Arrow. Need custom `AudioSource` or extract audio to files.

**Effort**: 3-5 weeks | **Risk**: Medium (impedance mismatch) | **Gain**: Lhotse's augmentation features

### Option C: Full migration to HF Datasets + custom audio layer

**Actions**: Convert all datasets to HF format, keep ~415 lines of audio seeking code as standalone package, rewrite pipelines.

**What you lose**: Remote range requests, column auto-discovery, batch-level caching, computed columns. Still need custom audio code.

**Effort**: 6-10 weeks + data conversion | **Risk**: High | **Gain**: HF ecosystem

---

## Recommendation: Option A

**The maintenance burden is overstated.** Excluding generated flatbuf code, convplayer, and CLI tools, the core is ~2,400 lines of straightforward Arrow/Polars plumbing. It's stable, it works, and no open-source library replaces its unique value proposition.

**Migration risk is high, reward is low.** You'd convert terabytes of data and rewrite every pipeline to end up with a system that performs worse at the specific things wsds was designed for -- and you'd STILL need ~500 lines of custom audio seeking code.

**The ecosystem argument doesn't apply here.** Datasets are internal (S3/Modal), not shared on HF Hub. The community benefits of HF Datasets don't materialize for this use case.

---

## Questions for You

1. **How often does wsds itself change?** If stable, maintenance is near-zero.
2. **Are you sharing datasets externally?** If yes, HF format has real value.
3. **Is pupyarrow (remote range requests) actively used?** It's the most complex component (~1,450 lines). If you always read locally, this could be dropped.
4. **Is convplayer still used?** Heavy dependencies (whisper, PIL, torchaudio) for a visualization tool.
5. **Is ws_feather_index.py the replacement for ws_index.py, or do both coexist?**
6. **How many datasets exist in wsds format?** Determines migration cost.
7. **What is the actual pain point?** Maintenance burden, onboarding difficulty, performance, or ecosystem compatibility? The answer changes the recommendation.
8. **Do you want Lhotse-style augmentation (mixing, speed perturbation, noise)?** If yes, Option B's thin adapter may be worth it for the audio pipeline specifically.

---

## Verification
- Review this plan against actual wsds source at `wsds/wsds/`
- Review usage patterns in `research/` notebooks/scripts
- Validate feature gaps by checking latest HF Datasets, Lhotse, and Lance docs
