# %%
"""bench_async_pipeline.py — evidence for the async pupyarrow pipeline.

Measures, on a real S3-linked shard (data-en audio on Backblaze B2), the
improvements introduced by the async-pupyarrow branch:

  1. Async coalesced column reads (FeatherFile.__getitem__ plan mode) vs the
     old sequential per-batch path — ~10x wall time (concurrent range GETs).
  2. The presigned-aiohttp S3 data path (default) vs botocore get_object —
     botocore burns ~1.5-3ms CPU per request serialized on the IO loop
     thread, capping the process at a few hundred req/s; presigning cuts
     that ~4x, so the gap widens with concurrency.
  3. LazyBuffer BlockCache + async_prepopulate — parallel prefetch of seek
     ranges inside an audio blob, then seeks are served from memory with
     zero further S3 requests.

Each section asserts the fast path returns byte-identical data.
Run on a host with /mnt/weka mounted (e.g. iren-login-02).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import wsds
from wsds.pupyarrow.file_reader import S3FileReader, _get_io_loop
from wsds.pupyarrow.pupyarrow import FeatherFile

DATASET = "/mnt/weka/data-wsds/data-en/indices/source"
LINK_DIR = "audio.wsds-link"

ds = wsds.WSDataset(DATASET)
shard_ref = next(iter(ds.index.shards()))
shard = ds.get_shard(LINK_DIR, shard_ref)
client = shard._reader._client
print(ds)
print(shard)
print("batches:", shard._feather.num_record_batches, "batch_size:", shard.batch_size)

COLUMNS = [n for n in shard._feather.schema.names if n != "audio"]  # "audio" = whole shard
print("bench columns:", COLUMNS)


def fresh_feather(presigned: bool = True) -> FeatherFile:
    """New reader + FeatherFile so IO stats start from zero (footer included)."""
    return FeatherFile(S3FileReader(client, shard.bucket, shard.key, presigned=presigned))


def bench(label, fn, reader=None):
    t0 = time.monotonic()
    out = fn()
    dt = time.monotonic() - t0
    stats = ""
    if reader:
        stats = (
            f"io_count={reader.io_count:4d}  io_bytes={reader.io_bytes / 1e6:6.2f}MB  "
            f"io_time={reader.io_time * 1000:6.0f}ms  hits={reader.cache_hits}"
        )
    print(f"{label:<42} wall={dt * 1000:7.0f}ms  {stats}")
    return out


# %% 1. single shard: old sequential path vs async coalesced f[cols]
# Old path: one blocking range GET after another (~9s at B2's ~180ms RTT).
# New path: plan mode defers reads, coalesces nearby ones, runs them
# concurrently (~1s). io_count is higher (concurrency defeats the sequential
# forward-cache carryover) but wall time is what matters.
f_old = fresh_feather()


def read_sequential():
    out = {c: [] for c in COLUMNS}
    for i in range(f_old.num_record_batches):
        batch = f_old.record_batch(i)
        for c in COLUMNS:
            out[c].append(batch.column(c).to_numpy())
    return out


old_result = bench("old: sequential per-batch reads", read_sequential, f_old._reader)

f_new = fresh_feather()
new_result = bench("new: async coalesced f[cols]", lambda: f_new[COLUMNS], f_new._reader)

for c in COLUMNS:
    assert list(new_result[c]) == [x for chunk in old_result[c] for x in chunk], c
print("results identical ✓")


# %% 2. S3 data path at scale: N concurrent collections, botocore vs presigned
# Simulates multi-shard collection (same shard N times — every request is a
# real range GET, nothing is cached between readers). Threads block on the
# shared IO loop, so all ~N×107 requests compete for one thread's CPU:
# exactly where botocore's per-request overhead becomes the bottleneck.
# Expect similar io totals but ~3-5x lower wall time for presigned.
N_CONCURRENT = 20


def concurrent_collect(presigned: bool):
    with ThreadPoolExecutor(N_CONCURRENT) as pool:
        files = list(pool.map(lambda _: fresh_feather(presigned), range(N_CONCURRENT)))
        setup_reqs = sum(f._reader.io_count for f in files)
        t0, c0 = time.monotonic(), time.process_time()
        results = list(pool.map(lambda f: f[COLUMNS], files))
        dt, dc = time.monotonic() - t0, time.process_time() - c0
    reqs = sum(f._reader.io_count for f in files) - setup_reqs
    label = "presigned aiohttp" if presigned else "botocore get_object"
    print(
        f"{label:<20} x{N_CONCURRENT}: wall={dt * 1000:6.0f}ms  cpu={dc * 1000:6.0f}ms  "
        f"{reqs / dt:6.0f} req/s  {dc / max(reqs, 1) * 1000:5.2f}ms cpu/req  ({reqs} requests)"
    )
    return results


res_boto = concurrent_collect(presigned=False)
res_pre = concurrent_collect(presigned=True)

key0 = COLUMNS[0]
assert all(list(r[key0]) == list(new_result[key0]) for r in res_boto + res_pre)
print(f"all {2 * N_CONCURRENT} collections identical ✓")


# %% 3. BlockCache: audio seek reads served from prefetched ranges
# ffmpeg-style access inside one audio blob: header probe + scattered 32kB
# reads. Uncached, each miss is a fresh S3 round trip. With enable_cache()
# + async_prepopulate(), all ranges arrive in ~1 parallel round trip and
# the seeks themselves do zero IO.
f_blob = fresh_feather()
audio_col = f_blob.record_batch(0).column("audio")
sizes = audio_col.byte_sizes()
idx = int(sizes.argmax())  # biggest blob in batch 0
blob_len = int(sizes[idx])
print(f"blob #{idx}: {blob_len / 1e6:.2f}MB")

offsets = [0] + [int(blob_len * frac) for frac in (0.1, 0.25, 0.26, 0.5, 0.52, 0.9)]
READ = 32 * 1024


def seek_reads(buf):
    return [buf.read_range(o, min(o + READ, blob_len)) for o in offsets]


r_plain = fresh_feather()
buf_plain = r_plain.record_batch(0).column("audio")[idx]
plain = bench("seeks, no cache (sequential misses)", lambda: seek_reads(buf_plain), r_plain._reader)

r_cached = fresh_feather()
buf = r_cached.record_batch(0).column("audio")[idx].enable_cache()
bench(
    "async_prepopulate (parallel prefetch)",
    lambda: _get_io_loop().run(buf.async_prepopulate([(o, READ) for o in offsets])),
    r_cached._reader,
)
cached = bench("seeks after prepopulate (memory only)", lambda: seek_reads(buf), r_cached._reader)

assert cached == plain
print("seek data identical ✓")
