"""Offline LRU eviction for the WSDS S3 block-sparse cache.

Merges per-worker Feather access logs, measures the cache (present-bits x SLOT --
reliable, unlike du/df on WEKA), and evicts to a target size by GLOBAL LRU: whole
cold objects -> unlink; partially-hot -> compact (offset-preserving rewrite + atomic
rename, so lock-free readers never see a torn mix). Blocks never seen in the logs are
treated as coldest.

This is the periodic sweep that bounds a `"cache": true` link (see WSS3Shard._resolve_cache):
mirrors live at <dataset_root>/<partition>/audio/<shard>.wsds.sparse (scanned recursively under
--root, named relative to it, as in the access logs) and the logs at <dataset_root>/audio/.access:

  python -m wsds.cache_evict --root /data/podcasts-en/source --target-gb 500

  (--logdir defaults to <root>/audio/.access; --dry-run previews; --consume-logs deletes
   logs after a successful pass). `wsds-cache` (wsds.cachectl) wraps this with per-dataset
   policy files and a systemd timer.
"""

import argparse
import glob
import os

import pyarrow.feather as feather

from wsds.pupyarrow.block_cache import EXT, SLOT, compact, present_blocks, read_layout


def merge_logs(logdir):
    """-> ({object: {block: last_ts}}, [logfiles])   (most-recent access wins)"""
    merged = {}
    files = sorted(glob.glob(os.path.join(logdir, "access-*.feather")))
    for f in files:
        try:
            t = feather.read_table(f)
        except Exception:
            continue
        for o, b, ts in zip(
            t.column("object").to_pylist(), t.column("block").to_pylist(), t.column("last_ts").to_pylist()
        ):
            d = merged.setdefault(o, {})
            if ts > d.get(b, -1):
                d[b] = ts
    return merged, files


def scan_cache(root):
    """-> [(name, path, present_set)]"""
    out = []
    for path in glob.glob(os.path.join(root, "**", "*" + EXT), recursive=True):
        name = os.path.relpath(path, root)[: -len(EXT)]
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            objsize, nblocks, bm_off = read_layout(fd)
            present = present_blocks(fd, nblocks, bm_off)
        except Exception:
            os.close(fd)
            continue
        os.close(fd)
        out.append((name, path, present))
    return out


def plan(mirrors, merged, target_bytes):
    scored = []  # (last_ts, name, block)
    for name, _, present in mirrors:
        ts_map = merged.get(name, {})
        for b in present:
            scored.append((ts_map.get(b, 0.0), name, b))
    scored.sort(reverse=True)  # most-recent first
    budget = max(0, target_bytes) // SLOT
    keep = {name: set() for name, _, _ in mirrors}
    for i, (_, name, b) in enumerate(scored):
        if i < budget:
            keep[name].add(b)
    return keep


def default_logdir(root):
    return os.path.join(root.rstrip("/"), "audio", ".access")  # WSS3Shard._resolve_cache's logdir


def evict_one(root, target_gb, logdir=None, dry_run=False, consume_logs=False, log=print):
    """Run one LRU eviction pass over `root`. Returns a summary dict.

    Pure engine (no argparse) so cachectl and cron/systemd can call it directly."""
    logdir = logdir or default_logdir(root)
    merged, logfiles = merge_logs(logdir)
    mirrors = scan_cache(root)
    cur = sum(len(p) for _, _, p in mirrors) * SLOT
    target = int(target_gb * 1e9)
    if log:
        log(
            f"{root}: {len(mirrors)} objects, {cur / 1e9:.2f} GB present; "
            f"logs {len(logfiles)} files / {len(merged)} objects; target {target_gb} GB"
        )
    res = {
        "root": root,
        "objects": len(mirrors),
        "gb_before": cur / 1e9,
        "target_gb": target_gb,
        "unlinked": 0,
        "compacted": 0,
        "gb_freed": 0.0,
        "dry_run": dry_run,
    }
    if cur <= target:
        if log:
            log("  under target, nothing to evict.")
        res["gb_after"] = cur / 1e9
        return res
    keep = plan(mirrors, merged, target)
    unlink = [m for m in mirrors if not keep[m[0]]]
    comp = [m for m in mirrors if 0 < len(keep[m[0]]) < len(m[2])]
    freed = sum(len(p) for n, _, p in unlink) * SLOT + sum((len(p) - len(keep[n])) for n, _, p in comp) * SLOT
    res.update(unlinked=len(unlink), compacted=len(comp), gb_freed=freed / 1e9)
    if log:
        log(f"  plan: unlink {len(unlink)} cold, compact {len(comp)} partial -> free ~{freed / 1e9:.2f} GB")
    if dry_run:
        if log:
            log("  [dry-run] no changes made.")
        res["gb_after"] = cur / 1e9
        return res
    for name, path, _ in unlink:
        try:
            os.unlink(path)
        except OSError:
            pass
    for name, path, _ in comp:
        try:
            compact(path, keep[name])
        except Exception as e:
            if log:
                log(f"  compact failed {name}: {e}")
    new = sum(len(p) for _, _, p in scan_cache(root)) * SLOT
    res["gb_after"] = new / 1e9
    if log:
        log(f"  cache now ~{new / 1e9:.2f} GB present")
    if consume_logs:
        for f in logfiles:
            try:
                os.unlink(f)
            except OSError:
                pass
        if log:
            log(f"  consumed {len(logfiles)} log files")
    return res


def main():
    ap = argparse.ArgumentParser(prog="python -m wsds.cache_evict")
    ap.add_argument("--root", required=True, help="dataset root holding */audio/*.sparse mirrors")
    ap.add_argument("--logdir", default=None, help="access logs (default <root>/audio/.access)")
    ap.add_argument("--target-gb", type=float, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--consume-logs", action="store_true")
    a = ap.parse_args()
    evict_one(a.root, a.target_gb, logdir=a.logdir, dry_run=a.dry_run, consume_logs=a.consume_logs)


if __name__ == "__main__":
    main()
