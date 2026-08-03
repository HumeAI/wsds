"""
Generate (and validate) a .wsds-link file that serves one column directory from S3.

A .wsds-link is a JSON file at the dataset root named `<name>.wsds-link`. ONE file
serves ONE column directory (e.g. `audio/`) across ALL its shards — the shard name is
filled in per read from the index, so you do NOT need one file per shard. To serve
several column dirs from S3, write one link per column dir.

At read time WSS3Shard builds each S3 key as:
    normpath(prefix / partition / subdir / <shard>.wsds)     (leading "../" stripped)
where `partition` comes from the index's shard refs ("." — i.e. nothing — for an
in-place index like bbc/source). So `prefix` must be the S3 path up to the column dir,
minus whatever the partition labels already contribute.

Naming: the link's filename stem is registered as a field of the dataset, so it should
match a column the link actually serves (e.g. `mp3.wsds-link` for a dataset whose audio
column is called `mp3`). A mismatched stem creates a phantom field that raises KeyError
when read — and can make `get_audio()` fail intermittently if the stem is an audio-like
name such as `audio`.

Credentials: by default none are embedded and boto3's ambient chain is used at read
time (env vars / ~/.aws profile / instance role). Pass --key-id/--app-key to embed
credentials in the link — ONLY do this with read-only keys, since link files usually
live on shared storage.

Usage:
    python support_scripts/make_s3_link.py \
        --s3-url s3://data-wsds/bbc/source/audio \
        --dataset /mnt/weka/data-wsds/bbc/source \
        --endpoint https://s3.us-east-005.backblazeb2.com \
        --out /mnt/weka/data-wsds/bbc/source/mp3.wsds-link   # omit --out for a dry run
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def build_key(prefix, partition, subdir, shard):
    """Reconstruct the S3 key exactly as WSS3Shard.from_link does."""
    parts = [p for p in (prefix, partition, subdir, f"{shard}.wsds") if p and p != "."]
    key = os.path.normpath("/".join(parts))
    while key.startswith("../"):
        key = key[3:]
    return key


def shard_refs_from_dataset(dataset: Path, subdir: str):
    """Prefer the index's shard refs (authoritative partitions); fall back to local listing."""
    idx = dataset / "index.sqlite3"
    if idx.exists():
        import sqlite3

        con = sqlite3.connect(f"file:{idx}?immutable=1", uri=True)
        cols = {r[0] for r in con.execute("SELECT name FROM pragma_table_info('shards')")}
        part_col = "partition" if "partition" in cols else ("dataset_path" if "dataset_path" in cols else "''")
        rows = con.execute(f"SELECT {part_col}, shard FROM shards").fetchall()
        return [(p or "", s) for p, s in rows], "index"
    local = dataset / subdir
    if local.is_dir():
        return [("", f.stem) for f in sorted(local.glob("*.wsds"))], "local-listing"
    return [], "none"


def discover_columns(dataset: Path, subdir: str, s3, bucket: str, key_path: str):
    """Column names served by this link: from a local shard if present, else from an
    S3 shard footer via a full download of the last 64KB (no pupyarrow dependency)."""
    local_dir = dataset / subdir
    local_shard = next(local_dir.glob("*.wsds"), None) if local_dir.is_dir() else None
    if local_shard is not None:
        import pyarrow as pa

        names = pa.RecordBatchFileReader(pa.memory_map(str(local_shard))).schema.names
        return sorted(c for c in names if c != "__key__"), f"local shard {local_shard.name}"

    listed = s3.list_objects_v2(Bucket=bucket, Prefix=key_path + "/", MaxKeys=5).get("Contents", [])
    first = next((o["Key"] for o in listed if o["Key"].endswith(".wsds")), None)
    if first is None:
        return None, None
    import io

    import pyarrow as pa

    head = s3.get_object(Bucket=bucket, Key=first, Range="bytes=0-4194303")["Body"].read()
    # An IPC file's schema lives right after the 8-byte magic preamble; stream-read it.
    reader = pa.ipc.open_stream(io.BytesIO(head[8:]))
    return sorted(c for c in reader.schema.names if c != "__key__"), f"s3 shard {first}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-url", required=True, help="s3://bucket/path/to/<column_dir>")
    ap.add_argument("--dataset", required=True, help="local dataset root the link belongs to")
    ap.add_argument("--endpoint", default=os.environ.get("WSDS_S3_ENDPOINT_URL"))
    ap.add_argument("--name", default=None, help="link filename stem (default: a column the link serves)")
    ap.add_argument("--key-id", default=None, help="embed this access key id (read-only keys only!)")
    ap.add_argument("--app-key", default=None, help="embed this secret key (read-only keys only!)")
    ap.add_argument("--out", default=None, help="write the link here; dry-run if omitted")
    ap.add_argument("--sample", type=int, default=8, help="how many shards to validate against S3")
    args = ap.parse_args()

    u = urlparse(args.s3_url)
    assert u.scheme == "s3", f"expected s3:// URL, got {args.s3_url}"
    bucket = u.netloc
    key_path = u.path.strip("/")
    prefix, subdir = os.path.split(key_path)
    dataset = Path(args.dataset)

    import boto3

    client_kwargs = {"endpoint_url": args.endpoint} if args.endpoint else {}
    if args.key_id and args.app_key:
        client_kwargs.update(aws_access_key_id=args.key_id, aws_secret_access_key=args.app_key)
    s3 = boto3.client("s3", **client_kwargs)

    link = {
        "loader": ["wsds.ws_s3_shard", "WSS3Shard"],
        "bucket": bucket,
        "prefix": prefix,
        "subdir": subdir,
    }
    if args.endpoint:
        link["endpoint_url"] = args.endpoint
    if args.key_id and args.app_key:
        link["aws_access_key_id"] = args.key_id
        link["aws_secret_access_key"] = args.app_key

    columns, col_source = discover_columns(dataset, subdir, s3, bucket, key_path)
    if columns:
        link["columns"] = columns
        print(f"columns discovered from {col_source}: {columns}")
    else:
        print("could not discover columns; omitting `columns` (WSDataset will discover them from S3 at open time)")

    # The filename stem becomes a dataset field, so it must be a column this link serves.
    # Prefer the audio column (usually what the link exists for), then the subdir name.
    if args.name:
        name = args.name
    elif columns:
        from wsds.ws_decode import AUDIO_FILE_KEYS

        audio_cols = [c for c in columns if c in AUDIO_FILE_KEYS]
        name = audio_cols[0] if audio_cols else (subdir if subdir in columns else columns[0])
    else:
        name = subdir
    if columns and name not in columns:
        print(f"WARNING: link name {name!r} is not among the served columns {columns} - "
              f"this registers a phantom field that raises KeyError when read")

    # validate: reconstruct keys exactly as WSS3Shard would and confirm they exist in S3
    refs, refs_source = shard_refs_from_dataset(dataset, subdir)
    print(f"shard refs from: {refs_source} ({len(refs)} shards)")
    ok = missing = 0
    for partition, shard in refs[: args.sample]:
        key = build_key(prefix, partition, subdir, shard)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            ok += 1
            print(f"  [ok] s3://{bucket}/{key}")
        except Exception:
            missing += 1
            print(f"  [MISSING] s3://{bucket}/{key}")
    if refs:
        print(f"validation: {ok}/{ok + missing} sampled shards resolve in S3")

    print("\n.wsds-link content:")
    print(json.dumps(link, indent=2))

    if args.out:
        if missing:
            print(f"\nREFUSING to write: {missing} sampled shards did not resolve (prefix/partition mismatch?)")
            raise SystemExit(1)
        Path(args.out).write_text(json.dumps(link, indent=2))
        print(f"\nwrote {args.out}")
        if Path(args.out).name != f"{name}.wsds-link":
            print(f"NOTE: recommended filename is {name}.wsds-link")
    else:
        print(f"\n(dry run) place this JSON at <dataset_root>/{name}.wsds-link to activate")


if __name__ == "__main__":
    main()
