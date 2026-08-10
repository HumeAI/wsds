"""
wsds index creation

The index is built in two phases:

`extract_partition_index` – scans the shards of one partition directory and caches an
episode-level index (`episode-list.feather`) inside each subdataset
`merge_partition_indices` – merges the cached extracts across partitions into a single
SQLite wsds index

Which subdatasets exist and how to index them is described by `SubdatasetSpec`.
`extract_batch_index` / `merge_batch_indices` are thin wrappers that preserve the
original data-pl delivery/batch interface (partition = batch dir, subdatasets =
`source` + `filtered_vad`).
"""

import json
import os
import time
import traceback
import typing
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.feather

from wsds import AtomicFile, WSDataset
from wsds.ws_index import WSDSIndexWriter

# schema-metadata key under which the cleaned field mapping is embedded in episode-list.feather
FIELDS_METADATA_KEY = b"wsds_fields"


@dataclass(frozen=True)
class SubdatasetSpec:
    """Describes how to index one subdataset (a set of column dirs holding the same rows).

    kind: subdataset directory name, relative to each partition dir (e.g. "source", "v4-vad_ws")
    segmented: True when rows are segments of source episodes (keys end in _NNN)
    key_column: optional column used to anchor `__key__` extraction to that column's
        (complete) column dir; without it sql_select picks one automatically, which may
        be an incomplete/in-progress dir whose missing shards would silently drop episodes
    duration_expr: SQL expression for per-episode audio duration (non-segmented datasets);
        None when no duration column exists (episodes get audio_duration = -1)
    speech_expr: SQL expression for per-segment speech duration (segmented datasets);
        None when unavailable (episodes get speech_duration = -1). Every column it
        references must exist in a complete column dir, otherwise shards missing that
        column are dropped from the index.
    segment_regex: regex extracting the episode name from a segment __key__
    vad_column: source-dataset column holding per-episode segment timestamps; stored in
        the index metadata so `sample["audio"]` can cut segments out of the source audio
    source_kind: kind of the source subdataset (segmented only); its episode extract
        supplies per-episode audio durations, and it is the computed-audio link target
    shard_filter: optional predicate on (partition, shard_name) restricting which shards
        are indexed (e.g. to skip shards with a foreign schema in a mixed column dir)
    """

    kind: str
    segmented: bool = False
    key_column: str | None = None
    duration_expr: str | None = "load_duration AS audio_duration"
    speech_expr: str | None = "tend - tstart AS speech_duration"
    segment_regex: str = r"(.*)_[0-9]+$"
    vad_column: str | None = None
    source_kind: str | None = None
    shard_filter: typing.Callable[[tuple[str, str]], bool] | None = None


# the original data-pl delivery/batch layout
SOURCE_SPEC = SubdatasetSpec(kind="source")
FILTERED_VAD_SPEC = SubdatasetSpec(
    kind="filtered_vad", segmented=True, vad_column="vad.npy", source_kind="source"
)
DEFAULT_SPECS = (SOURCE_SPEC, FILTERED_VAD_SPEC)


def clean_fields(fields: dict) -> dict:
    """Drop bookkeeping fields that should not be exposed through the index."""

    def col_name(v):
        # fields values are [(column_dir, column)] (normalized) or legacy (column_dir, column)
        spec = v if isinstance(v[0], str) else v[0]
        return spec[1]

    return {k: v for k, v in fields.items() if col_name(v) not in ("sample_source_id", "src_key")}


def extract_episodes(episode_idx: pl.DataFrame, segment_regex: str = r"(.*)_[0-9]+$") -> pl.DataFrame:
    """
    Aggregate segment-level data into episode-level data.

    Takes a DataFrame with segment keys (e.g. "episode_123_0", "episode_123_1")
    and aggregates them into episodes by extracting the base key and summing durations.
    """
    return (
        episode_idx.with_columns(
            pl.col("__key__").str.extract(segment_regex, 1),
            shard=pl.col("__shard_path__").str.extract(r"([^/]+).wsds$", 1),
        )
        .group_by("__key__", maintain_order=True)
        .agg(
            pl.sum("speech_duration"),
            pl.len().alias("segments"),
            pl.first("shard"),
            pl.first("offset"),
        )
    )


def make_shard_idx(
    sample_idx: pl.DataFrame, n_samples_expr: pl.Expr, partition: Path | str, shard_id_offset: int = 0
) -> pl.DataFrame:
    """
    Create a shard index from a sample/episode index.

    Groups samples by shard and computes aggregate statistics.
    """
    return (
        sample_idx.group_by("shard", maintain_order=True)
        .agg(n_samples_expr, pl.sum("audio_duration"))
        .with_row_index("shard_id", offset=shard_id_offset)
        .with_columns(
            partition=pl.lit(str(partition)),
        )
    )


def write_index(
    path: Path | str,
    shard_idx: pl.DataFrame,
    episode_idx: pl.DataFrame,
    fields: dict,
    source_path: str | None = None,
    vad_column: str | None = None,
    segmented: bool | None = None,
):
    """
    Write a wsds SQLite index file with shard and episode data.

    Args:
        path: Directory to write index.sqlite3 to
        shard_idx: DataFrame with shard information
        episode_idx: DataFrame with episode/file information
        fields: Field mapping dictionary
        source_path: Path to source dataset (for computed audio columns)
        vad_column: VAD column name; together with source_path it emits a computed
            audio column. Omit it for segmented datasets that get their audio from
            an `audio.wsds-link` file instead.
        segmented: Whether sample keys are segments (episode + offset suffix).
            Defaults to inferring from source_path+vad_column for compatibility.
    """
    audio_duration, speech_duration = episode_idx.select("audio_duration", "speech_duration").sum().row(0)
    with AtomicFile(f"{path}/index.sqlite3") as fname:
        with WSDSIndexWriter(fname) as index:
            metadata = {}
            if source_path and vad_column:
                metadata["computed_columns"] = {
                    "audio.wsds-computed": {
                        "dataset_dir": str(source_path),
                        "loader": ["wsds.ws_shard", "WSSourceAudioShard"],
                        "vad_column": vad_column,
                    }
                }
                fields = {k: v for k, v in fields.items()}
                fields["audio"] = ("audio.wsds-computed", "audio")
            metadata["segmented"] = bool(source_path and vad_column) if segmented is None else segmented
            metadata.update({"fields": fields, "audio_duration": audio_duration, "speech_duration": speech_duration})
            index.append_metadata(metadata)

        conn = dict(connection=f"sqlite:///{fname}", if_table_exists="append", engine="adbc")
        shard_idx.drop("audio_duration").write_database(table_name="shards", **conn)
        # duration columns are NOT NULL in the index schema; -1 marks "unknown"
        episode_idx.with_columns(
            pl.col("audio_duration").fill_null(-1), pl.col("speech_duration").fill_null(-1)
        ).write_database(table_name="files", **conn)


def _write_episode_list(out_file: Path | str, episode_idx: pl.DataFrame, fields: dict):
    """Write an episode extract with the cleaned field mapping embedded in the schema metadata."""
    table = episode_idx.to_arrow()
    metadata = dict(table.schema.metadata or {})
    metadata[FIELDS_METADATA_KEY] = json.dumps(fields).encode()
    with AtomicFile(out_file) as fname:
        pa.feather.write_feather(table.replace_schema_metadata(metadata), str(fname), compression="zstd")


def _read_episode_list(idx_file: Path | str) -> tuple[pl.DataFrame, dict | None]:
    """Read an episode extract; returns (episode_idx, fields) with fields None for
    legacy extracts that kept the field mapping in a sidecar fields.json."""
    table = pa.feather.read_table(str(idx_file))
    fields = None
    metadata = table.schema.metadata or {}
    if FIELDS_METADATA_KEY in metadata:
        fields = json.loads(metadata[FIELDS_METADATA_KEY])
    return pl.from_arrow(table), fields


def extract_subdataset_index(
    ds_path: Path | str,
    spec: SubdatasetSpec,
    source_idx: pl.DataFrame | None = None,
    overwrite: bool = False,
    shard_subsample: float = 1,
    out_file: Path | str | None = None,
) -> pl.DataFrame:
    """
    Extract an episode-level index for a single subdataset directory.

    Writes `episode-list.feather` (into `ds_path` unless `out_file` overrides it,
    e.g. for read-only datasets) and returns the episode index. For segmented
    subdatasets `source_idx` supplies per-episode audio durations.
    """
    ds_path = Path(ds_path)
    out_file = Path(out_file) if out_file else ds_path / "episode-list.feather"

    if out_file.exists() and not overwrite:
        print(f"Skipping, {out_file} already exists")
        episode_idx, _fields = _read_episode_list(out_file)
        return episode_idx

    start = time.perf_counter()
    ds = WSDataset(ds_path, ignore_index=True)
    print(f"Loaded dataset {ds.dataset_root} in {time.perf_counter() - start:.1f}s")

    fields = clean_fields(ds.fields)

    queries = ["__key__"]
    if spec.segmented:
        queries.append(spec.speech_expr or "NULL AS speech_duration")
    else:
        queries.append(spec.duration_expr or "NULL AS audio_duration")
    queries += ["__shard_path__", "__shard_offset__ AS offset"]

    start = time.perf_counter()
    if spec.segmented:
        if source_idx is None:
            raise ValueError(f"source_idx is required to extract segmented subdataset {ds_path}")
        segment_idx = ds.sql_select(
            *queries,
            shard_subsample=shard_subsample,
            shard_pipe=lambda df: extract_episodes(df, spec.segment_regex),
            key_column=spec.key_column,
            shard_filter=spec.shard_filter,
        )
        episode_idx = segment_idx.join(source_idx["__key__", "audio_duration"], on="__key__").with_columns(
            pl.col("speech_duration").cast(pl.Float32)
        )
        if len(episode_idx) < len(segment_idx):
            print(f"WARNING: dropped {len(segment_idx) - len(episode_idx)} episodes not found in the source index")
    else:
        episode_idx = ds.sql_select(
            *queries, shard_subsample=shard_subsample, key_column=spec.key_column, shard_filter=spec.shard_filter
        )
        episode_idx = episode_idx.with_columns(
            pl.col("audio_duration").cast(pl.Float32),
            speech_duration=pl.lit(None).cast(pl.Float32()),
            shard=pl.col("__shard_path__").str.extract(r"([^/]+).wsds$", 1),
        )

    _write_episode_list(out_file, episode_idx, fields)
    print(f"Extracted {len(episode_idx)} episodes from {ds.dataset_root} in {time.perf_counter() - start:.1f}s")
    return episode_idx


def extract_partition_index(
    partition_dir: Path | str,
    specs: tuple[SubdatasetSpec, ...] = DEFAULT_SPECS,
    overwrite: bool = False,
) -> tuple[str, str | None, str | None, str | None]:
    """
    Extract episode indices for all subdatasets of a single partition directory.

    Specs are processed in order, so a segmented spec can use the episode index
    of its `source_kind` extracted earlier in the same call.

    Returns:
        Tuple of (partition_dir, error_message, exception_repr, traceback_str) - error fields are None on success
    """
    partition_dir = Path(partition_dir)
    extracted: dict[str, pl.DataFrame] = {}

    for spec in specs:
        ds_path = partition_dir / spec.kind
        if not ds_path.exists():
            return str(partition_dir), f"error: {spec.kind} not found", None, None

        try:
            source_idx = None
            if spec.segmented and spec.source_kind:
                source_idx = extracted.get(spec.source_kind)
                if source_idx is None:
                    src_file = partition_dir / spec.source_kind / "episode-list.feather"
                    if src_file.exists():
                        source_idx, _fields = _read_episode_list(src_file)

            extracted[spec.kind] = extract_subdataset_index(ds_path, spec, source_idx=source_idx, overwrite=overwrite)
        except Exception as e:
            return str(partition_dir), f"error extracting {spec.kind} episodes", repr(e), traceback.format_exc()

    return str(partition_dir), None, None, None


def merge_partition_indices(
    partitions: list[Path | str],
    spec: SubdatasetSpec,
    dest: Path | str,
    duplicate_tolerance: float = 0.01,
) -> tuple[str, list[tuple[str, str, str | None, str | None]]]:
    """
    Merge cached episode extracts for `spec` across partitions into a wsds SQLite index.

    Duplicate episode names are resolved deterministically (the first occurrence in
    partition list order wins); duplicates whose audio durations differ by more than
    `duplicate_tolerance` seconds are reported in the returned error list.

    Args:
        partitions: List of partition directories (each containing a `spec.kind` subdir)
        spec: The subdataset to merge
        dest: Directory the index is written into (e.g. the subdataset root itself)

    Returns:
        Tuple of (dest, errors) where errors is a list of
        (file_path, error_message, exception_repr, traceback_str) tuples.
    """
    start = time.perf_counter()
    dst = Path(dest)
    print(f"Merging {spec.kind} to {dst}:")

    episode_idxs = []
    shard_idxs = []
    errors = []
    merged_fields = {}
    size = 0
    n_shards = 0

    for partition in partitions:
        ds_path = Path(partition) / spec.kind
        idx_file = ds_path / "episode-list.feather"
        if not idx_file.exists():
            errors.append((str(idx_file), "missing file", None, None))
            continue
        size += idx_file.stat().st_size

        try:
            episode_idx, fields = _read_episode_list(idx_file)
        except Exception as e:
            errors.append((str(idx_file), "read error", repr(e), traceback.format_exc()))
            continue

        if fields is None:
            # legacy extracts keep the field mapping in a sidecar file
            try:
                with open(ds_path / "fields.json") as f:
                    fields = json.load(f)
            except FileNotFoundError:
                errors.append((str(idx_file), "missing fields (no embedded metadata or fields.json)", None, None))
                fields = {}

        # create shard index
        shard_idx = make_shard_idx(
            episode_idx,
            n_samples_expr=pl.sum("segments").alias("n_samples")
            if spec.segmented
            else pl.len().alias("n_samples"),
            partition=os.path.relpath(ds_path, dst),
            shard_id_offset=n_shards,
        )
        n_shards += len(shard_idx)
        # replace shard names with unique indices
        episode_idx = episode_idx.rename({"__key__": "name"}).join(shard_idx.select("shard", "shard_id"), on="shard")
        episode_idxs.append(episode_idx)
        shard_idxs.append(shard_idx)

        merge_field_errors = [k for k, v in fields.items() if merged_fields.setdefault(k, v) != v]
        if merge_field_errors:
            errors.append((str(idx_file), "error merging fields", None, ", ".join(merge_field_errors)))

    if not episode_idxs:
        details = "; ".join(f"{path}: {msg}" for path, msg, _, _ in errors)
        raise ValueError(f"no readable {spec.kind} episode extracts in {len(partitions)} partition(s): {details}")

    # vertical_relaxed coerces to a common supertype so a merge can combine cached
    # extracts written by different code versions (e.g. Float32 vs Float64 speech_duration)
    # without crashing; for same-version extracts the schemas match and this is a no-op.
    merged_episode_idx = pl.concat(episode_idxs, how="vertical_relaxed").select(
        "name", "shard_id", "offset", "audio_duration", "speech_duration"
    )
    deduped = merged_episode_idx.unique(subset=["name"], keep="first", maintain_order=True)
    if len(deduped) < len(merged_episode_idx):
        duplicates = merged_episode_idx.filter(pl.col("name").is_duplicated())
        conflicts = (
            duplicates.group_by("name")
            .agg((pl.col("audio_duration").max() - pl.col("audio_duration").min()).alias("spread"))
            .filter(pl.col("spread") > duplicate_tolerance)
        )
        print(
            f"Dropping {len(merged_episode_idx) - len(deduped)} duplicate episodes "
            "(the first occurrence in partition order wins)"
        )
        if len(conflicts):
            errors.append(
                (
                    str(dst),
                    "conflicting duplicate episodes",
                    None,
                    f"{len(conflicts)} duplicated episodes have audio durations differing by more than "
                    f"{duplicate_tolerance}s, e.g.: " + ", ".join(conflicts["name"].head(10).to_list()),
                )
            )
    merged_episode_idx = deduped.sort("name")

    merged_shard_idx = pl.concat(shard_idxs, how="vertical_relaxed").with_columns(
        global_offset=pl.col("n_samples").cum_sum() - pl.col("n_samples"),
    )

    print(
        f"Merged {len(merged_episode_idx)} {spec.kind} episodes ({size / 1024 / 1024:.1f} MB) for {dst} in {time.perf_counter() - start:.2f} s"
    )

    dst.mkdir(exist_ok=True, parents=True)

    source_rel = f"../{spec.source_kind}" if spec.source_kind else None

    try:
        start = time.perf_counter()
        write_index(
            dst,
            merged_shard_idx,
            merged_episode_idx,
            merged_fields,
            vad_column=spec.vad_column if spec.segmented else None,
            source_path=source_rel,
            segmented=spec.segmented,
        )
        print(f"Saved index to {dst} in {time.perf_counter() - start:.2f} s")

    except Exception as e:
        errors.append((str(dst), "error saving index", repr(e), traceback.format_exc()))

    print("Skipped these indices due to errors:")
    for path, error, exc_repr, tb in errors:
        print("    ", path, "-", error, exc_repr or "")

    with open(dst / "indexing.log", "w") as f:
        for path, error, exc_repr, tb in errors:
            f.write(f"{path} - {error}")
            if exc_repr:
                f.write(f" {exc_repr}")
            f.write("\n")
            if tb:
                f.write(tb + "\n")

    return str(dst), errors


#
# Backwards-compatible wrappers for the original data-pl delivery/batch layout
#
def extract_batch_index(
    batch_path: Path | str, overwrite: bool = False
) -> tuple[str, str | None, str | None, str | None]:
    """
    Extract episode indices from a single batch directory containing
    'source' and 'filtered_vad' subdatasets.
    """
    return extract_partition_index(batch_path, DEFAULT_SPECS, overwrite=overwrite)


def merge_batch_indices(
    batches: list[Path | str],
    dataset_kind: str,
    dest_path: Path | str,
) -> tuple[str, list[tuple[str, str, str | None, str | None]]]:
    """
    Merge episode indices from multiple batches into a single wsds index
    written to `dest_path/dataset_kind`.
    """
    spec = {s.kind: s for s in DEFAULT_SPECS}[dataset_kind]
    return merge_partition_indices(batches, spec, Path(dest_path) / dataset_kind)
