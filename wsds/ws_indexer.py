"""
Core wsds/Polars indexing logic for extracting and merging episode indices.
"""

import json
import os
import time
import traceback
from pathlib import Path

import polars as pl

from wsds import AtomicFile, WSDataset
from wsds.ws_index import WSDSIndexWriter


def extract_episodes(episode_idx: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate segment-level data into episode-level data.

    Takes a DataFrame with segment keys (e.g. "episode_123_0", "episode_123_1")
    and aggregates them into episodes by extracting the base key and summing durations.
    """
    return (episode_idx
        .with_columns(
            pl.col('__key__').str.extract(r"(.*)_[0-9]+$", 1),
            shard = pl.col('__shard_path__').str.extract(r"([^/]+).wsds$", 1),
        )
        .group_by('__key__', maintain_order=True)
        .agg(
            pl.sum('speech_duration'),
            pl.len().alias('segments'),
            pl.first('shard'),
            pl.first('offset'),
        )
    )


def make_shard_idx(
    sample_idx: pl.DataFrame,
    n_samples_expr: pl.Expr,
    dataset_path: Path | str,
    shard_id_offset: int = 0
) -> pl.DataFrame:
    """
    Create a shard index from a sample/episode index.

    Groups samples by shard and computes aggregate statistics.
    """
    return (sample_idx
        .group_by('shard', maintain_order=True)
        .agg(
            n_samples_expr,
            pl.sum('audio_duration')
        )
        .with_row_index('shard_id', offset=shard_id_offset)
        .with_columns(
            dataset_path = pl.lit(str(dataset_path)),
        )
    )


def write_index(
    path: Path | str,
    shard_idx: pl.DataFrame,
    episode_idx: pl.DataFrame,
    fields: dict,
    source_path: str | None = None,
    vad_column: str | None = None
):
    """
    Write a wsds SQLite index file with shard and episode data.

    Args:
        path: Directory to write index.sqlite3 to
        shard_idx: DataFrame with shard information
        episode_idx: DataFrame with episode/file information
        fields: Field mapping dictionary
        source_path: Path to source dataset (for computed audio columns)
        vad_column: VAD column name (for segmented datasets)
    """
    audio_duration, speech_duration = episode_idx.select('audio_duration', 'speech_duration').sum().row(0)
    with AtomicFile(f'{path}/index.sqlite3') as fname:
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
                metadata['segmented'] = True
            else:
                metadata['segmented'] = False
            metadata.update({"fields": fields, 'audio_duration': audio_duration, 'speech_duration': speech_duration})
            index.append_metadata(metadata)

        conn = dict(connection=f'sqlite:///{fname}', if_table_exists='append', engine='adbc')
        shard_idx.drop('audio_duration').write_database(table_name='shards', **conn)
        episode_idx.with_columns(pl.col('speech_duration').fill_null(-1)).write_database(table_name='files', **conn)


def extract_batch_index(
    batch_path: Path | str,
    overwrite: bool = False
) -> tuple[str, str | None, str | None, str | None]:
    """
    Extract episode indices from a single batch directory.

    Processes both 'source' and 'filtered_vad' subdatasets, creating
    episode-list.feather files for each.

    Args:
        batch_path: Path to batch directory containing 'source' and 'filtered_vad'
        overwrite: If True, regenerate indices even if they exist

    Returns:
        Tuple of (batch_path, error_message, exception_repr, traceback_str) - error fields are None on success
    """
    batch = Path(batch_path)

    # Process source dataset
    ds_path = batch / 'source'
    out_file = ds_path / 'episode-list.feather'

    if out_file.exists() and not overwrite:
        print(f"Skipping, {out_file} already exists")
        try:
            source_idx = pl.read_ipc(out_file, memory_map=False)
        except Exception as e:
            return str(batch), "error reading source index:", repr(e), traceback.format_exc()
    else:
        if not ds_path.exists():
            return str(batch), "error: source not found", None, None

        try:
            start = time.perf_counter()
            source_ds = WSDataset(ds_path, ignore_index=True)
        except Exception as e:
            return str(batch), "error initializing source dataset", repr(e), traceback.format_exc()

        print(f"Loaded dataset {source_ds.dataset_dir} in {time.perf_counter() - start:.1f}s")

        # Update fields.json
        fields = {}
        for k, v in source_ds.fields.items():
            if isinstance(v[0], str) and v[1] in ["sample_source_id", "src_key"]:
                continue
            fields[k] = v
        with AtomicFile(ds_path / 'fields.json') as fname:
            with open(fname, 'w') as f:
                json.dump(fields, f)

        try:
            start = time.perf_counter()
            source_idx = (
                source_ds
                .sql_select('__key__', 'load_duration AS audio_duration', '__shard_path__', '__shard_offset__ AS offset', shard_subsample=1)
                .with_columns(
                    pl.col('audio_duration').cast(pl.Float32),
                    speech_duration=pl.lit(None).cast(pl.Float32()),
                    shard=pl.col('__shard_path__').str.extract(r"([^/]+).wsds$", 1),
                )
            )
            source_idx.write_ipc(batch / 'source/episode-list.feather', compression='zstd')

            print(f"Extracted {len(source_idx)} episodes from {source_ds.dataset_dir} in {time.perf_counter() - start:.1f}s")
        except Exception as e:
            return str(batch), "error extracting source episodes", repr(e), traceback.format_exc()

    # Process filtered_vad dataset
    ds_path = batch / 'filtered_vad'
    out_file = ds_path / 'episode-list.feather'

    if out_file.exists() and not overwrite:
        print(f"Skipping, {out_file} already exists")
    else:
        try:
            start = time.perf_counter()
            vad_ds = WSDataset(ds_path, ignore_index=True)
        except Exception as e:
            traceback.print_exc()
            print(f"Error initializing WSDataset at {batch / 'filtered_vad'}: {e}")
            return str(batch), "error initializing filtered_vad dataset", repr(e), traceback.format_exc()

        print(f"Loaded dataset {vad_ds.dataset_dir} in {time.perf_counter() - start:.1f}s")

        # Update fields.json
        fields = {}
        for k, v in vad_ds.fields.items():
            if isinstance(v[0], str) and v[1] in ["sample_source_id", "src_key"]:
                continue
            fields[k] = v
        with AtomicFile(ds_path / 'fields.json') as fname:
            with open(fname, 'w') as f:
                json.dump(fields, f)

        try:
            start = time.perf_counter()
            vad_idx = (
                vad_ds
                .sql_select('__key__', 'tend - tstart AS speech_duration', '__shard_path__', '__shard_offset__ AS offset', shard_subsample=1,
                    shard_pipe=extract_episodes)
                .join(source_idx['__key__', 'audio_duration'], on='__key__')
            )
            vad_idx.write_ipc(batch / 'filtered_vad/episode-list.feather', compression='zstd')

            print(f"Extracted {len(vad_idx)} episodes from {vad_ds.dataset_dir} in {time.perf_counter() - start:.1f}s")
        except Exception as e:
            return str(batch), "error extracting filtered_vad episodes", repr(e), traceback.format_exc()

    return str(batch), None, None, None


def merge_batch_indices(
    batches: list[Path | str],
    dataset_kind: str,
    dest_path: Path | str,
) -> tuple[str, list[tuple[str, str, str | None, str | None]]]:
    """
    Merge episode indices from multiple batches into a single wsds index.

    Args:
        batches: List of batch directory paths
        dataset_kind: 'source' or 'filtered_vad'
        dest_path: Destination directory for merged index

    Returns:
        Tuple of (dest_path, errors) where errors is a list of
        (file_path, error_message, exception_repr, traceback_str) tuples.
    """
    start = time.perf_counter()
    print(f"Merging to {dest_path}:")
    dst = Path(dest_path) / dataset_kind

    episode_idxs = []
    shard_idxs = []
    errors = []
    merged_fields = {}
    size = 0
    n_shards = 0

    for batch in batches:
        ds_path = Path(batch) / dataset_kind
        idx_file = ds_path / 'episode-list.feather'
        if idx_file.exists():
            size += idx_file.stat().st_size

            try:
                episode_idx = pl.read_ipc(idx_file, memory_map=False)
            except Exception as e:
                errors.append((str(idx_file), "read error", repr(e), traceback.format_exc()))
                continue

            # create shard index
            shard_idx = make_shard_idx(
                episode_idx,
                n_samples_expr=pl.len().alias('n_samples') if dataset_kind == 'source' else pl.sum('segments').alias('n_samples'),
                dataset_path=os.path.relpath(ds_path, dst),
                shard_id_offset=n_shards,
            )
            n_shards += len(shard_idx)
            # replace shard names with unique indices
            episode_idx = (episode_idx
                .rename({'__key__': 'name'})
                .join(shard_idx.select('shard', 'shard_id'), on='shard')
            )
            episode_idxs.append(episode_idx)
            shard_idxs.append(shard_idx)

            merge_field_errors = []
            with open(ds_path / 'fields.json') as f:
                for k, v in json.load(f).items():
                    if k not in merged_fields:
                        merged_fields[k] = v
                    else:
                        if v != merged_fields[k]:
                            merge_field_errors.append(k)
            if merge_field_errors:
                errors.append((str(idx_file), f"error merging fields", None, ', '.join(merge_field_errors)))
        else:
            errors.append((str(idx_file), "missing file", None, None))

    merged_episode_idx = (
        pl.concat(episode_idxs)
        .unique(subset=['name'])
        .sort('name')
        .select('name', 'shard_id', 'offset', 'audio_duration', 'speech_duration')
    )
    merged_shard_idx = (
        pl.concat(shard_idxs)
        .with_columns(
            global_offset=pl.col('n_samples').cum_sum() - pl.col('n_samples'),
        )
    )

    print(f"Merged {len(merged_episode_idx)} {dataset_kind} episodes ({size/1024/1024:.1f} MB) for {dest_path} in {time.perf_counter() - start:.2f} s")

    start = time.perf_counter()
    dst.mkdir(exist_ok=True, parents=True)

    merged_episode_idx.write_ipc(dst / 'episode-index.feather')
    merged_shard_idx.write_ipc(dst / 'shard-index.feather')
    print(f"Saved feather indices to {dst} in {time.perf_counter() - start:.2f} s")

    try:
        start = time.perf_counter()
        write_index(
            dst, merged_shard_idx, merged_episode_idx, merged_fields,
            vad_column='vad.npy' if dataset_kind == 'filtered_vad' else None,
            source_path='../source')
        print(f"Saved index to {dst} in {time.perf_counter() - start:.2f} s")

    except Exception as e:
        errors.append((str(dst), "error saving index", repr(e), traceback.format_exc()))

    print("Skipped these indices due to errors:")
    for path, error, exc_repr, tb in errors:
        print("    ", path, "-", error, exc_repr or "")

    with open(dst / 'indexing.log', 'w') as f:
        for path, error, exc_repr, tb in errors:
            f.write(f"{path} - {error}")
            if exc_repr:
                f.write(f" {exc_repr}")
            f.write("\n")
            if tb:
                f.write(tb + "\n")

    return str(dst), errors
