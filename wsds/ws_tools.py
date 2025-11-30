import functools
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import webdataset as wds

from . import WSSample, WSSink

commands = {}


def command(name_or_fun):
    name = None

    def decorator(f):
        commands[name or f.__name__] = f
        return f

    if isinstance(name_or_fun, str):
        name = name_or_fun
        return decorator
    else:
        return decorator(name_or_fun)


@command("list")
def _list(input_shard: str):
    """Lists keys in a wsds dataset or shard."""
    if (Path(input_shard) / "index.sqlite3").exists():
        # FIXME: implement keys
        pass
    else:
        has_invalid_batches = False
        reader = pa.RecordBatchFileReader(pa.memory_map(input_shard))
        batch_size = int(reader.schema.metadata[b'batch_size'])
        try:
            for i in range(reader.num_record_batches):
                b = reader.get_batch(i)
                if b.num_rows != batch_size and i != reader.num_record_batches - 1:
                    sys.stderr.write(f"Batch {i} has {b.num_rows} rows instead of {batch_size}\n")
                    has_invalid_batches = True
                for key in b["__key__"]:
                    print(key)
        except BrokenPipeError:
            pass
        if has_invalid_batches:
            sys.exit(1)


def inspect_dataset(input_path, verbose=True):
    """Displays metadata and schema of a wsds dataset."""
    from . import WSDataset

    ds = WSDataset(input_path)
    print(ds)
    if verbose:
        print("Metadata:")
        for k, v in ds.index.metadata.items():
            print(f"{k}: {v}")
        print()
        print("One sample:")
        for x in ds:
            print(x)
            break


def inspect_shard(input_path):
    reader = pa.RecordBatchFileReader(pa.memory_map(str(input_path)))
    print(f"Batches: {reader.num_record_batches}")
    print(f"Rows: {int(reader.schema.metadata[b'batch_size']) * reader.num_record_batches}")
    print(f"Schema:\n{reader.schema}")


@command
def inspect(input_path: str):
    """Displays metadata and schema of a wsds dataset or shard."""
    if Path(input_path).is_dir():
        if (Path(input_path) / "index.sqlite3").exists():
            inspect_dataset(input_path)
        else:
            segmentations = [x for x in Path(input_path).iterdir() if (x / "index.sqlite3").exists()]
            if segmentations:
                print(f"Found {len(segmentations)} segmentations:")
                print()
            for file in segmentations:
                inspect_dataset(str(file), verbose=False)
            if not segmentations:
                for shard in Path(input_path).glob("*.wsds"):
                    print(f"Inspecting first shard: {shard}")
                    inspect_shard(shard)
                    break
                else:
                    print("Nothing to inspect here...")
    elif input_path.endswith(".wsds"):
        inspect_shard(input_path)


def print_head(shard: str, n: int = 5):
    reader = pa.ipc.open_file(shard)
    batch = reader.get_batch(0)
    table = batch.to_pandas()
    print(table.head(n))


@command
def head_shard(input_path: str, n: int = 5):
    """Displays metadata and schema of a wsds dataset or shard."""
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas.io.formats.format")

    if Path(input_path).is_dir():
        for shard in Path(input_path).glob("*.wsds"):
            print(f"Inspecting first shard: {shard}")
            print_head(shard, n)
            break
    elif input_path.endswith(".wsds"):
        print_head(input_path, n)


@command
def dump_index(source_dataset: Path):
    """Dump the index of the given dataset in a readable format."""
    from . import WSDataset

    ds = WSDataset(source_dataset)

    try:
        for sample in ds.index.query(
            "SELECT name,s.shard,offset FROM files AS f, shards AS s WHERE s.shard_id == f.shard_id ORDER BY name,s.shard,offset;"
        ):
            print(*sample)
    except BrokenPipeError:
        pass


@command
class validate:
    @command("validate_shards")  # backwards compatibility
    @staticmethod
    def shards(dataset: Path, verbose=False, complete_in_progress=False):
        """Validate if subdirs have all the shards and if all their schemas match."""
        from .utils import list_all_shards
        from .ws_sink import indented

        shard_names = list_all_shards(dataset, verbose=True, print_missing=False)
        print()
        for subdir in Path(dataset).iterdir():
            if not subdir.is_dir():
                continue
            shards = [(Path(dataset_path) / subdir / shard).with_suffix(".wsds") for dataset_path, shard in shard_names]
            schemas = {shard: get_shard_schema(shard) for shard in shards}
            unique = set(s for s in schemas.values() if s)
            if len(unique) > 1:
                print(f"Found schema conflicts for {subdir}:\n")
                for schema in unique:
                    matching_shards = [shard for shard, shard_schema in schemas.items() if schema == shard_schema]
                    prefix = f"  in {len(matching_shards)} shards: "
                    print(indented(prefix, schema))
                    if verbose:
                        for shard in matching_shards:
                            print(indented(" " * len(prefix), shard))
                        print()
            if None not in schemas.values() and complete_in_progress:
                os.rename(subdir, str(subdir).replace('.in-progress', ''))

    @staticmethod
    def load_test_yaml(test_yaml_path: Path):
        """Load the expected artifacts, keys, and datatypes from test.yaml"""
        import yaml

        with open(test_yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("artifacts", {})

    @command("validate_artifacts")
    @staticmethod
    def validate_artifacts(dataset: Path, test_yaml: Path, verbose=False, check_index=False, n_shards: int = None):
        """
        Validate that:
        1. The dataset has a valid index and loads correctly.
        2. All expected artifacts exist as subdirectories.
        3. Each artifact's .wsds shards contain the correct columns.
        4. Each column matches the expected datatype defined in test.yaml.
        """
        from tqdm import tqdm

        from . import WSDataset

        RED = "\033[0;31m"
        GREEN = "\033[0;32m"
        YELLOW = "\033[1;33m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        dataset = Path(dataset)
        expected_artifacts = validate.load_test_yaml(test_yaml)

        if check_index:
            print(f"\n{BOLD}Running quick dataset check for:{RESET} {dataset}\n")

            index_path = dataset / "index.sqlite3"
            if not index_path.exists():
                print(f"{RED}✗ Missing index.sqlite3 at {index_path}{RESET}")
            else:
                print(f"{GREEN}✓ Found index.sqlite3{RESET}")

            try:
                ds = WSDataset(dataset)
                sample = ds.random_sample()
                print(f"{GREEN}✓ WSDataset loaded successfully{RESET}")
                print(f"Sample keys: {list(sample.keys())[:8]} ...\n")
                print(sample)
            except Exception as e:
                print(f"{RED}✗ Failed to load WSDataset: {e}{RESET}")
                print(f"{YELLOW}Skipping deeper validation for {dataset}{RESET}\n")
                return

        existing_subdirs = {d.name for d in dataset.iterdir() if d.is_dir()}
        expected_subdirs = set(expected_artifacts.keys())

        missing_artifacts = expected_subdirs - existing_subdirs
        extra_artifacts = existing_subdirs - expected_subdirs

        if missing_artifacts:
            print(f"{RED}Missing artifacts:{RESET} {', '.join(missing_artifacts)}")
        if extra_artifacts:
            print(f"{YELLOW}Extra artifacts found (not in YAML):{RESET} {', '.join(extra_artifacts)}")

        # --- Step 2: Validate keys and datatypes ---
        for artifact_name, artifact_spec in expected_artifacts.items():
            artifact_path = dataset / artifact_name
            if not artifact_path.exists():
                print(f"{YELLOW}Skipping missing artifact:{RESET} {artifact_name}")
                continue

            if "columns_data_types" in artifact_spec:
                expected_types = artifact_spec["columns_data_types"]
                expected_keys = set(expected_types.keys())

            shard_files = sorted(artifact_path.glob("*.wsds"))
            if n_shards is not None:
                shard_files = shard_files[:n_shards]

            artifact_failed = False

            for shard_file in tqdm(shard_files, desc=f"Validating {artifact_name}"):
                try:
                    import pyarrow as pa

                    reader = pa.ipc.open_file(shard_file)
                    schema = reader.schema
                    columns = {f.name: str(f.type) for f in schema}
                    actual_keys = set(columns.keys())

                except Exception as e:
                    artifact_failed = True
                    print(f"{RED}Could not read schema from {shard_file}: {e}{RESET}")
                    continue

                # --- Compare keys ---
                missing_keys = expected_keys - actual_keys
                extra_keys = actual_keys - expected_keys

                if missing_keys:
                    artifact_failed = True
                    print(f"{RED}{shard_file}: Missing keys {missing_keys}{RESET}")
                if extra_keys and verbose:
                    print(f"{YELLOW}{shard_file}: Extra keys {extra_keys}{RESET}")

                # --- Compare datatypes ---
                for key, expected_type in expected_types.items():
                    if key not in columns:
                        continue
                    actual_type = columns[key]
                    if expected_type.lower() not in actual_type.lower():
                        artifact_failed = True
                        print(
                            f"{RED}{shard_file}: Key '{key}' type mismatch "
                            f"(expected {expected_type}, got {actual_type}){RESET}"
                        )

            if artifact_failed:
                print(f"{RED}{BOLD}✗ {artifact_name} failed validation{RESET}")
            else:
                print(f"{GREEN}{BOLD}✓ {artifact_name} passed validation{RESET}")

        print(f"\n{BOLD}Validation complete.{RESET}\n")

    @staticmethod
    def keys(dataset: Path, verbose=False, skip_audio=True):
        """Validate __key__s against the index for all the shards in the dataset."""
        from collections import defaultdict

        from tqdm import tqdm

        from . import WSDataset

        dataset = Path(dataset)
        if next(dataset.iterdir()).suffix == ".wsds":
            subdirs = [dataset]
            dataset = dataset.parent
        else:
            subdirs = list(dataset.iterdir())
            if skip_audio:
                subdirs = [dir for dir in subdirs if dir.name != "audio"]

        ds = WSDataset(dataset)
        shards = ds.get_shard_list()
        missing_shards = defaultdict(int)
        for shard in tqdm(shards, desc=str(dataset)):
            expected_keys = generate_all_keys_for_shard(ds.index, shard)
            for subdir in subdirs:
                if not subdir.is_dir():
                    continue
                shard_fname = (subdir / shard).with_suffix(".wsds")
                if not shard_fname.exists():
                    missing_shards[subdir] += 1
                else:
                    if not pl.scan_ipc(shard_fname).select((pl.col("__key__") == expected_keys).all()).collect().item():
                        tqdm.write(f"Shard {shard} in {subdir} has keys that don't match the index.")
                    reader = pa.RecordBatchFileReader(pa.memory_map(str(shard_fname)))
                    batch_size = int(reader.schema.metadata[b'batch_size'])
                    for i in range(reader.num_record_batches - 1):
                        batch = reader.get_batch(i)
                        if len(batch) != batch_size:
                            tqdm.write(f"Batch {i} in shard {shard} in {subdir} has incorrect length.")
        for subdir, count in missing_shards.items():
            tqdm.write("")
            tqdm.write(f"{subdir}: missing {count} shards")

    @staticmethod
    def all(base_path, skip_audio=True):
        from tqdm import tqdm

        base_path = Path(base_path)
        all_segmentations = []
        for base_dataset in base_path.iterdir():
            if not base_dataset.is_dir():
                continue
            for segmentation in base_dataset.iterdir():
                if not segmentation.is_dir():
                    continue
                all_segmentations.append(segmentation)

        for segmentation in tqdm(all_segmentations):
            if not (segmentation / "index.sqlite3").exists():
                tqdm.write(f"{segmentation}: Missing index.sqlite3")
                continue
            subdirs = [d for d in segmentation.iterdir() if d.is_dir()]
            if not subdirs:
                tqdm.write(f"{segmentation}: Empty dataset!")
            validate.keys(segmentation, skip_audio=skip_audio)


def get_shard_schema(fname):
    fname = Path(fname)
    if not fname.exists():
        return None
    if fname.stat().st_size == 0:
        return None
    return repr(pa.RecordBatchFileReader(pa.memory_map(str(fname))).schema).split("-- schema metadata --")[0]


def generate_all_keys_for_shard(index, shard):
    N, shard_id = index.query("SELECT n_samples, shard_id FROM shards WHERE shards.shard = ?", shard).fetchone()
    files = index.query("SELECT name, offset FROM files WHERE files.shard_id == ?", shard_id).fetchall()
    df = pl.DataFrame(files, schema=["name", "offset"], orient="row")
    if not index.metadata["segmented"]:
        return df["name"]
    return (
        df.with_columns(N=pl.col("offset").extend_constant(N, 1).diff(null_behavior="drop"))
        .with_columns(seq=pl.arange("N").over("name", mapping_strategy="join"))
        .explode("seq")
        .select(__key__=pl.format("{}_{}", "name", pl.col("seq").cast(pl.String).str.zfill(3)))["__key__"]
    )


@command
def init(
    new_dataset: Path,
    source_dataset: Path | None = None,
    vad_column: str | None = None,
    num_workers: int = 32,
):
    """Initialize a new dataset, from scratch or from a segmentation of an existing one."""
    import multiprocessing

    from fastprogress import progress_bar

    from . import AtomicFile, WSDataset
    from .ws_index import WSDSIndexWriter

    new_dataset = Path(new_dataset)

    if source_dataset is not None:
        assert vad_column is not None, "vad_column must be specified when initializing from a source dataset"
    else:
        source_dataset = new_dataset

    ds = WSDataset(source_dataset)
    shard_extractor = functools.partial(extract_index_for_shard, source_dataset, vad_column=vad_column)
    all_shards = ds.get_shard_list(ignore_index = True)

    with AtomicFile(new_dataset / "index.sqlite3") as fname:
        with WSDSIndexWriter(fname) as index:
            with multiprocessing.Pool(num_workers) as p:
                for r in progress_bar(p.imap_unordered(shard_extractor, all_shards), total=len(all_shards)):
                    r["dataset_path"] = ""
                    try:
                        index.append(r)
                    except:
                        print("Failed to append records to index:", r)
                        raise

            index.append_metadata({"segmented": True if vad_column else False})

        if vad_column:
            with AtomicFile(new_dataset / "audio.wsds-link") as fname:
                with open(fname, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "dataset_dir": os.path.relpath(source_dataset, new_dataset),
                                "loader": ["wsds.ws_shard", "WSSourceAudioShard"],
                                "vad_column": vad_column,
                            }
                        )
                    )

@command
def init_split(
    splits_path: Path,
    new_dataset: Path,
    source_dataset: Path | None = None,
    vad_column: str | None = None,
    num_workers: int = 64,
    include_in_progress: bool = False,
):
    """Initialize a new dataset, from scratch or from a segmentation of an existing one."""
    if source_dataset is not None:
        assert vad_column is not None, "vad_column must be specified when initializing from a source dataset"
    else:
        source_dataset = new_dataset

    import multiprocessing

    from fastprogress import progress_bar

    from . import AtomicFile, WSDataset
    from .ws_index import WSDSIndexWriter

    with AtomicFile("index.sqlite3") as fname:
        with WSDSIndexWriter(fname) as index:

            splits = [x for x in Path(splits_path).iterdir() if x.is_dir()]

            for split in progress_bar(splits):
                ds = WSDataset(Path(split) / source_dataset)
                shard_extractor = functools.partial(extract_index_for_shard, Path(split) / source_dataset, vad_column=vad_column)
                all_shards = ds.get_shard_list(ignore_index = True)

                try:
                    with multiprocessing.Pool(num_workers) as p:
                        for r in p.imap_unordered(shard_extractor, all_shards):
                            r["dataset_path"] = Path(split) / new_dataset
                            try:
                                index.append(r)
                            except:
                                print("Failed to append records to index:", r)
                                raise
                except KeyError as err:
                    print(err)
                    print("Dataset fields:", ds.fields)

            index.append_metadata({"segmented": True if vad_column else False})
            ds = WSDataset(Path(splits[0]) / new_dataset)
            new_fields = {k:v for k,v in ds.fields.items() if v[1] not in ['sample_source_id', 'src_key']}
            if vad_column:
                index.append_metadata({"computed_columns": {
                    "audio.wsds-computed": {
                        "dataset_dir": os.path.relpath(source_dataset, new_dataset),
                        "loader": ["wsds.ws_shard", "WSSourceAudioShard"],
                        "vad_column": vad_column,
                    }
                }})
                new_fields['audio'] = ("audio.wsds-computed", "audio")
            index.append_metadata({"fields": new_fields})

def extract_index_for_shard(dataset, shard, vad_column=None):
    from . import WSDataset

    ds = WSDataset(dataset)
    index = []
    i = 0

    for s in ds.iter_shard(shard):
        key = s["__key__"]

        if not vad_column:
            n = 1
            speech_duration = -1
        else:
            vad = s[vad_column]
            n = len(vad)
            speech_duration = 0
            if vad.size > 0:
                speech_duration = float((vad[:, -1] - vad[:, -2]).sum())  # tend - tstart

        audio_duration = s['load_duration'] or s['est_duration'] or -1

        if (
            n > 0
        ):  # in derived datasets, skip files with no vad segments (they won't have samples and will never appear as keys)
            index.append((key, i, audio_duration, speech_duration))

        i += n
    return {
        "shard_name": shard[1],
        "index": index,
        "n_samples": i,
    }


@command
def _remove_columns(*fnames, remove: str = ""):
    """
    Remove one or more columns from .wsds shard files if they exist.

    Example:
        wsds _remove_columns /path/to/shards/*.wsds --remove transcription_parakeet-tdt-0-6b-v3_raw.txt
        wsds _remove_columns /path/to/shards/*.wsds --remove col1,col2,col3
    """
    import pyarrow as pa
    import tqdm

    from .ws_sink import AtomicFile

    remove_cols = [r.strip() for r in remove.split(",") if r.strip()]
    if not remove_cols:
        raise ValueError("You must specify at least one column to remove via --remove")

    for fname in tqdm.tqdm(fnames, desc=f"Removing {remove_cols}"):
        reader = pa.ipc.open_file(fname)
        table = reader.read_all()

        cols_to_drop = [c for c in table.column_names if c in remove_cols]
        if not cols_to_drop:
            continue

        table2 = table.drop(cols_to_drop)

        if table.schema != table2.schema:
            with AtomicFile(fname) as tmp:
                with pa.ipc.new_file(tmp, table2.schema.with_metadata(reader.schema.metadata)) as sink:
                    sink.write_table(table2)
