import functools
import json
import os
import tarfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
# import webdataset as wds

from wsds import WSSample, WSSink

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
        reader = pa.RecordBatchFileReader(pa.memory_map(input_shard))
        try:
            for i in range(reader.num_record_batches):
                for key in reader.get_batch(i)["__key__"]:
                    print(key)
        except BrokenPipeError:
            pass


def inspect_dataset(input_path, verbose=True):
    """Displays metadata and schema of a wsds dataset."""
    from . import WSDataset

    ds = WSDataset(input_path)
    print(ds)
    if verbose:
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
def shard_from_webdataset(
    input_shard: str,  # input shard URL/path
    output_shard: str,  # output shard URL/path
    batch_size: int = 16,  # batch size
    compression: str = "zstd",
    min_batch_size_bytes: int = 1024 * 1024,
    no_keys: bool = False,
    yt_data_specific: bool = False,
    audio_requires_sorting: bool = False,
    mixed_audio: bool = False,
    check_audio: bool = False,
):
    """Converts a WebDataset shard into wsds format.

    Supports yt_data_specific mode with optional sorting.
    """
    import gzip
    import io
    from pathlib import Path

    import soundfile as sf

    from hume_wsds.constants import ShardMapping
    from hume_wsds.utils import cast_types_for_storage

    out_dir = Path(output_shard).parents[0].name
    if out_dir == "audio":
        compression = "no-compression"

    AUDIO_KEYS = ["m4a", "mp3", "wav", "flac", "ogg", "opus"]

    def process(stream):
        # yt audio data requiring sorting
        def list_keys_tarfile(input_shard):
            if input_shard.endswith("gz"):
                o = gzip.open
            else:
                o = open

            all_samples = defaultdict(dict)
            audio_entries = []

            f = o(input_shard, "rb")
            tar = tarfile.TarFile(fileobj=f)

            for member in tar.getmembers():
                if "/" in member.name:
                    path, name = member.name.rsplit("/", 1)
                else:
                    path, name = "", member.name  # fallback if at root level
                name = name.replace("_comments", ".comments")
                key, field = name.split(".", 1)
                sample_key = f"{path}/{key}"

                all_samples[sample_key][field] = member

                if field in AUDIO_KEYS:
                    audio_entries.append(sample_key)

            return tar, all_samples, audio_entries

        def is_audio_valid(audio_bytes: bytes) -> bool:
            try:
                with io.BytesIO(audio_bytes) as f:
                    with sf.SoundFile(f) as sf_desc:
                        _ = sf_desc.frames  # force metadata read
                return True
            except Exception as e:
                print(f"[Warning] Corrupt audio detected: {e}")
                return False

        if out_dir == "audio" and audio_requires_sorting:
            tar, samples, audio_entries = list_keys_tarfile(input_shard)

            for sample_key in audio_entries:
                fields = samples[sample_key]
                new_s = {}
                new_s["__key__"] = sample_key

                for ak in AUDIO_KEYS:
                    if ak in fields:
                        if mixed_audio:
                            audio_bytes = tar.extractfile(fields[ak]).read()
                            if check_audio and not is_audio_valid(audio_bytes):
                                print(f"[Skipping] corrupt audio in {sample_key}")
                                continue

                            new_s["audio"] = audio_bytes
                            new_s["audio_type"] = ak

                        else:
                            audio_bytes = tar.extractfile(fields[ak]).read()
                            if check_audio and not is_audio_valid(audio_bytes):
                                print(f"[Skipping] corrupt audio in {sample_key}")
                                continue
                            new_s[ak] = audio_bytes

                if yt_data_specific:
                    for meta in ["info.json", "description", "comments.json"]:
                        if meta in fields:
                            try:
                                new_s[meta] = tar.extractfile(fields[meta]).read().decode("utf-8", errors="ignore")
                            except Exception:
                                new_s[meta] = ""
                        else:
                            new_s[meta] = ""

                    vtts = {}
                    for k, v in fields.items():
                        if k.endswith(".vtt"):
                            try:
                                vtts[k] = tar.extractfile(v).read().decode("utf-8", errors="ignore")
                            except Exception:
                                continue
                    new_s["vtt"] = json.dumps(vtts) if vtts else "{}"

                yield new_s

        # regular processing
        else:
            for s in stream:
                new_s = {}

                def get_or_empty(field, as_text=True):
                    if field in s:
                        val = s[field]
                        if isinstance(val, bytes):
                            return val.decode("utf-8", errors="ignore") if as_text else val
                        return val
                    return "" if as_text else b""

                # process fields
                for k, v in s.items():
                    if k.endswith(".json"):
                        try:
                            v = json.loads(v)
                            k = k[: -len(".json")]
                            v = cast_types_for_storage(v, float_cast="float16", int_cast="int32")
                        except Exception:
                            pass

                    if k == "brouhaha_mean":
                        if isinstance(v, dict):
                            if "mean_c50" in v:
                                new_s["c50"] = float(v["mean_c50"])
                            if "mean_snr" in v:
                                new_s["snr"] = float(v["mean_snr"])
                        continue

                    if k == "vad_silero_diarized_continuous":

                        def to_npy_bytes(array):
                            buf = io.BytesIO()
                            np.save(buf, array, allow_pickle=False)
                            return buf.getvalue()

                        vad_array, pause_dur, pause_energy, speakers = [], [], [], []
                        for item in v:
                            try:
                                start, end, spk, dur, energy = item
                                vad_array.append([start, end])
                                pause_dur.append(dur)
                                pause_energy.append(energy)
                                speakers.append(spk)
                            except Exception as e:
                                print(f"[Warning] Skipping diarization entry: {e}")
                                continue

                        new_s["diarized.vad.npy"] = to_npy_bytes(np.array(vad_array, dtype=np.float32))
                        new_s["diarized.pause_dur.npy"] = to_npy_bytes(np.array(pause_dur, dtype=np.float32))
                        new_s["diarized.pause_energy.npy"] = to_npy_bytes(np.array(pause_energy, dtype=np.float32))
                        new_s["diarized.speaker.npy"] = to_npy_bytes(np.array(speakers))
                        continue

                    new_s[k] = v
                renamed = {}
                for k, v in new_s.items():
                    # Find (out_dir, k) in ShardMapping
                    # otherwise use the original key k as default
                    target_key = ShardMapping.get((out_dir, k), k)
                    renamed[target_key] = v

                yield renamed

    if compression == "no-compression":
        compression = None

    if out_dir == "audio" and (yt_data_specific or audio_requires_sorting):
        iterator = process(None)
    else:
        ds = wds.WebDataset([input_shard], shardshuffle=False).compose(process)
        iterator = iter(ds)

    with WSSink(
        output_shard,
        batch_size=batch_size,
        compression=compression,
        min_batch_size_bytes=min_batch_size_bytes,
    ) as sink:
        for i, x in enumerate(iterator):
            drop_keys(x, "__url__", "__local_path__")
            if no_keys and "__key__" in x:
                del x["__key__"]
            sink.write(dict(sorted(x.items())))


@command
def drop_keys(dict, *keys):
    """Remove specified keys from the given dictionary."""
    for key in keys:
        if key in dict:
            del dict[key]


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
    def shards(dataset: Path, verbose=False):
        """Validate if subdirs have all the shards and if all their schemas match."""
        from .utils import list_all_shards
        from .ws_sink import indented

        shard_names = list_all_shards(dataset, verbose=True, print_missing=False)
        print()
        for subdir in Path(dataset).iterdir():
            if not subdir.is_dir():
                continue
            shards = [(subdir / shard).with_suffix(".wsds") for shard in shard_names]
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
    require_audio_duration: bool = True,
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
    shard_extractor = functools.partial(
        extract_index_for_shard,
        source_dataset,
        vad_column=vad_column,
        require_audio_duration=require_audio_duration,
    )
    all_shards = ds.get_shard_list()

    with AtomicFile(new_dataset / "index.sqlite3") as fname:
        with WSDSIndexWriter(fname) as index:
            with multiprocessing.Pool(num_workers) as p:
                for r in progress_bar(p.imap_unordered(shard_extractor, all_shards), total=len(all_shards)):
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
                                "loader": ["hume_wsds.ws_shard", "WSSourceAudioShard"],
                                "vad_column": vad_column,
                            }
                        )
                    )


def extract_index_for_shard(dataset, shard, vad_column=None, require_audio_duration=True):
    from . import WSDataset

    ds = WSDataset(dataset)
    index = []
    i = 0
    if isinstance(shard, (tuple, list)) and len(shard) == 2:
        dataset_path, shard_name = shard
    else:
        dataset_path, shard_name = "", shard

    # instead of passing `shard` directly, wrap it
    # WSSample(ds, shard_name, offset=0)
    sample = WSSample(ds, (dataset_path, shard_name), 0)

    for s in ds.sequential_from(sample, 0):
        try:
            key = str(s["__key__"])
        except IndexError:
            # this can only happen if we don't have an index yet and we got a sample that's out of bounds
            # because of lazyness the error in WSSample only happens on first access
            break

        if not vad_column:
            n = 1
            speech_duration = -1
        else:
            vad = s[vad_column]
            n = len(vad)
            speech_duration = 0
            if vad.size > 0:
                speech_duration = float((vad[:, -1] - vad[:, -2]).sum())  # tend - tstart

        # import ipdb; ipdb.set_trace()
        if not require_audio_duration:
            audio_duration = -1.0
        else:
            try:
                audio_reader = s.get_audio()
                meta = audio_reader.metadata
                audio_duration = _duration_seconds_from_metadata(meta)
                if audio_duration is None:
                    raise ValueError("could not infer duration from audio metadata")
            except Exception as e:
                print("Audio loading error:", e)
                print("         for sample:", s)
                raise

        if (
            n > 0
        ):  # in derived datasets, skip files with no vad segments (they won't have samples and will never appear as keys)
            index.append((key, i, audio_duration, speech_duration))

        i += n
    return {
        "shard_name": shard_name,
        "dataset_path": dataset_path,
        "index": index,
        "n_samples": i,
    }


def _duration_seconds_from_metadata(meta):
    for attr in ("duration_seconds_from_header", "duration", "duration_seconds"):
        val = getattr(meta, attr, None)
        if val is not None:
            return float(val)
    num_frames = getattr(meta, "num_frames", None) or getattr(meta, "num_samples", None)
    sample_rate = getattr(meta, "sample_rate", None)
    if num_frames is not None and sample_rate:
        return float(num_frames) / float(sample_rate)
    return None


@command
def _sort_columns(*fnames, add_prefix: str | None = None, dry_run: bool = False):
    import pyarrow as pa
    from tqdm import tqdm

    from .ws_sink import AtomicFile

    _renames_general = {
        "parakeet-tdt-0-6b-v3.txt": "transcription_parakeet-tdt-0-6b-v3_raw.txt",
        "parakeet_wordlevel.txt": "words_parakeet-tdt-0-6b-v3_raw.txt",
        "raw.spk_emb.npy": "v4-vad_ws_continuous_mvad.raw.spk_emb.npy",
        "raw.subvads.pyd": "v4-vad_ws_continuous_mvad.raw.subvads.pyd",
        "raw.vad.npy": "v4-vad_ws_continuous_mvad.raw.vad.npy",
        "txt": "transcription_wslang_turbo_raw.txt",
        "m4a": "audio",
        "mp3": "audio",
    }
    # 16k dtoks
    _renames_16k = {
        "dtok_level_1_16k.txt": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.dtok_level_1_16k.npy",
        "dtok_v2_ml_50hz_32x16384_graphemes_key16k.dtok_level_1_16k.txt": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.dtok_level_1_16k.npy",
        "source_start_end_time.txt": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.source_start_end_time.npy",
        "boundary_shift_energy.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.boundary_shift_energy.npy",
        "dtok_level_1_16k.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.dtok_level_1_16k.npy",
        "source_start_end_time.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.source_start_end_time.npy",
        "vad_boundary_shift.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.vad_boundary_shift.npy",
        "vad_original_duration.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.vad_original_duration.npy",
        "vad_original_gap.npy": "dtok_v2_ml_50hz_32x16384_graphemes_key16k.vad_original_gap.npy",
    }
    # 25hz dtoks
    _renames_25hz = {
        "boundary_shift_energy.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.boundary_shift_energy.npy",
        "dtok_level_1.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_level_1.npy",
        "dtok_level_2.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_level_2.npy",
        "dtok_global.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_global.npy",
        "source_start_end_time.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.source_start_end_time.npy",
        "vad_boundary_shift.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.vad_boundary_shift.npy",
        "vad_original_duration.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.vad_original_duration.npy",
        "vad_original_gap.npy": "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.vad_original_gap.npy",
    }

    for fname in tqdm(fnames):
        if "dtok_v2_ml_25hz" in fname.lower():
            _renames = _renames_25hz
        elif "dtok_v2_ml_50hz" in fname.lower():
            _renames = _renames_16k
        else:
            _renames = _renames_general

        reader = pa.ipc.open_file(fname)
        table = reader.read_all()
        table2 = table.select(sorted(table.column_names))
        if add_prefix:
            table2 = table2.rename_columns(
                [f"{add_prefix}{k}" if k != "__key__" else "__key__" for k in table2.column_names]
            )
        else:
            table2 = table2.rename_columns([k if k not in _renames else _renames[k] for k in table2.column_names])
        if dry_run:
            from .ws_sink import indented

            if table.schema != table2.schema:
                new = "NEW: "
                tqdm.write(indented("OLD: ", table.schema))
            else:
                new = "NO CHANGE: "
            tqdm.write(indented(new, table2.schema))
            # return
        else:
            if table.schema == table2.schema:
                continue
            with AtomicFile(fname) as fname:
                with pa.ipc.new_file(fname, table2.schema.with_metadata(reader.schema.metadata)) as sink:
                    sink.write_table(table2)
        # else:
        #     print(f"No changes needed: {fname}")


@command
def _rename_keys(*fnames, dry_run: bool = False):
    """Rename keys in wsds shards by stripping '/' or './' prefixes from key names."""
    import pyarrow as pa
    from tqdm import tqdm

    from .ws_sink import AtomicFile

    def strip_path_prefix(key):
        """Strip '/' or './' prefix from key name if present."""
        if key.startswith("./"):
            return key[2:]
        elif key.startswith("/"):
            return key[1:]
        return key

    for fname in tqdm(fnames):
        reader = pa.ipc.open_file(fname)
        table = reader.read_all()

        # Get the __key__ column and check for changes in one pass
        key_column = table["__key__"]
        changes = []
        new_keys = []

        for key in key_column:
            old_key = key.as_py()
            new_key = strip_path_prefix(old_key)
            new_keys.append(new_key)
            if old_key != new_key:
                changes.append((old_key, new_key))

        if dry_run:
            if changes:
                tqdm.write(f"{fname}: Would rename {len(changes)} keys")
                for old_key, new_key in changes[:2]:
                    tqdm.write(f"  '{old_key}' -> '{new_key}'")
                tqdm.write("...")
            else:
                tqdm.write(f"{fname}: NO CHANGE - no keys need renaming")
        else:
            if not changes:
                continue
            # Create new table with renamed keys
            table2 = table.set_column(0, "__key__", pa.array(new_keys))
            with AtomicFile(fname) as tmp:
                with pa.ipc.new_file(tmp, table2.schema.with_metadata(reader.schema.metadata)) as sink:
                    sink.write_table(table2)


@command
def _convert_datatype(*fnames, target_type="string", columns=""):
    """
    Convert all or specific columns in the given .wsds files to a specific Arrow datatype
    (e.g., string, float32, int64, etc.).

    Example:
        wsds _convert_datatype /path/to/shards/*.wsds --target_type string --columns transcription.txt
    """
    import pyarrow as pa
    import tqdm

    from .ws_sink import AtomicFile

    _type_map = {
        "string": pa.utf8(),
        "binary": pa.binary(),
        "float32": pa.float32(),
        "float16": pa.float16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
    }

    if target_type not in _type_map:
        raise ValueError(f"Unsupported target_type: {target_type}. Choose from {list(_type_map)}")

    target_arrow_type = _type_map[target_type]
    selected_cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None

    for fname in tqdm.tqdm(fnames, desc=f"Converting to {target_type}"):
        reader = pa.ipc.open_file(fname)
        table = reader.read_all()
        new_cols = {}

        for name in table.column_names:
            col = table[name]
            col_type = col.type

            if selected_cols and name not in selected_cols:
                new_cols[name] = col
                continue

            if col_type == target_arrow_type:
                new_cols[name] = col
                continue

            try:
                new_cols[name] = col.cast(target_arrow_type)
            except Exception as e:
                print(f"Failed to convert column '{name}' in {fname}: {e}")
                new_cols[name] = col

        table2 = pa.table(new_cols)
        if table.schema != table2.schema:
            with AtomicFile(fname) as tmp:
                with pa.ipc.new_file(tmp, table2.schema.with_metadata(reader.schema.metadata)) as sink:
                    sink.write_table(table2)


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


@command
class check_status:
    @staticmethod
    def all_datasets_progress(
        dataset: Path = None,
        test_yaml: Path = None,
        columns: list[str] = None,
        show_mvad: bool = False,
        refresh_interval: float = 2.0,
    ):
        """Display a live-updating table of shard completion across datasets."""
        import time

        from rich.console import Console
        from rich.live import Live
        from rich.table import Table

        console = Console()
        base_path = Path(dataset) if dataset else Path("/mnt/weka/data-wsds")
        artifacts = validate.load_test_yaml(test_yaml)
        ALLOWED_SHARDS = list(artifacts.keys())
        if columns:
            ALLOWED_SHARDS = [a for a in artifacts.keys() if a in columns]
        if show_mvad:
            artifacts["mvad"] = {"display_name": "mvad"}
            ALLOWED_SHARDS = ["mvad"] + ALLOWED_SHARDS

        def generate_table():
            # Collect dataset info for allowed shards only
            dataset_info = {}

            for dataset_dir in base_path.iterdir():
                if not dataset_dir.is_dir():
                    continue

                # Get ground truth total from source audio
                source_audio = dataset_dir / "source" / "audio"
                total_shards = len(list(source_audio.glob("*.wsds"))) if source_audio.exists() else 0

                if total_shards == 0:
                    continue

                # Look for version folders (v2*, v3*, v4*, etc.)
                for version_dir in dataset_dir.iterdir():
                    if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                        continue

                    for shard_dir in version_dir.iterdir():
                        if not shard_dir.is_dir() or shard_dir.name not in ALLOWED_SHARDS:
                            continue

                        completed = len(list(shard_dir.glob("*.wsds")))
                        if completed > 0:
                            if dataset_dir.name not in dataset_info:
                                dataset_info[dataset_dir.name] = {
                                    "type": version_dir.name,
                                    "total": total_shards,
                                    "shards": {},
                                }
                            dataset_info[dataset_dir.name]["shards"][shard_dir.name] = completed

            if show_mvad:
                for dataset_name in dataset_info.keys():
                    source_dir = base_path / dataset_name / "source"
                    num_mvad_shards = len(list(source_dir.glob("v*/*.wsds")))
                    dataset_info[dataset_name]["shards"]["mvad"] = num_mvad_shards

            # Build table
            table = Table(title="Dataset Shard Validation Status")
            table.add_column("Dataset", style="cyan", no_wrap=True)
            table.add_column("Type", style="cyan", width=22, overflow="fold", no_wrap=False)

            for shard_name in ALLOWED_SHARDS:
                display_name = artifacts[shard_name].get("display_name", shard_name)
                table.add_column(display_name, justify="center", width=18, overflow="fold", no_wrap=False)

            for dataset_name in sorted(dataset_info.keys()):
                info = dataset_info[dataset_name]
                dataset_type = info["type"]
                total = info["total"]
                shards = info["shards"]
                row = [dataset_name, dataset_type]

                for shard_name in ALLOWED_SHARDS:
                    if shard_name not in shards:
                        row.append("[red]-[/red]")
                        continue

                    completed = shards[shard_name]
                    pct = (completed / total * 100) if total > 0 else 0

                    # Color code based on percentage
                    if pct == 0 or pct > 100:
                        color = "red"
                    elif pct == 100:
                        color = "green"
                    else:
                        color = "yellow"

                    row.append(f"[{color}]{completed}/{total} ({pct:.0f}%)[/{color}]")

                table.add_row(*row)

            return table

        with Live(generate_table(), refresh_per_second=1 / refresh_interval, console=console) as live:
            try:
                while True:
                    time.sleep(refresh_interval)
                    live.update(generate_table())
            except KeyboardInterrupt:
                pass

    @staticmethod
    def dataset_progress(dataset: Path, refresh_interval: float = 5.0):
        """Display completion progress for each subdirectory."""
        import time

        from rich.console import Console
        from rich.live import Live
        from rich.table import Table

        from .utils import list_all_shards

        console = Console()

        def generate_table():
            table = Table(title=f"Dataset Shard Progress: {dataset}")
            table.add_column("Subdirectory", style="cyan")
            table.add_column("Completed", justify="right")
            table.add_column("Total", justify="right")
            table.add_column("Progress", justify="center")
            table.add_column("Status", justify="center")

            shard_names = list_all_shards(dataset, verbose=False, print_missing=False)
            total_shards = len(shard_names)

            for subdir in sorted(Path(dataset).iterdir()):
                if not subdir.is_dir():
                    continue

                completed = 0
                conflicts = False
                schemas = {}

                for shard_name in shard_names:
                    shard_path = (subdir / shard_name).with_suffix(".wsds")
                    if shard_path.exists():
                        try:
                            schema = get_shard_schema(shard_path)
                            if schema:
                                schemas[shard_name] = schema
                                completed += 1
                        except Exception:
                            pass

                # Check for schema conflicts
                unique_schemas = set(schemas.values())
                conflicts = len(unique_schemas) > 1

                progress_pct = (completed / total_shards * 100) if total_shards > 0 else 0
                progress_bar = f"[{'=' * int(progress_pct / 5):{20}}]"

                status = (
                    "⚠️ Conflict" if conflicts else ("✓ Complete" if completed == total_shards else "⏳ In Progress")
                )
                status_color = "red" if conflicts else ("green" if completed == total_shards else "yellow")

                table.add_row(
                    subdir.name,
                    str(completed),
                    str(total_shards),
                    f"{progress_bar} {progress_pct:.1f}%",
                    f"[{status_color}]{status}[/{status_color}]",
                )

            return table

        with Live(generate_table(), refresh_per_second=1 / refresh_interval, console=console) as live:
            try:
                while True:
                    time.sleep(refresh_interval)
                    live.update(generate_table())
            except KeyboardInterrupt:
                pass


def write_batch(batch, out_path, batch_size: int = 1, compression: str | None = None):
    # Use WSSink to control record batch sizes and avoid Arrow auto-chunking.
    with WSSink(out_path, batch_size=batch_size, compression=compression) as sink:
        for row in batch:
            sink.write(row)


@command
def shard_from_audio_dir(
    input_dir: str,
    output_dir: str,
    max_files_per_shard: int = 300,
    batch_size: int = 1,
    resume=True,
    init_index: bool = False,
    require_audio_duration: bool = True,
    key_fn: Callable[[str], str] | None = None,
    write_key_mapping: bool = False,
):
    """Write batched Feather (.wsds) shards with up to N audio files each.
    Note: could change to max_size instead of n file.

    Args:
        key_fn: Optional function to transform the file stem into a key.
                E.g., a hash function for obfuscation.
        write_key_mapping: If True and key_fn is provided, writes a JSON file
                           mapping transformed keys back to original stems.
    """

    from tqdm import tqdm

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
    all_files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)
    print(f"[INFO] Found {len(all_files):,} audio files under {input_dir}")

    MAX_ARROW_BYTES = 2_140_000_000  # ~2.1 GB Arrow cell limit
    # Note, if we skip the large file we would then want to chunk it smaller or re-encode.
    # then just rerun the sharding for those files.

    shard_idx = 0
    batch = []
    key_mapping = {}  # key -> original_stem

    def flush_batch():
        nonlocal shard_idx, batch
        if not batch:
            return
        out_path = output_dir / f"audio-{shard_idx:05d}.wsds"
        write_batch(batch, out_path, batch_size=batch_size, compression=None)
        shard_idx += 1
        batch = []

    for i, path in enumerate(tqdm(all_files, ncols=90, desc="Writing WSDS shards")):
        stem = path.stem
        key = key_fn(stem) if key_fn else stem
        if key_fn and write_key_mapping:
            key_mapping[key] = stem
        ext = path.suffix.lower().lstrip(".")
        try:
            audio_bytes = path.read_bytes()
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        size = len(audio_bytes)
        if size > MAX_ARROW_BYTES:
            print(f"[SKIP] {path.name}: {size / 1e6:.1f} MB exceeds 2 GB Arrow limit")
            continue

        batch.append({"__key__": key, "audio": audio_bytes, "audio_type": ext})

        if len(batch) >= max_files_per_shard:
            flush_batch()

    if batch:
        flush_batch()

    print(f"[DONE] Wrote {shard_idx} WSDS shards → {output_dir}")

    if key_mapping:
        mapping_path = output_dir / "key_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(key_mapping, f, indent=2)
        print(f"[INFO] Wrote key mapping ({len(key_mapping):,} entries) → {mapping_path}")

    if init_index:
        dataset_root = output_dir.parent if output_dir.name == "audio" else output_dir
        init(dataset_root, require_audio_duration=require_audio_duration)
