import functools
import gzip
import io
import json
import os

import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
from pathlib import Path
from collections import defaultdict
import webdataset as wds

from hume_wsds import WSSink, WSSample



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


@command
def inspect(input_path: str):
    """Displays metadata and schema of a wsds dataset or shard."""
    if (Path(input_path) / "index.sqlite3").exists():
        from . import WSDataset

        ds = WSDataset(input_path)
        print(ds)
    else:
        reader = pa.RecordBatchFileReader(pa.memory_map(input_path))
        print(f"Batches: {reader.num_record_batches}")
        print(f"Rows: {int(reader.schema.metadata[b'batch_size']) * reader.num_record_batches}")
        print(f"Schema:\n{reader.schema}")


@command
def shard_from_webdataset(
    input_shard: str,          # input shard URL/path
    output_shard: str,         # output shard URL/path
    batch_size: int = 16,      # batch size
    compression: str = 'zstd',
    min_batch_size_bytes: int = 1024*1024,
    no_keys: bool = False,
    yt_data_specific: bool = False,
    audio_requires_sorting: bool = False,
):
    """Converts a WebDataset shard into wsds format.

    Supports yt_data_specific mode with optional sorting.
    """
    import io, json, tarfile, gzip
    import numpy as np
    import webdataset as wds
    from pathlib import Path
    from collections import defaultdict
    from hume_wsds import WSSink
    from hume_wsds.utils import cast_types_for_storage
    from hume_wsds.constants import ShardMapping

    out_dir = Path(output_shard).parents[0].name
    if out_dir == 'audio':
        compression = 'no-compression'

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
        
            f = o(input_shard, 'rb')
            tar = tarfile.TarFile(fileobj=f)
        
            for member in tar.getmembers():
                if "/" in member.name:
                    path, name = member.name.rsplit('/', 1)
                else:
                    path, name = "", member.name  # fallback if at root level
                name = name.replace('_comments', '.comments')
                key, field = name.split('.', 1)
                sample_key = f'{path}/{key}'
        
                all_samples[sample_key][field] = member
        
                if field in AUDIO_KEYS:
                    audio_entries.append(sample_key)
        
            return tar, all_samples, audio_entries
            
        if out_dir == 'audio' and audio_requires_sorting:

            tar, samples, audio_entries = list_keys_tarfile(input_shard)
            
            for sample_key in audio_entries:
                fields = samples[sample_key]
                new_s = {}
                new_s["__key__"] = sample_key
            
                for ak in AUDIO_KEYS:
                    if ak in fields:
                        new_s['audio'] = tar.extractfile(fields[ak]).read()
                
                if yt_data_specific: 
                    for meta in ["info.json", "description", "comments.json"]:
                        if meta in fields:
                            try:
                                new_s[meta] = tar.extractfile(fields[meta]).read().decode("utf-8", errors="ignore")
                            except Exception:
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
                            k = k[:-len(".json")]
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

    if compression == 'no-compression':
        compression = None

    if out_dir == 'audio' and yt_data_specific:
        iterator = process(None)
    elif out_dir == 'audio' and audio_requires_sorting:
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
            drop_keys(x, '__url__', '__local_path__')
            if no_keys and '__key__' in x:
                del x['__key__']
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
def validate_shards(dataset, verbose=False):
    from .utils import list_all_shards

    list_all_shards(dataset, verbose)


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


def extract_index_for_shard(dataset, shard, vad_column=None):
    from torchcodec.decoders import AudioDecoder

    from . import WSDataset
    from .ws_audio import to_filelike
    ds = WSDataset(dataset)
    index = []
    i = 0

    # instead of passing `shard` directly, wrap it
    # WSSample(ds, shard_name, offset=0)
    sample = WSSample(ds, shard, 0)

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

        try:
            # FIXME: move this to ws_audio and add to autodecoders?
            decoder = AudioDecoder(to_filelike(s.get_audio()))
            audio_duration = decoder.metadata.duration_seconds_from_header
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
        "shard_name": shard,
        "index": index,
        "n_samples": i,
    }
