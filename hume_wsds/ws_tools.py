import functools
import json
from pathlib import Path
import os

import pyarrow as pa

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



@command('list')
def _list(input_shard:str):
    """Lists keys in a wsds dataset or shard."""
    if (Path(input_shard) / 'index.sqlite3').exists():
        # FIXME: implement keys
        pass
    else:
        reader = pa.RecordBatchFileReader(pa.memory_map(input_shard))
        try:
            for i in range(reader.num_record_batches):
                for key in reader.get_batch(i)['__key__']:
                    print(key)
        except BrokenPipeError:
            pass


@command
def inspect(input_path:str):
    """Displays metadata and schema of a wsds dataset or shard."""
    if (Path(input_path) / 'index.sqlite3').exists():
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
    input_shard:str,          # input shard URL/path
    output_shard:str,         # output shard URL/path
    batch_size:int=16,        # batch size
    compression:str='zstd',
    min_batch_size_bytes=1024*1024,
    no_keys:bool=False
):
    """Converts a WebDataset shard into wsds format.

    Tries to automatically determine good defaults for compression and batch size."""
    import webdataset as wds
    from hume_wsds import WSSink
    from hume_wsds.utils import cast_types_for_storage

    ds = wds.WebDataset([input_shard], shardshuffle=False)

    out_dir = Path(output_shard).parents[0].name

    if out_dir == 'audio': compression = 'no-compression'

    def process(stream):
        for s in stream:
            new_s = {}
            for k,v in s.items():
                if k.endswith('.json'):
                    v = json.loads(v)
                    k = k[:-len('.json')]
                    v = cast_types_for_storage(v, float_cast='float16', int_cast='int32')
                if out_dir == 'dtok_v2_ml_50hz_32x16384_graphemes_key16k' and k == 'dtok_level_1.npy':
                    k = 'dtok_level_1_16k.npy'
                # if k.endswith('.vad.npy'):
                #     v = list(np.load(io.BytesIO(v)))
                #     k = k[:-len('.npy')]
                new_s[k] = v
            yield new_s

    ds = ds.compose(process)
    # update the data format
    # if 'snr-c50' in output_shard:
    #     def fix_c50(stream):
    #         for s in stream:
    #             if 'snr_c50.npy' in s:
    #                 s['snr'], s['c50'] = s['snr_c50.npy']
    #                 del s['snr_c50.npy']
    #             yield s
    #     ds = ds.decode().compose(
    #         fix_c50,
    #     )

    if compression == 'no-compression': compression = None

    with WSSink(output_shard, batch_size=batch_size, compression=compression,
                min_batch_size_bytes=min_batch_size_bytes) as sink:
        for x in ds:
            drop_keys(x, '__url__', '__local_path__')
            if no_keys: del x['__key__']
            sink.write(dict(sorted(x.items())))

def drop_keys(dict, *keys):
    """Remove specified keys from the given dictionary."""
    for key in keys:
        if key in dict: del dict[key]

@command
def dump_index(source_dataset:Path):
    """Dump the index of the given dataset in a readable format."""
    from . import WSDataset

    ds = WSDataset(source_dataset)

    try:
        for sample in ds.index.query("SELECT name,s.shard,offset FROM files AS f, shards AS s WHERE s.shard_id == f.shard_id ORDER BY name,s.shard,offset;"):
            print(*sample)
    except BrokenPipeError:
        pass

@command
def validate_shards(dataset):
    from .utils import list_all_shards
    list_all_shards(dataset)

@command
def init(
    new_dataset:Path,
    source_dataset:Path | None = None,
    vad_column:str | None = None,
    num_workers:int = 32,
):
    """Initialize a new dataset, from scratch or from a segmentation of an existing one."""
    from . import WSDataset
    from fastprogress import progress_bar
    from .ws_index import WSDSIndexWriter
    from . import AtomicFile
    import multiprocessing

    new_dataset = Path(new_dataset)

    if source_dataset is not None:
        assert vad_column is not None, "vad_column must be specified when initializing from a source dataset"
    else:
        source_dataset = new_dataset

    ds = WSDataset(source_dataset)
    shard_extractor = functools.partial(extract_index_for_shard, source_dataset, vad_column=vad_column)
    all_shards = ds.get_shard_list()

    with AtomicFile(new_dataset / 'index.sqlite3') as fname:
        with WSDSIndexWriter(fname) as index:
            with multiprocessing.Pool(num_workers) as p:
                for r in progress_bar(p.imap_unordered(shard_extractor, all_shards), total=len(all_shards)):
                    try:
                        index.append(r)
                    except:
                        print("Failed to append records to index:", r)
                        raise

            index.append_metadata({
                'segmented': True if vad_column else False
            })

        if vad_column:
            with AtomicFile(new_dataset / 'audio.wsds-link') as fname:
                with open(fname, 'w') as f:
                    f.write(json.dumps({
                        "dataset_dir": os.path.relpath(source_dataset, new_dataset),
                        "loader": ["hume_wsds.ws_shard", "WSSourceAudioShard"],
                        "vad_column": vad_column,
                    }))

def extract_index_for_shard(dataset, shard, vad_column=None):
    from . import WSDataset
    from torchcodec.decoders import AudioDecoder
    from .ws_audio import to_filelike

    ds = WSDataset(dataset)
    index = []
    i = 0
    for s in ds.sequential_from(shard, 0):
        try:
            key = str(s['__key__'])
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
                speech_duration = float((vad[:,-1] - vad[:,-2]).sum()) # tend - tstart

        try:
            # FIXME: move this to ws_audio and add to autodecoders?
            decoder = AudioDecoder(to_filelike(s.get_audio()))
            audio_duration = decoder.metadata.duration_seconds_from_header
        except Exception as e:
            print("Audio loading error:", e)
            print("         for sample:", s)
            raise

        if n > 0: # in derived datasets, skip files with no vad segments (they won't have samples and will never appear as keys)
            index.append((key, i, audio_duration, speech_duration))

        i += n
    return {
        'shard_name': shard,
        'index': index,
        'n_samples': i,
    }
