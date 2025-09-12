from pathlib import Path
import json

import pyarrow as pa
import numpy as np

from hume_wsds import WSSink

__all__ = []
def command(f):
    __all__.append(f.__name__)
    return f



@command
def list(input_shard:str):
    """Lists keys in a wsds shard."""
    reader = pa.RecordBatchFileReader(pa.memory_map(input_shard))
    for i in range(reader.num_record_batches):
        for key in reader.get_batch(i)['__key__']:
            print(key)



@command
def inspect(input_shard:str):
    """Displays metadata and schema of a wsds shard."""
    reader = pa.RecordBatchFileReader(pa.memory_map(input_shard))
    print(f"Batches: {reader.num_record_batches}")
    print(f"Rows: {int(reader.schema.metadata[b'batch_size']) * reader.num_record_batches}")
    print(f"Schema:\n{reader.schema}")



@command
def from_webdataset(
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
                    v = cast_types_for_storage(v, float_cast=np.float16, int_cast=np.int32)
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

def cast_types_for_storage(obj, float_cast=np.float32, int_cast=np.int32, debug=False):
    """Cast nested JSON-like objects to more resctrictive float and integer types.

    By default PyArrow would cast to float64 and int64, which are not optimal for storage.
    This function casts all numbers to float32/int32 by default.
    """
    if type(obj) is float:
        return float_cast(obj)
    elif type(obj) is int:
        return int_cast(obj)
    elif type(obj) is dict:
        return {k:cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast) for k,v in obj.items()}
    elif type(obj) is list:
        return [cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast) for v in obj]
    else:
        if debug: print('unknown type:', type(obj), obj, type(obj) == float, float)
        return obj
