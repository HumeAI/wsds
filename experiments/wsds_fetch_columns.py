import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import pyarrow as pa
    from wsds_audio import timeit
    import json
    import numpy as np
    import torch


@app.cell
def _():
    reader = pa.RecordBatchFileReader(pa.memory_map('youtube-cc/vad_ws/accent_raw/stanford-000000.wsds'))
    reader.schema.names
    return


@app.cell
def _():
    @timeit
    def _test():
        reader = pa.RecordBatchFileReader('youtube-cc/vad_ws/accent_raw/stanford-000000.wsds')
        reader.schema
    return


@app.cell
def _():
    @timeit
    def _test():
        reader = pa.RecordBatchFileReader(pa.memory_map('youtube-cc/vad_ws/accent_raw/stanford-000000.wsds'))
        reader.schema
    return


@app.cell
def _():
    @timeit
    def _test():
        reader = pa.RecordBatchFileReader(pa.memory_map('youtube-cc/vad_ws_zstd/accent_raw/stanford-000000.wsds'))
        reader.schema
    return


@app.function
def get_columns(fname):
    if isinstance(fname, Path): fname = str(fname)
    reader = pa.RecordBatchFileReader(pa.memory_map(fname))
    return reader.schema.names


@app.cell
def _():
    [get_columns(f) for p in Path('youtube-cc/vad_ws/').iterdir() if p.is_dir() for f in [next(p.iterdir(), None)] if f]
    return


@app.function
def find_first_shard(path):
    return next(Path(path).iterdir(), None)


@app.function
def list_all_columns(ds_path):
    dupes = {}
    cols = {}
    for p in Path(ds_path).iterdir():
        if p.suffix == '.wsds-link':
            col = p.with_suffix('').name
            cols[col] = (p.name, col)
            continue
        if not p.is_dir(): continue
        # FIXME: this is quite expensive (adds up to 100ms on youtube-cc)
        # we should get one shard name from the sqlite index instead
        fname = next(p.iterdir(), None)
        if fname:
            for col in get_columns(fname):
                if col == '__key__': continue
                # seems like we should fix this during the original conversion
                if col in cols or col in dupes:
                    dupes[col] = True
                    if col in cols:
                        dirname = cols[col][0]
                        cols[f'{dirname}.{col}'] = (dirname, col)
                        del cols[col]
                    cols[f'{p.name}.{col}'] = (p.name, col)
                else:
                    cols[col] = (p.name, col)
    return dict(sorted(cols.items()))


@app.cell
def _():
    list_all_columns('youtube-cc/vad_ws/')
    return


@app.cell
def _():
    @timeit
    def _():
        list_all_columns('youtube-cc/vad_ws/')
    return


@app.cell
def _():
    @timeit
    def _():
        list_all_columns('youtube-cc/vad_ws_zstd/')
    return


@app.cell
def _():
    @timeit
    def _():
        list_all_columns('youtube-cc/vad_ws_zstd_128/')
    return


@app.cell
def _():
    @timeit
    def _():
        [get_columns(f) for p in Path('youtube-cc/vad_ws/').iterdir() if p.is_dir() for f in [next(p.iterdir(), None)] if f]
    return


@app.cell
def _():
    @timeit
    def _():
        [get_columns(f) for p in Path('youtube-cc/vad_ws_zstd/').iterdir() if p.is_dir() for f in [next(p.iterdir(), None)] if f]
    return


@app.cell
def _():
    @timeit
    def _():
        [get_columns(f) for p in Path('youtube-cc/vad_ws_zstd_128/').iterdir() if p.is_dir() for f in [next(p.iterdir(), None)] if f]
    return


@app.cell
def _():
    sample = {}
    _base = Path('youtube-cc/vad_ws')
    print(list_all_columns(_base))
    for k,(dir,col) in list_all_columns(_base).items():
        if k.endswith('.json'):
            _shard = find_first_shard(_base/dir)
            _reader = pa.RecordBatchFileReader(pa.memory_map(str(_shard)))
            _b = _reader.get_batch(0)
            # print(k, dir, _reader.schema)
            if k.endswith('json'):
                v = json.loads(bytes(_b[col][0].as_buffer()))
            else:
                v = _b[col][0]
            print(k, v)
            sample[k] = v
    return (sample,)


@app.cell
def _(sample):
    sample
    return


@app.cell
def _(sample):
    pa.RecordBatch.from_pylist([sample]).schema.field('tag_dict.json')
    return


@app.cell
def _(sample):
    type(pa.RecordBatch.from_pylist([sample])['tag_dict.json'])
    return


@app.cell
def _():
    torch.tensor(15, dtype=torch.float16).numpy()[()]
    return


@app.function
def cast_types_for_storage(obj, float_cast=np.float32, int_cast=np.int64):
    if type(obj) is float:
        return float_cast(obj)
    elif type(obj) is int:
        return int_cast(obj)
    elif type(obj) is dict:
        return {k:cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast) for k,v in obj.items()}
    elif type(obj) is list:
        return [cast_types_for_storage(v, float_cast=float_cast, int_cast=int_cast) for v in obj]
    else:
        print('unknown type:', type(obj), obj, type(obj) == float, float)
        return obj


@app.cell
def _(sample):
    cast_types_for_storage(sample, float_cast=np.float16, int_cast=np.int32)
    return


@app.cell
def _(sample):
    pa.RecordBatch.from_pylist([dict(sample, **{'tag_dict.json': {k:torch.tensor(v, dtype=torch.float16).numpy()[()] for k,v in sample['tag_dict.json'].items()}})])
    return


@app.cell
def _(sample):
    pa.RecordBatch.from_pylist([cast_types_for_storage(sample, float_cast=np.float16, int_cast=np.int32)])
    return


@app.cell
def _():
    import pyarrow.feather
    import pyarrow.compute as pc

    _arr = pyarrow.feather.read_table('/mnt/weka/rashish/research/personal/rashish/pyarrow_experiments/ipc_datasets/podcasts_subset_1000_zstd/pquality_scores_raw/you-must-remember-this-000000.pquality_scores_raw.feather')

    len(_arr), len(_arr.filter(pc.field('pq.json') > 7.5)), (_arr['pq.json'] > 7.5).get_total_buffer_size()
    return (pyarrow,)


@app.cell
def _():
    import polars as pl

    pl.scan_ipc('/mnt/weka/rashish/research/personal/rashish/pyarrow_experiments/ipc_datasets/podcasts_subset_1000_zstd/pquality_scores_raw/*.feather', include_file_paths='shard').select(pl.col('pq.json') > 7.5).collect()
    return (pl,)


@app.cell
def _(pl):
    pl.scan_ipc('/mnt/weka/rashish/research/personal/rashish/pyarrow_experiments/ipc_datasets/podcasts_subset_1000_zstd/transcription_wslang_raw/*.feather').drop('__key__').cast({'txt': pl.String})
    return


@app.cell
def _(pl):
    pl.concat([
    pl.scan_ipc('/mnt/weka/rashish/research/personal/rashish/pyarrow_experiments/ipc_datasets/podcasts_subset_1000_zstd/pquality_scores_raw/*.feather', include_file_paths='shard'), pl.scan_ipc('/mnt/weka/rashish/research/personal/rashish/pyarrow_experiments/ipc_datasets/podcasts_subset_1000_zstd/transcription_wslang_raw/*.feather').drop('__key__').cast({'txt': pl.String})
    ], how='horizontal').head(10).collect()
    return


@app.cell
def _():
    import io
    return (io,)


@app.cell
def _(io, pl, pyarrow):
    pyarrow.array(list(np.load(io.BytesIO(pl.scan_ipc('librilight/mvad/librilight-large-wo6454-flac-000013.wsds').collect()['raw.vad.npy'][0]))))
    return


@app.cell
def _(io, pl, pyarrow):
    pyarrow.array(np.load(io.BytesIO(pl.scan_ipc('librilight/mvad/librilight-large-wo6454-flac-000013.wsds').collect()['raw.vad.npy'][0])).tolist())
    return


@app.cell
def _(pl):
    pl.scan_ipc('librilight/mvad-zstd-native/librilight-large-wo6454-flac-000013.wsds').collect()['raw.vad'][0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
