import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import json, io

    from fastprogress import progress_bar
    from fastcore.script import call_parse

    import webdataset as wds

    from wsds import WSSink
    import pyarrow
    import numpy as np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Convert `youtube-cc` to the new WSDS format

    The tricky thing is that some of the shards are broken (the ones with ~ in the tar names) so we borrow them from the older dataset version.

    `parallel` uses all the cores to run the given command on files passed after `:::` replacing {} with the file name (see .

    ```{bash}
    parallel --plus --tag --bar \
        python convert-webdataset-to-wsds.py --compression no-compression {} youtube-cc/audio/{/.}.wsds \
        ::: /mnt/weka/data2/multimodal/core/youtube-cc/raw/audio/*.tar
            /mnt/weka/data/etts-data/v2/youtube-cc/audio/alux.com-00000*
            /mnt/weka/data/etts-data/v2/youtube-cc/audio/freecodecamp.org-0000*

    rm youtube-cc/audio/*~*
    ```

    ```{bash}
    parallel --plus --tag --bar \
        python convert-webdataset-to-wsds.py --compression no-compression {} youtube-cc/mvad/{/...}.wsds \
        ::: /mnt/weka/data2/multimodal/core/youtube-cc/processed-v3-pipeline-vad_ws/staging/mvad/*.tar.gz

    parallel --plus --tag --bar \
        python convert-webdataset-to-wsds.py --compression no-compression {} youtube-cc/mvad/{/..}.wsds \
        ::: /mnt/weka/data/etts-data/v2/youtube-cc/mvad/alux.com-00000*.tar.gz
            /mnt/weka/data/etts-data/v2/youtube-cc/mvad/freecodecamp.org-0000*.tar.gz

    rm youtube-cc/mvad/*~*
    ```
    """
    )
    return


@app.function
def drop_keys(x, *keys):
    for key in keys:
        if key in x: del x[key]


@app.function
def cast_types_for_storage(obj, float_cast=np.float32, int_cast=np.int64, debug=False):
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


@app.function
# @call_parse
def process_shard(
    input_shard:str,          # input shard URL/path
    output_shard:str,         # output shard URL/path
    batch_size:int=16,        # batch size
    compression:str='zstd',
    min_batch_size_bytes=1024*1024,
    no_keys:bool=False
):
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
    # if out_dir == 'tagger_raw_v0':
    #     pass
    # if 'txt-medium' in output_shard or 'snr-c50' in output_shard:
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


@app.cell
def _():
    src = '/mnt/weka/data2/multimodal/core/librilight/processed-v3-pipeline-vad_ws/staging/acoustic_scores_raw/librilight-large-6454-flac-000000.acoustic_scores_raw.tar.gz'
    return (src,)


@app.cell
def _(src):
    dst = 'librilight/accent_raw/librilight-large-wo6454-flac-000008.wsds'
    process_shard(src, dst)
    return (dst,)


@app.cell
def _(dst):
    print(f"{Path(dst).stat().st_size / 1024 / 1024:.2f}MB")
    _reader = pyarrow.RecordBatchFileReader(dst)
    print(_reader.get_batch(0).nbytes, _reader.schema.metadata[b'batch_size'])
    print(_reader.num_record_batches * int(_reader.schema.metadata[b'batch_size']), _reader.schema)
    print(_reader.get_batch(0).take([0]))
    return


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(mo, pl):
    def run_all_combinations(fun, *args, **cols):
        df = None
        for col in (pl.DataFrame({k: v}) for k,v in cols.items()):
            if df is None:
                df = col
            else:
                df = df.join(col, how='cross')
        results = pl.DataFrame(
            fun(*args, **row)
            for row in mo.status.progress_bar(df.iter_rows(named=True), total=len(df))
        )
        return pl.concat([df, results], how='horizontal')
    return (run_all_combinations,)


@app.cell
def _(run_all_combinations, src):
    def _test(src, **kwargs):
        process_shard(src, 'tmp.wsds', **kwargs)
        return dict(
            size_MB = Path('tmp.wsds').stat().st_size / 1024 / 1024
        )
    print(f"{Path(src).stat().st_size/1024/1024:.2f}MB")
    run_all_combinations(_test, src, compression=['no-compression', 'zstd'], batch_size=[16,128,1024,8192,65536])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
