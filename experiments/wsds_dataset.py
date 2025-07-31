import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")

with app.setup:
    import random, io, pickle
    from pathlib import Path
    from dataclasses import dataclass

    import webdataset
    import numpy as np
    import pyarrow as pa

    from wsds_fetch_columns import list_all_columns
    from wsds_extract_index import WSIndex
    from wsds_audio import WSDSAudio, AudioReader

    from wsds_audio import timeit


@app.class_definition
@dataclass(frozen=True, slots=True)
class WSSample:
    dataset: 'WSDataset'
    shard_name: str
    offset: int

    def get_key(self):
        return self.dataset.get_key(self.shard_name, self.offset)

    def keys(self):
        return self.dataset.fields.keys()

    def items(self):
        yield from ((k,self[k]) for k in self.dataset.fields.keys())

    def values(self):
        yield from (v for k,v in self.items())

    def __getitem__(self, field):
        return self.dataset.get_sample(self.shard_name, field, self.offset)

    def __repr__(self):
        r = f"WSSample({repr(self.dataset)}, shard_name={repr(self.shard_name)}, offset={repr(self.offset)}, fields={'{'}\n"
        for k,v in self.items():
            if hasattr(v, 'shape'):
                v = f"array(size={repr(v.shape)}, dtype={v.dtype})"
            else:
                v = repr(v)
                if len(v) > 1000: v = v[:1000]+'…'
            r += f"  {k} = {v},\n"
        r += "})\n"
        return r


@app.cell
def _():
    # lazy field creation
    @timeit
    def _test():
        reader = pa.RecordBatchFileReader(pa.memory_map('youtube-cc/vad_ws/atoks_raw/stanford-000000.wsds'))
        bs = int(reader.schema.metadata[b'batch_size'])
        offset = 10000
        # b = reader.get_batch(offset // bs)
        # src = b['atoks.npy'][offset % bs]
    return


@app.cell
def _():
    # lazy field creation
    @timeit
    def _test():
        reader = pa.RecordBatchFileReader(pa.memory_map('youtube-cc/source/audio/stanford-000000.wsds'))
        bs = int(reader.schema.metadata[b'batch_size'])
        offset = 56
        b = reader.get_batch(offset // bs)
        src = b['mp3'][offset % bs]
    return


@app.class_definition
class WSShard:
    def __init__(self, fname, shard_name=None):
        self.shard_name = shard_name
        self.fname = fname
        self.reader = pa.RecordBatchFileReader(pa.memory_map(fname))
        self.batch_size = int(self.reader.schema.metadata[b'batch_size'])

        # cache
        self._start = None
        self._end = None
        self._data = None

    def get_sample(self, column, offset):
        if self._data is None or offset < self._start or offset >= self._end:
            i = offset // self.batch_size
            if i >= self.reader.num_record_batches: return
            self._data = self.reader.get_batch(i)
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= len(self._data): return
        data = self._data[column][j]
        # FIXME: implement proper encoders and decoders
        if column.endswith('npy'):
            return np.load(io.BytesIO(data.as_buffer()))
        elif column.endswith('pyd'):
            return pickle.load(io.BytesIO(data.as_buffer()))
        elif column.endswith('txt'):
            return bytes(data.as_buffer()).decode('utf-8')
        else:
            return data

    def __repr__(self):
        r = f"WSShard({repr(self.fname)})"
        if self._data:
            r += f" # cached_region = [{self._start, self._end}]"
        return r


@app.cell
def _():
    _shard = WSShard('youtube-cc/vad_ws/atoks_raw/stanford-000000.wsds')
    _shard.get_sample('__key__', 0)
    print(_shard)
    for i in range(10):
        print('  ', _shard.get_sample('__key__', 510 + i))
    print(_shard)
    return


@app.class_definition
@dataclass(frozen=True, slots=True)
class WSSourceShard:
    shard_name:str
    source_dataset:'WSDataset'
    derived_dataset:'WSDataset'

    def get_sample(self, column, offset):
        # FIXME: using the global parse_key here confuses marimo compiler in strange ways
        file_name, segment_offset = self.derived_dataset.parse_key(
            self.derived_dataset.get_key(self.shard_name, offset)
        )
        print(file_name, segment_offset)
        source_sample = WSSample(self.source_dataset, *self.source_dataset.get_position(file_name))
        # print(_, offset, self.derived_dataset.get_key(self.shard_name, offset))
        # print(file_name, source_sample['raw.vad.npy'].shape)
        tstart, tend = source_sample['raw.vad.npy'][segment_offset]

        return WSDSAudio(AudioReader(source_sample['mp3']), tstart, tend)


@app.cell
def _():
    _shard = WSShard('youtube-cc/vad_ws/atoks_raw/stanford-000000.wsds')
    # this is fast since it reuses the first batch
    @timeit
    def _test():
        for i in range(512):
            _shard.get_sample('__key__', i)
    return


@app.cell
def _():
    _shard = WSShard('youtube-cc/vad_ws/atoks_raw/stanford-000000.wsds')
    # this is slow since it keeps alternating between batches
    @timeit
    def _test():
        for i in range(512):
            _shard.get_sample('__key__', i)
            _shard.get_sample('__key__', i+512)
    return


@app.cell
def _():
    _shard = WSShard('youtube-cc/vad_ws/atoks_raw/stanford-000000.wsds')
    _shard.get_sample('__key__', 22669), _shard.get_sample('__key__', 22670)
    return


@app.function
def make_key(src_file:str, segment_id:int):
    """Make a composite string key from source file name and sequential segment id.

    >>> make_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 1254)
    '20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_00001254'
    """

    assert(type(segment_id) == int)
    return f"{src_file}_{segment_id:08d}"


@app.function
def parse_key(key:str):
    """Parse a composite string key into the source file name and sequential segment id.

    >>> parse_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_00001254')
    ('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 1254)
    """

    src_file, segment_id = key.rsplit('_', 1)
    return src_file, int(segment_id)


@app.class_definition
class WSDataset:
    def __init__(self, dir, segmented=None):
        self.dir = dir
        self.fields = list_all_columns(self.dir)
        self.index = WSIndex(f"{self.dir}/index.sqlite3")
        self.segmented = (Path(self.dir)/'segmented').exists() if segmented is None else segmented
        self._open_shards = {}
        self._linked_datasets = {}

    def get_key(self, shard_name, offset):
        file_name, start_offset = self.index.query('''
        SELECT f.name, f.offset FROM files AS f, shards AS s
        WHERE s.shard = ? AND f.offset <= ? AND s.shard_id == f.shard_id
        ORDER BY f.offset DESC
        LIMIT 1''', shard_name, offset).fetchone()
        return make_key(file_name, offset - start_offset)

    def get_shard(self, subdir, shard_name):
        dir = f"{self.dir}/{subdir}"
        shard = self._open_shards.get(dir, None)
        if shard is None or shard.shard_name != shard_name:
            if subdir.endswith('.wsds-link'):
                # json.loads(Path(dir).read_text())
                link = dict(dataset_dir=f'{self.dir}/../source/', relation='mvad_source')
                if link['dataset_dir'] not in self._linked_datasets:
                    self._linked_datasets[link['dataset_dir']] = WSDataset(link['dataset_dir'])
                shard = WSSourceShard(shard_name, self._linked_datasets[link['dataset_dir']], self)
            else:
                shard = WSShard(f"{dir}/{shard_name}.wsds", shard_name = shard_name)
        self._open_shards[dir] = shard
        return shard

    def get_sample(self, shard_name, field, offset):
        subdir, column = self.fields[field]
        return self.get_shard(subdir, shard_name).get_sample(column, offset)

    def sequential_from(self, shard_name, start, end=None):
        if end is None:
            end = self.index.query('SELECT n_samples FROM shards WHERE shard = ?', shard_name).fetchone()[0]
        i = start
        while i < end:
            yield WSSample(self, shard_name, i)
            i += 1

    def parse_key(self, key):
        if self.segmented:
            return parse_key(key)
        else:
            return key, 0

    def get_position(self, key):
        file_name, offset = self.parse_key(key)
        shard_name, file_offset = self.index.query(
            'SELECT s.shard, offset FROM files AS f, shards AS s WHERE f.name = ? AND s.shard_id == f.shard_id',
            file_name).fetchone()
        return shard_name, file_offset + offset

    def random_position(self):
        shard_name, offset = self.index.query(
            'SELECT s.shard, offset FROM files AS f, shards AS s WHERE f.rowid = ? AND s.shard_id == f.shard_id',
            random.randrange(self.index.n_files)).fetchone()
        return shard_name, offset

    def __iter__(self, max_per_shard=None):
        while True:
            shard_name, start = self.random_position()
            end = start + max_per_shard if max_per_shard else None
            yield from self.sequential_from(shard_name, start, end)

    def __repr__(self):
        return f"WSDataset({repr(self.dir)}, segmented={self.segmented})"


@app.cell
def _(mo):
    ds = WSDataset('youtube-cc/vad_ws')
    for x in ds: break
    x.get_key(), mo.audio(x['audio'].load(16000).numpy(), 16000), x['transcription_wslang_raw.txt'], x['pquality_scores_raw.pq'], x['tag_dict'].as_py()
    return (ds,)


@app.cell
def _(ds):
    for _i in range(12):
        ds.get_key('khan-academy-000016', 12666 + _i)
    return


@app.cell
def _(ds):
    for _x in ds.index.query('''
            EXPLAIN QUERY PLAN SELECT f.name, f.offset FROM files AS f, shards AS s
            WHERE s.shard = ? AND f.offset <= ? AND s.shard_id == f.shard_id
            ORDER BY f.offset DESC
            LIMIT 1''', 'qwe', 0):
        print(_x)
    return


@app.cell
def _(ds):
    @timeit
    def _():
        ds.get_key('khan-academy-000016', 12667)
    ds.get_key('khan-academy-000016', 12667)
    return


@app.cell
def _(ds):
    ds.fields.keys()
    return


@app.cell
def _(ds):
    ds.fields['pquality_scores_raw.pq']
    return


@app.cell
def _(ds, mo):
    for _x in ds: break
    _x
    mo.audio(_x['audio'].load(16000).numpy(), 16000), _x['transcription_wslang_raw.txt']
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl
    import pandas as pd
    return pd, pl


@app.cell
def _(ds):
    {v[1]:k for k,v in ds.fields.items() if v[0] == 'pquality_scores_raw'}
    return


@app.cell
def _(pl):
    pl.sql_expr('pq > 4 AS DQ1').meta.output_name()
    return


@app.cell
def _(ds, pl):
    ds_path = 'youtube-cc/vad_ws'
    pl.concat([pl.scan_ipc(list(dir.glob('*.wsds'))).drop('__key__').rename({v[1]:k for k,v in ds.fields.items() if v[0] == dir.name}) for dir in Path(ds_path).iterdir() if dir.is_dir() and list(dir.glob('*.wsds'))], how='horizontal')#.select('pquality_scores_raw.pq').collect()
    return


@app.cell
def _(pl):
    # random access
    @timeit
    def _test():
        ds_path = 'youtube-cc/vad_ws'
        pl.concat([pl.scan_ipc((Path(ds_path)/dir).glob('*.wsds')) for dir in Path(ds_path).iterdir() if dir.is_dir()])
    return


@app.cell
def _(pl):
    pl.sql_expr('pquality_scores_raw$pq > 4')
    return


@app.cell
def _(pl):
    _e = pl.sql_expr('pq@pquality_scores_raw > 4')
    _e.meta.root_names(), _e.meta.output_name()
    return


@app.cell
def _(pl):
    pl.scan_ipc(list(Path('youtube-cc/vad_ws/pquality_scores_raw').glob('*.wsds')), include_file_paths="__shard__") \
        .sql("SELECT pq > 4 AS hq, __shard__ FROM self") \
        .sink_ipc(pl.PartitionParted('test-output', include_key=False, by='__shard__', file_path=lambda x: Path(x.keys[0].str_value).name),
                  mkdir=True, compression='zstd')
    # .sink_ipc('test-output.wsds', compression='zstd')
    return


@app.cell
def _(pl):
    pl.read_ipc('test-output/stanford-000000.wsds')
    return


@app.cell
def _(ds, pd):
    for _x in ds: break
    pd.DataFrame([{k:getattr(_x[k], 'shape', _x[k]) for k in _x.keys() if k !="audio"}])
    return


@app.cell
def _(ds):
    ds.fields
    return


@app.cell
def _(ds):
    # sequential access
    _iter = ds.__iter__()
    @timeit
    def _test():
        _x = next(_iter)
        _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_1.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_2.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_global.npy'], _x['atoks.npy']
    return


@app.cell
def _(ds):
    # random access
    @timeit
    def _test():
        _iter = iter(ds)
        _x = next(_iter)
        _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_1.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_2.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_global.npy'], _x['atoks.npy']
    return


@app.cell
def _(ds):
    # limited sequential access
    _iter = ds.__iter__(max_per_shard=64)
    @timeit
    def _test():
        _x = next(_iter)
        _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_1.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_2.npy'], _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_global.npy'], _x['atoks.npy']
    return


@app.cell
def _(ds):
    _iter = iter(ds)
    _x = next(_iter)
    _x['dtok_25hz_vocab_512x3_global_4p4_v2.dtok_level_1.npy']
    return


@app.cell
def _(ds):
    _iter = iter(ds)
    @timeit
    def _test():
        _x = next(_iter)
        for key in ds.fields.keys():
            _x[key]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
