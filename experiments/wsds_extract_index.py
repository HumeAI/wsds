import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium")

with app.setup:
    import tarfile
    import gzip
    import multiprocessing
    import functools

    import wsds

    import sqlite3
    from pathlib import Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# WSDS Dataset indices""")
    return


@app.cell
def _():
    test_shard = '/mnt/weka/data2/multimodal/core/youtube-cc/raw/audio/andrew-huberman-000000.tar'
    return (test_shard,)


@app.function
def list_keys_tarfile(input_shard):
    last_key = None
    if input_shard.endswith("gz"):
        o = gzip.open
    else:
        o = open
    with o(input_shard, 'rb') as f:
        for name in tarfile.TarFile(fileobj=f).getnames():
            path, name = name.rsplit('/', 1)
            key = name.split('.', 1)[0]
            if key != last_key: yield f'{path}/{key}'
            last_key = key


@app.cell
def _(test_shard):
    len(list(list_keys_tarfile(test_shard)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Build a key index for one shard

    Records the shard name, all keys and their positions. For segmented shards we are skipping sequentially numbered segments to reduce index size.
    """
    )
    return


@app.function
def shard_base_name(input_shard):
    name = input_shard.rsplit('/', 1)[1]
    if name.endswith('.gz'): return name.rsplit('.', 3)[0]
    else: return name.rsplit('.', 2)[0]


@app.function
# TODO:
# - add a function to generate based on mvad shards instead of downstream shards
def extract_index(input_shard, segmented=False):
    last_key = None
    index = []
    for i, key in enumerate(list_keys_tarfile(input_shard)):
        if segmented: key = key.rsplit('_', 1)[0]
        if key != last_key:
            index.append((key, i))
        last_key = key
    return {
        'shard_name': shard_base_name(input_shard),
        'index': index,
        'n_samples': i+1,
    }


@app.cell
def _():
    extract_index('/mnt/weka/data2/multimodal/core/youtube-cc/raw/audio/20vc-with-harry-stebbings-000000.tar')
    return


@app.cell
def _():
    extract_index('/mnt/weka/data2/multimodal/core/youtube-cc/processed-v3-pipeline-vad_ws/staging/transcription_wslang_raw/20vc-with-harry-stebbings-000000.transcription_wslang_raw.tar.gz', segmented=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Index all `youtube-cc` `transcription_wslang_raw` shards""")
    return


@app.cell
def _():
    shards = [str(x) for x in Path('/mnt/weka/data2/multimodal/core/youtube-cc/processed-v3-pipeline-vad_ws/staging/transcription_wslang_raw/').glob('*.transcription_wslang_raw.tar.gz') if '~' not in str(x)]
    print(len(shards))
    return (shards,)


@app.cell
def _(shards):
    extract_index(shards[0], segmented=True)
    return


@app.class_definition
# TODO:
# - add comulative segment duration to the index
# - add support for dataset splits (split on file-level or segment-level?)
class WSDSIndexWriter:
    def __init__(self, fname):
        self.fname = Path(fname)

    def __enter__(self):
        self.fname.unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.fname)
        print("opening", self.fname)

        self.conn.execute('''
        CREATE TABLE files (
            name TEXT PRIMARY KEY NOT NULL,
            shard_id INTEGER NOT NULL,
            offset INTEGER NOT NULL
        );''')
        self.conn.execute('''
        CREATE UNIQUE INDEX files_name ON files (name);
        ''')
        self.conn.execute('''
        CREATE UNIQUE INDEX files_shard_id_offset ON files (shard_id, offset);
        ''')
        self.conn.execute('''
        CREATE TABLE shards (
            shard_id INTEGER PRIMARY KEY,
            shard TEXT NOT NULL,
            n_samples INTEGER NOT NULL
        );''')
        self.conn.execute('''
        CREATE UNIQUE INDEX shard_name ON shards (shard);
        ''')

        return self

    def append(self, s):
        shard_id = self.conn.execute('INSERT INTO shards (shard, n_samples) VALUES (?, ?);',
                                     (s['shard_name'], s['n_samples'])).lastrowid
        for name, offset in s['index']:
            try:
                self.conn.execute('INSERT INTO files (name, shard_id, offset) VALUES (?, ?, ?);',
                                  (name, shard_id, offset))
            except sqlite3.IntegrityError as err:
                if err.args[0] == 'UNIQUE constraint failed: files.name':
                    old_shard = self.conn.execute('SELECT s.shard FROM shards AS s, files AS f WHERE f.name == ?', (name,)).fetchone()[0]
                    raise ValueError(f"Detected duplicate file name: {repr(name)} in shard \n{repr(s['shard_name'])}, previously seen in {repr(old_shard)}")
                raise

    def __exit__(self, exc_type, exc_value, traceback):
        print("exiting:", exc_type, exc_value, traceback)
        if exc_type is None:
            print("closing", self.fname)
            self.conn.commit()
            self.conn.close()


@app.cell
def _(mo, shards):
    with wsds.AtomicFile('youtube-cc/vad_ws/index.sqlite3') as fname:
        with WSDSIndexWriter(fname) as index:
            with multiprocessing.Pool(32) as p:
                for r in mo.status.progress_bar(p.imap_unordered(functools.partial(extract_index, segmented=True), shards), total=len(shards)):

                    index.append(r)
    return


@app.cell
def _(mo):
    def _():
        shards = [str(x) for x in Path('/mnt/weka/data2/multimodal/core/youtube-cc/raw/audio/').glob('*.tar') if '~' not in str(x)]
        with wsds.AtomicFile('youtube-cc/source/index.sqlite3') as fname:
            with WSDSIndexWriter(fname) as index:
                with multiprocessing.Pool(32) as p:
                    for r in mo.status.progress_bar(p.imap_unordered(functools.partial(extract_index, segmented=False), shards), total=len(shards)):
                        if r['shard_name'] == 'freecodecamp~org-000005':
                            print(r)
                        index.append(r)
    _()
    return


@app.class_definition
# TODO:
# - select random sample
# - how often we can reset the stream position and still get good performance
class WSIndex:
    def __init__(self, fname):
        self.fname = fname
        if not Path(fname).exists(): raise ValueError(f"WSIndex not found: {fname}")
        self.conn = sqlite3.connect(f'file:{fname}?immutable=1,ro=True', uri=True)

    @functools.cached_property
    def n_shards(self):
        return self.conn.execute('SELECT COUNT(n_samples) FROM shards;').fetchone()[0]

    @functools.cached_property
    def n_files(self):
        return self.conn.execute('SELECT COUNT(*) FROM files;').fetchone()[0]

    @functools.cached_property
    def n_samples(self):
        return self.conn.execute('SELECT SUM(n_samples) FROM shards;').fetchone()[0]

    def query(self, query, *args):
        return self.conn.execute(query, args)

    def __repr__(self):
        return f"WSIndex({repr(self.fname)})"


@app.cell
def _():
    _idx = WSIndex('rawvad_index_denorm.sqlite3')
    print(f"""{_idx = }
    {_idx.n_samples = }
    {_idx.n_files = }
    {_idx.n_shards = }
    """)
    return


@app.function
def make_key(src_file:str, segmentation_kind:str, segment_id:int):
    """Make a composite string key from source file name, segmentation kind and sequential segment id.

    >>> make_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 'rawvad', 1254)
    '20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_rawvad_00001254'
    """

    assert(type(segment_id) == int)
    assert('_' not in segmentation_kind)
    return f"{src_file}_{segmentation_kind}_{segment_id:08d}"


@app.function
def parse_key(key:str):
    """Parse a composite string key into the source file name, segmentation kind and sequential segment id.

    >>> parse_key('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf_rawvad_00001254')
    ('20VC with Harry Stebbings/[HGF1a_ULnik] Gina Gotthilf', 'rawvad', 1254)
    """

    src_file, segmentation_kind, segment_id = key.rsplit('_', 2)
    return src_file, segmentation_kind, int(segment_id)


@app.cell
def _():
    _in = '20VC with Harry Stebbings/[HGF1a_ULnik] There Has Never Been a Better Time to Invest in Latin American Startups -- Gina Gotthilf', 'rawvad', 1254
    assert parse_key(make_key(*_in)) == _in
    return


@app.cell
def _(mo):
    #make_key, parse_key

    import doctest

    failures, success = doctest.testmod(verbose=False)
    mo.md(f"Success: {success}, Failures: {failures}")
    return


@app.cell
def _(mo, needles):
    _query = 'SELECT shard, offset FROM files WHERE name == ?'
    for _n in mo.status.progress_bar(needles):
        with sqlite3.connect('file:rawvad_index_denorm.sqlite3?immutable=1', uri=True) as _connection:
            for _x in _connection.execute(_query, (_n,)):
                pass
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
