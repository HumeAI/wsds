import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    from hume_wsds import WSDataset
    import polars as pl
    import marimo as mo
    import numpy as np
    import moutils
    return WSDataset, alt, mo, np


@app.cell
def _():
    datasets = {
        'librilight': '/mnt/weka/data-wsds/librilight/v3-vad_ws',
        'fb_ears': '/mnt/weka/data-wsds/fb_ears/v4-vad_ws',
        'voxceleb': '/mnt/weka/data-wsds/voxceleb/v3-vad_ws',
        'voxceleb2': '/mnt/weka/data-wsds/voxceleb2/v3-vad_ws',
        'voxceleb2_2': '/mnt/weka/data-wsds/voxceleb2_2/v3-vad_ws',
        # 'podcasts-en': '/mnt/weka/data-wsds/podcasts-en/v3-vad_ws',
        'youtube-cc': '/mnt/weka/jpc/youtube-cc/vad_ws_continuous/',
        'wyndlabs_1M-it diarized': '/mnt/weka/data-wsds/wyndlabs_1M-it/v5-diarized_continuous/',
    }
    return (datasets,)


@app.cell
def _(datasets, mo):
    ds_path = mo.ui.dropdown(datasets)
    return (ds_path,)


@app.cell
def _(ds_path, mo):
    mo.md(f"""## Select a dataset: {ds_path}""")
    return


@app.cell
def _():
    from functools import lru_cache
    return


@app.cell
def _(ds):
    ds.random_sample()
    return


@app.cell
def _(WSDataset, ds_path, mo, np):
    mo.stop(not ds_path.value)
    ds = WSDataset(ds_path.value)
    keys = [k for k,v in ds.random_sample().items() if np.isscalar(v) and not isinstance(v, str)]
    txt_key = [k for k in ds.random_sample().keys() if k.endswith('txt')][0]
    keys += [
        f"LENGTH(CAST(`{txt_key}` AS string)) / (tend - tstart) AS cps",
        "tend - tstart AS duration",
    ]
    data = ds.sql('__key__', *keys).sample(10000)
    return data, ds


@app.cell
def _(data, ds_path, mo):
    mo.stop(not ds_path.value)
    x_key = mo.ui.dropdown(data.columns)
    y_key = mo.ui.dropdown(data.columns)
    return x_key, y_key


@app.cell
def _(data, mo):
    import plotly.express as px
    fig = px.scatter_matrix(data, dimensions=[k for k in data.columns if k not in {'__key__', 'tstart', 'tend', 'tmax', 'i'}])
    fig.update_traces(showupperhalf=False, diagonal_visible=False)
    fig = mo.ui.plotly(fig)
    mo.md(f"""
    ## Metrics overview

    For more interactive filtering see below.
    {fig}
    """)
    return


@app.cell
def _(mo, x_key, y_key):
    mo.md(f"""## Select metrics for X: {x_key} and Y: {y_key}""")
    return


@app.cell
def _(alt, data, mo, x_key, y_key):
    chart = mo.ui.altair_chart(alt.Chart(data).mark_point().encode(
        x=x_key.value,
        y=y_key.value,
    ))
    return (chart,)


@app.cell
def _(mo):
    def show_sample(_x):
        try:
            return mo.md(f"""
    &nbsp;
    /// admonition | {_x['__key__']}
    `full_transcript:` {_x.get_one_of('transcription_wslang_raw.txt', 'txt', 'transcription_wslang_continuous.txt')}  
    `pq`: {_x.get('pq')}  
    `language`: {_x.get('language_whisper.txt')}  
    {mo.audio(_x['audio'].load(24000).numpy(), rate=24000)
               if _x['audio'].tend - _x['audio'].tstart > 0 else _EMPTY}  
    """)
        except FileNotFoundError as _e:
            return mo.md(f"""
    &nbsp;
    /// error | **Missing shard**: {_e.args[1]}
    """)
    return (show_sample,)


@app.cell
def _():
    import json
    return


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(chart, ds, mo, show_sample):
    mo.vstack([show_sample(ds[_x]) for _x in chart.value['__key__'][:5]])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
