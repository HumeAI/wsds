import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from omegaconf import OmegaConf
    import marimo as mo
    from hume_wsds import WSDataset
    import polars as pl
    import random
    import time
    return WSDataset, mo, pl, time


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
    q = mo.query_params()
    ds_path = mo.ui.dropdown(datasets, value=q.get('dataset', None))

    N = 5

    mo.md(f"## Select a dataset: {ds_path}")
    return N, ds_path, q


@app.cell
def _(N, mo):
    btn = mo.ui.run_button('success', label=f'Show another {N} random samples')
    sql_query = mo.ui.code_editor(value="`language_whisper.txt` != 'en'", language="sql", debounce=200)
    return btn, sql_query


@app.cell
def _(WSDataset, ds_path, mo, q):
    mo.stop(not ds_path.value)
    ds = WSDataset(ds_path.value)
    q.set("dataset", ds_path.selected_key)
    return (ds,)


@app.cell
def _(ds_path, mo, sql_query):
    mo.stop(not ds_path.value)
    mo.vstack([
        mo.md("## SQL query:"),
        sql_query,
        mo.md("\nScroll to the bottom for some examples.")
    ], gap=1)
    return


@app.cell
def _(
    N,
    btn,
    ds,
    mo,
    pl,
    show_error,
    show_file_not_found_error,
    show_header,
    show_sample,
    sql_query,
    time,
):
    if btn.value: pass
    _start_time = time.time()
    mo.output.clear()
    with mo.status.spinner(title=f"Executing SQL on {ds.index.n_samples:,} samples...") as _spinner:
        _EMPTY = "empty segment\n"
        _first = True
        try:
            with mo.redirect_stdout():
                for _x in ds.filtered(sql_query.value, N=5):
                    if _first:
                        show_header(sql_query.value, ds.last_query_n_samples)
                        _first = False

                    _spinner.update(f'Displaying {N} random results...')
                    try:
                        show_sample(_x)
                    except FileNotFoundError as _e:
                        show_file_not_found_error(_e)
        except KeyError as _e:
            show_error(f"**KeyError**: Key `{_e.args[0]}` not available in this dataset.")
        except pl.exceptions.SQLInterfaceError as _e:
            show_error(f"**SQL Error**: {_e}")
        
    mo.output.append(mo.md(f"Computed in {time.time() - _start_time:.1f}s"))
    return


@app.cell
def _(N, ds, mo):
    def show_header(query, total):
        mo.output.append(mo.md(f"""
    ### Your query:
    ```sql
    {query}
    ```
    ### Your query matched {total} ({total/ds.index.n_samples*100:.2f}% of the dataset) results. Showing {N} random samples:
    """))

    def show_sample(_x):
        mo.output.append(mo.md(f"""
    &nbsp;
    /// admonition | {_x['__key__']}<br />
    `full_transcript:` {_x.get_one_of('transcription_wslang_raw.txt', 'txt', 'transcription_wslang_continuous.txt')}<br />
    `pq`: {_x.get('pq')}<br />
    `language`: {_x.get('language_whisper.txt')}<br />
    {mo.audio(_x['audio'].load(24000).numpy(), rate=24000)
               if _x['audio'].tend - _x['audio'].tstart > 0 else _EMPTY}<br />
    """))

    def show_file_not_found_error(_e):
        mo.output.append(mo.md(f"""
    &nbsp;
    /// error | **Missing shard**: {_e.args[1]}
    """))

    def show_error(err):
        mo.output.append(mo.md(f"""
    &nbsp;
    /// error | {err}

    Please check which keys are available at the bottom of the page."""))
    return show_error, show_file_not_found_error, show_header, show_sample


@app.cell
def _(btn, ds_path, mo):
    mo.stop(not ds_path.value)
    mo.center(btn)
    return


@app.cell
def _(ds, ds_path, mo):
    mo.stop(not ds_path.value)

    _sample = ds.random_sample()
    _r = ""
    for _k in _sample.keys():
        if not _k.startswith('dtok') and _k != 'audio':
            _kp = f"`{_k}`" if '.' in _k else _k
            try:
                _r += f"{_kp} = {_sample.__repr_field__(_k)}\n"
            except FileNotFoundError:
                pass

    mo.md(f"""
    ### Examples:

    ```sql
    -- find non-english samples (based on Whisper detections) in one of the English datasets:
    `language_whisper.txt` != 'en'
    -- keep in mind that dots in columns names need to be quoted with backticks

    -- metric based filtering:
    pq <= 3.0
    snr <= 20

    -- cps filtering
    LENGTH(CAST(txt AS string)) / (tend - tstart) NOT BETWEEN 5 AND 40
    -- caveat: this does not work on all datasets unfortunatelly because we don't yet have tstart end tend everywhere

    -- you can also do full text search and even mix it with boolean expressions:
    CAST(txt AS string) ILIKE '%hello%' AND pq < 3
    -- caveat: right now `txt` works on most datasets, but some use a different column name
    -- (like `transcription_wslang_continuous.txt` or `transcription_wslang_raw.txt`)
    ```

    ### Available columns:

    ```python
    {_r}
    ```
    Please remember to backtick-quote columns with dots in their names.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
