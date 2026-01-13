import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from omegaconf import OmegaConf
    import marimo as mo
    from hume_wsds import WSDataset
    import polars as pl
    import random
    import time
    from diff_match_patch import diff_match_patch

    import plotly.subplots as sp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde
    import numpy as np
    import os
    return (
        OmegaConf,
        WSDataset,
        diff_match_patch,
        gaussian_kde,
        go,
        make_subplots,
        mo,
        np,
        os,
        pl,
        sp,
        time,
    )


@app.cell
def _():
    # datasets = {
    #     'librilight': '/mnt/weka/data-wsds/librilight/v3-vad_ws',
    #     'fb_ears': '/mnt/weka/data-wsds/fb_ears/v4-vad_ws',
    #     'voxceleb': '/mnt/weka/data-wsds/voxceleb/v3-vad_ws',
    #     'voxceleb2': '/mnt/weka/data-wsds/voxceleb2/v3-vad_ws',
    #     'voxceleb2_2': '/mnt/weka/data-wsds/voxceleb2_2/v3-vad_ws',
    #     # 'podcasts-en': '/mnt/weka/data-wsds/podcasts-en/v3-vad_ws',
    #     'youtube-cc': '/mnt/weka/data-wsds/youtube-cc/v4-vad_ws_continuous',
    #     'wyndlabs_1M-en': '/mnt/weka/data-wsds/wyndlabs_1M-en/v4-vad_ws_continuous',
    #     'wyndlabs_1M-de': '/mnt/weka/data-wsds/wyndlabs_1M-de/v4-vad_ws_continuous',
    #     'wyndlabs_1M-fr': '/mnt/weka/data-wsds/wyndlabs_1M-fr/v4-vad_ws_continuous',
    #     # 'wyndlabs_1M-it diarized': '/mnt/weka/data-wsds/wyndlabs_1M-it/v5-diarized_continuous/',
    # }
    return


@app.cell
def _(os):
    root = "/mnt/weka/data-wsds/"
    datasets = {
        dir_name : os.path.join(root, dir_name) for dir_name in os.listdir(root)
    }
    return (datasets,)


@app.cell
def _(datasets, mo):
    # q = mo.query_params()
    ds_root = mo.ui.dropdown(datasets)
    return (ds_root,)


@app.cell
def _(ds_root, mo, os):
    ds_dirs = [d for d in os.listdir(ds_root.value) if d.startswith("v")]
    ds_artifacts = mo.ui.dropdown(ds_dirs)
    mo.md(f"## Select a dataset: {ds_root} {ds_artifacts}")
    return (ds_artifacts,)


@app.cell
def _(ds_artifacts, ds_root, mo, os):
    mo.stop(not ds_root.value or not ds_artifacts.value)
    ds_path = mo.ui.text(os.path.join(ds_root.value, ds_artifacts.value))
    return (ds_path,)


@app.cell
def _(WSDataset, ds_path, mo):
    mo.stop(not ds_path.value)

    ds = WSDataset(ds_path.value)
    # q.set("dataset", ds_path.selected_key)
    return (ds,)


@app.cell
def _(
    OmegaConf,
    advanced_mode,
    ds_path,
    generate_yaml_from_ui,
    mo,
    sql_editor,
    yaml_generate_sql_filter,
):
    mo.stop(not ds_path.value)

    filter_str = sql_editor.value if advanced_mode.value else generate_yaml_from_ui()
    filters = OmegaConf.create(filter_str).get('filters')
    keep_cols, accept_query, reject_query = yaml_generate_sql_filter(filters)
    return accept_query, filter_str, filters, keep_cols


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

    mo.callout(mo.md(f"""
    ## Available columns
    ```python
    {_r}
    ```
    """
    ))
    return


@app.cell
def _():
    # mo.stop(not ds_path.value)

    # _sample = ds.random_sample()
    # _r = ""
    # for _k in _sample.keys():
    #     if _k.startswith('dtok'):
    #         _kp = f"`{_k}`" if '.' in _k else _k
    #         try:
    #             _r += f"{_kp} = {_sample.__repr_field__(_k)}\n"
    #         except FileNotFoundError:
    #             pass

    # mo.callout(mo.md(f"""
    # ## Available columns
    # ```python
    # {_r}
    # ```
    # Make sure that the **text** and **language** columns are available for the current dataset.
    # """
    # ))
    return


@app.cell
def _(ds, ds_path, mo):
    mo.stop(not ds_path.value)

    _sample = ds.random_sample()
    txt_options = []
    lang_options = []
    for _k in _sample.keys():
        if "language_" in _k:
            lang_options.append(_k)
        elif '.txt' in _k:
            txt_options.append(_k)
    return lang_options, txt_options


@app.cell
def _(ds_path, lang_options, mo, txt_options):
    mo.stop(not ds_path.value)

    N = mo.ui.number(value=100000)
    txt_column = mo.ui.dropdown(options=txt_options, value=txt_options[0])
    language_column = mo.ui.dropdown(options=lang_options, value=lang_options[0])
    secondary_transcripts = mo.ui.multiselect(options=txt_options)

    mo.callout(
        mo.vstack([
            mo.md("## Presets"),
            mo.md("---"),
            mo.md(f"**Sample size**: {N}"),
            mo.md(f"**Primary transcript**: {txt_column}"),
            mo.md(f"**Secondary transcript(s)**: {secondary_transcripts}"),
            mo.md(f"**Language column**: {language_column}"),
        ])
    )
    return N, language_column, secondary_transcripts, txt_column


@app.cell
def _(ds_path, mo):
    mo.stop(not ds_path.value)

    # Update secondary transcript options to exclude the currently selected primary transcript
    # secondary_txt_options = [opt for opt in txt_options if opt != txt_column.value]
    # secondary_transcripts.options = secondary_txt_options

    # # Clear selected values if they're no longer valid options
    # current_values = secondary_transcripts.value
    # valid_values = [val for val in current_values if val in secondary_txt_options]
    # if len(valid_values) != len(current_values):
    #     secondary_transcripts.value = valid_values
    return


@app.cell
def _(ds_path, mo):
    mo.stop(not ds_path.value)

    explorer_enabled = mo.ui.checkbox(value=True, label="")
    audio_enabled = mo.ui.checkbox(value=True, label="")
    show_rejected_samples = mo.ui.checkbox(value=False, label="")
    show_dtoks = mo.ui.checkbox(value=False, label="")
    show_diffs = mo.ui.checkbox(value=False, label="")
    return (
        audio_enabled,
        explorer_enabled,
        show_diffs,
        show_dtoks,
        show_rejected_samples,
    )


@app.cell
def _(mo):
    def display_on(text, condition):
        return mo.md(f"**{text}**" if condition.value else text)
    return (display_on,)


@app.cell
def _(
    audio_enabled,
    display_on,
    ds_path,
    explorer_enabled,
    mo,
    show_diffs,
    show_dtoks,
    show_rejected_samples,
):
    mo.stop(not ds_path.value)

    mo.callout(
        mo.vstack([
            mo.md("## Display settings"),
            mo.md("---"),
            mo.hstack([explorer_enabled, display_on("Data Explorer", explorer_enabled)], justify='start', align='center', gap=1),
            mo.hstack([audio_enabled, display_on("Audio samples", audio_enabled)], justify='start', align='center', gap=1),
            mo.hstack([mo.md("&nbsp;&nbsp;&nbsp;&nbsp;"), show_rejected_samples, display_on("Show rejected samples", show_rejected_samples)], justify='start', align='center', gap=1),
            mo.hstack([mo.md("&nbsp;&nbsp;&nbsp;&nbsp;"), show_dtoks, display_on("Show dtoks", show_dtoks)], justify='start', align='center', gap=1),
            mo.hstack([mo.md("&nbsp;&nbsp;&nbsp;&nbsp;"), show_diffs, display_on("Show diffs", show_diffs)], justify='start', align='center', gap=1),
        ])
    )
    return


@app.cell
def _(ds_path, mo):
    mo.stop(not ds_path.value)

    # Create checkboxes for each filter
    pq_enabled = mo.ui.checkbox(value=True, label="")
    snr_enabled = mo.ui.checkbox(value=True, label="")
    cps_enabled = mo.ui.checkbox(value=True, label="")
    dur_enabled = mo.ui.checkbox(value=True, label="")
    lang_enabled = mo.ui.checkbox(value=True, label="")
    return cps_enabled, dur_enabled, lang_enabled, pq_enabled, snr_enabled


@app.cell
def _(mo):
    # Create toggle for advanced mode
    advanced_mode = mo.ui.switch(label="Advanced Mode", value=False)
    return (advanced_mode,)


@app.cell
def _(
    cps_enabled,
    cps_range,
    dur_enabled,
    duration_range,
    lang_enabled,
    language_column,
    language_input,
    pq_enabled,
    pq_threshold,
    snr_enabled,
    snr_threshold,
    txt_column,
):
    def generate_yaml_from_ui():
        yaml_str = "filters:\n"
        tab = " " * 2
        if pq_enabled.value:
            yaml_str += tab + f"min_pq: pq >= {pq_threshold.value}\n"
        if snr_enabled.value:
            yaml_str += tab + f"min_snr: snr >= {snr_threshold.value}\n"
        if cps_enabled.value:
            yaml_str += tab + f"cps_range: LENGTH(CAST(`{txt_column.value.strip()}` AS string)) / (tend - tstart) BETWEEN {cps_range.value[0]} AND {cps_range.value[1]}\n"
        if dur_enabled.value:
            yaml_str += tab + f"dur_range: (tend - tstart) BETWEEN {duration_range.value[0]} AND {duration_range.value[1]}\n"
        if lang_enabled.value:
            quoted_langs = ', '.join([f"'{lang}'" for lang in language_input.value])
            yaml_str += tab + f"language: CAST(`{language_column.value.strip()}` AS STRING) IN ({quoted_langs})"
        return yaml_str
    return (generate_yaml_from_ui,)


@app.cell
def _(
    advanced_mode,
    cps_enabled,
    display_on,
    dur_enabled,
    lang_enabled,
    language_column,
    mo,
    pq_enabled,
    snr_enabled,
    txt_column,
):
    # Create UI elements for each filter
    pq_threshold = mo.ui.number(start=0, stop=20, value=6, step=0.1)
    snr_threshold = mo.ui.number(start=0, stop=100, value=10, step=1)
    cps_range = mo.ui.range_slider(start=0, stop=50, value=[5, 30], step=0.5)
    duration_range = mo.ui.range_slider(start=0, stop=30, value=[0.25, 30], step=0.05)
    langs = ['en', 'ja', 'fr', 'es', 'de', 'it', 'pt', 'ar', 'pl', 'ko', 'ru']
    language_input = mo.ui.multiselect(options=langs, value=['en'])

    # SQL editor for advanced mode
    sql_editor = mo.ui.code_editor(
        value=f"""filters:
        min_snr: snr >= 10.0
        min_pq: pq >= 5.0
        cps_range: LENGTH(CAST(`{txt_column.value.strip()}` AS string)) / (tend - tstart) BETWEEN 5 AND 40
        duration_range: tend - tstart BETWEEN 1 AND 30
        language: CAST(`{language_column.value.strip()}` AS STRING) IN ('en', 'fr')""",  # Initial SQL
        language="yaml",
        label="Custom .yaml filter. Keys are descriptive column names, values are SQL expressions."
    )

    # Display the UI elements
    mo.callout(
        mo.vstack([
            mo.hstack([mo.md("## Filters"), advanced_mode], justify="space-between", align="center"),
            mo.md("---"),
            # Conditionally show simple or advanced
            sql_editor if advanced_mode.value else mo.vstack([
                mo.hstack([pq_enabled, display_on("PQ Min", pq_enabled), pq_threshold], justify="start", align="center"),
                mo.hstack([snr_enabled, display_on("SNR Min", snr_enabled), snr_threshold], justify="start", align="center"),
                mo.hstack([cps_enabled, display_on("CPS Range", cps_enabled), cps_range], justify="start", align="center"),
                mo.hstack([dur_enabled, display_on("Duration Range", dur_enabled), duration_range], justify="start", align="center"),
                mo.hstack([lang_enabled, display_on("Language", lang_enabled), language_input], justify="start", align="center"),
            ])
        ])
    )
    return (
        cps_range,
        duration_range,
        language_input,
        pq_threshold,
        snr_threshold,
        sql_editor,
    )


@app.cell
def _():
    import re

    def generate_sql_with_aliases(conditions_dict):
        # Extract formulas and create SELECT with aliases
        select_clauses = []
        where_clauses = []

        for alias, condition in conditions_dict.items():
            # Split on first occurrence of comparison operator
            for op in ['BETWEEN', '>=', '<=', '!=', '<>', '=', '>', '<', 'IN', 'LIKE', 'ILIKE', 'IS']:
                if f' {op} ' in condition.upper():
                    parts = re.split(f'\\s+{op}\\s+', condition, maxsplit=1, flags=re.IGNORECASE)
                    formula = parts[0].strip()
                    rest = parts[1].strip() if len(parts) > 1 else ''

                    select_clauses.append(f"{formula} AS {alias}")
                    where_clauses.append(f"{alias} {op} {rest}")
                    break
            else:
                # No operator found, just use as-is
                select_clauses.append(f"{condition} AS {alias}")

        return select_clauses, where_clauses
    return (generate_sql_with_aliases,)


@app.cell
def _(ds_path, mo):
    mo.stop(not ds_path.value)
    filter_btn = mo.ui.run_button('success', label='Filter samples', full_width=True)
    sample_btn = mo.ui.run_button('success', label=f'Sample 10 more random audios', full_width=True)
    mo.center(filter_btn)
    return filter_btn, sample_btn


@app.cell
def _(generate_sql_with_aliases, language_column, txt_column):
    def yaml_generate_sql_filter(filters):
        keep_cols, conditions = generate_sql_with_aliases(filters)
        base_cols = {
            '__key__': '__key__',
            'pq': 'pq',
            'snr': 'snr',
            'cps': f"LENGTH(CAST(`{txt_column.value.strip()}` AS STRING)) / (tend - tstart)",
            'duration': '(tend - tstart)',
            'language': f"CAST(`{language_column.value.strip()}` AS STRING)"
        }
        base_cols = [f"{v} AS {k}" for k,v in base_cols.items() if k not in filters.keys()]
        keep_cols = keep_cols + base_cols
        where = " AND ".join(conditions) if conditions else "1=1"
        select = f"SELECT * FROM self WHERE ({where})"
        reject = f"SELECT * FROM self WHERE NOT ({where})"
        return keep_cols, select, reject
    return (yaml_generate_sql_filter,)


@app.cell
def _(
    cps_enabled,
    cps_range,
    dur_enabled,
    duration_range,
    lang_enabled,
    language_column,
    language_input,
    pq_enabled,
    pq_threshold,
    snr_enabled,
    snr_threshold,
    txt_column,
):
    def generate_audio_sql_filter(filters):
        accept_query = " AND ".join(filters.values())
        reject_query = f"NOT ({accept_query})"
        return accept_query, reject_query

    def generate_sql_filter():
        cols = {
            '__key__': '__key__',
            # 'idx': 'i',
            'pq': 'pq',
            'snr': 'snr',
            'cps': f"LENGTH(CAST(`{txt_column.value.strip()}` AS STRING)) / (tend - tstart)",
            'duration': '(tend - tstart)',
            'language': f"CAST(`{language_column.value.strip()}` AS STRING)"
        }

        conditions = []
        if pq_enabled.value: conditions.append(f"pq >= {pq_threshold.value}")
        if snr_enabled.value: conditions.append(f"snr >= {snr_threshold.value}")
        if cps_enabled.value: conditions.append(f"cps BETWEEN {cps_range.value[0]} AND {cps_range.value[1]}")
        if dur_enabled.value: conditions.append(f"duration BETWEEN {duration_range.value[0]} AND {duration_range.value[1]}")
        if lang_enabled.value:
            quoted_langs = ', '.join([f"'{lang}'" for lang in language_input.value])
            conditions.append(f"language IN ({quoted_langs})")

        where = " AND ".join(conditions) if conditions else "1=1"
        keep_cols = [f"{v} AS {k}" for k,v in cols.items()]
        select = f"SELECT * FROM self WHERE ({where})"
        reject = f"SELECT * FROM self WHERE NOT ({where})"
        return keep_cols, select, reject
    return (generate_audio_sql_filter,)


@app.cell
def _(
    diff_match_patch,
    filter_str,
    gaussian_kde,
    go,
    language_column,
    make_subplots,
    mo,
    np,
    secondary_transcripts,
    show_diffs,
    show_dtoks,
    sp,
    txt_column,
):
    def filter_df(df, query, n_samples=100000):
        filtered_df = df.sql(query)
        samples = len(filtered_df)
        duration = filtered_df['duration'].sum() / 3600

        filtered_df = filtered_df.sample(min(n_samples, len(filtered_df)))
        return filtered_df, samples, duration

    def plot_stats(pre_df, post_df):
        fig = sp.make_subplots(rows=2, cols=2,
                               subplot_titles=("SNR", "PQ", "Duration", "CPS"),
                               specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                      [{"type": "scatter"}, {"type": "scatter"}]])

        for col, row, c in [("snr", 1, 1), ("pq", 1, 2), ("duration", 2, 1), ("cps", 2, 2)]:
            for df, name, color in [(pre_df, "Pre-filter", "green"), (post_df, "Post-filter", "red")]:
                # Filter out NaN and inf values
                data = df[col].drop_nulls()
                data = data.filter(data.is_finite())  # Removes NaN, +inf, -inf

                if len(data) < 2:  # Need at least 2 points for KDE
                    continue

                kde = gaussian_kde(data.to_numpy())

                if name == "Pre-filter":
                    x = np.linspace(data.quantile(0.005), data.quantile(0.995), 200) if col != "duration" else np.linspace(data.quantile(0.005), data.quantile(0.975), 200)
                else:
                    x = np.linspace(data.min(), data.max(), 200)
                y = kde(x)
                fig.add_trace(go.Scatter(x=x, y=y, name=name, fill='tozeroy',
                                        line_color=color, opacity=0.6,
                                        showlegend=(row==1 and c==1)),
                             row=row, col=c)

        fig.update_layout(height=600)
        return fig

    def lang_comparison(pre_df, post_df):
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                            subplot_titles=("Pre-filter", "Post-filter"))

        pre_counts = pre_df['language'].value_counts()
        post_counts = post_df['language'].value_counts()

        fig.add_trace(go.Pie(labels=pre_counts['language'], values=pre_counts['count'], textinfo='label+percent'),
                      row=1, col=1)
        fig.add_trace(go.Pie(labels=post_counts['language'], values=post_counts['count'], textinfo='label+percent'), row=1, col=2)
        fig.update_traces(textposition='inside')
        fig.update_layout(height=400, showlegend=True)
        return fig

    def show_header(query, stats):
        return mo.vstack([
            # Query section with better formatting
            mo.vstack([
                mo.md("### 🔍 Your Query"),
                mo.ui.code_editor(filter_str, language="yaml"),
            ]),

            # Match statistics in cards side by side
            mo.hstack([
                mo.callout(
                    mo.vstack([
                        mo.md("#### Samples Matched"),
                        mo.md(f"# {stats['n_filtered']:,}"),
                        mo.md(f"**{stats['n_filtered']/stats['n_total']*100:.1f}%** of {stats['n_total']:,} total")
                    ], align="center"),
                    kind="success"
                ),
                mo.callout(
                    mo.vstack([
                        mo.md("#### Hours Matched"),
                        mo.md(f"# {stats['filtered_duration']:,.1f}h"),
                        mo.md(f"**{stats['filtered_duration']/stats['total_duration']*100:.1f}%** of {stats['total_duration']:,.1f}h total")
                    ], align="center"),
                    kind="success"
                )
            ], justify="center", align="center", gap=3),

            # Filter percentages table with better title
            mo.md("### 📊 Filter Impact by Metric (% kept by filter)"),
            mo.ui.table(stats['kept_by_filter']),

            # Statistics plots with better spacing
            mo.md("\n\n"),
            mo.md("### 📈 Distribution Comparison"),
            mo.md(f"#### Based on {stats['n_sample']} random samples <span style='color:green;'>**Pre-**</span> and <span style='color:red;'>**Post-**</span> filtering"),
            mo.md("---")
        ], gap=1)

    def show_ds(ds, title):
        res = []
        color = "green" if "accept" in title.lower() else "red"
        icon = "✅" if "accept" in title.lower() else "❌"
        res.append(
            mo.md(f"## {icon} <span style='color:{color};'>{title} samples</span>"),
        )
        for _x in ds:
            try:
                res.append(show_sample(_x))
            except FileNotFoundError as _e:
                res.append(show_file_not_found_error(_e))
        return mo.vstack(res)

    def clip_middle(s, max_len=50):
        if len(s) <= max_len:
            return s
        half = (max_len - 3) // 2
        return s[:half] + "..." + s[-half:]

    def token_entry(_x, col):
        return mo.md(f"```{_x.get(col)}```")

    def format_transcript(_x, col):
        val = _x.get(col, "").lower()
        if val.startswith("b'") or val.startswith('b"'):
            val = val[2:-1]
        return val

    def show_sample(_x):
        if 'audio' not in _x:
            show_error(_x['__key__'])

        # Build transcript display
        primary_text = format_transcript(_x, txt_column.value)
        transcript_display = [mo.md(f"`{txt_column.value}:` {primary_text}")]

        # Add secondary transcripts if any are selected
        if secondary_transcripts.value:
            for secondary_txt in secondary_transcripts.value:
                transcript_text = format_transcript(_x, secondary_txt)
                if transcript_text:
                    if show_diffs.value:
                        dmp = diff_match_patch()
                        diffs = dmp.diff_main(primary_text, transcript_text)
                        dmp.diff_cleanupSemantic(diffs)
                        html_diff = dmp.diff_prettyHtml(diffs)
                        transcript_display.append(
                            mo.vstack([
                                mo.md(f"**`{secondary_txt}`:**"),
                                mo.Html(html_diff)
                            ])
                        )
                    else:
                        transcript_display.append(mo.md(f"`{secondary_txt}:` {transcript_text}"))
        transcript_display = mo.vstack(transcript_display)

        _EMPTY = "empty segment\n"
        sample_content = mo.md(f"""
    &nbsp;
    /// admonition | {clip_middle(_x['__key__'])}
    <span style='color:blue;'>`lang:` {_x.get(language_column.value)} • `cps:` {len(_x.get(txt_column.value)) / (_x.get('tend') - _x.get('tstart') + 1e-6):.2f} • `snr:` {_x.get('snr'):.2f} • `pq:` {_x.get('pq'):.2f}</span>\n
    {transcript_display}
    {mo.audio(_x['audio'].load(24000).numpy(), rate=24000)
               if _x['audio'].tend - _x['audio'].tstart > 0 else _EMPTY}
    """)

        # Conditionally add dtok accordion
        if show_dtoks.value:
            dtok_cols = {
                "Global": token_entry(_x, "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_global.npy"),
                "L1": token_entry(_x, "dtok_v2_ml_50hz_32x16384_graphemes_key16k.dtok_level_1_16k.npy"),
                "L2": token_entry(_x, "dtok_v2_ml_25hz_32x16384_graphemes_v2_encoder.dtok_level_2.npy"),
            }
            dtok_ui = mo.accordion(dtok_cols)
            return mo.vstack([sample_content, dtok_ui])
        else:
            return sample_content

    def show_file_not_found_error(_e):
        return mo.md(f"""
    &nbsp;
    /// error | **Missing shard**: {_e.args[1]}
    """)

    def show_error(err):
        mo.output.append(mo.md(f"""
    &nbsp;
    /// error | {err}

    Please check which keys are available at the bottom of the page."""))
    return (
        filter_df,
        lang_comparison,
        plot_stats,
        show_ds,
        show_error,
        show_header,
    )


@app.cell
def _(
    N,
    accept_query,
    ds,
    explorer_enabled,
    filter_btn,
    filter_df,
    filters,
    keep_cols,
    lang_comparison,
    mo,
    pl,
    plot_stats,
    show_error,
    show_header,
    time,
):
    mo.stop(not filter_btn.value or not explorer_enabled.value)

    _start_time = time.time()
    mo.output.clear()
    with mo.status.spinner(title=f"Executing SQL on {ds.index.n_samples:,} samples...") as _spinner:
        _EMPTY = "empty segment\n"
        try:
            with mo.redirect_stdout():
                # get % kept by filter
                kept_by_filter = ds.sql(*[f"{v} AS {k}" for k,v in filters.items()])
                kept_by_filter = kept_by_filter.select((pl.all().sum() / len(kept_by_filter) * 100).round(1))

                # sample data and plot metrics
                df = pl.DataFrame(ds.sql(*keep_cols))
                n_total, total_duration = len(df), df['duration'].sum() / 3600
                post_df, n_filtered, filtered_duration = filter_df(df, accept_query, n_samples=N.value)
                # reject_df, _, _ = filter_df(df, reject_query, n_samples=N.value)
                pre_df = df.sample(N.value)

                metrics_fig = plot_stats(pre_df, post_df)
                lang_fig = lang_comparison(pre_df, post_df)

                stats = {
                    "kept_by_filter": kept_by_filter,
                    "filtered_duration": filtered_duration,
                    "total_duration": total_duration,
                    "n_filtered": n_filtered,
                    "n_total": n_total,
                    "n_sample": N.value,
                }

                mo.output.append(
                    mo.callout(
                        mo.vstack([
                            show_header(accept_query, stats),
                            metrics_fig,
                            mo.md("### 🌐 Language distribution"),
                            mo.md("---"),
                            lang_fig,
                        ])
                    )
                )
        except pl.exceptions.SQLInterfaceError as _e:
            show_error(f"**SQL Error**: {_e}")
    return


@app.cell
def _():
    # mo.stop(not filter_btn.value or not explorer_enabled.value)

    # # Merge dfs with filter column
    # accept_with_label = accept_df.with_columns(pl.lit("accept").alias("filter"))
    # reject_with_label = reject_df.with_columns(pl.lit("reject").alias("filter"))
    # merged_df = pl.concat([accept_with_label, reject_with_label])

    # # Create data explorer with filter as color
    # mo.callout(
    #     mo.md(f"""
    #     ## Data Explorer:
    #     {mo.ui.data_explorer(merged_df, color="filter")}
    #     """)
    # )
    return


@app.cell
def _(
    audio_enabled,
    ds,
    filter_btn,
    filters,
    generate_audio_sql_filter,
    mo,
    pl,
    sample_btn,
    show_ds,
    show_error,
    show_rejected_samples,
    time,
):
    mo.stop(not (filter_btn.value or sample_btn.value) or not audio_enabled.value)

    audio_accept_query, audio_reject_query = generate_audio_sql_filter(filters)

    _start_time = time.time()
    mo.output.clear()
    with mo.status.spinner(title=f"Loading audio from 10 random samples...") as _spinner:
        _EMPTY = "empty segment\n"
        # _first = True
        try:
            with mo.redirect_stdout():
                accept_ds = ds.filtered(audio_accept_query, N=10)

                # Show accepted samples
                display_items = [show_ds(accept_ds, "Accepted")]

                # Conditionally show rejected samples
                if show_rejected_samples.value:
                    reject_ds = ds.filtered(audio_reject_query, N=10)
                    display_items.append(show_ds(reject_ds, "Rejected"))

                mo.output.append(
                    mo.callout(
                        mo.hstack(display_items, gap=2)
                    )
                )
        except KeyError as _e:
            show_error(f"**KeyError**: Key `{_e.args[0]}` not available in this dataset.")
        except pl.exceptions.SQLInterfaceError as _e:
            show_error(f"**SQL Error**: {_e}")

    # mo.output.append(mo.md(f"Computed in {time.time() - _start_time:.1f}s"))
    return


@app.cell
def _(audio_enabled, filter_btn, mo, sample_btn):
    mo.stop(not filter_btn.value or not audio_enabled.value)
    mo.center(sample_btn)
    return


if __name__ == "__main__":
    app.run()
