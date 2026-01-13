import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from omegaconf import OmegaConf
    import marimo as mo
    from hume_wsds import WSDataset
    import polars as pl
    import polars_distance
    import random
    import time

    import plotly.subplots as sp
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde
    import numpy as np
    import os
    return (
        OmegaConf,
        WSDataset,
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
    Make sure that the **text** and **language** columns are available for the current dataset.
    """
    ))
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

    N = 100000
    txt_column = mo.ui.dropdown(options=txt_options, value=txt_options[0])
    language_column = mo.ui.dropdown(options=lang_options, value=lang_options[0])
    secondary_transcript = mo.ui.dropdown(options=txt_options, value=txt_options[-1])
    show_dtoks = mo.ui.checkbox(value=False, label="")

    mo.callout(
        mo.vstack([
            mo.md("## Presets"),
            mo.md("---"),
            mo.md(f"**Primary transcript**: {txt_column}"),
            mo.md(f"**Secondary transcript**: {secondary_transcript}"),
            mo.md(f"**Language column**: {language_column}"),
            mo.md(f"**Show dtoks**: {show_dtoks}")
        ])
    )
    return N, language_column, secondary_transcript, show_dtoks, txt_column


@app.cell
def _(mo):
    def display_on(text, condition):
        return mo.md(f"**{text}**" if condition.value else text)
    return (display_on,)


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
    secondary_transcript,
    snr_enabled,
    snr_threshold,
    txt_column,
):
    import re

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

    def yaml_generate_sql_filter(filters):
        keep_cols, conditions = generate_sql_with_aliases(filters)
        base_cols = {
            '__key__': '__key__',
            'pq': 'pq',
            'snr': 'snr',
            'cps': f"LENGTH(CAST(`{txt_column.value.strip()}` AS STRING)) / (tend - tstart)",
            'duration': '(tend - tstart)',
            't1': f"CAST(`{txt_column.value.strip()}` AS STRING)",
            't2': f"CAST(`{secondary_transcript.value.strip()}` AS STRING)",
            'language': f"CAST(`{language_column.value.strip()}` AS STRING)",
        }
        base_cols = [f"{v} AS {k}" for k,v in base_cols.items() if k not in filters.keys()]
        keep_cols = keep_cols + base_cols
        where = " AND ".join(conditions) if conditions else "1=1"
        select = f"SELECT * FROM self WHERE ({where})"
        reject = f"SELECT * FROM self WHERE NOT ({where})"
        return keep_cols, select, reject

    def generate_audio_sql_filter(filters):
        accept_query = " AND ".join(filters.values())
        reject_query = f"NOT ({accept_query})"
        return accept_query, reject_query
    return (
        generate_audio_sql_filter,
        generate_yaml_from_ui,
        yaml_generate_sql_filter,
    )


@app.cell
def _(
    filter_str,
    gaussian_kde,
    go,
    language_column,
    make_subplots,
    mo,
    np,
    secondary_transcript,
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
        res.append(
            mo.md(f"## <span style='color:blue;'>{title} samples</span>"),
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

    def show_sample(_x):
        if 'audio' not in _x:
            show_error(_x['__key__'])

        # Build transcript display
        transcript_display = [mo.md(f"`{txt_column.value}:` {_x.get(txt_column.value)}")]

        # Add secondary transcripts if any are selected
        if secondary_transcript.value:
            transcript_text = _x.get(secondary_transcript.value)
            transcript_display.append(mo.md(f"`{secondary_transcript.value}:` {transcript_text}"))
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
    mo.stop(True)

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
def _(ds_path, mo, secondary_transcript):
    mo.stop(not ds_path.value)

    cer_btn = mo.ui.run_button("success", label="View highest CER samples", full_width=True)
    mo.callout(
        mo.vstack([
            mo.md("## 🔍 CER Analysis"),
            mo.md("---"),
            mo.md("**View samples with highest Character Error Rate (CER)** between primary and secondary transcripts."),
            mo.md(f"**Secondary transcript**: `{secondary_transcript.value}`" if secondary_transcript.value else "**No secondary transcript selected**"),
            mo.center(cer_btn)
        ]),
        kind="info"
    )
    return (cer_btn,)


@app.cell
def _(
    accept_query,
    cer_btn,
    ds,
    filters,
    generate_audio_sql_filter,
    keep_cols,
    mo,
    pl,
    secondary_transcript,
):
    mo.stop(not cer_btn.value or not secondary_transcript.value)

    base_audio_query, _ = generate_audio_sql_filter(filters)

    mo.output.clear()
    with mo.status.spinner(title=f"Computing CER...") as _spinner:
        cer_df = (
            ds.sql(*keep_cols)    # get all filter cols
            .sql(accept_query)    # filter by metrics
            .sample(10000)
            .with_columns(
                cer = pl.col('t1').dist_str.levenshtein(pl.col('t2'))
                    / (pl.col('t1').str.len_chars() + pl.col('t2').str.len_chars())
            )
            .filter(pl.col('cer').is_finite())
            .sort('cer', descending=True)
        )
        mo.output.append(cer_df[['__key__', 'cer', 't1', 't2']])
    return base_audio_query, cer_df


@app.cell
def _(
    base_audio_query,
    cer_btn,
    cer_df,
    ds,
    mo,
    pl,
    secondary_transcript,
    show_ds,
    show_error,
):
    mo.stop(not cer_btn.value or not secondary_transcript.value)

    mo.output.clear()
    with mo.status.spinner(title=f"Sampling audio...") as _spinner:
        # Get top 100 keys meeting filtering criteria
        top_keys = cer_df.head(10)['__key__'].to_list()
        random_keys = cer_df.head(250).sample(10)['__key__'].to_list()

        # Create SQL IN clause
        top_keys_list = ', '.join(f"'{key}'" for key in top_keys)
        random_keys_list = ', '.join(f"'{key}'" for key in random_keys)

        top_audio_query = base_audio_query + f' AND __key__ IN ({top_keys_list})'
        random_audio_query = base_audio_query + f' AND __key__ IN ({random_keys_list})'

        top_audio_data = ds.filtered(top_audio_query, N=5)
        random_audio_data = ds.filtered(random_audio_query, N=5)

        _EMPTY = "empty segment\n"
        # _first = True
        try:
            with mo.redirect_stdout():
                # Show accepted samples
                top_items = show_ds(top_audio_data, "Top CER Audio")
                random_items = show_ds(random_audio_data, "High CER Audio (random from top 250)")

                mo.output.append(
                    mo.callout(
                        mo.hstack([
                            top_items, random_items
                        ], gap=2)
                    )
                )
        except KeyError as _e:
            show_error(f"**KeyError**: Key `{_e.args[0]}` not available in this dataset.")
        except pl.exceptions.SQLInterfaceError as _e:
            show_error(f"**SQL Error**: {_e}")
    return


@app.cell
def _(mo):
    # Live search by query for primary and secondary transcript
    search_query = mo.ui.text(placeholder="Enter search query...", full_width=True)
    search_enabled = mo.ui.checkbox(value=False, label="Enable live search")
    return search_enabled, search_query


@app.cell
def _(
    ds_path,
    mo,
    search_enabled,
    search_query,
    secondary_transcript,
    txt_column,
):
    mo.stop(not ds_path.value)

    mo.callout(
        mo.vstack([
            mo.md("## 🔍 Live Transcript Search"),
            mo.md("---"),
            mo.hstack([search_enabled, mo.md("**Enable live search**")], justify="start", align="center", gap=1),
            mo.md(f"**Search in**: `{txt_column.value}`" + (f" and `{secondary_transcript.value}`" if secondary_transcript.value else "")),
            search_query if search_enabled.value else mo.md("*Enable search to see search box*")
        ])
    )
    return


@app.cell
def _(
    ds,
    ds_path,
    filters,
    generate_audio_sql_filter,
    language_column,
    mo,
    search_enabled,
    search_query,
    secondary_transcript,
    show_error,
    txt_column,
):
    mo.stop(not ds_path.value or not search_enabled.value or not search_query.value.strip())

    audio_accept_query, _ = generate_audio_sql_filter(filters)

    query = search_query.value.strip()
    mo.output.clear()

    with mo.status.spinner(title=f"Searching for '{query}' in filtered transcripts..."):
        try:
            # Get pre-filtered samples count (before search)
            filtered_keys = ds.sql_filter(audio_accept_query)
            total_filtered = len(filtered_keys)

            # Search conditions combining filters + transcript search
            primary_condition = f"({audio_accept_query}) AND CAST(`{txt_column.value.strip()}` AS STRING) ILIKE '%{query}%'"
            primary_keys = ds.sql_filter(primary_condition)
            primary_matches = len(primary_keys)

            secondary_matches = 0
            if secondary_transcript.value:
                secondary_condition = f"({audio_accept_query}) AND CAST(`{secondary_transcript.value.strip()}` AS STRING) ILIKE '%{query}%'"
                secondary_keys = ds.sql_filter(secondary_condition)
                secondary_matches = len(secondary_keys)

            # Get samples for display
            def get_samples(keys, limit=10):
                return [ds[key] for key in keys.sample(min(limit, len(keys))) if ds[key]]

            primary_samples = get_samples(primary_keys) if primary_matches > 0 else []
            secondary_samples = get_samples(secondary_keys) if secondary_matches > 0 else []

            # Highlight query in text
            def highlight_text(text, query):
                if not query or not text:
                    return text
                import re
                return re.sub(f'({re.escape(query)})', r'<mark style="background-color: yellow; font-weight: bold;">\1</mark>', text, flags=re.IGNORECASE)

            # Custom sample display with highlighting
            def show_highlighted_sample(sample, transcript_col):
                transcript_text = sample.get(transcript_col)
                highlighted_text = highlight_text(transcript_text, query)

                return mo.md(f"""
                &nbsp;
                /// admonition | {sample['__key__'][:50]}...
                <span style='color:blue;'>`lang:` {sample.get(language_column.value)} • `snr:` {sample.get('snr', 0):.2f} • `pq:` {sample.get('pq', 0):.2f}</span>

                {mo.md(highlighted_text)}

                {mo.audio(sample['audio'].load(24000).numpy(), rate=24000) if sample['audio'].tend - sample['audio'].tstart > 0 else "empty segment"}
                """)

            # Display results
            mo.output.append(mo.callout(mo.vstack([
                mo.md(f"## 🔍 Search Results for: '{query}'"),
                mo.md("---"),
                mo.hstack([
                    mo.callout(mo.vstack([
                        mo.md("#### Primary Transcript"),
                        mo.md(f"**{primary_matches:,}** matches"),
                        mo.md(f"**{primary_matches/total_filtered*100:.1f}%** of {total_filtered:,} filtered")
                    ], align="center"), kind="success"),
                    mo.callout(mo.vstack([
                        mo.md("#### Secondary Transcript"),
                        mo.md(f"**{secondary_matches:,}** matches"),
                        mo.md(f"**{secondary_matches/total_filtered*100:.1f}%** of {total_filtered:,} filtered")
                    ], align="center"), kind="success"),
                    mo.callout(mo.vstack([
                        mo.md("#### Total Matches"),
                        mo.md(f"**{primary_matches + secondary_matches:,}** samples"),
                        mo.md(f"**{(primary_matches + secondary_matches)/total_filtered*100:.1f}%** of filtered")
                    ], align="center"), kind="info")
                ], justify="center", gap=2),
                mo.md("---"),
                mo.hstack([
                    mo.vstack([
                        mo.md(f"### 📝 Primary (`{txt_column.value}`)"),
                        mo.md(f"*{len(primary_samples)} of {primary_matches:,} matches*"),
                        mo.vstack([show_highlighted_sample(s, txt_column.value) for s in primary_samples]) or mo.md("*No matches*")
                    ]),
                    mo.vstack([
                        mo.md(f"### 📝 Secondary (`{secondary_transcript.value}`)"),
                        mo.md(f"*{len(secondary_samples)} of {secondary_matches:,} matches*"),
                        mo.vstack([show_highlighted_sample(s, secondary_transcript.value) for s in secondary_samples]) or mo.md("*No matches*")
                    ])
                ], justify="space-between", gap=3)
            ])))

        except Exception as e:
            show_error(f"**Search Error**: {str(e)}")
    return


if __name__ == "__main__":
    app.run()
