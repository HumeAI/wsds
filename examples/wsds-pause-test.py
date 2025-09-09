import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import hume_wsds as wsds
    import marimo as mo
    import math


@app.cell
def _():
    ds = wsds.WSDataset('/mnt/weka/wsds/youtube-cc/vad_ws/')

    class WSSourceAudioEndPauseShard(wsds.WSSourceAudioShard):
        def get_timestamps(self, segment_offset):
            _, tend = super().get_timestamps(segment_offset)
            try:
                nstart, _ = super().get_timestamps(segment_offset+1)
            except IndexError:
                nstart = math.inf
            return tend, min(tend + 2, nstart)

    ds.add_computed('audio-pause', dataset_dir='../source', loader=WSSourceAudioEndPauseShard, vad_column='raw.vad.npy')
    return (ds,)


@app.cell
def _(ds):
    btn = mo.ui.run_button('success', label=f'Show 5 random samples')
    mo.vstack([
        mo.md(f'## Exploring `{ds.dir}` end-pauses\n&nbsp;\n'),
        mo.center(btn),
    ])
    return (btn,)


@app.cell
def _(btn, ds):
    if btn.value:
        mo.output.clear()
        for _ in range(5):
            for _x in ds: break
            pause = _x['audio-pause'].load(16000).numpy()
            mo.output.append(mo.md(f"""
    &nbsp;
    ## `{_x.get_key()}`  
    `full_transcript:` {_x['transcription_wslang_raw.txt']}  
    `last 2s` {mo.audio(_x['audio'].load(16000).numpy()[:,-32000:], rate=16000, normalize=False)}  
    `pause {pause.shape[-1]/16000:.1f}s` {mo.audio(pause, rate=16000, normalize=False) if pause.size > 0 else "no pause"}  
    """))
    return


if __name__ == "__main__":
    app.run()
