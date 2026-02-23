from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torchaudio
import whisper
from PIL import Image

HEADER = """
<html>
<body>
<div class="help-box">
<audio id="player" src="snd.m4a" controls=""></audio>
Press <kbd>space</kbd> or <kbd>esc</kbd> to play/pause.<br>
Click or drag anywhere to play from a specific position.<br />
You can always see the exact timestamp in the top right corner.<br />
You can share a specific moment by copying the URL from your browser bar (make sure you pause first).<br />
Use <kbd>[</kbd> and <kbd>]</kbd> to selectively mute the left or right channel.<br />
</div>
<div id=timer>00:01:32.41</div>
<div id="player-box">
<style>
    .help-box {
        background: #98E;
        padding: 4px 7px;
        box-sizing: border-box;
        width: 560px;
        margin: 5px auto;
        border-radius: 7px;
    }
    #timer {
        position: fixed;
        top: 5px; right: 5px;
        font-family: monospace;
    }
    #player-box {
        position: relative;
    }
    #playhead {
        position: absolute;
        width: 100%;
        top: 0;
        border-top: 1px solid red;
        pointer-events: none;
        scroll-margin-top: 50px;
    }
    .middle-box {
        margin: auto;
        width: 600px;
        display: flex;
    }
    .col-separator {
        min-width: 10px;
        flex-grow: 1;
    }
    .col {
        position: relative;
    }
    .col-right {
        text-align: right;
    }
    .label {
        width: 100%;
        box-sizing: border-box;
        border: 2px solid #c2c2c2;
        border-width: 2px 0px;
        scroll-margin-top: 50px;
        display: block;
        font-size: 80%;
        margin-bottom: 8px;
    }
    .col-right .label {
        padding-right: 5px;
    }
    .col-left .label {
        padding-left: 5px;
    }
    body.chn-silenced-0 .col-left {
        opacity: .5;
    }
    body.chn-silenced-1 .col-right {
        opacity: .5;
    }
    .label b {
        margin-right: 10px;
    }
    #player {
         display: block;
         margin: 5px auto;
    }
    kbd {
      background-color: #eee;
      border-radius: 3px;
      border: 1px solid #b4b4b4;
      box-shadow:
        0 1px 1px rgba(0, 0, 0, 0.2),
        0 2px 0 0 rgba(255, 255, 255, 0.7) inset;
      color: #333;
      display: inline-block;
      font-size: 0.85em;
      font-weight: 700;
      line-height: 1;
      padding: 2px 4px;
      white-space: nowrap;
    }
</style>
"""

FOOTER = """
</div>
<div id="playhead"></div>
<script>
    console.log('Startup...', window.location.hash.substr(1));
    var body = document.body;
    var player = document.getElementById("player");
    var playhead = document.getElementById("playhead");
    var playerbox = document.getElementById("player-box");

    if (window.location.hash.substr(0,1) == '#') {
        player.currentTime = parseFloat(window.location.hash.substr(1));
        setTimeout(() => {
            playhead.scrollIntoView({ behavior: "smooth" });
        }, 50);
    }

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const ac = new AudioContext();
    const track = ac.createMediaElementSource(player);

    const splitter = ac.createChannelSplitter(2);
    track.connect(splitter);

    const gainNodes = [ac.createGain(), ac.createGain()];
    window.gains = gainNodes;

    splitter.connect(gainNodes[0], 0);
    splitter.connect(gainNodes[1], 1);

    const merger = ac.createChannelMerger(1);
    gainNodes[0].connect(merger, 0, 0);
    gainNodes[1].connect(merger, 0, 0);

    merger.connect(ac.destination);

    var clickable_turns = false;
    var was_paused;
    var track_mouse = false;

    function mousedown(ev) {
        if (ev.buttons != 1) return; // left mouse button

        var el = ev.target;
        while (el) {
            if (el.className == 'labels') return;
            if (clickable_turns) {
                if (el.dataset.tstart) {
                    var start = el.dataset.tstart;
                    player.currentTime = start
                    return;
                }
            }
            el = el.parentElement;
        }

        was_paused = player.paused;
        track_mouse = true;
        mousemove(ev);
    }

    function mousemove(ev) {
        if (!track_mouse) return;

        player.currentTime = (ev.pageY - playerbox.offsetTop) / pixels_per_second;
        player.pause();
        ev.preventDefault();
    }

    function mouseup(ev) {
        if (!track_mouse) return;

        if (was_paused) {
            window.location.hash = player.currentTime;
        } else {
            player.play();
        }

        track_mouse = false;
    }

    document.getElementById('player-box').addEventListener('mousedown', mousedown);
    document.getElementById('player-box').addEventListener('mousemove', mousemove);
    document.getElementById('player-box').addEventListener('mouseup', mouseup);
    function togglePlayback() {
    	if (ac.state === 'suspended') {
    		ac.resume();
    	}

        if (player.paused) player.play()
        else {
            player.pause();
            window.location.hash = player.currentTime;
        }
    }
    var all_chunks = Array.from(document.querySelectorAll('[data-tstart]'));
    var current_element = null;
    var skipping_to = null;
    document.addEventListener('keydown', (ev) => {
        console.log(ev.key);
        if (ev.key == "Escape" || ev.key == " ") { togglePlayback(); ev.preventDefault(); }
        if (ev.key == "ArrowUp" || ev.key == "ArrowDown") {
            var up = ev.key == "ArrowUp";
            var upi = -1;
            if (current_element !== null) {
                var t = player.currentTime + (up ? -.1 : .1);
                all_chunks.some((n, i) => { if (n.dataset.tstart < t) { upi = i; return false; } else { return true; }});
            }
            var el = all_chunks[up ? upi : upi + 1];
            if (el) {
                current_element = el;
                skipping_to = el.dataset.tstart - .5;
                player.playbackRate = 3;
                // player.currentTime = el.dataset.tstart;
                el.scrollIntoView({ behavior: "smooth" });
            }
            ev.preventDefault();
        }
        if (ev.key == "[" || ev.key == "]") {
            const chn = ev.key == '[' ? 0 : 1;
            console.log('switching channel: ', chn, 'from:', gains[chn].gain.value);
            gains[chn].gain.value = 1 - gains[chn].gain.value;
            body.classList.toggle('chn-silenced-'+chn, gains[chn].gain.value == 0);
        }
        var digit = parseInt(ev.key);
        if (current_element && digit) {
            var radios = current_element.querySelectorAll("input[type=radio]")
            console.log(digit, radios);
            if (digit <= radios.length) {
                radios[digit-1].checked = true;
            }
        }
    });

    String.prototype.paddingLeft = function (paddingValue) {
       return String(paddingValue + this).slice(-Math.max(paddingValue.length, this.length));
    };
    timer = document.getElementById("timer");
    function movePlayhead() {
        var t = player.currentTime;
        if (skipping_to != null && t >= skipping_to) {
            skipping_to = null;
            player.playbackRate = 1;
        }
        playhead.style.top = t * pixels_per_second + "px";
        t *= 1000;
        var m = (Math.floor(t / 60000) % 60).toString(),
            s = (Math.floor(t / 1000) % 60).toString().paddingLeft("00"),
            ms = (Math.floor(t) % 1000).toString().paddingLeft("000");
        timer.textContent = `${m}:${s}.${ms}`;
        requestAnimationFrame(movePlayhead);
    }
    movePlayhead();

    console.log('Startup complete!');
</script>
<script src="/lblr.js"></script>
</body>
</html>
"""

CSV = """<div id="vote_results"></div>
<script>
  function calcResults() {
    results = {}; document.querySelectorAll('input[type=radio]').forEach((x) => { if(x.checked) results[x.name] = x.value }); results;
    let out = "";
    //let stats = "";
    //totals = {}
    //Object.entries(results).forEach(([k, v]) => { out += `${k},${v}\n`; totals[v] = (totals[v] || 0) + 1});
    //Object.entries(totals).forEach(([k, v]) => { stats += `<strong>${k}:</strong> ${v}\n`; });
    //result_div = document.getElementById('vote_results');
    //result_div.innerHTML = `${stats}<br /><a href="${encodeURI("data:text/csv;charset=utf-8," + out)}" target="_blank">Download CSV results</a>`;
  }
  document.addEventListener('change', calcResults);
  calcResults()
</script>"""


def mel_img(snd: torch.Tensor, sr: int, mel_min: float = -1, mel_max: float = 2) -> np.ndarray:
    mel = whisper.log_mel_spectrogram(torchaudio.functional.resample(snd, sr, 16000))
    return torch.clamp((mel - mel_min) / (mel_max - mel_min) * 255, 0, 255).numpy().astype(np.uint8)


def ticks_img(h: int) -> np.ndarray:
    ticks = np.full((10, h), 255, dtype=np.uint8)
    ticks[5:, :: h // 10] = 150
    ticks[:, 0] = 50
    ticks[4:, h // 2] = 50
    return ticks


from collections import defaultdict

from .utils import is_notebook


class ColumnList:
    def __init__(self, player: ConvPlayer, side: str) -> None:
        self.player = player
        self.side = side
        self.styles: defaultdict[str, str] = defaultdict(str)
        self.content: defaultdict[str, list[str]] = defaultdict(list)

    def _get_fname(self, name: str, fmt: str) -> str:
        return f"{name}-{self.side}-{len(self.content) + 1:03d}.{fmt}"

    def _save_img(self, name: str, img: np.ndarray, fmt: str = "png") -> str:
        fname = self._get_fname(name, fmt)
        Image.fromarray(img.T).save(self.player.path / fname)
        return fname

    def append(self, name: str, content: str) -> None:
        self.content[name].append(content)

    def put_html(self, name: str, t_start: float, html: str, t_len: Optional[float] = None, bg: str = "#eee", flex: int = 10, width: Optional[float] = None) -> None:
        if width:
            self.styles[name] = f' style="width: {width}px"'
        else:
            self.styles[name] = f' style="flex: {flex}"'
        y = t_start * self.player.pixels_per_second
        height = "" if t_len is None else f" height:{t_len * self.player.pixels_per_second}px;"
        self.append(
            name,
            f'<div class="col-{name}-{self.side}-html label" data-tstart="{t_start}" style="position: absolute; top: {y}px;{height} background-color: {bg};">{html}</div>',
        )

    def put_img(self, name: str, t: float, img: np.ndarray, fmt: str = "png", scalex: float = 1, scaley: float = 1) -> None:
        fname = self._save_img(name, img, fmt=fmt)
        w, h = img.shape
        # BUG: put_html expects (name, t_start, html, ...) but here the HTML string is
        # passed as t_start and `html` is missing entirely. The `t` parameter is also unused.
        self.put_html(  # type: ignore[call-arg]
            name,
            f'<img src="{fname}" width={w / scalex} height={h / scaley} class="col-{name}-{self.side}-img">',  # type: ignore[arg-type]
            width=w / scalex,
        )

    def append_img(self, name: str, img: np.ndarray, fmt: str = "png", scalex: float = 1, scaley: float = 1, repeat_y: bool = False) -> None:
        fname = self._save_img(name, img, fmt=fmt)
        w, h = img.shape
        if not repeat_y:
            self.append(
                name, f'<img src="{fname}" width={w / scalex} height={h / scaley} class="col-{name}-{self.side}-img">'
            )
        else:
            self.append(
                name,
                f'<div class="col-{name}-{self.side}-img" style="height: 100%; width: {w}px; background: url(\'{fname}\') repeat-y; background-size: {w / scalex}px {h / scaley}px;"></div>',
            )

    def __str__(self) -> str:
        lines = []
        cols_iter: Iterable[tuple[str, list[str]]] = self.content.items()
        if self.side == "right":
            cols_iter = reversed(list(self.content.items()))
        for name, c in cols_iter:
            lines.append(
                f'<div class="col col-{self.side} col-{name}-{self.side}"{self.styles[name]}>'
                + ("\n".join(self.content[name]))
                + "</div>"
            )
        return "\n".join(lines)


class ConvPlayer:
    def __init__(self, path: Union[str, Path], snd: torch.Tensor, sr: int, rmdir: bool = False, pixels_per_second: int = 50) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.path = path
        self.pixels_per_second = pixels_per_second

        self.left = ColumnList(self, "left")
        self.right = ColumnList(self, "right")

        if rmdir and path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)

        self._add_html()
        self._add_audio(snd, sr)

    def _add_html(self) -> None:
        self.html = open(self.path / "index.html", "w")
        self.html.write(HEADER)

    def _add_audio(self, snd: torch.Tensor, sr: int) -> None:
        torchaudio.save(self.path / "snd.m4a", snd, sr)
        ticks = ticks_img(self.pixels_per_second * 2)
        self.left.append_img("ticks", ticks, scaley=2, repeat_y=True)
        self.right.append_img("ticks", ticks[::-1], scaley=2, repeat_y=True)
        for i, snd in enumerate(torch.split(snd, 5 * 60 * sr)):
            mels = mel_img(snd, sr)
            for c, i in zip([self.left, self.right], mels):
                c.append_img("mel", i, scaley=100 / self.pixels_per_second)

    def close(self, zip: bool = False, show: bool = False) -> None:
        self.html.write('<div class="middle-box">\n')
        self.html.write(str(self.left) + "\n")
        self.html.write('<div class="col-separator"></div>')
        self.html.write(str(self.right) + "\n")
        self.html.write(f"<script>pixels_per_second = {self.pixels_per_second}</script>")
        self.html.write(FOOTER)
        self.html.close()
        if show:
            if is_notebook():
                from IPython.display import HTML, display

                display(HTML(f'<a href="{self.path}/index.html">View player</a>'))
            else:
                print(f"{self.path}/index.html")
