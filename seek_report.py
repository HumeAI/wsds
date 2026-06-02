"""Emit a Markdown seek-accuracy report (for the GitHub Actions job summary).

Runs the same synthetic-signal measurements as ``test_audio_seek`` and prints a
Markdown table of the seek error -- in output samples -- for every
codec x sample-rate x seek-position. Pipe it into ``$GITHUB_STEP_SUMMARY`` so it
renders on the workflow run page; it also reads fine on a terminal.

    python seek_report.py >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import sys

import test_audio_seek as ts


def _cell(err: float, conf: float) -> str:
    if conf < ts.MIN_CONFIDENCE:
        return f"{err:+.1f} ⚠️"  # unreliable measurement (tones lost to codec)
    mark = "✅" if abs(err) <= ts.TOL_SAMPLES else "❌"
    return f"{err:+.2f} {mark}"


def _table(rows: list[tuple[str, list[str]]]) -> str:
    head = "| case | " + " | ".join(f"{p:g}s" for p in ts.POSITIONS) + " |"
    sep = "|" + "---|" * (1 + len(ts.POSITIONS))
    body = "\n".join(f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows)
    return f"{head}\n{sep}\n{body}\n"


def main(out=sys.stdout) -> int:
    out.write("## Audio seek accuracy\n\n")
    if ts.ENCODER_CLS is None:
        out.write("_No audio encoder backend available; report skipped._\n")
        return 0

    backend = ts.ENCODER_CLS.__module__.split(".")[0]
    out.write(
        f"Backend `{backend}`. Each cell is the seek error in output samples at "
        f"that position (tolerance ±{ts.TOL_SAMPLES:g}; ✅ pass, ❌ fail, "
        f"⚠️ low-confidence measurement).\n\n"
    )

    n_pass = n_total = 0
    plan = [("native", ((c, sr, None) for c, *_ in ts.CODECS for sr in ts.ENCODE_RATES))]
    plan.append(("resampled to 16 kHz",
                 ((c, sr, 16000) for c in ts.RESAMPLE_CODECS for sr in ts.RESAMPLE_RATES)))

    for title, cases in plan:
        rows = []
        for codec, sr, out_sr in cases:
            blob = ts.fixture(codec, sr)
            if blob is None:
                continue
            cells = []
            for pos in ts.POSITIONS:
                err, conf = ts.measure(blob, pos, sample_rate=out_sr)
                if conf >= ts.MIN_CONFIDENCE:
                    n_total += 1
                    n_pass += abs(err) <= ts.TOL_SAMPLES
                cells.append(_cell(err, conf))
            rows.append((f"{codec} @ {sr}", cells))
        out.write(f"### {title}\n\n{_table(rows)}\n")

    out.write(f"**{n_pass}/{n_total} measurements within ±{ts.TOL_SAMPLES:g} samples.**\n")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
