"""Seek-accuracy tests built on a synthetic multi-tone "phase comb" signal.

The signal is a sum of incommensurate sine waves, band-limited so it survives
every codec we ship through (lowest real sample rate is 8 kHz -> 4 kHz Nyquist,
and lossy codecs low-pass aggressively below that). Because it is defined in
continuous time, the analytic reference is exact at *any* sample rate and *any*
seek position -- no external decoder / ffmpeg ground truth is needed.

A seek error manifests as a time offset between the decoded window and the
analytic reference. We recover that offset (with sub-sample, parabolic
precision) by normalized cross-correlation. Lossy codecs wreck amplitude but
preserve the phase of surviving tones, so the offset stays measurable.

The signal is encoded in-memory through every available codec; codecs whose
encoder is missing in the active backend are skipped. No audio fixtures are
committed -- everything is generated at test time.
"""

from __future__ import annotations

import io
import unittest

import numpy as np

# Tones in 300-3000 Hz: above DC/high-pass filtering, below the 8 kHz-file
# Nyquist (4 kHz) with margin so even low-bitrate 8 kHz mp3/aac pass them.
# Deliberately non-harmonic / incommensurate so the waveform is aperiodic over
# the whole file and its autocorrelation has a single sharp peak -> the seek
# offset is unambiguous.
TONES = (313.7, 547.3, 941.1, 1583.9, 2293.7, 2971.3)
PHASES = np.random.default_rng(0).uniform(0.0, 2.0 * np.pi, len(TONES))
PEAK = 0.7  # headroom below clipping for filter/resampler overshoot

DURATION = 35.0          # seconds; covers the deepest seek (30 s) + window
WINDOW = 1.0             # length of each decoded segment, seconds
ENCODE_RATES = (44100, 16000, 8000)
# Positions straddle the 5 s boundary -- below it the decoder reads from the
# start, at/above it the seek path runs (a real branch in audio_codec.py).
POSITIONS = (0.5, 2.0, 4.9, 5.1, 10.0, 30.0)
TOL_SAMPLES = 2.0        # a correct seek lands within 2 samples
MIN_CONFIDENCE = 0.5     # cross-correlation peak must be this strong to trust

# Subset exercised through the resample-to-16k path (opus/aac decode at a fixed
# internal rate, so resampling them muddies the measurement).
RESAMPLE_CODECS = ("mp3", "flac", "wav", "vorbis")
RESAMPLE_RATES = (44100, 8000)

# (name, container format, explicit encoder, lossless)
# aac is muxed into mp4 (not raw ADTS): that matches how wsds stores aac and
# carries the gapless edit-list metadata needed for sample-accurate seeking.
CODECS = (
    ("mp3", "mp3", None, False),
    ("aac", "mp4", None, False),
    ("opus", "ogg", "libopus", False),
    ("vorbis", "ogg", "libvorbis", False),
    ("flac", "flac", None, True),
    ("wav", "wav", None, True),
)


def render(sr: int, t0: float, n: int) -> np.ndarray:
    """Render ``n`` samples of the test signal starting at continuous time ``t0``."""
    t = t0 + np.arange(n) / sr
    x = np.zeros(n, dtype=np.float64)
    for f, ph in zip(TONES, PHASES):
        x += np.sin(2.0 * np.pi * f * t + ph)
    return (PEAK / len(TONES) * x).astype(np.float32)


def _encoder_cls():
    """Return a torchaudio-style streaming encoder class, or None if unavailable."""
    try:
        from humecodec import MediaEncoder

        return MediaEncoder
    except ImportError:
        pass
    try:
        from torchaudio.io import StreamWriter

        return StreamWriter
    except ImportError:
        return None


ENCODER_CLS = _encoder_cls()


def encode(sig: np.ndarray, sr: int, fmt: str, encoder: str | None):
    """Encode mono float samples to container bytes; return None if unsupported."""
    import torch

    out = io.BytesIO()
    waveform = torch.from_numpy(sig).reshape(-1, 1)  # (frames, channels)
    try:
        enc = ENCODER_CLS(out, fmt)
        kwargs = dict(sample_rate=sr, num_channels=1, format="flt")
        if encoder is not None:
            kwargs["encoder"] = encoder
        enc.add_audio_stream(**kwargs)
        with enc.open():
            enc.write_audio_chunk(0, waveform)
    except Exception:
        return None
    return out.getvalue()


def seek_error_samples(decoded: np.ndarray, sr: float, t_start: float,
                       max_offset: float = 0.15, win: float = 0.25):
    """Measure how far a decoded window's true start is from ``t_start``.

    Returns ``(error_samples, confidence)``. A positive error means the decoded
    audio starts *later* than requested (the seek landed late). ``confidence``
    is the normalized cross-correlation peak in ``[0, 1]``; values well below 1
    mean the measurement is unreliable (e.g. tones lost to a codec low-pass).
    """
    x = np.asarray(decoded, dtype=np.float64)
    n = min(len(x), int(win * sr))
    x = x[:n]
    x = x - x.mean()
    s = int(max_offset * sr)
    # Reference spans [t_start - max_offset, t_start - max_offset + (n + 2s)/sr]
    # so lag l in [0, 2s] corresponds to a true offset of (l - s) samples.
    ref = render(sr, t_start - s / sr, n + 2 * s).astype(np.float64)
    ref = ref - ref.mean()

    segs = np.lib.stride_tricks.sliding_window_view(ref, n)  # (2s+1, n)
    raw = segs @ x
    norms = np.linalg.norm(segs, axis=1) * (np.linalg.norm(x) + 1e-12) + 1e-12
    cc = raw / norms

    lag = int(np.argmax(cc))
    confidence = float(cc[lag])
    sub = 0.0
    if 0 < lag < 2 * s:
        ym1, y0, yp1 = cc[lag - 1], cc[lag], cc[lag + 1]
        denom = ym1 - 2.0 * y0 + yp1
        if denom != 0.0:
            sub = 0.5 * (ym1 - yp1) / denom
    return (lag - s + sub), confidence


def measure(blob: bytes, t_start: float, sample_rate=None):
    """Decode a window via the wsds seek path and return ``(error, confidence)``."""
    from wsds.ws_audio import WSAudioEpisode

    episode = WSAudioEpisode(io.BytesIO(blob))
    samples = episode.read_segment(start=t_start, end=t_start + WINDOW,
                                   sample_rate=sample_rate)
    # read_segment returns a (channels, frames) tensor with a .sample_rate
    # attribute; render the reference at the rate actually returned (native or
    # resampled).
    out_sr = float(samples.sample_rate)
    return seek_error_samples(samples[0].float().cpu().numpy(), out_sr, t_start)


# Cache encoded fixtures so each (codec, rate) is encoded only once.
_FIXTURES: dict = {}


def fixture(codec: str, sr: int):
    key = (codec, sr)
    if key not in _FIXTURES:
        name, fmt, encoder, _lossless = next(c for c in CODECS if c[0] == codec)
        sig = render(sr, 0.0, int(sr * DURATION))
        _FIXTURES[key] = encode(sig, sr, fmt, encoder)
    return _FIXTURES[key]


@unittest.skipIf(ENCODER_CLS is None, "no audio encoder backend available")
class AudioSeekTest(unittest.TestCase):
    """wsds must seek to the requested time, sample-accurately, for every codec."""

    def _assert_accurate(self, codec, encode_sr, positions, sample_rate=None):
        blob = fixture(codec, encode_sr)
        if blob is None:
            self.skipTest(f"{codec}@{encode_sr} not encodable in this backend")
        for t in positions:
            with self.subTest(codec=codec, encode_sr=encode_sr, t=t,
                              out_sr=sample_rate):
                err, conf = measure(blob, t, sample_rate=sample_rate)
                self.assertGreaterEqual(
                    conf, MIN_CONFIDENCE,
                    f"unreliable measurement (conf={conf:.2f}) for {codec}"
                    f"@{encode_sr} t={t}")
                self.assertLessEqual(
                    abs(err), TOL_SAMPLES,
                    f"{codec}@{encode_sr} t={t}s out_sr={sample_rate}: "
                    f"seek off by {err:.2f} samples (conf={conf:.2f})")

    def test_seek_accurate_native(self):
        """Every codec must seek sample-accurately at its native rate."""
        for codec, _fmt, _enc, _lossless in CODECS:
            for sr in ENCODE_RATES:
                self._assert_accurate(codec, sr, POSITIONS)

    def test_seek_accurate_resampled_to_16k(self):
        """Seeking while resampling to 16 kHz must also be sample-accurate."""
        for codec in RESAMPLE_CODECS:
            for sr in RESAMPLE_RATES:
                self._assert_accurate(codec, sr, POSITIONS, sample_rate=16000)


if __name__ == "__main__":
    unittest.main()
