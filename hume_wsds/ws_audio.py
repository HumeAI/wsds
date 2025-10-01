from __future__ import annotations

import io
import typing
from dataclasses import dataclass

import pyarrow as pa


def to_filelike(src: typing.Any) -> typing.BinaryIO:
    """Coerces files, byte-strings and PyArrow binary buffers into file-like objects."""
    if hasattr(src, "read"):  # an open file
        return src
    # if not an open file then we assume some kind of binary data in memory
    if hasattr(src, "as_buffer"):  # PyArrow binary data
        return pa.BufferReader(src.as_buffer())
    return io.BytesIO(src)


def load_segment(src, start, end, sample_rate=None):
    """Efficiently loads an audio segment from `src` (see below) `tstart` to `tend` seconds while
    optionally resampling it to `sample_rate`.

    `src` can be one of:
    - a file-like object
    - a byte string
    - a PyArrow binary buffer in memory"""
    return AudioReader(src).read_segment(start, end, sample_rate=sample_rate)


@dataclass(slots=True)
class AudioReader:
    """A lazy seeking-capable audio reader for random-access to recordings stored in WSDS shards."""

    src: typing.Any
    reader: "torchaudio.io.StreamReader" | None = None  # noqa: F821
    sample_rate: int | None = None
    skip_samples: int = 0

    def __repr__(self):
        return f"AudioReader(src={type(self.src)}, sample_rate={self.sample_rate})"

    # we materialize the reader on first use
    def get_reader(self, sample_rate=None):
        # import here to avoid pulling in the heavy torch dependency unless we really need it
        import torchaudio

        if self.sample_rate is not None:
            assert not sample_rate or sample_rate == self.sample_rate, "please use a consistent sample rate"

        if self.reader is None:
            # FIXME: move to AudioDecoder from torchcodec
            reader = torchaudio.io.StreamReader(src=to_filelike(self.src))
            info = reader.get_src_stream_info(0)

            # mp3 has encoder delays that are not handled well when seeking (http://mp3decoders.mp3-tech.org/decoders_lame.html)
            if info.codec == "mp3":
                self.skip_samples = 1105

            if sample_rate is None:
                sample_rate = info.sample_rate

            # fetch 32 seconds because we likely need 30s at maximum but the seeking may be imprecise (and we seek 1s early)
            # FIXME: check if we can get away with some better settings here (-1, maybe 10s + concatenate the chunks in a loop)
            reader.add_basic_audio_stream(frames_per_chunk=int(32 * sample_rate), sample_rate=sample_rate)

            self.reader = reader
            self.sample_rate = sample_rate

        return self.reader, self.sample_rate

    def read_segment(self, start, end, sample_rate=None):
        reader, sample_rate = self.get_reader(sample_rate)
        # rought seek
        reader.seek(max(0, start - 1), "key")
        reader.fill_buffer()
        (chunk,) = reader.pop_chunks()
        # tight crop (seems accurate down to 1 sample in my tests)
        prefix = int((start - chunk.pts) * sample_rate) + self.skip_samples if start > 0 else 0
        assert prefix >= 0
        samples = chunk[prefix : prefix + int((end - start) * sample_rate)].mT
        # clear out any remaining data
        while chunk is not None:
            (chunk,) = reader.pop_chunks()

        samples.sample_rate = sample_rate
        return samples


@dataclass(frozen=True, slots=True)
class WSAudio:
    """A lazy reference to a single sample from a segmented audio file."""

    audio_reader: AudioReader
    tstart: float
    tend: float

    def load(self, sample_rate=None, pad_to_seconds=None):
        samples = self.audio_reader.read_segment(self.tstart, self.tend, sample_rate)
        sample_rate = samples.sample_rate
        if pad_to_seconds is not None:
            import torch

            padding = int(pad_to_seconds * sample_rate - samples.shape[-1])
            samples = torch.nn.functional.pad(samples, (0, padding))
            samples.sample_rate = sample_rate
        return samples

    def _display_(self):
        import marimo
        samples = self.load()
        return marimo.audio(samples.numpy(), rate=samples.sample_rate)

    def _ipython_display_(self):
        from IPython.display import display, Audio
        samples = self.load()
        display(Audio(samples.numpy(), rate=samples.sample_rate))
