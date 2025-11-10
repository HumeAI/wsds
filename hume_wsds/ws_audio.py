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


class CompatAudioDecoder:
    def __init__(self, src, sample_rate):
        import torchaudio

        if not hasattr(torchaudio, "io"):
            raise ImportError("You need either torchaudio<2.9 or torchcodec installed")
        self.src = src
        self.reader = torchaudio.io.StreamReader(src=to_filelike(self.src))
        self.metadata = self.reader.get_src_stream_info(0)

        if sample_rate is None:
            sample_rate = self.metadata.sample_rate

        self.sample_rate = sample_rate

        # fetch 32 seconds because we likely need 30s at maximum but the seeking may be imprecise (and we seek 1s early)
        # FIXME: check if we can get away with some better settings here (-1, maybe 10s + concatenate the chunks in a loop)
        self.reader.add_basic_audio_stream(frames_per_chunk=int(32 * sample_rate), sample_rate=sample_rate)

    def get_samples_played_in_range(self, tstart=0, tend=None):
        # rought seek
        self.reader.seek(max(0, tstart - 1), "key")

        if tend == None:
            import torch
            chunks = []
            more_data = True
            while more_data:
                if self.reader.fill_buffer() == 1:
                    more_data = False
                (chunk,) = self.reader.pop_chunks()
                if chunk is None:
                    break
                chunks.append(chunk)
            prefix = int((tstart - chunks[0].pts) * self.sample_rate)
            return torch.cat(chunks)[prefix:].mT

        self.reader.fill_buffer()
        (chunk,) = self.reader.pop_chunks()
        # tight crop (seems accurate down to 1 sample in my tests)
        prefix = int((tstart - chunk.pts) * self.sample_rate)
        assert prefix >= 0
        if tend:
            samples = chunk[prefix : prefix + int((tend - tstart) * self.sample_rate)].mT
        else:
            samples = chunk[prefix:].mT
        # clear out any remaining data
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()
        return samples


def marimo_audio_mp3(samples):
    from io import BytesIO

    import marimo
    from torchcodec.encoders import AudioEncoder

    out = BytesIO()
    AudioEncoder(samples, sample_rate=samples.sample_rate).to_file_like(out, "mp3")

    return marimo.audio(out.getvalue())


@dataclass(slots=True)
class AudioReader:
    """A lazy seeking-capable audio reader for random-access to recordings stored in wsds shards."""

    src: typing.Any
    reader: CompatAudioDecoder | None = None
    sample_rate: int | None = None
    skip_samples: int = 0

    def __repr__(self):
        return f"AudioReader(src={type(self.src)}, sample_rate={self.sample_rate})"

    def unwrap(self):
        """Return the raw audio bytes"""
        if hasattr(self.src, "as_buffer"):
            return self.src.as_buffer().to_pybytes()
        elif isinstance(self.src, (bytes, bytearray)):
            return self.src
        else:
            raise TypeError(f"Unsupported AudioReader src type: {type(self.src)}")

    # we materialize the reader on first use
    def get_reader(self, sample_rate=None):
        sample_rate_switch = False
        if self.sample_rate is not None:
            sample_rate_switch = self.sample_rate != sample_rate

        if self.reader is None or sample_rate_switch:
            try:
                from torchcodec.decoders import AudioDecoder
            except ImportError:
                AudioDecoder = CompatAudioDecoder

            reader = AudioDecoder(to_filelike(self.src), sample_rate=sample_rate)
            # mp3 has encoder delays that are not handled well when seeking (http://mp3decoders.mp3-tech.org/decoders_lame.html)
            if reader.metadata.codec == "mp3":
                self.skip_samples = 1105

            if sample_rate is None:
                sample_rate = reader.metadata.sample_rate

            self.reader = reader
            self.sample_rate = sample_rate

        return self.reader, self.sample_rate

    @property
    def metadata(self):
        reader, sample_rate = self.get_reader()
        return reader.metadata

    def read_segment(self, start=0, end=None, sample_rate=None):
        reader, sample_rate = self.get_reader(sample_rate)
        seek_adjustment = self.skip_samples / sample_rate if start > 0 else 0
        _samples = reader.get_samples_played_in_range(
            start + seek_adjustment, end + seek_adjustment if end is not None else None
        )
        if hasattr(_samples, "data"):
            samples = _samples.data
        samples.sample_rate = sample_rate
        return samples

    def _display_(self):
        samples = self.read_segment()
        return marimo_audio_mp3(samples)

    def _ipython_display_(self):
        from IPython.display import Audio, display

        samples = self.read_full()
        display(Audio(samples.numpy(), rate=samples.sample_rate))


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

    @property
    def metadata(self):
        return self.audio_reader.metadata

    def _display_(self):
        samples = self.load()
        return marimo_audio_mp3(samples)

    def _ipython_display_(self):
        from IPython.display import Audio, display

        samples = self.load()
        display(Audio(samples.numpy(), rate=samples.sample_rate))
