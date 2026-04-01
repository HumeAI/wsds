from __future__ import annotations

import typing
from dataclasses import dataclass

from .audio_codec import audio_to_html, create_decoder, encode_mp3
from .pupyarrow import pupyarrow



@dataclass()
class WSAudioEpisode:
    """A lazy seeking-capable audio reader for random-access to recordings stored in wsds shards.

    >>> from wsds import WSDataset
    >>> ds = WSDataset("librilight/source")
    >>> audio = ds[0].get_audio()
    >>> audio.load().shape
    torch.Size([1, 17884909])
    >>> audio.read_segment(start=2, end=5).shape
    torch.Size([1, 48000])
    >>> audio.read_segment(start=2, end=5, sample_rate=8000).shape
    torch.Size([1, 24000])
    """

    src: typing.Any
    _decoder: typing.Any = None
    _sample_rate: int | None = None
    skip_samples: int = 0

    def __repr__(self):
        return f"WSAudioEpisode(src={type(self.src)}, sample_rate={self._sample_rate})"

    def unwrap(self):
        """Return the raw audio bytes"""
        if hasattr(self.src, "as_buffer"):
            return self.src.as_buffer().to_pybytes()
        elif isinstance(self.src, (bytes, bytearray)):
            return self.src
        elif isinstance(self.src, pupyarrow.LazyBuffer):
            return self.src.read()
        else:
            raise TypeError(f"Unsupported src type: {type(self.src)}")

    to_bytes = unwrap

    def get_decoder(self, sample_rate=None):
        """Lazily creates/caches decoder via audio_codec.create_decoder()."""
        sample_rate_switch = False
        if self._sample_rate is not None:
            sample_rate_switch = self._sample_rate != sample_rate

        if self._decoder is None or sample_rate_switch:
            decoder = create_decoder(to_filelike(self.src), sample_rate=sample_rate)
            # mp3 has encoder delays that are not handled well when seeking
            if decoder.metadata.codec == "mp3":
                self.skip_samples = 1105

            if sample_rate is None:
                sample_rate = decoder.metadata.sample_rate

            self._decoder = decoder
            self._sample_rate = sample_rate

        return self._decoder, self._sample_rate

    @property
    def metadata(self):
        decoder, sample_rate = self.get_decoder()
        return decoder.metadata

    @property
    def sample_rate(self):
        _, sr = self.get_decoder()
        return sr

    def read_segment(self, start=0, end=None, sample_rate=None):
        decoder, sample_rate = self.get_decoder(sample_rate)
        seek_adjustment = self.skip_samples / sample_rate if start > 0 else 0
        _samples = decoder.get_samples_played_in_range(
            start + seek_adjustment, end + seek_adjustment if end is not None else None
        )
        if hasattr(_samples, "data"):
            samples = _samples.data
        else:
            samples = _samples
        samples.sample_rate = sample_rate
        return samples

    def load(self, sample_rate=None):
        samples = self.read_segment(sample_rate=sample_rate)
        return samples

    def _repr_html_(self):
        return audio_to_html(self.read_segment())

    def _display_(self):
        import marimo

        return marimo.audio(encode_mp3(self.read_segment()))


@dataclass(frozen=True)
class WSAudioSegment:
    """A lazy reference to a single sample from a segmented audio file.
    """

    episode: WSAudioEpisode
    tstart: float
    tend: float

    def __repr__(self) -> str:
        return f"WSAudioSegment(episode={self.episode}, tstart={self.tstart!s}, tend={self.tend!s})"

    @property
    def duration(self) -> float:
        """Duration of the audio segment in seconds."""
        return self.tend - self.tstart

    def with_context(self, before: float = 0, after: float = 0) -> "WSAudioSegment":
        """Return a new WSAudioSegment with expanded timestamps to include surrounding context.

        Args:
            before: Seconds of context to add before the segment start (will not go below 0)
            after: Seconds of context to add after the segment end

        Returns:
            A new WSAudioSegment instance with adjusted timestamps
        """
        return WSAudioSegment(
            episode=self.episode,
            tstart=max(0, self.tstart - before),
            tend=self.tend + after,
        )

    def with_timestamps(self, tstart: float | None = None, tend: float | None = None) -> "WSAudioSegment":
        """Return a new WSAudioSegment with modified timestamps.

        Args:
            tstart: New start time in seconds (None to keep current)
            tend: New end time in seconds (None to keep current)

        Returns:
            A new WSAudioSegment instance with the specified timestamps
        """
        return WSAudioSegment(
            episode=self.episode,
            tstart=tstart if tstart is not None else self.tstart,
            tend=tend if tend is not None else self.tend,
        )

    def load(self, sample_rate=None, pad_to_seconds=None):
        samples = self.episode.read_segment(self.tstart, self.tend, sample_rate)
        sample_rate = samples.sample_rate
        if pad_to_seconds is not None:
            import torch

            padding = int(pad_to_seconds * sample_rate - samples.shape[-1])
            samples = torch.nn.functional.pad(samples, (0, padding))
            samples.sample_rate = sample_rate
        return samples

    @property
    def metadata(self):
        return self.episode.metadata

    def _repr_html_(self):
        return audio_to_html(self.load())

    def _display_(self):
        import marimo

        return marimo.audio(encode_mp3(self.load()))
