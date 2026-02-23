from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .audio_codec import audio_to_html, create_decoder, encode_mp3, to_filelike

if TYPE_CHECKING:
    import torch


def load_segment(src: object, start: float, end: Optional[float], sample_rate: Optional[int] = None) -> torch.Tensor:
    """Efficiently loads an audio segment from `src` (see below) `tstart` to `tend` seconds while
    optionally resampling it to `sample_rate`.

    `src` can be one of:
    - a file-like object
    - a byte string
    - a PyArrow binary buffer in memory"""
    return AudioReader(src).read_segment(start, end, sample_rate=sample_rate)


@dataclass()
class AudioReader:
    """A lazy seeking-capable audio reader for random-access to recordings stored in wsds shards."""

    src: object
    _decoder: Any = None
    _sample_rate: Optional[int] = None
    skip_samples: int = 0

    def __repr__(self) -> str:
        return f"AudioReader(src={type(self.src)}, sample_rate={self._sample_rate})"

    def unwrap(self) -> bytes:
        """Return the raw audio bytes"""
        if hasattr(self.src, "as_buffer"):
            return self.src.as_buffer().to_pybytes()
        elif isinstance(self.src, (bytes, bytearray)):
            return self.src
        else:
            raise TypeError(f"Unsupported AudioReader src type: {type(self.src)}")

    def get_decoder(self, sample_rate: Optional[int] = None) -> tuple[Any, int]:
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

        assert self._sample_rate is not None
        return self._decoder, self._sample_rate

    @property
    def metadata(self) -> Any:
        decoder, sample_rate = self.get_decoder()
        return decoder.metadata

    @property
    def sample_rate(self) -> Optional[int]:
        _, sr = self.get_decoder()
        return sr

    def read_segment(self, start: float = 0, end: Optional[float] = None, sample_rate: Optional[int] = None) -> torch.Tensor:
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

    def load(self, sample_rate: Optional[int] = None) -> torch.Tensor:
        samples = self.read_segment(sample_rate=sample_rate)
        return samples

    def _repr_html_(self) -> str:
        return audio_to_html(self.read_segment())

    def _display_(self) -> object:
        import marimo

        return marimo.audio(encode_mp3(self.read_segment()))  # type: ignore[arg-type]


@dataclass(frozen=True)
class WSAudio:
    """A lazy reference to a single sample from a segmented audio file."""

    audio_reader: AudioReader
    tstart: float
    tend: float

    @property
    def duration(self) -> float:
        """Duration of the audio segment in seconds."""
        return self.tend - self.tstart

    def with_context(self, before: float = 0, after: float = 0) -> "WSAudio":
        """Return a new WSAudio with expanded timestamps to include surrounding context.

        Args:
            before: Seconds of context to add before the segment start (will not go below 0)
            after: Seconds of context to add after the segment end

        Returns:
            A new WSAudio instance with adjusted timestamps
        """
        return WSAudio(
            audio_reader=self.audio_reader,
            tstart=max(0, self.tstart - before),
            tend=self.tend + after,
        )

    def with_timestamps(self, tstart: Optional[float] = None, tend: Optional[float] = None) -> "WSAudio":
        """Return a new WSAudio with modified timestamps.

        Args:
            tstart: New start time in seconds (None to keep current)
            tend: New end time in seconds (None to keep current)

        Returns:
            A new WSAudio instance with the specified timestamps
        """
        return WSAudio(
            audio_reader=self.audio_reader,
            tstart=tstart if tstart is not None else self.tstart,
            tend=tend if tend is not None else self.tend,
        )

    def load(self, sample_rate: Optional[int] = None, pad_to_seconds: Optional[float] = None) -> torch.Tensor:
        samples = self.audio_reader.read_segment(self.tstart, self.tend, sample_rate)
        sample_rate = samples.sample_rate  # type: ignore[attr-defined]
        if pad_to_seconds is not None:
            import torch

            padding = int(pad_to_seconds * sample_rate - samples.shape[-1])
            samples = torch.nn.functional.pad(samples, (0, padding))
            samples.sample_rate = sample_rate  # type: ignore[attr-defined]
        return samples

    @property
    def metadata(self) -> Any:
        return self.audio_reader.metadata

    def _repr_html_(self) -> str:
        return audio_to_html(self.load())

    def _display_(self) -> object:
        import marimo

        return marimo.audio(encode_mp3(self.load()))  # type: ignore[arg-type]
