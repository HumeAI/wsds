"""Audio codec layer: encoding, decoding, and format utilities.

This module contains all audio encoding/decoding logic, separated from the
data model layer in ws_audio.py. It provides:
- Decoder backends (TorchFFmpegAudioDecoder, CompatAudioDecoder)
- A factory for creating decoders with automatic backend selection
- MP3 encoding with multi-backend fallback
- HTML audio rendering utility
"""

from __future__ import annotations

import io
import typing

import pyarrow as pa


def to_filelike(src: typing.Any) -> typing.BinaryIO:
    """Coerces files, byte-strings and PyArrow binary buffers into file-like objects."""
    if hasattr(src, "read"):  # an open file
        return src
    # if not an open file then we assume some kind of binary data in memory
    if hasattr(src, "as_buffer"):  # PyArrow binary data
        return pa.BufferReader(src.as_buffer())
    return io.BytesIO(src)


class TorchFFmpegAudioDecoder:
    def __init__(self, src, sample_rate):
        from torchffmpeg import MediaDecoder

        if hasattr(src, "_optimal_read_size"):
            buffer_size = src._optimal_read_size
        else:
            buffer_size = 128 * 1024
        self.src = src
        self.reader = MediaDecoder(to_filelike(self.src), buffer_size=buffer_size)
        self.metadata = self.reader.get_src_stream_info(self.reader.default_audio_stream)

        if sample_rate is None:
            sample_rate = int(self.metadata.sample_rate)

        self.sample_rate = sample_rate

        self.reader.add_basic_audio_stream(
            frames_per_chunk=int(32 * sample_rate),
            sample_rate=sample_rate,
            decoder_option={"threads": "4", "thread_type": "frame"},
        )

    def get_samples_played_in_range(self, tstart=0, tend=None):
        import torch

        self.reader.seek(max(0, tstart - 1), "key")

        if tend is None:
            chunks = []
            more_data = True
            while more_data:
                if self.reader.fill_buffer() == 1:
                    more_data = False
                (chunk,) = self.reader.pop_chunks()
                if chunk is not None:
                    chunks.append(chunk)
            prefix = int((tstart - chunks[0].pts) * self.sample_rate)
            if prefix < 0:
                prefix = 0
            return torch.cat(chunks)[prefix:].mT

        self.reader.fill_buffer()
        (chunk,) = self.reader.pop_chunks()
        prefix = int((tstart - chunk.pts) * self.sample_rate)
        if prefix < 0:
            prefix = 0
        if tend:
            samples = chunk[prefix : prefix + int((tend - tstart) * self.sample_rate)].mT
        else:
            samples = chunk[prefix:].mT
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()
        return samples


class CompatAudioDecoder:
    def __init__(self, src, sample_rate):
        import torchaudio

        if not hasattr(torchaudio, "io"):
            raise ImportError("You need either torchaudio<2.9 or torchcodec installed")
        self.src = src
        if hasattr(src, "_optimal_read_size"):
            buffer_size = src._optimal_read_size
        else:
            buffer_size = 128 * 1024
        self.reader = torchaudio.io.StreamReader(src=to_filelike(self.src), buffer_size=buffer_size)
        self.metadata = self.reader.get_src_stream_info(0)

        if sample_rate is None:
            sample_rate = self.metadata.sample_rate

        self.sample_rate = sample_rate

        # fetch 32 seconds because we likely need 30s at maximum but the seeking may be imprecise (and we seek 1s early)
        # FIXME: check if we can get away with some better settings here (-1, maybe 10s + concatenate the chunks in a loop)
        self.reader.add_basic_audio_stream(
            frames_per_chunk=int(32 * sample_rate),
            sample_rate=sample_rate,
            decoder_option={"threads": "4", "thread_type": "frame"},
        )

    def get_samples_played_in_range(self, tstart=0, tend=None):
        # rought seek
        self.reader.seek(max(0, tstart - 1), "key")

        if tend is None:
            import torch

            chunks = []
            more_data = True
            while more_data:
                if self.reader.fill_buffer() == 1:
                    more_data = False
                (chunk,) = self.reader.pop_chunks()
                chunks.append(chunk)
            prefix = int((tstart - chunks[0].pts) * self.sample_rate)
            if prefix < 0:
                prefix = 0
            return torch.cat(chunks)[prefix:].mT

        self.reader.fill_buffer()
        (chunk,) = self.reader.pop_chunks()
        # tight crop (seems accurate down to 1 sample in my tests)
        prefix = int((tstart - chunk.pts) * self.sample_rate)
        if prefix < 0:
            prefix = 0
        if tend:
            samples = chunk[prefix : prefix + int((tend - tstart) * self.sample_rate)].mT
        else:
            samples = chunk[prefix:].mT
        # clear out any remaining data
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()
        return samples


def create_decoder(src, sample_rate=None):
    """Factory: tries torchffmpeg -> torchcodec -> torchaudio, returns a decoder instance.

    Args:
        src: A file-like object or bytes-like source for audio data.
        sample_rate: Optional target sample rate for resampling.

    Returns:
        A decoder instance with .metadata, .sample_rate, and .get_samples_played_in_range() interface.
    """
    try:
        from torchffmpeg import MediaDecoder as _  # noqa: F401

        AudioDecoder = TorchFFmpegAudioDecoder
    except ImportError:
        try:
            from torchcodec.decoders import AudioDecoder
        except ImportError:
            AudioDecoder = CompatAudioDecoder

    return AudioDecoder(src, sample_rate=sample_rate)


def decode_segment(src, start=0, end=None, sample_rate=None):
    """One-shot decode: creates decoder, reads segment, returns tensor with .sample_rate attr.

    Handles MP3 skip_samples compensation automatically.

    Args:
        src: Audio source (file-like, bytes, or PyArrow buffer).
        start: Start time in seconds.
        end: End time in seconds (None for rest of file).
        sample_rate: Optional target sample rate.

    Returns:
        A torch.Tensor with a .sample_rate attribute.
    """
    filelike = to_filelike(src)
    decoder = create_decoder(filelike, sample_rate)

    skip_samples = 0
    if decoder.metadata.codec == "mp3":
        skip_samples = 1105

    if sample_rate is None:
        sample_rate = decoder.metadata.sample_rate

    seek_adjustment = skip_samples / sample_rate if start > 0 else 0
    samples = decoder.get_samples_played_in_range(
        start + seek_adjustment, end + seek_adjustment if end is not None else None
    )
    if hasattr(samples, "data"):
        samples = samples.data
    samples.sample_rate = sample_rate
    return samples


def encode_mp3(samples) -> bytes:
    """Encode a torch tensor to MP3 bytes.

    Tries torchffmpeg -> torchcodec -> torchaudio as encoder backends.

    Args:
        samples: A torch.Tensor with a .sample_rate attribute. Shape: (channels, frames).

    Returns:
        MP3-encoded bytes.
    """
    out = io.BytesIO()
    try:
        from torchffmpeg import MediaEncoder

        sample_rate = int(samples.sample_rate)
        # samples is (channels, frames), write_audio_chunk expects (frames, channels)
        waveform = samples.mT.float().contiguous()
        enc = MediaEncoder(out, "mp3")
        enc.add_audio_stream(sample_rate=sample_rate, num_channels=waveform.size(1), format="flt")
        with enc.open():
            enc.write_audio_chunk(0, waveform)
    except ImportError:
        try:
            from torchcodec.encoders import AudioEncoder

            AudioEncoder(samples, sample_rate=int(samples.sample_rate)).to_file_like(out, "mp3")
        except ImportError:
            import torchaudio

            torchaudio.save(out, samples, int(samples.sample_rate), format="mp3")

    return out.getvalue()


def audio_to_html(samples) -> str:
    """Encode samples to an HTML <audio> tag with base64 MP3 data.

    Args:
        samples: A torch.Tensor with a .sample_rate attribute.

    Returns:
        An HTML string with an embedded audio player.
    """
    import base64

    mp3_data = base64.b64encode(encode_mp3(samples)).decode("ascii")
    return f'<audio controls src="data:audio/mp3;base64,{mp3_data}"></audio>'
