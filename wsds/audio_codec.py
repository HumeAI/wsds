"""Audio codec layer: encoding, decoding, and format utilities.

This module contains all audio encoding/decoding logic, separated from the
data model layer in ws_audio.py. It provides:
- AudioDecoder: unified decoder with automatic backend selection (humecodec or torchaudio)
- encode_audio(): multi-backend encoder (humecodec -> torchcodec -> torchaudio)
- HTML audio rendering utility
"""

from __future__ import annotations

import io
import traceback
import typing

import pyarrow as pa


class AudioDecoder:
    """Unified audio decoder that works with humecodec or torchaudio backends."""

    def __init__(self, reader, metadata, sample_rate, codec_delay=0):
        self.reader = reader
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.debug = False
        self.codec_delay = codec_delay
        self.init_skip_samples = getattr(metadata, 'start_skip_samples', 0) or 0

    def get_samples_played_in_range(self, tstart=0, tend=None, margin=.25):
        import torch

        chunk = True
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()

        seek_adj = self.init_skip_samples / self.metadata.sample_rate if tstart > 0 else 0
        tstart += seek_adj
        if tend is not None:
            tend += seek_adj

        # rough seek
        self.reader.seek(max(0, tstart - margin), "key")

        chunks = []
        more_data = True
        while more_data:
            if self.reader.fill_buffer() == 1:
                more_data = False
            (chunk,) = self.reader.pop_chunks()
            chunks.append(chunk)
            if tend is not None and chunk.pts + chunk.shape[0] / self.sample_rate > tend + margin:
                break

        # If the seek landed at (or near) the start of the stream, treat it
        # the same as tstart=0: ignore the first-chunk PTS so that the timeline
        # matches what ffmpeg CLI / torchaudio produce for a full decode.
        seek_landed_at_start = tstart == 0 or chunks[0].pts < margin
        chunk0_pts = 0.0 if seek_landed_at_start else chunks[0].pts
        prefix = round(tstart * self.sample_rate) - round(chunk0_pts * self.sample_rate)

        if self.debug:
            import torch as _t
            total_samples = sum(c.shape[0] for c in chunks)
            print(f"    [decode] codec={self.metadata.codec} sr={self.sample_rate} "
                  f"tstart_orig={tstart - seek_adj:.4f} tstart_adj={tstart:.4f} "
                  f"seek_adj={seek_adj:.6f} (init_skip={self.init_skip_samples} codec_delay={self.codec_delay}) "
                  f"chunk0.pts={chunks[0].pts:.6f} chunk0_pts_used={chunk0_pts:.6f} "
                  f"n_chunks={len(chunks)} total_samples={total_samples} prefix={prefix}", flush=True)

        if prefix < 0:
            if self.debug:
                print(f"    [trim] negative prefix {prefix}, clamping to 0", flush=True)
            prefix = 0
        samples = torch.cat(chunks)
        if tend is not None:
            return samples[prefix : prefix + round(tend * self.sample_rate) - round(tstart * self.sample_rate)].mT
        else:
            return samples[prefix:].mT

def _create_reader_humecodec(src, buffer_size):
    from humecodec import MediaDecoder

    reader = MediaDecoder(src=src, buffer_size=buffer_size)
    metadata = reader.get_src_stream_info(reader.default_audio_stream)
    return reader, metadata


def _create_reader_torchaudio(src, buffer_size):
    from torchaudio.io import StreamReader

    reader = StreamReader(src=src, buffer_size=buffer_size)
    metadata = reader.get_src_stream_info(reader.default_audio_stream)
    return reader, metadata


def _create_decoder_torchcodec(src, sample_rate):
    """Create a torchcodec-backed decoder that matches the AudioDecoder interface."""
    from types import SimpleNamespace

    from torchcodec.decoders import AudioDecoder as TorchcodecDecoder

    # torchcodec accepts bytes but not BytesIO
    decoder = TorchcodecDecoder(src, sample_rate=sample_rate)
    metadata = decoder.metadata

    class TorchcodecAdapter:
        def __init__(self):
            self.metadata = metadata
            self.sample_rate = sample_rate if sample_rate is not None else int(metadata.sample_rate)

        def get_samples_played_in_range(self, tstart=0, tend=None):
            return decoder.get_samples_played_in_range(tstart, tend)

    return TorchcodecAdapter()


_STREAMING_BACKENDS = [
    (_create_reader_humecodec, "humecodec"),
    (_create_reader_torchaudio, "torchaudio.io"),
]

_chosen_backend = None


def create_decoder(src, sample_rate=None):
    """Factory: tries humecodec -> torchaudio -> torchcodec, returns a decoder instance.

    Args:
        src: A file-like object for audio data.
        sample_rate: Optional target sample rate for resampling.

    Returns:
        A decoder with .metadata, .sample_rate, and .get_samples_played_in_range().
    """
    global _chosen_backend

    buffer_size = getattr(src, "_optimal_read_size", 128 * 1024)

    if _chosen_backend is not None:
        if _chosen_backend == "torchcodec":
            return _create_decoder_torchcodec(src, sample_rate)
        reader, metadata = _chosen_backend(src, buffer_size)
    else:
        for factory, module in _STREAMING_BACKENDS:
            try:
                reader, metadata = factory(src, buffer_size)
                _chosen_backend = factory
                break
            except ImportError:
                continue
        else:
            # Fall back to torchcodec (different API, no streaming reader)
            try:
                decoder = _create_decoder_torchcodec(src, sample_rate)
                _chosen_backend = "torchcodec"
                return decoder
            except ImportError:
                raise ImportError("Neither humecodec, torchaudio, nor torchcodec is installed.")

    if sample_rate is None:
        sample_rate = int(metadata.sample_rate)

    reader.add_basic_audio_stream(
        frames_per_chunk=int(1 * sample_rate),
        sample_rate=sample_rate,
        decoder_option={"threads": "4", "thread_type": "frame"},
    )

    # Get codec_delay from the decoder (available after add_audio_stream opens the codec)
    codec_delay = 0
    try:
        out_info = reader.get_out_stream_info(0)
        codec_delay = getattr(out_info, 'codec_delay', 0) or 0
    except Exception:
        pass

    return AudioDecoder(reader, metadata, sample_rate, codec_delay=codec_delay)



def encode_audio(samples, format="mp3", sample_rate=None, bitrate=None) -> bytes:
    """Encode a torch tensor to audio bytes.

    Tries humecodec -> torchcodec -> torchaudio as encoder backends.

    >>> from wsds import WSDataset
    >>> audio = WSDataset("librilight/source")[0].get_audio()
    >>> samples = audio.read_segment(start=0, end=2.0, sample_rate=16000)
    >>> mp3 = encode_audio(samples, format="mp3")
    >>> mp3[:3] == b"ID3" or mp3[:2] in (b"\\xff\\xfb", b"\\xff\\xf3")
    True
    >>> ogg = encode_audio(samples, format="ogg")  # doctest: +SKIP
    >>> ogg[:4] == b"OggS"  # doctest: +SKIP
    True

    Args:
        samples: A torch.Tensor with a .sample_rate attribute. Shape: (channels, frames).
        format: Output format, e.g. "mp3", "ogg" (Opus). Default: "mp3".
        sample_rate: Target sample rate (defaults to samples.sample_rate).
        bitrate: Bitrate in bps. Only used for formats that support it (e.g. Opus).

    Returns:
        Encoded audio bytes.
    """
    if sample_rate is None:
        sample_rate = int(samples.sample_rate)

    out = io.BytesIO()
    try:
        from humecodec import MediaEncoder

        waveform = samples.mT.float().contiguous()
        enc = MediaEncoder(out, format)
        stream_kwargs = dict(sample_rate=sample_rate, num_channels=waveform.size(1), format="flt")
        if format == "ogg":
            from humecodec import CodecConfig

            stream_kwargs.update(encoder="libopus", encoder_format="flt")
            if bitrate:
                stream_kwargs["codec_config"] = CodecConfig(bit_rate=bitrate)
        enc.add_audio_stream(**stream_kwargs)
        with enc.open():
            enc.write_audio_chunk(0, waveform)
    except ImportError:
        try:
            from torchcodec.encoders import AudioEncoder

            AudioEncoder(samples, sample_rate=sample_rate).to_file_like(out, format)
        except ImportError:
            import torchaudio

            torchaudio.save(out, samples, sample_rate, format=format)

    return out.getvalue()


def audio_to_html(samples) -> str:
    """Encode samples to an HTML <audio> tag with base64 MP3 data.

    Args:
        samples: A torch.Tensor with a .sample_rate attribute.

    Returns:
        An HTML string with an embedded audio player.
    """
    import base64

    mp3_data = base64.b64encode(encode_audio(samples, format="mp3")).decode("ascii")
    return f'<audio controls src="data:audio/mp3;base64,{mp3_data}"></audio>'
