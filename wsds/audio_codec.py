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

import numpy as np
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
        # Codecs where flush produces unreliable output (wrong skip_samples,
        # wrong frame sizes). For these, always read from the start and trim.
        # (vorbis was here too, but its post-seek transition-frame pts is now
        # corrected in humecodec's StreamProcessor, so it seeks accurately.)
        codec_name = getattr(metadata, 'codec', '') or ''
        self._seek_unreliable = codec_name in ('wmav2', 'wmapro')
        # Raw MPEG audio formats: timestamp seek does sequential scan,
        # byte-offset seek with our own index is much faster.
        self._use_byte_index = codec_name in ('mp3', 'mp2', 'mp1')
        self._packet_index = None
        # On-demand demuxer-index seeding (ogg/vorbis, which has no native seek
        # table). Rather than add every episode point up front (each
        # av_add_index_entry is an O(n) sorted insert -> O(n*m)), we keep the
        # full index and add only a small window of points around each requested
        # seek target as segments are read. See set_seed_index / _seed_around.
        self._seed_pts = None
        self._seed_positions = None
        self._seed_added = None
        self._seed_window = 4

    def _build_index(self):
        """Build a sparse packet index for byte-offset seeking."""
        if self._packet_index is not None:
            return
        from ._timing import record
        try:
            with record("build_packet_index"):
                idx = self.reader.build_packet_index(
                    self.reader.default_audio_stream, 128 * 1024)
            if idx and len(idx) > 1:
                self._packet_index = idx
        except Exception:
            self._packet_index = []

    def _indexed_seek(self, target_time):
        """Seek via byte offset using the packet index. Returns the index entry's PTS or None."""
        self._build_index()
        if not self._packet_index:
            return None
        # Find last entry with pts <= target_time
        best = self._packet_index[0]
        for entry in self._packet_index:
            if entry.pts_seconds <= target_time:
                best = entry
            else:
                break
        self.reader.seek_to_byte_offset(best.pos)
        return best.pts_seconds

    def get_samples_played_in_range(self, tstart=0, tend=None, margin=.25):
        import torch

        chunk = True
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()

        # For short seeks and unreliable codecs, read from the start.
        # This avoids seek accuracy issues for tstart < 5s (tiny cost) and
        # codec flush bugs for wmav2/wmapro/vorbis.
        read_from_start = self._seek_unreliable or tstart < 5.0

        # Only adjust for start_skip_samples when actually seeking — when
        # reading from start, the decoder applies skip_samples automatically.
        seek_adj = 0.0
        index_pts = None
        if not read_from_start:
            # For raw MPEG formats, use indexed byte seek (fast, avoids sequential scan).
            # No seek_adj needed: the index PTS and decoded audio are both in
            # the raw timeline (skip_samples is not applied after byte seek).
            if self._use_byte_index:
                index_pts = self._indexed_seek(tstart - margin)
            else:
                # Timestamp seek: the demuxer applies start_skip_samples at
                # pts=0 but not after seeking, so adjust tstart to compensate.
                seek_adj = self.init_skip_samples / self.metadata.sample_rate
                tstart += seek_adj
                if tend is not None:
                    tend += seek_adj

        if index_pts is None:
            # Fall back to timestamp seek (or read from start)
            seek_target = 0.0 if read_from_start else max(0, tstart - margin)
            if not read_from_start:
                # Seed just the AVIndexEntry points around this target (ogg/vorbis)
                # so the seek brackets in ~1 read; accumulates across seq reads.
                self._seed_around(seek_target)
            self.reader.seek(seek_target, "key")

        chunks = []
        more_data = True
        while more_data:
            if self.reader.fill_buffer() == 1:
                more_data = False
            (chunk,) = self.reader.pop_chunks()
            chunks.append(chunk)
            if tend is not None:
                chunk_end_pts = chunk.pts + chunk.shape[0] / self.sample_rate
                if index_pts is not None:
                    # PTS not updated by demuxer after byte seek — estimate from index
                    elapsed = sum(c.shape[0] for c in chunks) / self.sample_rate
                    chunk_end_pts = index_pts + elapsed
                if chunk_end_pts > tend + margin:
                    break

        # Determine the reference PTS for trimming
        if read_from_start:
            chunk0_pts = 0.0
        elif index_pts is not None:
            # Byte seek: demuxer PTS is stale, use our index entry
            chunk0_pts = index_pts
        else:
            chunk0_pts = chunks[0].pts
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
        # Unwrap humecodec Chunk (a torch.Tensor subclass) to its plain `_elem`
        # tensor before cat: otherwise every torch op on a Chunk goes through
        # __torch_dispatch__ -> tree_map(unwrap, ...), which allocates ~14 cyclic
        # pytree objects per decode and feeds the GC-collection latency spikes.
        samples = torch.cat([getattr(c, "_elem", c) for c in chunks])
        if tend is not None:
            return samples[prefix : prefix + round(tend * self.sample_rate) - round(tstart * self.sample_rate)].mT
        else:
            return samples[prefix:].mT

    def add_seek_points(self, positions, pts_seconds):
        """Seed the demuxer's seek index with precomputed (byte position, pts)
        pairs so a subsequent timestamp seek() brackets the target and converges
        in ~1 read instead of a full binary/secant search across the file
        (dramatic for Ogg/Vorbis, which has no container index). `positions` are
        byte offsets in the input (blob-relative when decoding via a LazyBuffer
        over the audio blob); `pts_seconds` are in seconds. Accuracy is
        unchanged — the seek still reads the landing page to position exactly.

        Requires humecodec >= 0.8 (the `add_seek_points` backend method); returns
        False and no-ops on older builds or the torchcodec backend.
        """
        fn = getattr(self.reader, "add_seek_points", None)
        if fn is None:
            return False
        fn([int(p) for p in positions], [float(t) for t in pts_seconds])
        return True

    def set_seed_index(self, positions, pts_seconds):
        """Store a precomputed (byte-position, pts) index for ON-DEMAND demuxer
        seeding. Points are added lazily in a small window around each seek
        target (see _seed_around) rather than all at once, so the per-seek cost
        stays O(window) even for multi-hour episodes with tens of thousands of
        points. Use only for formats WITHOUT a native seek table (ogg/vorbis);
        never for mp4/mov (the moov already indexes them and av_add_index_entry
        is O(n) per insert against its millions of native entries).

        `pts_seconds` MUST be ascending — the seek-index generator emits points
        in blob-scan order (ascending pts) — so we store them as-is (no sort) and
        binary-search with np.searchsorted in _seed_around."""
        self._seed_pts = np.asarray(pts_seconds, dtype=np.float64)
        self._seed_positions = np.asarray(positions, dtype=np.int64)
        self._seed_added = set()

    def _seed_around(self, target_time):
        """Add the few seed-index points bracketing target_time to the demuxer's
        seek index (idempotent per point, accumulates across seeks). No-op unless
        a seed index was set via set_seed_index."""
        if self._seed_pts is None or self._seed_pts.size == 0:
            return
        i = int(np.searchsorted(self._seed_pts, target_time, side="right"))
        lo = max(0, i - self._seed_window)
        hi = min(self._seed_pts.size, i + self._seed_window)
        sel = [k for k in range(lo, hi) if k not in self._seed_added]
        if sel:
            self._seed_added.update(sel)
            self.add_seek_points(self._seed_positions[sel].tolist(),
                                 self._seed_pts[sel].tolist())


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
