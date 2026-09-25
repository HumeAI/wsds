"""Audio codec layer: encoding, decoding, and format utilities.

This module contains all audio encoding/decoding logic, separated from the
data model layer in ws_audio.py. It provides:
- AudioDecoder: unified decoder with automatic backend selection (humecodec or torchaudio)
- encode_audio(): multi-backend encoder (humecodec -> torchcodec -> torchaudio)
- HTML audio rendering utility
"""

from __future__ import annotations

import io

import numpy as np


class AudioDecoder:
    """Unified audio decoder that works with humecodec or torchaudio backends."""

    def __init__(self, reader, metadata, sample_rate, codec_delay=0, in_matroska=False):
        self.reader = reader
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.debug = False
        self.codec_delay = codec_delay
        self.init_skip_samples = getattr(metadata, 'start_skip_samples', 0) or 0
        # wmav2/wmapro: flush produces unreliable output (wrong skip_samples,
        # wrong frame sizes) — always read from the start and trim.
        codec_name = getattr(metadata, 'codec', '') or ''
        self._seek_unreliable = codec_name in ('wmav2', 'wmapro')
        # vorbis in matroska/webm: container timestamps carry the stream's
        # UNDECLARED priming (no CodecDelay element) as a constant per-file
        # offset (measured +2..+16 ms), on a ~1 ms grid. Without knowing the
        # offset, mid-stream seeks land that far off — so read from the start.
        # With a known offset (measured at indexing time as the first decoded
        # frame's pts; see set_pts_offset) seeks become grid-accurate (<1 ms):
        # the offset feeds the same seek_adj compensation mp3 uses for its
        # declared start_skip_samples.
        # (OGG vorbis needs none of this: granule-derived pts + humecodec's
        # post-seek transition-frame fix make it sample-exact.)
        self._in_matroska = in_matroska
        self._matroska_vorbis = in_matroska and codec_name == 'vorbis'
        self._pts_offset_samples = None
        # Raw MPEG audio formats: no native seek table, and the demuxer's
        # byte<->time interpolation assumes CBR (badly wrong on VBR). Seed the
        # demuxer's AVIndexEntry list (from an offline index or a one-time
        # packet-index scan) so timestamp seeks land on a known-correct entry.
        # NEVER use seek_to_byte_offset here: after a raw byte seek the demuxer
        # stamps packets with wrong pts and, via AVFMT_GENERIC_INDEX, records
        # those (pts, pos) pairs in its internal seek index — poisoning every
        # subsequent timestamp seek on this decoder (measured: a read-from-start
        # after a byte seek resumed at the byte-seek position instead of 0).
        self._needs_seed_index = codec_name in ('mp3', 'mp2', 'mp1')
        self._seed_scan_done = False
        # On-demand demuxer-index seeding (ogg/vorbis, which has no native seek
        # table). Rather than add every episode point up front (each
        # av_add_index_entry is an O(n) sorted insert -> O(n*m)), we keep the
        # full index and add only a small window of points around each requested
        # seek target as segments are read. See set_seed_index / _seed_around.
        self._seed_index = None                 # ws_seek_index.SeekIndex or None
        self._seed_added = None
        self._seed_window = 4

    def _ensure_seed_index(self):
        """Make sure a seed index exists for formats that need one (mp3/mp2/mp1).

        If no precomputed index was attached (WSAudioEpisode.set_seek_index),
        run a one-time sequential packet-index scan and feed it to
        set_seed_index. The scan reads the whole blob (the cost the offline
        index exists to avoid) but yields exact per-packet (pos, pts) pairs
        even for VBR streams."""
        if self._seed_index is not None or self._seed_scan_done:
            return
        self._seed_scan_done = True
        from .ws_seek_index import PTS_UNIT, SeekIndex
        try:
            idx = self.reader.build_packet_index(self.reader.default_audio_stream, 128 * 1024)
            if idx and len(idx) > 1:
                # Scan positions are blob-relative -> audio_offset 0.
                self.set_seed_index(SeekIndex(
                    np.asarray([e.pos for e in idx], dtype=np.uint64),
                    np.asarray([round(e.pts_seconds / PTS_UNIT) for e in idx], dtype=np.uint32),
                    audio_offset=0))
        except Exception:
            pass

    def get_samples_played_in_range(self, tstart=0, tend=None, margin=.25):
        import torch

        chunk = True
        while chunk is not None:
            (chunk,) = self.reader.pop_chunks()

        # For short seeks and unreliable codecs, read from the start.
        # This avoids seek accuracy issues for tstart < 5s (tiny cost), codec
        # flush bugs for wmav2/wmapro, and matroska-vorbis pts offsets when no
        # measured offset is available (see _matroska_vorbis in __init__).
        read_from_start = (self._seek_unreliable or tstart < 5.0
                           or (self._matroska_vorbis and self._pts_offset_samples is None))

        # Only adjust for the start-of-stream skip when actually seeking — when
        # reading from start, the decoder applies skip_samples automatically
        # (and the trim below runs in the content timeline directly).
        seek_adj = 0.0
        if not read_from_start:
            # Timestamp seek: the demuxer applies start_skip_samples at pts=0
            # but not after seeking (mp3 encoder delay), and matroska-vorbis
            # timestamps carry the undeclared priming as a constant offset —
            # adjust tstart to compensate for either.
            skip = self.init_skip_samples + (self._pts_offset_samples or 0)
            seek_adj = skip / self.metadata.sample_rate
            tstart += seek_adj
            if tend is not None:
                tend += seek_adj

        seek_target = 0.0 if read_from_start else max(0, tstart - margin)
        if not read_from_start and self._needs_seed_index:
            # mp3/mp2/mp1: seeds are REQUIRED for accuracy (VBR breaks the
            # demuxer's CBR byte<->time estimate) — scan once if none attached.
            self._ensure_seed_index()
        # Seed the AVIndexEntry points around the target (ogg/vorbis, mp3) so
        # the seek brackets in ~1 read; accumulates across sequential reads.
        # This runs for read-from-start too: ffmpeg's generic seek REFUSES
        # (returns EPERM) any target below the demuxer's first index entry, and
        # after a deep seeded crop the index only has entries near past targets
        # — seeking back to 0 then needs the pts-0 seed point to exist.
        self._seed_around(seek_target)
        self.reader.seek(seek_target, "key")

        chunks = []
        eof = False
        empty_pops = 0
        while True:
            if not eof and self.reader.fill_buffer() == 1:
                eof = True
            (chunk,) = self.reader.pop_chunks()
            if chunk is None:
                if eof:
                    break                # buffer fully drained
                # A fill produced no decoded audio yet: mid-stream seek/decode
                # hiccup (e.g. flac "read_timestamp() failed in the middle").
                # Skip it — dereferencing None.pts kills the DataLoader worker
                # and with it the whole DDP run. The cap guards against a
                # wedged no-progress decoder: a hung worker is worse than a
                # crash.
                empty_pops += 1
                if empty_pops > 65536:
                    raise ValueError(
                        f"decoder made no progress after {empty_pops} empty "
                        f"pops (codec={self.metadata.codec}); wedged stream?")
                continue
            empty_pops = 0
            chunks.append(chunk)
            if tend is not None:
                chunk_end_pts = chunk.pts + chunk.shape[0] / self.sample_rate
                if chunk_end_pts > tend + margin:
                    break

        if not chunks:
            # Data-level failure (bad seek target / corrupt stream), not a bug:
            # raise ValueError so sample-skipping callers can drop this read.
            raise ValueError(
                f"decoder produced no samples for range [{tstart:.3f}, {tend}] "
                f"(codec={self.metadata.codec}, read_from_start={read_from_start})")

        # Determine the reference PTS for trimming. Post-seek pts are reliable:
        # timestamp seeks land on real packets (seeded AVIndexEntry / native
        # index) and the demuxer stamps pts from the landing entry.
        if read_from_start:
            chunk0_pts = 0.0
        else:
            chunk0_pts = chunks[0].pts
        prefix = round(tstart * self.sample_rate) - round(chunk0_pts * self.sample_rate)

        if self.debug:
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

    def set_pts_offset(self, offset_seconds):
        """Container-timeline offset: the pts of the FIRST decoded frame at
        stream start (0.0 when timestamps and content agree). Matroska-vorbis
        files with undeclared priming carry it as a constant shift on every
        timestamp; knowing it turns their seeks from ~+-20ms into grid-accurate
        (<1ms), so the read-from-start fallback is no longer needed. Measured
        once at indexing time and delivered via WSAudioEpisode.set_seek_index."""
        self._pts_offset_samples = round(float(offset_seconds) * self.metadata.sample_rate)

    def set_seed_index(self, index):
        """Store a SeekIndex (see wsds.ws_seek_index) for ON-DEMAND demuxer
        seeding. Points are added lazily in a small window around each seek
        target (see _seed_around) rather than all at once, so the per-seek cost
        stays O(window) even for multi-hour episodes with tens of thousands of
        points — and the index's raw zero-copy views are never converted
        wholesale. Use only for formats WITHOUT a native seek table (ogg/vorbis,
        mp3/mp2/mp1); never for mp4/mov (the moov already indexes them and
        av_add_index_entry is O(n) per insert against its millions of native
        entries)."""
        self._seed_index = index
        self._seed_added = set()

    def _seed_around(self, target_time):
        """Add the few seed-index points bracketing target_time to the demuxer's
        seek index (idempotent per point, accumulates across seeks). No-op unless
        a seed index was set via set_seed_index."""
        idx = self._seed_index
        if idx is None or len(idx) == 0:
            return
        i = idx.search(target_time)
        lo = max(0, i - self._seed_window)
        hi = min(len(idx), i + self._seed_window)
        sel = [k for k in range(lo, hi) if k not in self._seed_added]
        if sel:
            self._seed_added.update(sel)
            self.add_seek_points([idx.pos_at(k) for k in sel],
                                 [idx.pts_at(k) for k in sel])


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


def _sniff_matroska(src) -> bool:
    """True if `src` (a seekable file-like) starts with the EBML magic, i.e. is
    a matroska/webm container. humecodec doesn't expose the container format in
    its metadata, and AudioDecoder needs it: matroska pts after a mid-stream
    seek are too coarse to trim vorbis crops accurately (see _seek_unreliable)."""
    try:
        pos = src.tell()
        src.seek(0)
        head = src.read(4)
        src.seek(pos)
        return head == b"\x1a\x45\xdf\xa3"
    except Exception:
        return False


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
    in_matroska = _sniff_matroska(src)

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

    return AudioDecoder(reader, metadata, sample_rate, codec_delay=codec_delay, in_matroska=in_matroska)



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
        if format in ("ogg", "webm", "mka"):
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
