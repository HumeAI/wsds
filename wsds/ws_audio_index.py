"""Audio seek index generation and efficient segment loading using humecodec.

This module provides:
- `generate_audio_seek_index`: Scans audio shards using pupyarrow to get byte
  offsets, then uses humecodec's `build_packet_index` to build a seek index
  for each audio blob. The index is stored in a new wsds shard.
- `load_audio_segment`: Given a WSSample (containing seek index columns) and a
  timestamp + duration, opens the shard file directly via `shard.get_reader()`
  and decodes only the needed portion using `process_packet()`.

The seek index shard contains these columns per sample:
- `__key__`: The sample key (matching the audio shard)
- `seek_index_audio_offset`: int64 byte offset of the audio blob inside the audio shard
- `seek_index_audio_length`: int64 byte length of the audio blob
- `seek_index_positions`: list[int64] absolute byte positions inside the shard,
    spaced every ~512kB
- `seek_index_pts_seconds`: list[float64] corresponding timestamps in seconds
- `seek_index_duration`: float64 total audio duration in seconds
"""

from __future__ import annotations

import io
import typing
from pathlib import Path

import humecodec
import polars as pl
import torch

from .pupyarrow.file_reader import LocalFileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray, LazyBuffer

SEEK_RESOLUTION_BYTES = 512 * 1024  # 512kB between seek points

# Header/footer bytes cached in the index and served locally so a decoder's
# open fetches nothing from the store — a cold seek then touches only the target
# blocks. The decoder is opened with a SMALL avio buffer (DECODE_BUFFER_BYTES):
# find_stream_info then reads only ~8kB of header (a big buffer inflates it to
# ~132kB), and no read-ahead is needed because the block-cache layer already
# provides it (whole 128kB blocks, served locally). The Ogg duration probe still
# scans back a ~64kB window (FOOTER_PROBE_BYTES) but only needs the real last
# page, so we cache the last FOOTER_CACHE_BYTES and zero-fill the rest.
HEADER_CACHE_BYTES = 8192
FOOTER_CACHE_BYTES = 8192
FOOTER_PROBE_BYTES = 65536
DECODE_BUFFER_BYTES = 4096


def _measure_header_len(audio_bytes, buffer_size=DECODE_BUFFER_BYTES, cap=131072):
    """Bytes the decoder actually reads at the head during open (find_stream_info
    + first-packet confirm) with `buffer_size` — cache exactly this so a cold
    open touches 0 shard blocks. Self-adjusts to setup-header size (codebook
    complexity) vs a fixed guess. The consumer opens with the same buffer_size.
    Falls back to HEADER_CACHE_BYTES on failure."""
    reads = []

    class _T:
        def __init__(s): s.pos = 0; s.n = len(audio_bytes)
        def read(s, k=-1):
            if k is None or k < 0: k = s.n - s.pos
            k = min(k, s.n - s.pos)
            d = audio_bytes[s.pos:s.pos + k]; reads.append((s.pos, len(d))); s.pos += len(d); return d
        read1 = read
        def seek(s, o, w=0): s.pos = o if w == 0 else (s.pos + o if w == 1 else s.n + o); return s.pos
        def tell(s): return s.pos
        def size(s): return s.n
        def readable(s): return True
        def seekable(s): return True

    try:
        r = humecodec.MediaDecoder(src=_T(), buffer_size=buffer_size)
        info = r.get_src_stream_info(r.default_audio_stream)
        r.add_basic_audio_stream(frames_per_chunk=int(info.sample_rate),
                                 sample_rate=int(info.sample_rate))
    except Exception:
        return min(cap, len(audio_bytes), HEADER_CACHE_BYTES)
    half = max(1, len(audio_bytes) // 2)
    return min(cap, max((a + n for a, n in reads if a < half), default=HEADER_CACHE_BYTES))


def _ogg_footer(buf, search_cap=65536, fallback=FOOTER_CACHE_BYTES):
    """The real footer the Ogg duration probe needs: bytes from the last page
    carrying a VALID granule (>=0) to EOF (typically ~2-4kB). Non-Ogg / not
    found -> last `fallback` bytes. Searches only the last `search_cap` bytes."""
    if buf[:4] != b"OggS":
        return bytes(buf[-min(fallback, len(buf)):])
    end = len(buf)
    lo = max(0, end - search_cap)
    pos = buf.rfind(b"OggS", lo)
    while pos != -1:
        # granule position = int64 LE at [pos+6, pos+14); -1 (0xFFFF..) == no packet
        if pos + 14 <= end and int.from_bytes(buf[pos + 6:pos + 14], "little", signed=True) >= 0:
            return bytes(buf[pos:])
        pos = buf.rfind(b"OggS", lo, pos)
    return bytes(buf[-min(fallback, len(buf)):])


def generate_audio_seek_index(
    audio_shard_path: str | Path,
    output_path: str | Path,
    resolution: int = SEEK_RESOLUTION_BYTES,
    key_column: str = "__key__",
    audio_column: str = "audio",
    compression: str | None = "zstd",
):
    """Generate a seek index shard for an audio shard.

    For each audio sample in the shard, opens it with humecodec.MediaDecoder
    and calls `build_packet_index(resolution=...)` to get a sparse index of
    byte positions and pts values. These are stored as absolute shard-file
    offsets so that a reader can later seek directly within the shard.

    Args:
        audio_shard_path: Path to the source .wsds shard containing audio.
        output_path: Path for the output seek index .wsds shard.
        resolution: Minimum byte distance between index entries (default 512KB).
        key_column: Name of the key column in the source shard.
        audio_column: Name of the audio column in the source shard.
        compression: Compression for the output shard (default "zstd").
    """
    from .ws_sink import WSSink

    audio_shard_path = Path(audio_shard_path)
    output_path = Path(output_path)

    reader = LocalFileReader(audio_shard_path)
    feather = FeatherFile(reader)

    rows = []

    for batch_idx in range(feather.num_record_batches):
        batch = feather.record_batch(batch_idx)
        key_col = batch.column(key_column)
        audio_col = batch.column(audio_column)

        if not isinstance(audio_col, LazyBinaryArray):
            raise TypeError(f"Expected binary column for '{audio_column}', got {type(audio_col).__name__}")

        # Absolute file offset of the data buffer backing this column
        data_buf_offset = audio_col._data_buffer._offset

        for i in range(batch.num_rows):
            key = key_col[i]
            if isinstance(key, bytes):
                key = key.decode("utf-8")

            # Absolute byte offset and length of this audio element in the shard file
            elem_start = int(audio_col.offsets[i])
            elem_end = int(audio_col.offsets[i + 1])
            audio_offset = data_buf_offset + elem_start
            audio_length = elem_end - elem_start

            # Read the audio blob and build a packet index via humecodec
            audio_buf = audio_col[i]
            audio_bytes = audio_buf._read_all()

            decoder = humecodec.MediaDecoder(io.BytesIO(audio_bytes), buffer_size=len(audio_bytes))
            decoder.add_audio_stream(frames_per_chunk=-1)
            packet_index = decoder.build_packet_index(resolution=resolution)

            # Store positions as absolute shard-file offsets, pts as seconds
            seek_positions = [audio_offset + entry.pos for entry in packet_index]
            seek_pts_seconds = [entry.pts_seconds for entry in packet_index]

            # Compute total duration from the last packet
            if packet_index:
                last = packet_index[-1]
                duration = last.pts_seconds + last.duration_seconds
            else:
                duration = 0.0

            rows.append(
                {
                    key_column: key,
                    "seek_index_audio_offset": audio_offset,
                    "seek_index_audio_length": audio_length,
                    "seek_index_positions": seek_positions,
                    "seek_index_pts_seconds": seek_pts_seconds,
                    "seek_index_duration": duration,
                    # served locally at decode time so open touches 0 shard blocks
                    "seek_index_header": audio_bytes[:_measure_header_len(audio_bytes)],
                    "seek_index_footer": _ogg_footer(audio_bytes),  # real last page (~2-4kB)
                }
            )

    feather.close()

    with WSSink(str(output_path), compression=compression) as sink:
        for row in rows:
            sink.write(row)


class _HFOverlayReader:
    """Wraps a FileReader and serves the audio blob's first ``len(header)`` and
    last ``len(footer)`` bytes from in-memory buffers cached in the seek index,
    so a decoder's open-time reads (header/setup + Ogg duration probe) never hit
    the underlying store. Offsets are absolute; the blob spans
    ``[audio_offset, audio_offset + audio_length)``. Non-read attributes are
    delegated to the wrapped reader."""

    def __init__(self, reader, audio_offset, audio_length, header, footer):
        self._reader = reader
        self._ao = int(audio_offset)
        self._al = int(audio_length)
        self._header = header
        self._footer = footer
        self._H = len(header)
        self._F = len(footer)

    def read(self, offset: int, length: int) -> bytes:
        s = offset - self._ao          # blob-relative start
        e = s + length
        L = self._al
        fc_start = L - self._F                                  # real footer bytes
        # zeros cover the duration-probe window before the real last page
        fz_start = max(self._H, L - FOOTER_PROBE_BYTES) if self._F else L
        if s < 0 or length <= 0 or (s >= self._H and e <= fz_start):
            return self._reader.read(offset, length)           # pure middle / OOB
        if e <= self._H:
            return self._header[s:e]                            # pure header
        if s >= fc_start:
            return self._footer[s - fc_start:e - fc_start]      # pure footer
        out = bytearray(length)                                 # spans regions
        he = min(e, self._H)
        if s < he:
            out[0:he - s] = self._header[s:he]
        fs = max(s, fc_start)
        if self._F and fs < e:
            out[fs - s:length] = self._footer[fs - fc_start:e - fc_start]
        # [fz_start, fc_start) left as zeros (duration probe, never fetched)
        ms, me = max(s, self._H), min(e, fz_start)
        if ms < me:
            out[ms - s:me - s] = self._reader.read(self._ao + ms, me - ms)
        return bytes(out)

    def __getattr__(self, name):
        return getattr(self._reader, name)


def load_audio_segment(
    sample: typing.Any,
    timestamp: float,
    duration: float,
    sample_rate: int | None = None,
    audio_column: str = "audio",
) -> torch.Tensor:
    """Efficiently load an audio segment using the seek index.

    Opens the shard file directly via `shard.get_reader()` and creates a
    `LazyBuffer` view over just the audio blob region. Only the bytes needed
    for header probing and the requested segment are read — no full-blob
    download. Works for all shard sources (local, S3, Modal).

    Args:
        sample: A WSSample (or dict-like) containing at minimum:
            - "seek_index_audio_offset": int64 byte offset of the blob in the shard
            - "seek_index_audio_length": int64 byte length of the blob
            - "seek_index_positions": list[int64] absolute byte positions in the shard
            - "seek_index_pts_seconds": list[float64] timestamps in seconds
        timestamp: Start time in seconds.
        duration: Duration in seconds.
        sample_rate: Target sample rate for resampling. None keeps the native rate.
        audio_column: Name of the audio column (used to locate the shard).

    Returns:
        Torch tensor of shape (channels, samples) with decoded audio.
    """
    from .audio_codec import create_decoder

    audio_offset = sample["seek_index_audio_offset"]
    audio_length = sample["seek_index_audio_length"]
    seek_positions = sample["seek_index_positions"]      # absolute shard offsets
    seek_pts_seconds = sample["seek_index_pts_seconds"]

    # LazyBuffer view over just the audio blob (blob-relative offsets); reads are
    # sparse — only the header + the seeked segment are fetched.
    column_dir, _ = sample.dataset.fields[audio_column][0]
    shard = sample.dataset.get_shard(column_dir, sample.shard_ref)
    file_reader = shard.get_reader()

    # Serve cached header/footer locally (if present) so the decoder's open-time
    # reads fetch 0 blocks from the store — cold seek touches only target blocks.
    def _hf(name):
        try:
            v = sample[name]
        except (KeyError, TypeError):
            return b""
        return bytes(v) if v else b""
    header, footer = _hf("seek_index_header"), _hf("seek_index_footer")
    if header or footer:
        file_reader = _HFOverlayReader(file_reader, audio_offset, audio_length, header, footer)

    audio_view = LazyBuffer(file_reader, audio_offset, audio_length)
    # Small avio buffer: keeps find_stream_info's header read to ~8kB (a large
    # buffer inflates it) — read-ahead is redundant with the block-cache layer.
    audio_view._optimal_read_size = DECODE_BUFFER_BYTES

    # Per-codec decoder (mp4->moov/timestamp, vorbis->corrected granule seek,
    # wma->read-from-start, mp3->byte index). Seed the demuxer index so the
    # timestamp seek converges in ~1 read instead of scanning the whole file.
    dec = create_decoder(audio_view, sample_rate)
    rel_pos = [p - audio_offset for p in seek_positions]      # blob-relative
    dec.add_seek_points(rel_pos, seek_pts_seconds)
    # Also seed the byte-index (mp3/mp2/mp1) path so it doesn't rescan to build.
    if getattr(dec, "_use_byte_index", False):
        from types import SimpleNamespace
        dec._packet_index = [SimpleNamespace(pts_seconds=float(t), pos=int(p))
                             for p, t in zip(rel_pos, seek_pts_seconds)]

    return dec.get_samples_played_in_range(timestamp, timestamp + duration)
