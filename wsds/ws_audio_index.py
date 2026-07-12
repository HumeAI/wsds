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
                }
            )

    feather.close()

    with WSSink(str(output_path), compression=compression) as sink:
        for row in rows:
            sink.write(row)


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
    audio_view = LazyBuffer(file_reader, audio_offset, audio_length)

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
