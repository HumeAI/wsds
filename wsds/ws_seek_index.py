"""First-class audio seek index.

One episode's precomputed seek index, decoded from the `audio.wsds_seek_index` struct
column that the indexers write next to an audio column (a sibling column dir with the
same shard names and row order). Canonical schema:

    audio_offset     uint64        absolute byte offset of the audio blob in the shard file
    audio_length     uint64        blob length
    moov_offset      uint64        absolute offset of the mp4 `moov` box (0 if n/a)
    moov_size        uint32        its size (0 if n/a)
    seek_pts         list<uint32>  pts per seek point, 100us units, ascending
    seek_pos         list<uint64>  ABSOLUTE shard-file byte offset of each seek point's packet
    pts_offset       uint32        pts of the FIRST decoded frame at stream start, 100us units
                                   (matroska/webm undeclared vorbis priming; 0 elsewhere)
    flags            uint32        seek hazards + probe verdict, see FLAG_*
    edit_media_time  uint32        mp4 edit-list media_time (encoder-delay trim), media timescale
    header_bytes     uint32        what a decoder open touches from the blob start (0 = unknown)
    footer_bytes     uint32        tail a duration probe needs (ogg only; 0 = none)

Older indexes lack the last four fields and may carry `file_header`/`file_footer` byte
blobs instead of the extents; those are read with defaults and the blobs are ignored
(they were 91% of the index and nothing reads them: a stored header cannot make an mp4
open zero-fetch anyway, that needs the whole moov, which the index only points at).

`seek_pos` is absolute (not blob-relative) so `pos // 128KiB` is directly the block-cache
addressing unit; subtract `audio_offset` for blob-relative offsets (`pos_at`).

The arrays are kept as RAW zero-copy numpy views straight out of the arrow batch --
attaching an index allocates nothing regardless of its size. Unit conversion happens per
point (`pos_at`/`pts_at`), so consumers touching a small window around a seek target pay
O(window), never O(index).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PTS_UNIT = 1e-4   # seek_pts / pts_offset are stored as uint32 counts of 100us

# ---- seek hazards + verification (the ACCURACY GATE) -----------------------------------
# Structural hazards are read out of the mp4 `moov` at indexing time (free). The probe
# actually decodes at an indexed seek point and cross-correlates the audio against a
# decode-from-the-start of the same window; it runs on a deterministic sample of every
# container type. A row whose probe FAILED is content-misaligned by more than 10 ms when
# seeking, so consumers must not seek by it: `usable` is False and WSAudioEpisode then
# decodes such an episode from the start, exactly as if no index existed.
FLAG_EDIT_SHIFT = 1 << 0   # edit list media_time != 0 -> seek_pts shifted to presentation time
FLAG_IRREGULAR = 1 << 1    # non-uniform frame grid (>10 distinct sample durations)
FLAG_ROLL = 1 << 2         # roll distance < -1: a seek needs several pre-roll frames
FLAG_PROBE_DONE = 1 << 3   # this row was verified by a decode probe
FLAG_PROBE_FAILED = 1 << 4 # probe disagreed with the index -> consumers must decode from the start


@dataclass(frozen=True, slots=True)
class SeekIndex:
    seek_pos: np.ndarray             # uint64 view: absolute shard-file offsets
    seek_pts: np.ndarray             # uint32 view: 100us units, ascending
    audio_offset: int                # subtract from seek_pos for blob-relative
    pts_offset: float | None = None  # seconds; None = not measured (older indexes)
    moov_offset: int = 0             # ABSOLUTE shard offset of the mp4 `moov` box (0 = none / not mp4)
    moov_size: int = 0               # its size in bytes (0 = none); lets a prefetcher fetch the sample
                                     # table in the same parallel batch as the audio ranges
    audio_length: int = 0            # blob length (0 = unknown, older indexes)
    flags: int = 0                   # FLAG_*
    edit_media_time: int = 0
    header_bytes: int = 0            # extents: see module docstring
    footer_bytes: int = 0

    @classmethod
    def from_struct(cls, struct_scalar) -> "SeekIndex | None":
        """Decode the index column's arrow StructScalar (as returned by WSSample.get_raw)
        into zero-copy views. Returns None for empty rows (nothing indexed); raises on any
        other (= non-canonical) schema."""
        pos = struct_scalar["seek_pos"].values

        def _opt(name, default):
            try:
                v = struct_scalar[name].as_py()
                return default if v is None else v
            except KeyError:                   # older indexes predate the field
                return default

        moov_offset = int(_opt("moov_offset", 0))
        moov_size = int(_opt("moov_size", 0))
        if len(pos) == 0 and moov_size == 0:
            return None
        # mp4 rows may carry only the moov pointer (older extractors wrote empty seek lists for
        # mp4): still an index -- a prefetcher fetches the moov in parallel and plans from it.
        seek_pos = pos.to_numpy(zero_copy_only=False)
        seek_pts = struct_scalar["seek_pts"].values.to_numpy(zero_copy_only=False)
        po = _opt("pts_offset", None)
        return cls(
            seek_pos, seek_pts,
            int(struct_scalar["audio_offset"].as_py()),
            None if po is None else po * PTS_UNIT,
            moov_offset, moov_size,
            int(_opt("audio_length", 0)),
            int(_opt("flags", 0)),
            int(_opt("edit_media_time", 0)),
            int(_opt("header_bytes", 0)),
            int(_opt("footer_bytes", 0)),
        )

    # ---- the gate ----
    @property
    def probed(self) -> bool:
        return bool(self.flags & FLAG_PROBE_DONE)

    @property
    def usable(self) -> bool:
        """May a decoder seek by this index? False when the indexing-time probe found the
        index misaligned with the audio (FLAG_PROBE_FAILED): then read from the start."""
        return not (self.flags & FLAG_PROBE_FAILED)

    def __len__(self) -> int:
        return len(self.seek_pos)

    def search(self, t_seconds: float) -> int:
        """Index of the first seek point PAST t_seconds (binary search in raw units)."""
        return int(np.searchsorted(self.seek_pts, t_seconds / PTS_UNIT, side="right"))

    def pos_at(self, k: int) -> int:
        """Blob-relative byte offset of seek point k."""
        return int(self.seek_pos[k]) - self.audio_offset

    def pts_at(self, k: int) -> float:
        """Timestamp of seek point k, in seconds."""
        return float(self.seek_pts[k]) * PTS_UNIT

    def plan_ranges(self, t0: float, t1: float, margin_s: float = 2.0, probe_bytes: int = 1 << 19,
                    pad_bytes: int = 3 << 17) -> list[tuple[int, int]]:
        """Blob-relative byte ranges a decoder will touch to read [t0, t1]: the container
        header (plus the moov when it sits elsewhere) and the audio span bracketed by the
        seek points around the crop. Disjoint, sorted, merged when closer than 64 KiB --
        what a prefetcher should fetch in parallel before opening the decoder.

        `probe_bytes` widens the head range past `header_bytes` because ffmpeg's open
        probes a few frames past the container header; `pad_bytes` extends the audio span
        past the bracketing seek point for decoder read-ahead (measured on B2: 128 KB left
        21% of crops paying a second round trip, 384 KB removed all of them)."""
        length = int(self.audio_length)
        if length <= 0:
            raise ValueError("plan_ranges needs audio_length (index predates the field)")
        hdr = int(self.header_bytes) or min(65536, length)
        plan = [(0, min(hdr + probe_bytes, length))]
        if self.moov_size:
            mo = self.moov_offset - self.audio_offset
            if mo + self.moov_size > hdr:                     # non-faststart mp4: a separate range
                plan.append((mo, mo + self.moov_size))
        if len(self):
            i = max(0, self.search(max(0.0, t0 - margin_s)) - 1)
            j = min(len(self) - 1, int(np.searchsorted(self.seek_pts, (t1 + margin_s) / PTS_UNIT, side="left")))
            a = self.pos_at(i)
            b = self.pos_at(j) if j > i else length
            plan.append((max(0, a), min(length, b + pad_bytes)))
        else:
            plan.append((0, length))
        return _merge(plan)

    def __repr__(self) -> str:
        span = f"{self.pts_at(0):.1f}..{self.pts_at(len(self) - 1):.1f}s" if len(self) else "empty"
        return (f"SeekIndex({len(self)} points, {span}, pts_offset={self.pts_offset}, "
                f"flags={self.flags:#x}, usable={self.usable})")


def _merge(ranges, gap=65536):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


__all__ = ["SeekIndex", "PTS_UNIT", "FLAG_EDIT_SHIFT", "FLAG_IRREGULAR", "FLAG_ROLL",
           "FLAG_PROBE_DONE", "FLAG_PROBE_FAILED"]
