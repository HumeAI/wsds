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
    stts_count       list<uint32>  mp4 sample-clock map, only when FLAG_STTS_MISMATCH: the audio
    stts_delta       list<uint32>    trak's stts run-length table...
    media_timescale  uint32          ...its timescale...
    frame_samples    uint32          ...the samples the decoder emits per packet (2048 with SBR)...
    first_frame_samples uint32       ...how many the FIRST output frame carries (a priming trim
                                      makes it short)...
    first_packet     uint32          ...and which packet that first output came from.
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
FLAG_STTS_MISMATCH = 1 << 5   # mp4: the stts durations are not what the decoder emits per packet, so
                              # the container clock drifts from the sample clock -- the row carries a
                              # SampleClock and reads must go through it (see SampleClock)
FLAG_SAMPLE_CLOCK_OK = 1 << 6 # ...and a packet-indexed crop reproduced the full decode (verified at
                              # build time, so a consumer does not have to trust the table blindly)


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
    sample_clock: "SampleClock | None" = None   # set iff FLAG_STTS_MISMATCH (see SampleClock)

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
            _sample_clock(struct_scalar, _opt),
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

    @property
    def pts_lies(self) -> bool:
        """This file's container timestamps do not match the audio (FLAG_STTS_MISMATCH): a pts
        seek lands wrong and `sample_clock` is the only correct way in."""
        return bool(self.flags & FLAG_STTS_MISMATCH)

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


def _sample_clock(struct_scalar, _opt):
    """A SampleClock from the row's stts map, or None when the row has no map (the common case)
    or predates the fields."""
    try:
        counts = struct_scalar["stts_count"].values
    except KeyError:
        return None
    if len(counts) == 0:
        return None
    return SampleClock(
        counts.to_numpy(zero_copy_only=False),
        struct_scalar["stts_delta"].values.to_numpy(zero_copy_only=False),
        int(_opt("media_timescale", 0)), int(_opt("edit_media_time", 0)),
        int(_opt("frame_samples", 0)), int(_opt("first_frame_samples", 0)),
        int(_opt("first_packet", 0)),
    )


class SampleClock:
    """An mp4 whose CONTAINER clock disagrees with its SAMPLE clock, and the map between them.

    ffmpeg stamps each packet's pts from the `stts` table; the AAC decoder emits a fixed frame
    per packet regardless (1024 samples, 2048 with SBR). Where the two disagree, every pts-based
    seek -- ffmpeg's native mp4 seek and a seek_pts index built from the same table alike --
    returns audio displaced from a full decode by the cumulative difference. Measured in the
    corpus: a mid-file packet of 361 ticks in a 1024 grid (-15 ms from there on), SBR behind a
    1024-ish table (-67 -> -113 ms, stepping at each bad packet), byte sizes written as durations
    (-4 to -21 ms, changing every second), an undeclared 2112-sample priming trim (a constant
    -47.9 ms). 0.6-0.8% of mp4 episodes on most datasets, 15.5% on a podcast-heavy one.

    So reads do not seek by time at all: they seek to a PACKET (by that packet's own presentation
    pts), read back which packet they landed on from the first chunk's pts, and count samples from
    that packet's first sample. Packet k:

        presentation pts = (cumsum(stts)[k] - edit_media_time) / media_timescale
        first sample     = 0 for k <= first_packet,
                           else first_frame_samples + (k - first_packet - 1) * frame_samples

    A 44.1 kHz file whose stts says 1024 per packet but whose decoder trims 2112 samples of
    priming (first output frame 960 samples, from packet 2): true time runs 2112 samples behind
    the container clock, so a pts seek to 10 s returns audio from 10.048 s.

    >>> c = SampleClock(counts=[5000], deltas=[1024], timescale=44100, media_time=0,
    ...                 frame=1024, first=960, p0=2)
    >>> c.true_samples(2), c.true_samples(3), c.true_samples(4)
    (0, 960, 1984)
    >>> k = c.packet_of_sample(441000)           # sample 441000 = 10.000 s on the sample clock
    >>> k, c.true_samples(k) <= 441000 < c.true_samples(k + 1)
    (432, True)
    >>> round(c.pres[k] * 44100) - c.true_samples(k)    # its container pts is 2112 samples late
    2112
    >>> round(c.seek_target(441000), 4)          # so we seek to the PACKET's own pts
    10.031
    >>> c.packet_of_pts(c.pres[432] + 1e-6)      # and read the landing packet back from chunk0.pts
    432

    An stts that lies mid-file: one packet of 361 ticks in a 1024 grid steps the container clock
    663 samples early for everything after it.

    >>> c = SampleClock(counts=[100, 1, 100], deltas=[1024, 361, 1024], timescale=44100,
    ...                 media_time=0, frame=1024, first=1024, p0=0)
    >>> round((c.true_samples(150) / 44100 - c.pres[150]) * 44100)
    663
    """

    __slots__ = ("pres", "n", "frame", "first", "p0")

    def __init__(self, counts, deltas, timescale, media_time, frame, first, p0):
        d = np.repeat(np.asarray(deltas, np.int64), np.asarray(counts, np.int64))
        self.pres = (np.concatenate([[0], np.cumsum(d)]) - int(media_time)) / float(timescale)
        self.n = len(d)
        self.frame = int(frame)
        self.first = int(first)
        self.p0 = int(p0)

    def true_samples(self, k: int) -> int:
        """First sample of packet k on the full-decode timeline."""
        return 0 if k <= self.p0 else self.first + (k - self.p0 - 1) * self.frame

    def packet_of_sample(self, s: int) -> int:
        return self.p0 if s < self.first else min(self.n - 1, self.p0 + 1 + (s - self.first) // self.frame)

    def packet_of_pts(self, pts: float) -> int:
        """Which packet a post-seek chunk came from, by its pts."""
        return max(0, int(np.searchsorted(self.pres, pts + 1e-9, side="right") - 1))

    def seek_target(self, s: int) -> float:
        """Presentation pts to hand the demuxer so a BACKWARD seek lands on the packet holding
        sample s."""
        return max(0.0, float(self.pres[self.packet_of_sample(s)]))

    def __repr__(self) -> str:
        return f"SampleClock({self.n} packets, frame={self.frame}, first={self.first}, p0={self.p0})"
