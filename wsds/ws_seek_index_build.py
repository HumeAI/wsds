"""Build `audio.wsds_seek_index` rows and shards (see ws_seek_index for the schema).

Per audio blob, WITHOUT downloading the shard:
  * mp4/m4a/mov: no packet scan. The `moov` sample tables ARE the seek table: a few ranged
    reads (box headers + the moov) give `seek_pts`/`seek_pos` at SLOT resolution plus the moov
    location and the structural hazards (edit list, roll distance, irregular frame grid).
  * everything else (mp3, ogg, webm/mka, flac, wav): the blob is read once and humecodec's
    `build_packet_index` scanned for (pos, pts) points; the matroska pts offset and the ogg
    footer extent are measured on the same bytes.
A deterministic 1-in-`probe_share` sample of rows of EVERY container type is then VERIFIED by
a decode probe (`probe_index`): the same audio window reached by a direct seek and by
sequential decoding is cross-correlated; a disagreement over PROBE_TOL_S sets
FLAG_PROBE_FAILED -- the accuracy gate consumers honour (SeekIndex.usable).

`index_shard(reader, out_path)` works over any pupyarrow FileReader (a local .wsds, a block
cache mirror, an S3 object) and writes the index shard with the same keys and row order as
the audio shard, so it can sit next to it as a sibling column dir. Rows that cannot be
indexed (null blob, tiny blob, bytes not present in a mirror, decode failure) get EMPTY_INDEX.

CLI: `wsds build-seek-index <dataset> [--audio_col audio] [--num_workers N] [--probe_share 10]`
"""

from __future__ import annotations

import io
import os
import struct
import zlib

import numpy as np

from .ws_seek_index import FLAG_EDIT_SHIFT, FLAG_IRREGULAR, FLAG_PROBE_DONE, FLAG_PROBE_FAILED, FLAG_ROLL

SLOT = 128 * 1024          # seek-point spacing = the block cache's addressing unit
HEADER_BYTES = 8192        # header-extent fallback when the open cannot be traced
MP4_SNIFF_BYTES = 64
PROBE_SHARE = 10           # probe 1 row in PROBE_SHARE, chosen by key hash
PROBE_TOL_S = 0.010        # content misalignment we refuse: 10 ms, far below a syllable
SCAN_TYPES = ("mp3", "ogg", "oga", "opus", "webm", "mkv", "mka", "flac", "wav")
MOOV_TYPES = ("m4a", "mp4", "m4b", "mov", "aac")
COLUMN = "audio.wsds_seek_index"
BATCH_SIZE = 64

EMPTY_INDEX = {
    "audio_offset": 0, "audio_length": 0, "moov_offset": 0, "moov_size": 0,
    "seek_pts": [], "seek_pos": [], "pts_offset": 0,
    "flags": 0, "edit_media_time": 0, "header_bytes": 0, "footer_bytes": 0,
}


def index_type():
    import pyarrow as pa

    return pa.struct([
        ("audio_offset", pa.uint64()),
        ("audio_length", pa.uint64()),
        ("moov_offset", pa.uint64()),
        ("moov_size", pa.uint32()),
        ("seek_pts", pa.list_(pa.uint32())),
        ("seek_pos", pa.list_(pa.uint64())),
        ("pts_offset", pa.uint32()),
        ("flags", pa.uint32()),
        ("edit_media_time", pa.uint32()),
        ("header_bytes", pa.uint32()),
        ("footer_bytes", pa.uint32()),
    ])


def index_schema(column=COLUMN):
    import pyarrow as pa

    return pa.schema([("__key__", pa.string()), (column, index_type())], metadata={"batch_size": str(BATCH_SIZE)})


def probe_due(key, share=PROBE_SHARE):
    """Deterministic 1-in-`share` sample, stable across reruns and workers."""
    return share > 0 and zlib.crc32(key.encode("utf-8")) % share == 0


# ---- mp4 structure -------------------------------------------------------------------
def _boxes(buf, start, end):
    p = start
    while p + 8 <= end:
        size = int.from_bytes(buf[p:p + 4], "big")
        typ = buf[p + 4:p + 8]
        hl = 8
        if size == 1:
            size = int.from_bytes(buf[p + 8:p + 16], "big")
            hl = 16
        elif size == 0:
            size = end - p
        if size < hl:
            return
        yield typ, hl, p + hl, min(end, p + size)
        p += size


def find_moov(buf):
    """(offset, size) of the top-level `moov` box in `buf`, or (0, 0)."""
    i, n = 0, len(buf)
    while i + 8 <= n:
        size = int.from_bytes(buf[i:i + 4], "big")
        typ = buf[i + 4:i + 8]
        if size == 1:
            if i + 16 > n:
                break
            size = int.from_bytes(buf[i + 8:i + 16], "big")
        if typ == b"moov":
            return i, size
        if size < 8:
            break
        i += size
    return 0, 0


def mp4_media_timescale(moov):
    """Media timescale of the first audio track (mdhd), or 0."""
    for t, hl, a, b in _boxes(moov, 0, len(moov)):
        if t != b"trak":
            continue
        ts = None
        handler = None
        for t2, _, a2, b2 in _boxes(moov, a, b):
            if t2 != b"mdia":
                continue
            for t3, _, a3, b3 in _boxes(moov, a2, b2):
                if t3 == b"mdhd":
                    v = moov[a3]
                    ts = struct.unpack(">I", moov[a3 + (20 if v == 1 else 12):a3 + (24 if v == 1 else 16)])[0]
                elif t3 == b"hdlr":
                    handler = moov[a3 + 8:a3 + 12]
        if handler == b"soun" and ts:
            return int(ts)
    return 0


def mp4_edit_media_time(moov):
    """media_time of the first edit-list entry (media timescale units), 0 if none/empty.
    A non-zero value is the encoder-delay trim: presentation pts = media pts - media_time/timescale."""
    i = moov.find(b"elst")
    if i < 0:
        return 0
    try:
        ver = moov[i + 4]
        cnt = struct.unpack(">I", moov[i + 8:i + 12])[0]
        if cnt < 1:
            return 0
        mt = struct.unpack(">Qq", moov[i + 12:i + 28])[1] if ver == 1 else struct.unpack(">Ii", moov[i + 12:i + 20])[1]
        return max(0, int(mt))
    except Exception:
        return 0


def mp4_roll_distance(moov):
    """`roll` sample-group distance (-1 for ordinary AAC), 0 if absent."""
    j = moov.find(b"sgpd")
    if j < 0 or moov[j + 8:j + 12] != b"roll":
        return 0
    try:
        return int(struct.unpack(">h", moov[j + 20:j + 22])[0])
    except Exception:
        return 0


def grid_flags(times):
    """FLAG_IRREGULAR when the frame grid is not uniform (edited/concatenated files)."""
    if times is None or len(times) < 3:
        return 0
    d = np.diff(times)
    return FLAG_IRREGULAR if len(np.unique(np.round(d, 6))) > 10 else 0


def mp4_sample_table(moov):
    """First audio track of an mp4 `moov` payload -> (times_s, offsets, sizes) numpy arrays, or None."""
    for t, hl, a, b in _boxes(moov, 0, len(moov)):
        if t != b"trak":
            continue
        timescale = None
        handler = None
        stbl = None
        for t2, _, a2, b2 in _boxes(moov, a, b):
            if t2 != b"mdia":
                continue
            for t3, _, a3, b3 in _boxes(moov, a2, b2):
                if t3 == b"mdhd":
                    v = moov[a3]
                    timescale = struct.unpack(">I", moov[a3 + (20 if v == 1 else 12):a3 + (24 if v == 1 else 16)])[0]
                elif t3 == b"hdlr":
                    handler = moov[a3 + 8:a3 + 12]
                elif t3 == b"minf":
                    for t4, _, a4, b4 in _boxes(moov, a3, b3):
                        if t4 == b"stbl":
                            stbl = (a4, b4)
        if handler != b"soun" or stbl is None or not timescale:
            continue
        stts = stsc = stsz = stco = None
        co64 = False
        for t5, _, a5, b5 in _boxes(moov, stbl[0], stbl[1]):
            if t5 == b"stts":
                stts = (a5, b5)
            elif t5 == b"stsc":
                stsc = (a5, b5)
            elif t5 == b"stsz":
                stsz = (a5, b5)
            elif t5 in (b"stco", b"co64"):
                stco = (a5, b5)
                co64 = t5 == b"co64"
        if None in (stts, stsc, stsz, stco):
            return None

        def u32(a, n):
            return np.frombuffer(moov, dtype=">u4", count=n, offset=a).astype(np.int64)

        n_stts = struct.unpack(">I", moov[stts[0] + 4:stts[0] + 8])[0]
        e = u32(stts[0] + 8, 2 * n_stts).reshape(-1, 2)
        deltas = np.repeat(e[:, 1], e[:, 0])
        n = int(deltas.size)
        times = np.concatenate([[0], np.cumsum(deltas)[:-1]]) / float(timescale)
        fixed = struct.unpack(">I", moov[stsz[0] + 4:stsz[0] + 8])[0]
        n_sz = struct.unpack(">I", moov[stsz[0] + 8:stsz[0] + 12])[0]
        sizes = np.full(n_sz, fixed, dtype=np.int64) if fixed else u32(stsz[0] + 12, n_sz)
        n_co = struct.unpack(">I", moov[stco[0] + 4:stco[0] + 8])[0]
        if co64:
            chunks = np.frombuffer(moov, dtype=">u8", count=n_co, offset=stco[0] + 8).astype(np.int64)
        else:
            chunks = u32(stco[0] + 8, n_co)
        n_sc = struct.unpack(">I", moov[stsc[0] + 4:stsc[0] + 8])[0]
        sc = u32(stsc[0] + 8, 3 * n_sc).reshape(-1, 3)
        per_chunk = np.empty(n_co, dtype=np.int64)
        for i in range(n_sc):
            c0 = sc[i, 0] - 1
            c1 = (sc[i + 1, 0] - 1) if i + 1 < n_sc else n_co
            per_chunk[c0:c1] = sc[i, 1]
        m = min(n, int(per_chunk.sum()), sizes.size)
        chunk_of = np.repeat(np.arange(n_co), per_chunk)[:m]
        first_in_chunk = np.concatenate([[0], np.cumsum(per_chunk)[:-1]])
        csum = np.concatenate([[0], np.cumsum(sizes[:m])])
        offsets = chunks[chunk_of] + (csum[np.arange(m)] - csum[first_in_chunk[chunk_of]])
        return times[:m], offsets, sizes[:m]
    return None


def mp4_seek_entries(moov, audio_offset, resolution=SLOT):
    """(seek_pts 100us, seek_pos ABSOLUTE) at >= `resolution` byte spacing, first sample included."""
    tab = mp4_sample_table(moov)
    if tab is None:
        return None
    times, offsets, _ = tab
    pts, pos = [], []
    last = None
    for t, o in zip(times.tolist(), offsets.tolist()):
        if last is None or o - last >= resolution:
            pts.append(int(round(t * 10000)))
            pos.append(int(audio_offset + o))
            last = o
    return pts, pos


# ---- measurements on the raw bytes (scan formats) --------------------------------------
class _TracingSource:
    """A seekable file-like over bytes that records every read (to measure a decoder open)."""

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.n = len(data)
        self.reads = []

    def read(self, k=-1):
        if k is None or k < 0:
            k = self.n - self.pos
        k = min(k, self.n - self.pos)
        d = self.data[self.pos:self.pos + k]
        self.reads.append((self.pos, len(d)))
        self.pos += len(d)
        return d

    read1 = read

    def seek(self, o, w=0):
        self.pos = o if w == 0 else (self.pos + o if w == 1 else self.n + o)
        return self.pos

    def tell(self):
        return self.pos

    def size(self):
        return self.n

    def readable(self):
        return True

    def seekable(self):
        return True


def measure_header_len(audio_bytes, cap=131072):
    """Bytes a small-buffer decoder open actually touches in the blob's first half."""
    import humecodec

    src = _TracingSource(audio_bytes)
    try:
        r = humecodec.MediaDecoder(src=src, buffer_size=8192)
        info = r.get_src_stream_info(r.default_audio_stream)
        r.add_basic_audio_stream(frames_per_chunk=int(info.sample_rate), sample_rate=int(info.sample_rate))
    except Exception:
        return min(cap, len(audio_bytes), HEADER_BYTES)
    half = max(1, len(audio_bytes) // 2)
    return min(cap, max((a + n for a, n in src.reads if a < half), default=HEADER_BYTES))


def ogg_footer(buf, search_cap=65536):
    """The last complete Ogg page (what a duration probe reads). Empty for non-ogg."""
    if buf[:4] != b"OggS":
        return b""
    end = len(buf)
    lo = max(0, end - search_cap)
    pos = buf.rfind(b"OggS", lo)
    while pos != -1:
        if pos + 14 <= end and int.from_bytes(buf[pos + 6:pos + 14], "little", signed=True) >= 0:
            return bytes(buf[pos:])
        pos = buf.rfind(b"OggS", lo, pos)
    return b""


def measure_pts_offset(audio_bytes):
    """pts of the first decoded frame at stream start (matroska/webm only: undeclared vorbis
    priming shifts every container timestamp; WSAudioEpisode feeds this into the seek
    compensation)."""
    import humecodec

    if audio_bytes[:4] != b"\x1a\x45\xdf\xa3":
        return 0.0
    try:
        d = humecodec.MediaDecoder(io.BytesIO(audio_bytes), buffer_size=64 * 1024)
        info = d.get_src_stream_info(d.default_audio_stream)
        d.add_basic_audio_stream(frames_per_chunk=256, sample_rate=int(info.sample_rate))
        for _ in range(64):
            if d.fill_buffer() == 1:
                break
            (c,) = d.pop_chunks()
            if c is not None:
                return max(0.0, float(c.pts))
    except Exception:
        pass
    return 0.0


# ---- the decode probe (accuracy gate) -------------------------------------------------
def _decode_window(dec, start, dur, sr, max_chunks=6000):
    """Samples covering exactly [start, start+dur) as mono float32, decoding forward from the
    decoder's current position. Empty if the decoder is already past `start` or the stream
    ends first."""
    got = []
    first = None
    for _ in range(max_chunks):
        if dec.fill_buffer() == 1:
            break
        (c,) = dec.pop_chunks()
        if c is None:
            continue
        pts = float(c.pts)
        x = np.asarray(c.data if hasattr(c, "data") else c, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=0 if x.shape[0] <= 8 else 1)
        end = pts + len(x) / sr
        if end <= start:
            continue
        if first is None:
            if pts > start + 1e-6:          # already past the window: cannot align exactly
                return np.zeros(0, np.float32)
            first = pts
        got.append(x)
        if end >= start + dur:
            break
    if not got or first is None:
        return np.zeros(0, np.float32)
    y = np.concatenate(got)
    off = int(round((start - first) * sr))
    return y[off:off + int(dur * sr)]


def lag_seconds(ref, test, sr, max_lag_s=1.0):
    """Seconds by which `test` content is displaced from `ref` (both nominally the same window),
    sample-accurate via FFT cross-correlation. Positive = test holds audio from LATER in the
    file. NaN when the comparison is not meaningful (silence, too short, no confident peak)."""
    n = min(len(ref), len(test))
    if n < sr // 8:
        return float("nan")
    a = ref[:n].astype(np.float64)
    b = test[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return float("nan")
    m = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a, m) * np.conj(np.fft.rfft(b, m)), m)
    cc = np.concatenate([cc[-(n - 1):], cc[:n]])          # lags -(n-1)..n-1
    lim = int(max_lag_s * sr)
    mid = n - 1
    lo, hi = max(0, mid - lim), min(len(cc), mid + lim + 1)
    k = int(np.argmax(cc[lo:hi])) + lo
    if cc[k] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) < 0.3:
        return float("nan")
    return (k - mid) / sr


def probe_index(src, idx, tol=PROBE_TOL_S, sr=16000, win=0.5, pad=1.0):
    """Verify that seeking returns the RIGHT AUDIO, not merely a plausible timestamp.

    Both checks compare the SAME nominal window [t+pad, t+pad+win], reached two ways, so a
    frame-boundary landing cancels out and only real content misalignment shows up:
      A. decode from the start vs a direct seek to an early point t_ref
         (catches a whole-file timeline shift: priming, edit lists, wrong pts mapping);
      B. seek 5 s short of a late point and decode forward vs a direct seek to it
         (catches seeks that land wrong deep in the file).
    Returns (flags, worst_lag_seconds, target); an unmeasurable check is NaN, never a failure."""
    import humecodec

    pts = idx.get("seek_pts") or []
    if len(pts) < 2:
        return 0, float("nan"), float("nan")
    dur = float(pts[-1]) * 1e-4
    t_ref = min(20.0, max(1.0, 0.25 * dur))
    t_deep = float(pts[int(len(pts) * 0.75)]) * 1e-4
    lags = []
    try:
        src.seek(0)
        dec = humecodec.MediaDecoder(src=src, buffer_size=256 * 1024)
        dec.get_src_stream_info(dec.default_audio_stream)
        dec.add_basic_audio_stream(frames_per_chunk=sr, sample_rate=sr)
        ref = _decode_window(dec, t_ref + pad, win, sr)          # A: sequential ground truth
        dec.seek(t_ref)
        test = _decode_window(dec, t_ref + pad, win, sr)         # A: direct seek
        lags.append((lag_seconds(ref, test, sr), t_ref))
        if t_deep > t_ref + 10.0:
            dec.seek(max(0.0, t_deep - 5.0))
            ref2 = _decode_window(dec, t_deep + pad, win, sr)    # B: short back-seek + forward
            dec.seek(t_deep)
            test2 = _decode_window(dec, t_deep + pad, win, sr)   # B: direct deep seek
            lags.append((lag_seconds(ref2, test2, sr), t_deep))
    except Exception:
        return FLAG_PROBE_DONE, float("nan"), float("nan")
    usable = [(lag, t) for lag, t in lags if lag == lag]
    if not usable:
        return FLAG_PROBE_DONE, float("nan"), float("nan")
    worst, target = max(usable, key=lambda p: abs(p[0]))
    return FLAG_PROBE_DONE | (FLAG_PROBE_FAILED if abs(worst) > tol else 0), worst, target


# ---- builders ---------------------------------------------------------------------------
def build_index_for_mp4(lb, audio_offset, blob_len, resolution=SLOT):
    """mp4/m4a blob behind a LazyBuffer: box headers + the moov via ranged reads -> index row.
    None when the moov is not within the first 16 MB (the caller then scans the blob)."""
    lb.read_range(0, min(MP4_SNIFF_BYTES, blob_len))       # sniff only; the moov is pointed at, not stored
    pos = 0
    moov = None
    moov_rel = moov_size = 0
    while pos + 8 <= blob_len and pos < (16 << 20):
        h = lb.read_range(pos, min(pos + 16, blob_len))
        if len(h) < 8:
            break
        size = int.from_bytes(h[:4], "big")
        typ = h[4:8]
        hl = 8
        if size == 1:
            size = int.from_bytes(h[8:16], "big")
            hl = 16
        if typ == b"moov":
            moov_rel, moov_size = pos, size
            moov = lb.read_range(pos + hl, min(blob_len, pos + size))
            break
        if size < hl:
            break
        pos += size
    if moov is None:
        return None
    ent = mp4_seek_entries(moov, audio_offset, resolution)
    pts, spos = ent if ent else ([], [])
    # Structural hazards, straight out of the moov we just read (no extra bytes).
    mt = mp4_edit_media_time(moov)
    roll = mp4_roll_distance(moov)
    flags = 0
    tab = mp4_sample_table(moov)
    flags |= grid_flags(tab[0] if tab is not None else None)
    if roll < -1:
        flags |= FLAG_ROLL
    if mt:
        # Edit-list priming trim. Measured on 900 corpus episodes: the RAW sample-table pts
        # already agree with the decoder better than an edit-shifted version would (median
        # |error| 10 ms vs 23 ms), so the pts are left alone and this is a hazard marker only;
        # `edit_media_time` keeps the value for consumers that want it.
        flags |= FLAG_EDIT_SHIFT
    faststart = moov_rel + moov_size <= max(1 << 20, moov_size + (1 << 16))
    header_bytes = int(moov_rel + moov_size) if faststart else int(min(MP4_SNIFF_BYTES, blob_len))
    return {
        "audio_offset": int(audio_offset), "audio_length": int(blob_len),
        "header_bytes": header_bytes, "footer_bytes": 0,
        "moov_offset": int(audio_offset + moov_rel), "moov_size": int(moov_size),
        "seek_pts": pts, "seek_pos": spos, "pts_offset": 0,
        "flags": int(flags), "edit_media_time": int(mt),
    }


def build_index_for_blob(audio_bytes, audio_offset, resolution=SLOT):
    """Any container humecodec can demux: one sequential packet scan -> index row."""
    import humecodec

    dec = humecodec.MediaDecoder(io.BytesIO(audio_bytes), buffer_size=len(audio_bytes))
    dec.add_audio_stream(frames_per_chunk=-1)
    packet_index = dec.build_packet_index(resolution=resolution)

    seek_pts, seek_pos = [], []
    for e in packet_index:
        if e.pts_seconds is None or e.pts_seconds < 0:
            continue
        seek_pts.append(int(round(e.pts_seconds * 10000)))     # 100us units
        seek_pos.append(int(audio_offset + e.pos))             # absolute shard offset

    moov_rel = moov_size = 0
    if audio_bytes[4:8] == b"ftyp" or b"ftyp" in audio_bytes[:16]:
        moov_rel, moov_size = find_moov(audio_bytes[: min(len(audio_bytes), 1 << 20)])
        if moov_size == 0 and audio_bytes[4:8] == b"ftyp":
            moov_rel, moov_size = find_moov(audio_bytes)

    return {
        "audio_offset": int(audio_offset),
        "audio_length": len(audio_bytes),
        "header_bytes": int(measure_header_len(audio_bytes)),
        "footer_bytes": int(len(ogg_footer(audio_bytes))),
        "moov_offset": int(audio_offset + moov_rel) if moov_size else 0,
        "moov_size": int(moov_size),
        "seek_pts": seek_pts,
        "seek_pos": seek_pos,
        "pts_offset": int(round(measure_pts_offset(audio_bytes) * 10000)),   # 100us
        "flags": 0, "edit_media_time": 0,
    }


def index_row(lb, key, audio_type=None, probe_share=PROBE_SHARE, stats=None, min_blob_bytes=SLOT, resolution=SLOT):
    """Index one blob (a pupyarrow LazyBuffer over the audio shard) -> index row dict.
    mp4 by moov, everything else by scan; probed when `probe_due(key)`. Blobs under
    `min_blob_bytes` (default: one cache block) get EMPTY_INDEX: nothing to seek within.
    `resolution`: minimum byte spacing of seek points (default: one cache block)."""
    stats = stats if stats is not None else {}
    atype = (audio_type or "").lower()
    idx = None
    blob = None
    if lb.length < min_blob_bytes:
        stats["tiny"] = stats.get("tiny", 0) + 1
        return dict(EMPTY_INDEX)
    if atype in MOOV_TYPES or (atype not in SCAN_TYPES and lb.read_range(4, 8) == b"ftyp"):
        idx = build_index_for_mp4(lb, lb.offset, lb.length, resolution)
        if idx is not None:
            stats["mp4"] = stats.get("mp4", 0) + 1
    if idx is None:
        blob = lb.read_range(0, lb.length)
        idx = build_index_for_blob(blob, lb.offset, resolution)
        stats["scan"] = stats.get("scan", 0) + 1
    if probe_due(key, probe_share):
        src = io.BytesIO(blob) if blob is not None else lb
        pf, lag, _ = probe_index(src, idx)
        idx["flags"] |= pf
        stats["probed"] = stats.get("probed", 0) + 1
        if pf & FLAG_PROBE_FAILED:
            stats["probe_failed"] = stats.get("probe_failed", 0) + 1
        if lag == lag:
            stats["worst_lag"] = max(stats.get("worst_lag", 0.0), abs(lag))
    return idx


def index_shard(reader, out_path, audio_col="audio", type_col="audio_type", probe_share=PROBE_SHARE,
                column=COLUMN, log=None, min_blob_bytes=SLOT, resolution=SLOT):
    """Index every row of the audio shard behind `reader` (any pupyarrow FileReader, or a path)
    into `out_path` (an Arrow IPC shard, zstd, same keys and row order). Returns a stats dict."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from .pupyarrow.file_reader import FileReader, LocalFileReader
    from .pupyarrow.pupyarrow import FeatherFile

    if not isinstance(reader, FileReader):
        reader = LocalFileReader(reader)
    ff = FeatherFile(reader)
    st = {"rows": 0, "indexed": 0, "empty": 0, "failed": 0}
    keys, rows = [], []
    for bi in range(ff.num_record_batches):
        rb = ff.record_batch(bi)
        kcol = rb.column("__key__")
        acol = rb.column(audio_col)
        tcol = rb.column(type_col) if type_col in rb.schema.names else None
        compressed = bool(ff._parse_record_batch_info(bi).compression)
        for j in range(rb.num_rows):
            k = kcol[j]
            k = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            keys.append(k)
            st["rows"] += 1
            lb = acol[j]
            if lb is None or compressed:
                # a compressed IPC body has no stable byte offsets: nothing to seek by
                rows.append(dict(EMPTY_INDEX))
                st["empty"] += 1
                continue
            at = tcol[j] if tcol is not None else None
            at = at.decode() if isinstance(at, bytes) else (str(at) if at is not None else None)
            try:
                row = index_row(lb, k, at, probe_share, st, min_blob_bytes, resolution)
            except Exception as e:                 # bytes not present (mirror), decode failure
                if log:
                    log(f"  {k}: {type(e).__name__}: {str(e)[:100]}")
                rows.append(dict(EMPTY_INDEX))
                st["failed"] += 1
                continue
            rows.append(row)
            st["indexed"] += row["audio_length"] > 0
    schema = index_schema(column)
    tbl = pa.table({"__key__": pa.array(keys, pa.string()), column: pa.array(rows, type=index_type())}, schema=schema)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = f"{out_path}.{os.getpid()}.tmp"
    with ipc.new_file(tmp, schema, options=ipc.IpcWriteOptions(compression="zstd")) as w:
        for b in tbl.to_batches(max_chunksize=BATCH_SIZE):
            # concat_arrays: a sliced array keeps its parent's 64-byte-rounded buffer capacity,
            # which pyarrow then records as the uncompressed length in a compressed IPC file
            # and polars rejects. Copying the slice gives exactly-sized buffers.
            w.write_batch(pa.RecordBatch.from_arrays([pa.concat_arrays([c]) for c in b.columns], schema=b.schema))
    os.replace(tmp, out_path)
    st["out_bytes"] = os.path.getsize(out_path)
    return st


def open_audio_reader(dataset_root, partition, shard, audio_col="audio"):
    """The reader for a shard's audio column, wherever the bytes are: a full local .wsds, a block
    cache mirror (.sparse) next to it, or the linked (S3) shard's own reader."""
    from .pupyarrow.block_cache import EXT, MirrorFileReader
    from .pupyarrow.file_reader import LocalFileReader

    base = os.path.join(str(dataset_root), partition or "", audio_col, f"{shard}.wsds")
    if os.path.exists(base):
        return LocalFileReader(base)
    if os.path.exists(base + EXT):
        return MirrorFileReader(base + EXT)
    from . import WSDataset

    ds = WSDataset(dataset_root)
    link = f"{audio_col}.wsds-link"
    if link in ds.computed_columns:
        return ds.get_shard(link, (partition, shard))._reader
    raise FileNotFoundError(base)


def _index_one(args):
    root, partition, shard, audio_col, probe_share, force = args
    out = os.path.join(str(root), partition or "", f"{audio_col}.wsds_seek_index", f"{shard}.wsds")
    if os.path.exists(out) and not force:
        return {"shard": shard, "status": "exists"}
    try:
        reader = open_audio_reader(root, partition, shard, audio_col)
        st = index_shard(reader, out, audio_col=audio_col, probe_share=probe_share)
        return {"shard": shard, "status": "ok", **st}
    except Exception as e:
        return {"shard": shard, "status": f"ERR {type(e).__name__}: {str(e)[:120]}"}
