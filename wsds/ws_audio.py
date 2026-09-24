from __future__ import annotations

import typing
from dataclasses import dataclass

from .audio_codec import audio_to_html, create_decoder, encode_audio
from .pupyarrow import pupyarrow



@dataclass()
class WSAudioEpisode:
    """A lazy seeking-capable audio reader for random-access to recordings stored in wsds shards.

    >>> from wsds import WSDataset
    >>> ds = WSDataset("librilight/source")
    >>> audio = ds[0].get_audio()
    >>> audio.load().shape
    torch.Size([1, 17884909])
    >>> audio.read_segment(start=2, end=5).shape
    torch.Size([1, 48000])
    >>> audio.read_segment(start=2, end=5, sample_rate=8000).shape
    torch.Size([1, 24000])
    """

    src: typing.Any
    _decoder: typing.Any = None
    _sample_rate: int | None = None
    _seek_index: typing.Any = None      # ws_seek_index.SeekIndex or None

    def __repr__(self):
        return f"WSAudioEpisode(src={type(self.src)}, sample_rate={self._sample_rate})"

    def unwrap(self):
        """Return the raw audio bytes"""
        if hasattr(self.src, "as_buffer"):
            return self.src.as_buffer().to_pybytes()
        elif isinstance(self.src, (bytes, bytearray)):
            return self.src
        elif isinstance(self.src, pupyarrow.LazyBuffer):
            return self.src.read()
        else:
            raise TypeError(f"Unsupported src type: {type(self.src)}")

    to_bytes = unwrap

    def set_seek_index(self, index):
        """Attach a precomputed SeekIndex (see wsds.ws_seek_index) so seeks skip the
        expensive on-open work: `build_packet_index` for mp3/mp2/mp1 (which reads the WHOLE
        episode -- the dominant deep-seek cost on a long podcast), the Ogg/Vorbis demuxer
        bisection, and matroska-vorbis's read-from-start fallback (via the index's measured
        pts_offset).

        Usually there is no need to call this: WSSample.get_audio auto-attaches the dataset's
        `<audio-col>.wsds_seek_index` column when present. Applied on the next decoder
        (re)creation. An index whose indexing-time probe FAILED (SeekIndex.usable is False)
        is kept for range planning but never fed to the decoder: the episode then reads from
        the start, exactly as with no index."""
        self._seek_index = index
        if self._decoder is not None:
            self._apply_seek_index()

    def _apply_seek_index(self):
        index, d = self._seek_index, self._decoder
        if index is None or d is None or len(index) == 0 or not index.usable:
            return
        # Formats without a native seek table (ogg/vorbis, mp3/mp2/mp1) -> seed the demuxer's
        # AVIndexEntry list (humecodec>=0.8) so timestamp seeks bracket the target via a
        # known-correct entry instead of the demuxer's interpolating bisection (ogg) / CBR
        # byte<->time estimate (mp3, badly wrong on VBR). For mp3 this also skips the on-open
        # build_packet_index scan that reads the whole blob. Seeding is INCREMENTAL:
        # set_seed_index just stores the index, and a small window of points near each
        # requested seek target is added on demand (AudioDecoder._seed_around), so per-seek
        # cost is O(window) regardless of episode length.
        #
        # mp4/mov (aac/alac) are SKIPPED: the moov already carries a full sample table, so
        # seeding is redundant AND each av_add_index_entry is an O(n) sorted insert against
        # its millions of native entries (a 36 h aac took 41 s for 16k points). Native mp4
        # seeking is already fast; the index still serves range planning for mp4.
        # Matroska/webm carry cues: seeding them lands seeks 130-880 ms LATE (measured), so
        # they are never seeded either; a measured pts_offset is all vorbis-in-matroska needs.
        codec = getattr(getattr(d, "metadata", None), "codec", "") or ""
        if getattr(d, "_in_matroska", False):
            # matroska/webm, ANY codec: never seed (cross-checked on webm/opus: a seeded seek
            # landed 880 ms late). The only thing the index contributes is the measured pts
            # offset for undeclared vorbis priming.
            if getattr(d, "_matroska_vorbis", False) and index.pts_offset is not None and hasattr(d, "set_pts_offset"):
                d.set_pts_offset(index.pts_offset)
        elif codec in ("mp3", "mp2", "mp1"):
            if hasattr(d, "set_seed_index"):
                d.set_seed_index(index)
        elif (codec in ("vorbis", "opus") and hasattr(d, "set_seed_index")
              and not getattr(d, "_seek_unreliable", False)):
            d.set_seed_index(index)                       # ogg: no container index, seeding replaces bisection

    def get_decoder(self, sample_rate=None):
        """Lazily creates/caches decoder via audio_codec.create_decoder()."""
        requested_sr = sample_rate or (self._decoder and self._decoder.metadata.sample_rate)
        if self._decoder is None or requested_sr != self._sample_rate:
            self.src.seek(0)
            self._decoder = create_decoder(self.src, sample_rate=sample_rate)
            self._sample_rate = sample_rate or self._decoder.metadata.sample_rate
            self._apply_seek_index()
        return self._decoder, self._sample_rate

    @property
    def metadata(self):
        decoder, sample_rate = self.get_decoder()
        return decoder.metadata

    @property
    def sample_rate(self):
        _, sr = self.get_decoder()
        return sr

    def read_segment(self, start=0, end=None, sample_rate=None):
        decoder, sample_rate = self.get_decoder(sample_rate)
        samples = decoder.get_samples_played_in_range(start, end)
        if hasattr(samples, "data"):
            samples = samples.data
        samples.sample_rate = sample_rate
        return samples

    def load(self, sample_rate=None):
        samples = self.read_segment(sample_rate=sample_rate)
        return samples

    def _repr_html_(self):
        return audio_to_html(self.read_segment())

    def _display_(self):
        import marimo

        return marimo.audio(encode_audio(self.read_segment()))


@dataclass(frozen=True)
class WSAudioSegment:
    """A lazy reference to a single sample from a segmented audio file.
    """

    episode: WSAudioEpisode
    tstart: float
    tend: float

    def __repr__(self) -> str:
        return f"WSAudioSegment(episode={self.episode}, tstart={self.tstart!s}, tend={self.tend!s})"

    @property
    def duration(self) -> float:
        """Duration of the audio segment in seconds."""
        return self.tend - self.tstart

    def with_context(self, before: float = 0, after: float = 0) -> "WSAudioSegment":
        """Return a new WSAudioSegment with expanded timestamps to include surrounding context.

        Args:
            before: Seconds of context to add before the segment start (will not go below 0)
            after: Seconds of context to add after the segment end

        Returns:
            A new WSAudioSegment instance with adjusted timestamps
        """
        return WSAudioSegment(
            episode=self.episode,
            tstart=max(0, self.tstart - before),
            tend=self.tend + after,
        )

    def with_timestamps(self, tstart: float | None = None, tend: float | None = None) -> "WSAudioSegment":
        """Return a new WSAudioSegment with modified timestamps.

        Args:
            tstart: New start time in seconds (None to keep current)
            tend: New end time in seconds (None to keep current)

        Returns:
            A new WSAudioSegment instance with the specified timestamps
        """
        return WSAudioSegment(
            episode=self.episode,
            tstart=tstart if tstart is not None else self.tstart,
            tend=tend if tend is not None else self.tend,
        )

    def load(self, sample_rate=None, pad_to_seconds=None):
        samples = self.episode.read_segment(self.tstart, self.tend, sample_rate)
        sample_rate = samples.sample_rate
        if pad_to_seconds is not None:
            import torch

            padding = int(pad_to_seconds * sample_rate - samples.shape[-1])
            samples = torch.nn.functional.pad(samples, (0, padding))
            samples.sample_rate = sample_rate
        return samples

    @property
    def metadata(self):
        return self.episode.metadata

    def _repr_html_(self):
        return audio_to_html(self.load())

    def _display_(self):
        import marimo

        return marimo.audio(encode_audio(self.load()))
