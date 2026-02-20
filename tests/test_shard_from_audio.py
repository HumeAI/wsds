import struct

import pyarrow as pa

from wsds.ws_tools import shard_from_audio_dir


def make_wav(path, num_samples=100, sample_rate=16000, num_channels=1):
    """Write a minimal valid WAV file."""
    bits_per_sample = 16
    data_size = num_samples * num_channels * (bits_per_sample // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        num_channels,
        sample_rate,
        sample_rate * num_channels * bits_per_sample // 8,
        num_channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        data_size,
    )
    pcm = b"\x00\x01" * num_samples * num_channels
    path.write_bytes(header + pcm)


def _collect_shards(output_dir):
    """Read all .wsds shards in output_dir, return list of (keys, audio_bytes, audio_types) per shard."""
    shards = []
    for shard_path in sorted(output_dir.glob("*.wsds")):
        reader = pa.ipc.open_file(str(shard_path))
        table = reader.read_all()
        keys = table.column("__key__").to_pylist()
        audio = [v.as_py() for v in table.column("audio")]
        audio_types = table.column("audio_type").to_pylist()
        shards.append((keys, audio, audio_types))
    return shards


class TestShardFromAudioDir:
    def test_basic_sharding(self, tmp_path):
        """Files are split into correct number of shards and content matches."""
        input_dir = tmp_path / "audio_in"
        output_dir = tmp_path / "audio_out"
        input_dir.mkdir()

        stems = [f"clip_{i:03d}" for i in range(5)]
        original_bytes = {}
        for stem in stems:
            p = input_dir / f"{stem}.wav"
            make_wav(p, num_samples=50 + len(stem))
            original_bytes[stem] = p.read_bytes()

        shard_from_audio_dir(str(input_dir), str(output_dir), max_files_per_shard=2)

        shards = _collect_shards(output_dir)
        # 5 files / 2 per shard = 3 shards
        assert len(shards) == 3

        all_keys = []
        all_audio = {}
        all_types = []
        for keys, audio, audio_types in shards:
            all_keys.extend(keys)
            for k, a in zip(keys, audio):
                all_audio[k] = a
            all_types.extend(audio_types)

        assert sorted(all_keys) == sorted(stems)
        for stem in stems:
            assert all_audio[stem] == original_bytes[stem]
        assert all(t == "wav" for t in all_types)

    def test_key_prefix(self, tmp_path):
        """key_prefix is prepended to each key."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        make_wav(input_dir / "hello.wav")

        shard_from_audio_dir(str(input_dir), str(output_dir), key_prefix="dataset1")

        shards = _collect_shards(output_dir)
        keys = shards[0][0]
        assert keys == ["dataset1/hello"]

    def test_key_fn(self, tmp_path):
        """key_fn transforms the key."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        make_wav(input_dir / "original.wav")

        shard_from_audio_dir(
            str(input_dir), str(output_dir), key_fn=lambda s: s.upper()
        )

        shards = _collect_shards(output_dir)
        keys = shards[0][0]
        assert keys == ["ORIGINAL"]

    def test_key_fn_with_prefix(self, tmp_path):
        """key_fn receives the prefixed stem."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        make_wav(input_dir / "file.wav")

        shard_from_audio_dir(
            str(input_dir),
            str(output_dir),
            key_prefix="pfx",
            key_fn=lambda s: s.replace("/", "__"),
        )

        shards = _collect_shards(output_dir)
        keys = shards[0][0]
        assert keys == ["pfx__file"]

    def test_oversized_files_skipped(self, tmp_path, monkeypatch):
        """Files exceeding the Arrow byte limit are skipped."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        make_wav(input_dir / "small.wav", num_samples=10)
        make_wav(input_dir / "big.wav", num_samples=100)

        small_size = (input_dir / "small.wav").stat().st_size
        large_size = (input_dir / "big.wav").stat().st_size

        # Pick a fake limit between the two file sizes so big.wav gets skipped
        fake_limit = (small_size + large_size) // 2

        # Patch the read_bytes to attach a fake size, then patch len check via
        # a wrapper around shard_from_audio_dir that lowers MAX_ARROW_BYTES.
        # Since MAX_ARROW_BYTES is a local, we instead wrap the whole function
        # by replacing it with one that sets a lower limit.
        import wsds.ws_tools as mod

        orig_code = mod.shard_from_audio_dir.__code__

        # Replace the constant in the code object's co_consts
        new_consts = tuple(
            fake_limit if c == 2_140_000_000 else c for c in orig_code.co_consts
        )
        new_code = orig_code.replace(co_consts=new_consts)
        monkeypatch.setattr(mod.shard_from_audio_dir, "__code__", new_code)

        shard_from_audio_dir(str(input_dir), str(output_dir))

        shards = _collect_shards(output_dir)
        all_keys = [k for keys, _, _ in shards for k in keys]
        assert "small" in all_keys
        assert "big" not in all_keys

    def test_empty_input_dir(self, tmp_path):
        """Empty input directory produces no shards."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        shard_from_audio_dir(str(input_dir), str(output_dir))

        assert list(output_dir.glob("*.wsds")) == []

    def test_subdirectory_files(self, tmp_path):
        """Audio files in subdirectories use relative path as key."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        sub = input_dir / "speaker1"
        sub.mkdir(parents=True)

        make_wav(sub / "utt.wav")

        shard_from_audio_dir(str(input_dir), str(output_dir))

        shards = _collect_shards(output_dir)
        keys = shards[0][0]
        assert keys == ["speaker1/utt"]

    def test_shard_naming(self, tmp_path):
        """Shard files are named audio-NNNNN.wsds."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        for i in range(4):
            make_wav(input_dir / f"f{i}.wav")

        shard_from_audio_dir(str(input_dir), str(output_dir), max_files_per_shard=2)

        shard_names = sorted(p.name for p in output_dir.glob("*.wsds"))
        assert shard_names == ["audio-00000.wsds", "audio-00001.wsds"]

    def test_init_index_creates_audio_subdir(self, tmp_path):
        """When init_index=True, shards are written to audio/ subdirectory and index is created."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "dataset"
        input_dir.mkdir()

        # Create some test audio files
        for i in range(3):
            make_wav(input_dir / f"file{i}.wav")

        shard_from_audio_dir(
            str(input_dir),
            str(output_dir),
            max_files_per_shard=2,
            init_index=True,
            require_audio_duration=False,  # Skip audio duration requirement for test
        )

        # Shards should be in output_dir/audio/
        audio_dir = output_dir / "audio"
        assert audio_dir.exists()
        assert audio_dir.is_dir()

        # Check shards are in the audio subdirectory
        shard_files = sorted(audio_dir.glob("*.wsds"))
        assert len(shard_files) == 2  # 3 files / 2 per shard = 2 shards

        # Check index was created at dataset root
        index_file = output_dir / "index.sqlite3"
        assert index_file.exists()

    def test_init_index_with_audio_named_output(self, tmp_path):
        """When init_index=True and output_dir is already named 'audio', don't create nested audio/audio/."""
        input_dir = tmp_path / "in"
        dataset_root = tmp_path / "dataset"
        output_dir = dataset_root / "audio"
        input_dir.mkdir()

        make_wav(input_dir / "test.wav")

        shard_from_audio_dir(
            str(input_dir),
            str(output_dir),
            init_index=True,
            require_audio_duration=False,
        )

        # Shards should be in output_dir (which is already named 'audio')
        shard_files = sorted(output_dir.glob("*.wsds"))
        assert len(shard_files) == 1

        # Check we didn't create audio/audio/
        nested_audio = output_dir / "audio"
        assert not nested_audio.exists()

        # Index should be at dataset_root (parent of audio/)
        index_file = dataset_root / "index.sqlite3"
        assert index_file.exists()

