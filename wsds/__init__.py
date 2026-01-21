"""
# wsds dataset library

Usage example:
>>> from wsds import WSDataset
>>> dataset = WSDataset("librilight/v3-vad_ws")
>>> for sample in dataset.random_samples(5):
>>>     print(sample['__key__'], sample['txt'])

"""

from .ws_dataset import WSDataset
from .ws_sample import WSSample
from .ws_shard import WSSourceAudioShard
from .ws_audio import extract_segment_ffmpeg
from .ws_sink import AtomicFile, SampleFormatChanged, WSSink

__all__ = [
    WSDataset,
    WSSample,
    WSSourceAudioShard,
    extract_segment_ffmpeg,
    AtomicFile,
    SampleFormatChanged,
    WSSink,
]
