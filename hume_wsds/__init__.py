"""
# wsds dataset library

Usage example:
>>> from hume_wsds import WSDataset
>>> dataset = WSDataset("librilight/v3-vad_ws")
>>> for sample in dataset.random_samples(5):
>>>     print(sample['__key__'], sample['txt'])

"""
from hume_wsds.ws_dataset import WSDataset
from hume_wsds.ws_sample import WSSample
from hume_wsds.ws_shard import WSSourceAudioShard
from hume_wsds.ws_sink import AtomicFile, SampleFormatChanged, WSSink

__all__ = [
    WSDataset,
    WSSample,
    WSSourceAudioShard,
    AtomicFile,
    SampleFormatChanged,
    WSSink,
]
