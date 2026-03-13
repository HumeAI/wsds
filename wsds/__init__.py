"""

.. include:: ../README.md
.. include:: ../docs/dataset-structure.md

"""

from .ws_dataset import WSDataset
from .ws_sample import WSSample
from .ws_shard import WSSourceAudioShard
from .ws_sink import AtomicFile, SampleFormatChanged, WSSink

__all__ = [
    WSDataset,
    WSSample,
    WSSourceAudioShard,
    AtomicFile,
    SampleFormatChanged,
    WSSink,
]
