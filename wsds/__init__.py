"""

.. include:: ../README.md
.. include:: ../docs/dataset-structure.md

"""

from .ws_dataset import WSDataset
from .ws_meta import WSMetaDataset
from .ws_sample import WSSample
from .ws_shard import WSSourceAudioShard
from .ws_sink import AtomicFile, SampleFormatChanged, WSSink

__all__ = [
    WSDataset,
    WSMetaDataset,
    WSSample,
    WSSourceAudioShard,
    AtomicFile,
    SampleFormatChanged,
    WSSink,
]
