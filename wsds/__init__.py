"""
## wsds — Web-Scale DataSets

**wsds** is a multimodal dataset library that combines the power of SQL querying with native support for speech, audio, and video data. Built for large-scale machine learning workflows, it lets you work with massive datasets efficiently, regardless of where you store your data (SSDs, HDDs, Weka, S3).

>>> from wsds import WSDataset
>>> dataset = WSDataset("librilight/v3-vad_ws")
>>> print(str(dataset))
WSDataset('librilight/v3-vad_ws', segmented=True)
     Audio duration: 52.69 k hours
    Speech duration: 47.44 k hours
   Number of shards: 623
  Number of samples: 22 662 659
<BLANKLINE>

### Key Features

- **SQL Queries on Sharded Data** — Filter and select across your entire dataset using familiar SQL syntax, powered by Polars. Only the columns and shards you need are loaded.

>>> dataset.sql_select('`transcription_wslang_raw.txt`', 'snr', 'tend - tstart as duration')
INFO: to speed things up wsds is loading a random 24.08% subset of the shards, pass shard_subsample=1 to force it to load the whole dataset
shape: (5_271_939, 3)
┌─────────────────────────────────┬──────────┬───────────┐
│ transcription_wslang_raw.txt    ┆ snr      ┆ duration  │
│ ---                             ┆ ---      ┆ ---       │
│ str                             ┆ f16      ┆ f32       │
╞═════════════════════════════════╪══════════╪═══════════╡
│  This is a liberal box recordi… ┆ 70.0625  ┆ 1.331058  │
│  or liberty box recordings dur… ┆ 66.25    ┆ 1.962457  │
│  For more information or to vo… ┆ 65.6875  ┆ 3.276451  │
│  The Elder Eddas of Semen-Sekh… ┆ 51.09375 ┆ 4.863482  │
│  Translated by Erasmus B. Ande… ┆ 70.1875  ┆ 1.843002  │
│ …                               ┆ …        ┆ …         │
│  I stared about me.             ┆ 66.1875  ┆ 1.433472  │
│  and then pointing to the huge… ┆ 64.75    ┆ 3.703003  │
│  It was there. Where it is now… ┆ 73.75    ┆ 3.651855  │
│  He shrugged his shoulders, to… ┆ 65.0     ┆ 9.4198    │
│  the first chance, and he made… ┆ 62.0     ┆ 11.501709 │
└─────────────────────────────────┴──────────┴───────────┘

- **Random Access & Indexing** — Optional SQLite-based indexing enables fast random access by key or integer index across shards.

>>> x = dataset['large/1259/lettersofjaneausten_etk_librivox_64kb_mp3/lettersofjaneausten_22_austen_64kb_032']

- **Lazy, On-Demand Loading** — Samples are dict-like objects that load fields only when accessed, keeping memory usage minimal even for terabyte-scale datasets.

>>> x['transcription_wslang_raw.txt'], x['dbu']
(' The Sherers, I believe, are now really going to go. Joseph has had a bed here the last two nights, and I do not know whether this is not the day of moving. Mrs. Sherer called yesterday to take leave. The weather looks worse again.', -26.34375)

- **Native Audio & Multimodal Support** — First-class handling of speech and audio data, including segmented datasets with voice activity detection and computed columns that reference source audio.

>>> x['audio']
WSAudio(audio_reader=AudioReader(src=<class '_io.BytesIO'>, sample_rate=None), tstart=614.46246, tend=627.3976)

- **Sharded Architecture** — Data is stored in `.wsds` files (PyArrow IPC format) organized by column type into subdirectories, enabling efficient columnar access patterns.

- **Atomic Writes** — The `WSSink` context manager provides safe, batched, compressed writes with atomic commit semantics.

- **Flexible Data Linking** — Computed columns and `.wsds-link` files let you compose datasets without duplicating data, referencing columns across dataset boundaries.

### Quick Start

```bash
pip install git+https://github.com/HumeAI/wsds.git
```

```python
from wsds import WSDataset

ds = WSDataset("my_dataset")

# SQL filtering
subset = ds.filtered("duration > 5.0 AND language = 'en'")

# Lazy sample access
sample = ds[0]
audio = sample["audio"]   # loaded on demand
text = sample["text"]     # loaded on demand
```

```bash
# CLI tools
wsds inspect my_dataset
wsds validate shards my_dataset
```


# wsds dataset library

Usage example:
>>> from wsds import WSDataset
>>> dataset = WSDataset("librilight/v3-vad_ws", rng=42)
>>> for sample in dataset.random_samples(5): print(f"{sample['transcription_wslang_raw.txt'].strip()} [{sample['__key__']}]")
while trying to screw my courage up to the point of making a verbless explanation of my difficulty. Someone pushed through the crowd, and to my great relief began speaking to me. It was Manzoor the Mayor. As best I could, I explained that I had lost my way and had found it necessary to come down for the purpose of making inquiries. I knew that it was awful French, but hoped that it would be intelligible, in part at least. [medium/3020/high_adventure_mv_0810_librivox_64kb_mp3/highadventure_04_hall_64kb_021]
I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. I. A. I. I. [large/11466/lifeandlilliangish_1805_librivox_64kb_mp3/lifeandlilliangish_30_paine_64kb_027]
or [large/10244/historyteachernov1909_1807_librivox_64kb_mp3/historyteachernov1909_20_various_64kb_269]
The other reports, with which I will not trouble the reader, told the same story. Gardeners, carpenters, shoemakers, boatmen, all complained of the same grievances. [large/3549/sophisms_1005_librivox_64kb_mp3/sophismsofprotectionists_11_bastiat_64kb_145]
"'What do you mean?' asked Phoebe, her voice full of antagonism. "'Mean?' said Hiram, sidling after her. "'I mean it's time we set up a partnership. I've waited long enough. I need somebody to look after the children. You suit me pretty well, and I'd guess you'd be well enough fixed with me.' [large/3157/phoebedeane_1701_librivox_64kb_mp3/phoebedeane_01_hill_64kb_043]

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
