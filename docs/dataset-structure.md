## wsds Dataset Structure

This document explains the physical layout and logical data model of a WSDS dataset.

### Overview

A WSDS dataset is a **table** where:

- **Rows** are samples (split across **shards**)
- **Columns** are fields (split across **column directories**)

This design enables selective loading (only read the columns you need) and parallel processing (column directories are independent).

### Visual Layout

Based on the `data-pl` dataset:

```
data-pl/
│
├── indices/
│   └── source/                                 INDEX (spans all partitions)
│       └── index.sqlite3
│
├── pl_batch_0/
│   └── source/                                 PARTITION (pl_batch_0/source)
│       ├── audio_metadata/                     COLUMN DIRECTORY
│       │   ├── shard_..._pl_0_audio.wsds       ┐
│       │   ├── shard_..._pl_1000_audio.wsds    │ SHARDS
│       │   └── ...                             ┘
│       ├── audio_info/                         COLUMN DIRECTORY
│       │   ├── shard_..._pl_0_audio.wsds       ┐ same shard names,
│       │   └── ...                             ┘ different columns
│       ├── audio_filter_music/                 COLUMN DIRECTORY
│       │   └── ...
│       └── audio_lid1_filter/                  COLUMN DIRECTORY
│           └── ...
│
├── pl_batch_1/
│   └── source/                                 PARTITION (pl_batch_1/source)
│       └── ...                                 (same column dirs & schema)
│
└── pl_batch_2/
    └── source/                                 PARTITION (pl_batch_2/source)
        └── ...
```

### How It Fits Together

Think of the full dataset as a wide table reconstructed by **joining column directories horizontally** and **stacking shards vertically**:

```
                     ┌─────────────────────────────────────────────────────────────────────┐
                     │                     Logical Table (the dataset)                     │
                     ├──────────────┬──────────┬──────────┬────────────┬───────────────────┤
                     │  __key__     │  title   │ duration │  lid       │  vad.npy          │
                     │              │ (from    │ (from    │ (from      │ (from             │
                     │              │ audio_   │ audio_   │ audio_lid1_│ filtered_         │
                     │              │ info/)   │ info/)   │ filter/)   │ vad/)             │
  ┌────────────────┐ ├──────────────┼──────────┼──────────┼────────────┼───────────────────┤
  │ shard_...      │ │  00000208-.. │ "My V.." │  763.6   │  "pl"      │ [[0.1,3.2], ...]  │
  │ _pl_0_audio    │ │  0000039e-.. │ "Pod.."  │  545.9   │  "pl"      │ [[0.0,5.1], ...]  │
  │ (224 rows)     │ │  ...         │  ...     │  ...     │  ...       │  ...              │
  ├────────────────┤ ├──────────────┼──────────┼──────────┼────────────┼───────────────────┤
  │ shard_...      │ │  00000467-.. │ "How.."  │  425.8   │  "pl"      │ [[0.5,2.8], ...]  │
  │ _pl_1000_audio │ │  ...         │  ...     │  ...     │  ...       │  ...              │
  │ (240 rows)     │ │              │          │          │            │                   │
  └────────────────┘ └──────────────┴──────────┴──────────┴────────────┴───────────────────┘
        SHARDS                                        COLUMNS
     (vertical)                          (spread across COLUMN DIRECTORIES)
```

**Key invariant:** Shards with the same name across different column directories contain the same rows in the same order. `shard_..._pl_0_audio.wsds` in `audio_info/` and in `audio_metadata/` both hold the same 224 samples, so row 42 in each file refers to the same audio file.

### Components in Detail

#### Partitions

A partition is a relative path between the dataset root and the column directories. In `data-pl`, the partition strings would be something like `../../pl_batch_0/source`, `../../pl_batch_1/source`, etc. The index stores these relative paths per shard so a single index can span data spread across multiple directories. Shard paths are constructed as `dataset_root / partition / column_dir / shard_name.wsds`.

#### Column Directories

Each subdirectory within a partition groups related columns together. In the `data-pl` source dataset:

| Column directory | Example columns |
|---|---|
| `audio_metadata/` | `__key__`, `decoder_success`, `est_duration`, `load_err` |
| `audio_info/` | `title`, `duration`, `language`, `tags`, `upload_date` |
| `audio_filter_music/` | `music_pass`, `max_music_prob`, `speech_detection.npy` |
| `audio_lid1_filter/` | `lid`, `lid_probability`, `language_pass` |
| `filtered_vad/` | `vad.npy` |

Column directories are **independent** — you can:

- Load only the ones you need (skip `audio_filter_music/` when you only want transcripts)
- Store different column directories on different backends (local disk, S3, Modal volumes)
- Add new column directories without touching existing ones

#### Shards (`.wsds` files)

Each shard is an [Apache Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format) file containing a sequence of **RecordBatches**. Internally:

```
┌─────────────────────────────────────────┐
│   shard_64701231_pl_0_audio.wsds        │
├─────────────────────────────────────────┤
│  Schema: {__key__: utf8,                │
│           title: binary,                │
│           duration: float64, ...}       │
│  Metadata: {batch_size: "16"}           │
├─────────────────────────────────────────┤
│  RecordBatch 0  (rows 0-15)             │
│  RecordBatch 1  (rows 16-31)            │
│  ...                                    │
│  RecordBatch 13 (rows 208-223)          │
└─────────────────────────────────────────┘
```

- **Batch caching**: When a sample is read, the entire batch containing it is loaded and cached. Sequential reads within the same batch are instant.
- **Compression**: Batches are typically zstd-compressed on disk.
- **Column encoding**: Column names indicate how to decode the binary data. Suffixed names use file-extension conventions (`.txt` → UTF-8 string, `.npy` → NumPy array, `.pyd` → pickle). Audio columns should be called `audio`.

#### Columns

A column is a named field stored inside a shard. Every shard in a column directory has the same set of columns (same Arrow schema). Every shard always includes a special `__key__` column that uniquely identifies each sample and is used to verify consistency across column directories.

#### Index (`index.sqlite3`)

The index maps sample keys and global integer offsets to their physical location:

```
global_index  ──▶  (partition_path, shard_name, local_offset)
sample_key    ──▶  (partition_path, shard_name, local_offset)
```

It stores:

| Table | Purpose |
|---|---|
| `shards` | Shard name, sample count, cumulative global offset, and partition path |
| `files` | Source file name → shard + offset (with audio/speech durations) |
| `metadata` | JSON blob with field mappings, computed columns, durations |

The index can live outside the partitions it references (as in `data-pl/indices/`) and use relative paths to span all of them. In `data-pl`, the source index references shards across all `pl_batch_*/source` partitions.

Without an index, the dataset can still be iterated sequentially but cannot be randomly accessed by key.

#### Computed Columns (Links)

A computed column defines a **virtual column directory** — columns that are derived on-the-fly from another dataset rather than stored locally. In `data-pl`, the segmented `filtered_vad` dataset has a computed column that links back to the `source` dataset to extract audio segments:

```json
{
    "dataset_dir": "../source",
    "loader": ["wsds.ws_shard", "WSSourceAudioShard"],
    "vad_column": "vad.npy"
}
```

When you access `sample["audio"]` on a segment, WSDS looks up the source audio file and extracts just the time range for that segment using the VAD timestamps. No audio is duplicated on disk.

Links can point to local directories, S3 buckets, or Modal volumes.

### Sample Access Flow

When you access a field on a sample, here's what happens:

```
sample["lid"]
       │
       ▼
  ┌──────────────────────────────────┐
  │  Look up field location:         │
  │  "lid" is in column directory    │
  │  "audio_lid1_filter/",           │
  │  column "lid"                    │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  Open shard file:                │
  │  audio_lid1_filter/              │
  │    shard_64701231_pl_0_audio.wsds│
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  Load batch containing           │
  │  this sample's offset            │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │  Decode and return value:        │
  │  Return "pl"                     │
  └──────────────────────────────────┘
```

Fields are **lazily loaded** — accessing `sample["lid"]` never touches the `audio_info/` or `audio_metadata/` column directories. This makes it efficient to work with large multimodal datasets where you may only need a subset of columns at any given time. The library is also optimized for sequential access to samples and caches all expensive operations which makes it great for dataloaders.
