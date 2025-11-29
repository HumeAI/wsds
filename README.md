# WSDS

wsds merges SQL querying capabilities with native support for multimodal data (speech and video) in a single data
format and a unified API. It uses shards for efficiency and to support very-scalable parallel data processing.

wsds has a powerful database query engine integrated into it (built on top of Polars). This makes database-style
operations like duplicate detection, group by operations and aggregations very fast and easy to write.
This tight integration let's you run both SQL queries and efficient dataloaders directly on your data without any
conversion or importing.

## Getting Started

```bash
# create environment
conda create -n wsds python=3.10
conda activate wsds

# install hume_wsds
pip install https://github.com/HumeAI/wsds.git
```

## Tests

To run tests you currently need a copy of the `librilight` dataset. The tests can be run with:
```
WSDS_DATASET_PATH=/path/to/the/librilight/folder python tests.py
```
