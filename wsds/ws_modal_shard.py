import os
import typing
from typing import TYPE_CHECKING, Optional, Tuple

from .pupyarrow.file_reader import ModalFileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray
from .ws_decode import decode_sample
from .ws_shard import WSShardInterface

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


class WSModalShard(WSShardInterface):
    """A shard reader that loads data from a Modal Volume via range requests.

    Uses ModalFileReader (gRPC ``VolumeGetFile2`` with ``start``/``len``) so
    that only the IPC footer and the specific batch(es) needed are fetched,
    rather than downloading the entire shard file."""

    def __init__(self, dataset: "WSDataset", volume_name: str, path: str, shard_ref: Optional[Tuple[str, str]]=None):
        self.dataset = dataset
        self.shard_ref = shard_ref
        self.volume_name = volume_name
        self.path = path

        self._reader = ModalFileReader.from_name(volume_name, path)
        self._feather = FeatherFile(self._reader)
        self.batch_size = int(self._feather.schema.custom_metadata["batch_size"])

        # cache
        self._start = None
        self._end = None
        self._batch = None

    @classmethod
    def from_link(cls, link, dataset, shard_ref):
        """Create a Modal shard from a link spec.

        The volume path is built as ``<prefix>/<partition>/<column_dir>/<shard>.wsds``.
        ``column_dir`` comes from the link spec (required when the volume mirrors the
        local dataset directory layout with per-column subdirectories)."""
        partition, shard = shard_ref
        prefix = link.get("prefix", "")
        column_dir = link.get("subdir", "")
        parts = [p for p in (prefix, partition, column_dir, f"{shard}.wsds") if p]
        path = os.path.normpath("/".join(parts))
        # Strip leading "../" — partition is relative to the index but
        # volume paths are absolute from the volume root.
        while path.startswith("../"):
            path = path[3:]
        return cls(dataset, link["volume_name"], path, shard_ref=shard_ref)

    @classmethod
    def get_columns(cls, link, dataset):
        """Return columns provided by this Modal link."""
        if "columns" in link:
            return {col: col for col in link["columns"]}
        columns = cls._discover_columns(link)
        return {col: col for col in columns if col != "__key__"}

    @classmethod
    def _discover_columns(cls, link):
        """Read one shard's footer from the Modal Volume to discover column names."""
        import modal

        vol = modal.Volume.from_name(link["volume_name"])
        prefix = link["prefix"]
        for entry in vol.listdir(prefix):
            if entry.path.endswith(".wsds"):
                reader = ModalFileReader.from_name(link["volume_name"], entry.path)
                feather = FeatherFile(reader)
                names = feather.schema.names
                reader.close()
                return names
        raise ValueError(f"No .wsds files found in modal volume '{link['volume_name']}' at prefix '{prefix}'")

    def _modal_path(self) -> str:
        return f"modal://{self.volume_name}/{self.path}"

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._batch is None or offset < self._start or offset >= self._end:
            i = offset // self.batch_size
            if i >= self._feather.num_record_batches:
                raise IndexError(f"{offset} is out of range for shard {self._modal_path()}")
            self._batch = self._feather.record_batch(i)
            if i < self._feather.num_record_batches - 1:
                if self._batch.num_rows < self.batch_size:
                    raise ValueError(
                        f"Batch {i} in shard {self._modal_path()} is incomplete "
                        f"(has only {self._batch.num_rows} rows instead of {self.batch_size})"
                    )
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= self._batch.num_rows:
            raise IndexError(f"{offset} is out of range for shard {self._modal_path()}")
        try:
            col = self._batch.column(column)
        except KeyError:
            raise KeyError(f"column {column} not found in shard {self._modal_path()}")
        data = col[j]
        try:
            if isinstance(col, LazyBinaryArray):
                data._optimal_read_size = 2 * 1024 * 1024
                return decode_sample(column, data)
        except Exception as e:
            raise ValueError(f"Failed to decode column {column} in shard {self._modal_path()} (offset {offset}): {e}")
        return data

    def __repr__(self):
        r = f"WSModalShard('{self._modal_path()}')"
        if self._batch:
            r += f" # cached_region = [{self._start}, {self._end}]"
        return r
