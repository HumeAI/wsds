import typing
from typing import TYPE_CHECKING, Optional, Tuple

from .pupyarrow.file_reader import ModalFileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray
from .ws_decode import decode_sample
from .ws_s3_shard import build_link_key
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
        self._batch = None

    @classmethod
    def from_link(cls, link, dataset, shard_ref):
        """Create a Modal shard from a link spec.

        The volume path is built as ``<prefix>/<partition>/<column_dir>/<shard>.wsds``.
        ``column_dir`` comes from the link spec (required when the volume mirrors the
        local dataset directory layout with per-column subdirectories)."""
        partition, shard = shard_ref
        path = build_link_key(link.get("prefix", ""), partition, link.get("subdir", ""), shard)
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

    def _num_batches(self) -> int:
        return self._feather.num_record_batches

    def _get_batch(self, index: int):
        return self._feather.record_batch(index)

    def _shard_name(self) -> str:
        return self._modal_path()

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._batch is None or offset < self._start or offset >= self._end:
            self._batch = self._locate_batch(offset)

        j = offset - self._start
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
