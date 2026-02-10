import os
import typing
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .pupyarrow.file_reader import S3FileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray
from .utils import WSShardMissingError
from .ws_decode import decode_sample
from .ws_shard import WSShardInterface

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


class WSS3Shard(WSShardInterface):
    """A shard reader that loads data from S3 via boto3 range requests.

    Uses pupyarrow's FeatherFile with an S3File wrapper so that only the
    IPC footer and the specific batch(es) needed are fetched, rather than
    downloading the entire shard file."""

    def __init__(self, dataset: "WSDataset", bucket: str, key: str, shard_name=None, s3_client=None):
        self.dataset = dataset
        self.shard_name = shard_name
        self.bucket = bucket
        self.key = key

        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        self._reader = S3FileReader(s3_client, bucket, key)
        try:
            self._feather = FeatherFile(self._reader)
        except s3_client.exceptions.ClientError as err:
            raise WSShardMissingError.from_s3(s3_client, bucket, key, err)
        self.batch_size = int(self._feather.schema.custom_metadata["batch_size"])

        # cache
        self._start = None
        self._end = None
        self._batch = None

    @classmethod
    def from_s3_url(cls, dataset: "WSDataset", url: str, shard_name=None, s3_client=None):
        """Construct from an s3://bucket/key URL."""
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            raise ValueError(f"expected s3:// URL, got: {url}")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return cls(dataset, bucket, key, shard_name=shard_name, s3_client=s3_client)

    @classmethod
    def get_columns(cls, link, dataset):
        """Return columns provided by this S3 link."""
        if "columns" in link:
            return {col: col for col in link["columns"]}
        columns = cls._discover_columns_from_s3(link)
        return {col: col for col in columns if col != "__key__"}

    @classmethod
    def from_link(cls, link, dataset, shard_name):
        """Create an S3 shard from a link spec."""
        dataset_path, shard = shard_name
        prefix = link["prefix"]
        key = f"{prefix}/{dataset_path}/{shard}.wsds" if dataset_path else f"{prefix}/{shard}.wsds"
        s3_client = cls._make_s3_client(link.get("endpoint_url"))
        return cls(dataset, link["bucket"], os.path.normpath(key), shard_name=shard_name, s3_client=s3_client)

    @classmethod
    def _make_s3_client(cls, endpoint_url=None):
        import boto3

        endpoint_url = endpoint_url or os.environ.get("WSDS_S3_ENDPOINT_URL")
        kwargs = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        return boto3.client("s3", **kwargs)

    @classmethod
    def _discover_columns_from_s3(cls, link):
        """Read one shard's footer from S3 to discover column names."""
        s3_client = cls._make_s3_client(link.get("endpoint_url"))
        bucket = link["bucket"]
        prefix = link["prefix"]
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".wsds"):
                reader = S3FileReader(s3_client, bucket, obj["Key"])
                feather = FeatherFile(reader)
                return feather.schema.names
        raise ValueError(f"No .wsds files found in s3://{bucket}/{prefix}")

    def _s3_path(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._batch is None or offset < self._start or offset >= self._end:
            i = offset // self.batch_size
            if i >= self._feather.num_record_batches:
                raise IndexError(f"{offset} is out of range for shard {self._s3_path()}")
            self._batch = self._feather.record_batch(i)
            if i < self._feather.num_record_batches - 1:
                if self._batch.num_rows < self.batch_size:
                    raise ValueError(
                        f"Batch {i} in shard {self._s3_path()} is incomplete "
                        f"(has only {self._batch.num_rows} rows instead of {self.batch_size})"
                    )
            self._start = i * self.batch_size
            self._end = self._start + self.batch_size

        j = offset % self.batch_size
        if j >= self._batch.num_rows:
            raise IndexError(f"{offset} is out of range for shard {self._s3_path()}")
        try:
            col = self._batch.column(column)
        except KeyError:
            raise KeyError(f"column {column} not found in shard {self._s3_path()}")
        data = col[j]
        try:
            if isinstance(col, LazyBinaryArray):
                data._optimal_read_size = 2 * 1024 * 1024
                return decode_sample(column, data)
        except Exception as e:
            raise ValueError(f"Failed to decode column {column} in shard {self._s3_path()} (offset {offset}): {e}")
        return data

    def __repr__(self):
        r = f"WSS3Shard('{self._s3_path()}')"
        if self._batch:
            r += f" # cached_region = [{self._start}, {self._end}]"
        return r
