import atexit
import os
import re
import typing
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse

from .pupyarrow.file_reader import S3FileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray
from .utils import WSShardMissingError
from .ws_decode import decode_sample
from .ws_shard import WSShardInterface

if TYPE_CHECKING:
    from .ws_dataset import WSDataset


def create_s3_client(link=None):
    """Create a shared aiobotocore S3 client.

    Reads `endpoint_url` and AWS credentials from `link` (falling back to the
    WSDS_S3_ENDPOINT_URL env var for the endpoint). Returns the entered client
    and its context manager (for cleanup). The client should be shared across
    all S3FileReader instances.

    signature_version is pinned to SigV4: for non-AWS endpoints botocore may
    otherwise presign SigV2-style URLs, which e.g. Backblaze B2 rejects.
    """
    from aiobotocore.session import AioSession
    from botocore.config import Config

    from .pupyarrow.file_reader import _get_io_loop

    link = link or {}
    kwargs = {"config": Config(max_pool_connections=50, signature_version="s3v4")}
    endpoint_url = link.get("endpoint_url") or os.environ.get("WSDS_S3_ENDPOINT_URL")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    for k in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token", "region_name"):
        if link.get(k):
            kwargs[k] = link[k]
    if "region_name" not in kwargs and endpoint_url:
        # SigV4 embeds the region in the credential scope, so it must match
        # the endpoint. Derive it from "s3.<region>.<provider>" hostnames
        # (e.g. s3.us-east-005.backblazeb2.com); real regions contain "-",
        # which also excludes bare hosts like s3.amazonaws.com.
        m = re.match(r"https?://s3\.([a-z0-9-]+)\.", endpoint_url)
        if m and "-" in m.group(1):
            kwargs["region_name"] = m.group(1)
    ctx = AioSession().create_client("s3", **kwargs)
    client = _get_io_loop().run(ctx.__aenter__())

    def _cleanup():
        try:
            _get_io_loop().run(ctx.__aexit__(None, None, None))
        except Exception:
            pass

    atexit.register(_cleanup)
    return client, ctx


def build_link_key(prefix: str, partition: str, subdir: str, shard: str) -> str:
    """Construct the storage key/path for a shard as link readers resolve it:
    normpath(prefix / partition / subdir / <shard>.wsds) with leading "../"
    stripped — partitions are relative to the index, but bucket/volume paths
    are absolute from their root. Shared by WSS3Shard, WSModalShard and
    support_scripts/make_s3_link.py (which validates the exact keys reads use).
    """
    parts = [p for p in (prefix, partition, subdir, f"{shard}.wsds") if p]
    key = os.path.normpath("/" + "/".join(parts)).lstrip("/")
    return key


class WSS3Shard(WSShardInterface):
    """A shard reader that loads data from S3 via aiobotocore range requests.

    Uses pupyarrow's FeatherFile with an S3FileReader so that only the
    IPC footer and the specific batch(es) needed are fetched, rather than
    downloading the entire shard file."""

    def __init__(self, dataset: "WSDataset", bucket: str, key: str, shard_ref: Optional[Tuple[str, str]]=None, s3_client=None, presigned: Optional[bool]=None):
        self.dataset = dataset
        self.shard_ref = shard_ref
        self.bucket = bucket
        self.key = key

        if s3_client is None:
            s3_client, _ = create_s3_client()

        self._reader = S3FileReader(s3_client, bucket, key, presigned=presigned)
        try:
            self._feather = FeatherFile(self._reader)
        except s3_client.exceptions.ClientError as err:
            raise WSShardMissingError.from_s3(s3_client, key, bucket, err)
        self.batch_size = int(self._feather.schema.custom_metadata["batch_size"])

        # cache
        self._batch = None

    @classmethod
    def from_s3_url(cls, dataset: "WSDataset", url: str, shard_ref: Optional[Tuple[str, str]]=None, s3_client=None):
        """Construct from an s3://bucket/key URL."""
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            raise ValueError(f"expected s3:// URL, got: {url}")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return cls(dataset, bucket, key, shard_ref=shard_ref, s3_client=s3_client)

    @classmethod
    def get_columns(cls, link, dataset):
        """Return columns provided by this S3 link."""
        if "columns" in link:
            return {col: col for col in link["columns"]}
        columns = cls._discover_columns_from_s3(link)
        return {col: col for col in columns if col != "__key__"}

    @classmethod
    def from_link(cls, link, dataset, shard_ref):
        """Create an S3 shard from a link spec."""
        partition, shard = shard_ref
        key = build_link_key(link.get("prefix", ""), partition, link.get("subdir", ""), shard)
        s3_client, _ = create_s3_client(link)
        return cls(dataset, link["bucket"], key, shard_ref=shard_ref, s3_client=s3_client, presigned=link.get("presigned"))

    @classmethod
    def _discover_columns_from_s3(cls, link):
        """Read one shard's footer from S3 to discover column names."""
        from .pupyarrow.file_reader import _get_io_loop

        bucket = link["bucket"]
        prefix = link["prefix"]
        s3_client, _ = create_s3_client(link)

        async def _discover():
            response = await s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
            for obj in response.get("Contents", []):
                if obj["Key"].endswith(".wsds"):
                    reader = S3FileReader(s3_client, bucket, obj["Key"])
                    feather = FeatherFile(reader)
                    return feather.schema.names
            raise ValueError(f"No .wsds files found in s3://{bucket}/{prefix}")

        return _get_io_loop().run(_discover())

    def _s3_path(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def _num_batches(self) -> int:
        return self._feather.num_record_batches

    def _get_batch(self, index: int):
        return self._feather.record_batch(index)

    def _shard_name(self) -> str:
        return self._s3_path()

    def _batch_row_counts(self) -> list[int]:
        # one concurrent round of header reads instead of a sequential GET per batch
        import asyncio

        from .pupyarrow.file_reader import _get_io_loop

        async def _fetch():
            return await asyncio.gather(
                *(self._feather.async_record_batch(i) for i in range(self._feather.num_record_batches))
            )

        return [b.num_rows for b in _get_io_loop().run(_fetch())]

    def get_sample(self, column: str, offset: int) -> typing.Any:
        if self._batch is None or offset < self._start or offset >= self._end:
            self._batch = self._locate_batch(offset)

        j = offset - self._start
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
