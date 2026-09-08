import atexit
import os
import re
import threading
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


_s3_clients: dict[tuple[str | None, ...], tuple[typing.Any, typing.Any]] = {}
_s3_clients_pid: int | None = None
_s3_clients_atexit_pid: int | None = None
_s3_clients_lock = threading.Lock()


def _close_s3_clients():
    """Close this process's shared aiobotocore clients at interpreter exit."""
    if _s3_clients_pid != os.getpid():
        return

    from .pupyarrow.file_reader import _get_io_loop

    for _, ctx in _s3_clients.values():
        try:
            _get_io_loop().run(ctx.__aexit__(None, None, None))
        except Exception:
            pass
    _s3_clients.clear()


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

    global _s3_clients_pid, _s3_clients_atexit_pid

    link = link or {}
    endpoint_url = link.get("endpoint_url") or os.environ.get("WSDS_S3_ENDPOINT_URL")
    client_key = (
        endpoint_url,
        link.get("aws_access_key_id"),
        link.get("aws_secret_access_key"),
        link.get("aws_session_token"),
        link.get("region_name"),
    )

    with _s3_clients_lock:
        pid = os.getpid()
        if _s3_clients_pid != pid:
            # A DataLoader worker may be forked after its parent opened S3.
            # Its inherited client is tied to the parent's event loop, so do
            # not reuse or close it from the child process.
            _s3_clients.clear()
            _s3_clients_pid = pid
            _s3_clients_atexit_pid = None

        existing = _s3_clients.get(client_key)
        if existing is not None:
            return existing

        if _s3_clients_atexit_pid != pid:
            atexit.register(_close_s3_clients)
            _s3_clients_atexit_pid = pid

        kwargs = {"config": Config(max_pool_connections=50, signature_version="s3v4")}
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
        _s3_clients[client_key] = (client, ctx)
        return client, ctx


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
        self._start = None
        self._end = None
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
        prefix = link.get("prefix", "")
        column_dir = link.get("subdir", "")
        parts = [p for p in (prefix, partition, column_dir, f"{shard}.wsds") if p]
        # Prepend "/" so normpath collapses any initial ../ against the root,
        # then strip it back off — S3 keys are relative to the bucket root.
        key = os.path.normpath("/" + "/".join(parts)).lstrip("/")
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
