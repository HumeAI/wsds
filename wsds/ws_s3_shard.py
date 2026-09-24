import atexit
import os
import re
import threading
import typing
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse

from .pupyarrow.file_reader import CachedS3FileReader, LocalFileReader, S3FileReader
from .pupyarrow.pupyarrow import FeatherFile, LazyBinaryArray, LazyStringArray
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


class _LazyS3Client:
    """Stands in for an S3 client; materialises the real one (create_s3_client, shared per
    process and endpoint/credentials) on first use.

    Attribute access is the trigger, so `client.get_object(...)`, `client.exceptions.ClientError`
    and friends all work unchanged, while a shard whose bytes are all local -- a full .wsds copy
    or a fully populated cache mirror -- never builds one. Resolve it on a normal thread: building
    the client goes through the IO loop, so touching the handle from a coroutine on that loop
    deadlocks (CachedS3FileReader.client and the tier-3 path below do this). Building a client is the dominant cost
    of opening a cached shard otherwise (SSL context + connection pool, seconds the first time in
    a process, milliseconds after) and every client retained by a reader is ~0.3 MB.
    """
    __slots__ = ("_link", "_client")

    def __init__(self, link=None):
        self._link = dict(link or {})
        self._client = None

    def _materialise(self):
        if self._client is None:
            self._client, _ = create_s3_client(self._link)
        return self._client

    def __getattr__(self, name):                  # only for names not in __slots__
        return getattr(self._materialise(), name)

    def __reduce__(self):                         # picklable: rebuild the handle, not the client
        return (_LazyS3Client, (self._link,))


class WSS3Shard(WSShardInterface):
    """A shard reader that loads data from S3 via aiobotocore range requests.

    Uses pupyarrow's FeatherFile with an S3FileReader so that only the
    IPC footer and the specific batch(es) needed are fetched, rather than
    downloading the entire shard file.

    Three tiers, decided per shard at open (see _resolve_cache):
      1. a FULL local .wsds at the shard's own column path under the dataset root is always
         preferred (free and always correct);
      2. with `"cache": true` in the .wsds-link, a block-sparse mirror next to it
         (<path>.sparse) serves cached blocks locally and fills misses from S3;
      3. otherwise plain S3 range requests.
    """

    def __init__(self, dataset: "WSDataset", bucket: str, key: str, shard_ref: Optional[Tuple[str, str]]=None,
                 s3_client=None, presigned: Optional[bool]=None, cache: Optional[dict]=None):
        self.dataset = dataset
        self.shard_ref = shard_ref
        self.bucket = bucket
        self.key = key
        # Read-through cache config (see _resolve_cache). from_link resolves it from the
        # .wsds-link `cache` field.
        self._cache = cache if cache is not None else self._resolve_cache({}, dataset)

        self._local_path = self._find_local_shard(dataset, key, self._cache.get("subdir", "audio"), shard_ref)
        if self._local_path is not None:
            self._reader = LocalFileReader(self._local_path)          # tier 1: full local shard
            self._feather = FeatherFile(self._reader)
        else:
            # WSDS_SKIP_S3=1: refuse ALL S3 access with a ValueError (a training pipeline's
            # per-sample skip handles it) -- when set we would rather error out than incur S3
            # latency for shards with no local copy.
            if os.environ.get("WSDS_SKIP_S3"):
                raise ValueError(f"S3 access disabled (WSDS_SKIP_S3): s3://{bucket}/{key}")
            if s3_client is None:
                s3_client = _LazyS3Client()          # built only if a byte actually misses
            root = self._cache.get("root")
            if root:
                # tier 2: mirror = the shard's own column path, <partition>/<subdir>/<shard>.wsds.sparse
                self._reader = CachedS3FileReader(s3_client, bucket, key, root, logdir=self._cache.get("logdir"),
                                                  name=self._shard_relpath(key, self._cache.get("subdir", "audio"), shard_ref),
                                                  presigned=presigned)
            else:
                # tier 3: cold S3. S3FileReader uses the client inside coroutines on the IO loop,
                # where a lazy handle cannot be resolved (see _LazyS3Client): resolve it now.
                if isinstance(s3_client, _LazyS3Client):
                    s3_client = s3_client._materialise()
                self._reader = S3FileReader(s3_client, bucket, key, presigned=presigned)
            try:
                self._feather = FeatherFile(self._reader)
            except s3_client.exceptions.ClientError as err:
                raise WSShardMissingError.from_s3(s3_client, key, bucket, err)
        self.batch_size = int(self._feather.schema.custom_metadata["batch_size"])

        # cache
        self._batch = None

    @staticmethod
    def _resolve_cache(link, dataset):
        """Effective read-through cache config. Declarative and OPINIONATED:
        `"cache": true` in the .wsds-link is the only switch, everything else follows
        the dataset's normal column layout (symlink pieces elsewhere if the bytes must
        live on another volume):

          - the audio column of a shard lives where every other column of that shard
            lives: <dataset_root>/<partition>/<subdir>/<shard>.wsds (`subdir` is the
            column dir the .wsds-link names, default "audio"; partition is "" for
            unpartitioned datasets, giving the plain <dataset_root>/audio/<shard>.wsds);
          - tier 1: a FULL local shard at that path is ALWAYS preferred, cache on or off;
          - tier 2: the block-sparse mirror is that same path + ".sparse", i.e. it sits
            next to the partition's metadata shards. Extensions differ, so eviction
            (globs *.sparse) and tier-1 (looks at .wsds) never confuse the two, and
            shard names that repeat across partitions never collide;
          - access logs live at <dataset_root>/<subdir>/.access (the eviction
            service consumes them; point it at the dataset base to see all partitions).
        """
        subdir = link.get("subdir") or "audio"
        if not link.get("cache"):
            return {"subdir": subdir}                    # block cache off; tier 1 still applies
        droot = WSS3Shard._dataset_root(dataset)
        if not droot:
            return {"subdir": subdir}
        return {"subdir": subdir, "root": droot, "logdir": os.path.join(droot, subdir, ".access")}

    @staticmethod
    def _dataset_root(dataset):
        root = getattr(dataset, "dataset_root", None)
        return str(root) if root else None

    @staticmethod
    def _shard_relpath(key, subdir="audio", shard_ref=None):
        """The shard's audio column path relative to the dataset root:
        <partition>/<subdir>/<shard>.wsds (partition may climb with `..`, as index dirs
        such as indices/source reference ../../<delivery>/<batch>/source)."""
        partition, shard = shard_ref if shard_ref else ("", os.path.basename(key)[:-5])
        return os.path.normpath(os.path.join(partition or "", subdir, f"{shard}.wsds"))

    @staticmethod
    def _find_local_shard(dataset, key, subdir="audio", shard_ref=None):
        """Resolve a full local .wsds shard for this S3 key, or None: the shard's own
        column path <dataset_root>/<partition>/<subdir>/<shard>.wsds (see _resolve_cache)."""
        root = WSS3Shard._dataset_root(dataset)
        if not root:
            return None
        cand = os.path.join(root, WSS3Shard._shard_relpath(key, subdir, shard_ref))
        return cand if os.path.exists(cand) else None

    def close(self):
        """Release this shard's reader: its read buffers, its sparse-mirror fd and the S3
        reader if one was opened. Without it, evicting a shard from the open-shard cache
        frees nothing (measured: +58 MB per 1000 crops of retained read buffers)."""
        r = getattr(self, "_reader", None)
        self._reader = None
        self._feather = None
        self._batch = None
        if r is not None:
            try:
                r.close()
            except Exception:
                pass

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
        s3_client = _LazyS3Client(link)   # a local or fully mirrored shard never builds a client
        return cls(dataset, link["bucket"], key, shard_ref=shard_ref, s3_client=s3_client,
                   presigned=link.get("presigned"), cache=cls._resolve_cache(link, dataset))

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
        if data is None or isinstance(col, LazyStringArray):
            # nulls and string columns already materialize to Python values
            return data
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
