import io
import pickle
import typing

import numpy as np
import pyarrow as pa

from .ws_audio import WSAudioEpisode

AUDIO_FILE_KEYS = frozenset(
    [
        "audio",  # recommended so all shards can have the same columns
        "flac",
        "mp3",
        "sox",
        "wav",
        "m4a",
        "ogg",
        "wma",
        "opus",  # fallback for old datasets
    ]
)


def to_filelike(src: typing.Any) -> typing.BinaryIO:
    """Coerces files, byte-strings and PyArrow binary buffers into file-like objects."""
    if hasattr(src, "read"):  # an open file
        return src
    # if not an open file then we assume some kind of binary data in memory
    if hasattr(src, "as_buffer"):  # PyArrow binary data
        return pa.BufferReader(src.as_buffer())
    return io.BytesIO(src)


# --- ".arr": variable-length numeric arrays stored NATIVELY in pyarrow --------
# Unlike ".npy" (a binary blob decoded with np.load, which parses the header via
# ast.literal_eval and allocates cyclic garbage every read), a ".arr" column is a
# native `list<fixed_size_list<T, K>>` (2-D) or `list<T>` (1-D). Reads are a
# zero-copy `flatten().to_numpy()` — ~10x faster, no per-read cyclic allocation.

def arr_column_type(sample) -> pa.DataType:
    """Native pyarrow type for a ".arr" column value (numpy 1-D or 2-D)."""
    a = np.asarray(sample)
    et = pa.from_numpy_dtype(a.dtype)
    if a.ndim == 1:
        return pa.list_(et)                       # list<T>
    if a.ndim == 2:
        return pa.list_(pa.list_(et, a.shape[1]))  # list<fixed_size_list<T, K>>
    raise ValueError(f".arr supports 1-D/2-D arrays, got ndim={a.ndim}")


def decode_arr(scalar, arrow_type) -> np.ndarray:
    """Read a native ".arr" list scalar as a numpy array (no np.load / as_py)."""
    inner = arrow_type.value_type
    if pa.types.is_fixed_size_list(inner):
        vals = scalar.values.flatten().to_numpy(zero_copy_only=False)
        return vals.reshape(-1, inner.list_size)
    return scalar.values.to_numpy(zero_copy_only=False)


def decode_sample(column: str, data):
    """Decode a binary column value from a file-like object based on column name.

    Handles .npy (numpy), .pyd (pickle), .txt (UTF-8 string), .json, and audio columns.
    Column can be a bare name ("audio") or dotted ("source.mp3").
    """
    ext = column.rsplit(".", 1)[-1] if "." in column else column
    fd = to_filelike(data)
    if ext == "npy":
        return np.load(fd)
    elif ext == "pyd":
        return pickle.load(fd)
    elif ext == "txt":
        return fd.read().decode("utf-8")
    elif ext == "json":
        import json
        return json.load(fd)
    elif ext in AUDIO_FILE_KEYS:
        return WSAudioEpisode(fd)
    else:
        return fd.read()


def encode_value(column: str, value):
    """Encode a Python value to bytes based on column name. Mirrors decode_sample."""
    if isinstance(value, bytes):
        return value
    ext = column.rsplit(".", 1)[-1] if "." in column else column
    if ext == "npy":
        buf = io.BytesIO()
        np.save(buf, value)
        return buf.getvalue()
    elif ext == "arr":
        # keep the numpy array; WSSink builds the native list column (arr_column_type)
        return np.asarray(value)
    elif ext == "pyd":
        return pickle.dumps(value)
    elif ext == "json":
        import json
        return json.dumps(value).encode("utf-8")
    elif ext in AUDIO_FILE_KEYS:
        if isinstance(value, np.ndarray):
            import torch
            value = torch.from_numpy(value)
        if hasattr(value, 'shape'):  # torch tensor
            from .audio_codec import encode_audio
            fmt = ext if ext != "audio" else "ogg"
            return encode_audio(value, format=fmt)
        return value.to_bytes()
    else:
        return value


def get_audio(sample, audio_columns=None):
    """Find and return the first audio column value from a dict-like sample.

    Args:
        sample: A dict-like object (e.g. WSSample) supporting `keys()` and `__getitem__`.
        audio_columns: Optional list of column names to try. Defaults to AUDIO_FILE_KEYS.

    Returns:
        The audio value (typically a WSAudioEpisode or WSAudioSegment).

    Raises:
        KeyError: If no audio column is found in the sample.
    """
    candidates = audio_columns or AUDIO_FILE_KEYS
    for col in candidates:
        if col in sample:
            return sample[col]
    raise KeyError(f"No audio column found (tried {list(candidates)}), available keys: {list(sample.keys())}")
