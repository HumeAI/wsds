import pickle

import numpy as np

from .ws_audio import AudioReader

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


def decode_sample(column: str, data):
    """Decode a binary column value from a file-like object based on column name.

    Handles .npy (numpy), .pyd (pickle), .txt (UTF-8 string), and audio columns.
    Must only be called on binary columns.
    """
    if column.endswith("npy"):
        return np.load(data)
    elif column.endswith("pyd"):
        return pickle.load(data)
    elif column.endswith("txt"):
        return data if isinstance(data, str) else data.read().decode("utf-8")
    elif column in AUDIO_FILE_KEYS:
        return AudioReader(data)
    raise ValueError(f"Unknown binary column type: {column}")


def get_audio(sample, audio_columns=None):
    """Find and return the first audio column value from a dict-like sample.

    Args:
        sample: A dict-like object (e.g. WSSample) supporting `keys()` and `__getitem__`.
        audio_columns: Optional list of column names to try. Defaults to AUDIO_FILE_KEYS.

    Returns:
        The audio value (typically an AudioReader or WSAudio).

    Raises:
        KeyError: If no audio column is found in the sample.
    """
    candidates = audio_columns or AUDIO_FILE_KEYS
    for col in candidates:
        if col in sample:
            return sample[col]
    raise KeyError(f"No audio column found (tried {list(candidates)}), available keys: {list(sample.keys())}")
