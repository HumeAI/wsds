"""Env-gated phase timing for the audio-fetch hot path.

Disabled by default (unset ``WSDS_TIMING``): ``record()`` returns a shared no-op
context manager (~130 ns/call — negligible next to the ms-scale read/decode it
wraps). When enabled, each process accumulates per-phase ``[count, total_s,
max_s]`` and periodically writes ``$WSDS_TIMING_OUT.<pid>`` as JSON so stats
survive even if DataLoader worker processes are killed. Aggregate by summing all
``<out>.*`` files.

    from ._timing import record
    with record("blob_decode"):
        ...
"""
import atexit
import json
import os
import time

_ENABLED = bool(os.environ.get("WSDS_TIMING"))
_OUT = os.environ.get("WSDS_TIMING_OUT")
_stats: dict[str, list] = {}   # phase -> [count, total_s, max_s]
_n = 0


class _Noop:
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NOOP = _Noop()


class _Timer:
    __slots__ = ("phase", "_t")

    def __init__(self, phase):
        self.phase = phase

    def __enter__(self):
        self._t = time.perf_counter()

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t
        s = _stats.get(self.phase)
        if s is None:
            _stats[self.phase] = [1, dt, dt]
        else:
            s[0] += 1
            s[1] += dt
            if dt > s[2]:
                s[2] = dt
        global _n
        _n += 1
        if _n % 500 == 0:
            _flush()
        return False


def _flush():
    if _OUT and _stats:
        tmp = f"{_OUT}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(_stats, f)
        os.replace(tmp, f"{_OUT}.{os.getpid()}")


if _ENABLED:
    def record(phase):
        return _Timer(phase)

    atexit.register(_flush)
else:
    def record(phase):        # zero-work fast path: shared no-op context manager
        return _NOOP


def get_stats():
    return _stats
