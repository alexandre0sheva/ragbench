from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class Timer:
    elapsed_ms: float = 0.0


@contextmanager
def timer() -> Iterator[Timer]:
    t = Timer()
    start = perf_counter()
    try:
        yield t
    finally:
        t.elapsed_ms = (perf_counter() - start) * 1000

