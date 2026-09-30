"""Dependency-free parallel helpers (importable from model and system code without import cycles)."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def warm_up_imports(modules: Iterable[str]) -> list[str]:
    """Import `modules` now, on the calling thread; returns the ones that could not be imported.

    Call this before starting worker threads. The OpenAI SDK (3.x) looks at `sys.modules["httpx"]` whenever it
    builds a request and assumes the module is complete; if another thread is in the middle of the *first* import
    of `httpx` (Chroma does that, lazily) the request fails with "partially initialized module 'httpx'". Importing
    such libraries up front, single-threaded, removes the window.
    """
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            logger.debug("Optional module %s is not installed; skipping warm-up.", name)
            missing.append(name)
    return missing


def ordered_parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    on_done: Callable[[int, int], None] | None = None,
    thread_name_prefix: str = "ragbench",
) -> list[R]:
    """Apply `fn` to every item on up to `workers` threads and return the results in input order.

    `on_done(completed, total)` is called from the calling thread as items finish (in completion order). The first
    exception stops new work from starting and is re-raised once running items have finished.
    """
    work = list(items)
    total = len(work)
    if workers <= 1 or total <= 1:
        results = []
        for done, item in enumerate(work, start=1):
            results.append(fn(item))
            if on_done:
                on_done(done, total)
        return results
    ordered: list[Any] = [None] * total
    with ThreadPoolExecutor(max_workers=min(workers, total), thread_name_prefix=thread_name_prefix) as pool:
        futures: dict[Future[R], int] = {pool.submit(fn, item): index for index, item in enumerate(work)}
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                ordered[futures[future]] = future.result()
                if on_done:
                    on_done(done, total)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return ordered


def evenly_sample(items: Sequence[T], count: int) -> list[T]:
    """`count` items spread evenly across the sequence, first and last included (all of them when fewer exist)."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    return [items[round(i * last / (count - 1))] for i in range(count)]


