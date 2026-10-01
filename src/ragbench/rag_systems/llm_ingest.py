"""Ingestion-time LLM calls that must not take a whole run down when one of them fails."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TypeVar

from ragbench.runtime.parallel import ordered_parallel_map

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def map_llm_calls(fn: Callable[[T], R], items: Iterable[T], *, workers: int, what: str, thread_name_prefix: str) -> tuple[list[R | None], int]:
    """`fn` over `items` (in parallel, order kept) with one slot per item: its result, or None when its call raised.

    Returns the slots and the number of failures. The caller degrades the failed items (no context, a lead-text summary), because one
    flaky call should not lose a run. But when *every* call failed, nothing worked (typically a bad key or model name), so the first
    error is raised instead of quietly producing a run that is not the system it claims to be.
    """

    def guarded(item: T) -> tuple[R | None, Exception | None]:
        try:
            return fn(item), None
        except Exception as exc:  # noqa: BLE001  (any client error is a failed call for this one item)
            logger.warning("LLM call for %s failed (%s: %s); degrading that item", what, type(exc).__name__, exc)
            return None, exc

    outcomes = ordered_parallel_map(guarded, items, workers=workers, thread_name_prefix=thread_name_prefix)
    errors = [error for _, error in outcomes if error is not None]
    if errors and len(errors) == len(outcomes):
        raise errors[0]
    return [result for result, _ in outcomes], len(errors)
