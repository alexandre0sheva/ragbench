"""What one question has spent so far, so that a question which fails partway still charges what it paid for.

`evaluation.question_runner` opens a meter around every question. Each paid call made inside it (the cache-wrapped LLM, embedding and
decision models, and an agent harness that stops early) adds its cost, on the question's own thread or on a worker thread that copied its
context (`ordered_parallel_map` does). Only the failure path reads the meter: a question that finishes reports its cost from its own trace.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ragbench.models.cost import CostBreakdown


class SpendMeter:
    """A thread-safe running total of the cost noted while it is active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cost = CostBreakdown()

    def add(self, cost: CostBreakdown) -> None:
        with self._lock:
            self._cost = self._cost.plus(cost)

    @property
    def cost(self) -> CostBreakdown:
        with self._lock:
            return self._cost


_ACTIVE: ContextVar[SpendMeter | None] = ContextVar("ragbench_spend_meter", default=None)


@contextmanager
def metering() -> Iterator[SpendMeter]:
    """Count the spend noted inside the block (a nested block gets its own meter)."""
    meter = SpendMeter()
    token = _ACTIVE.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE.reset(token)


def note_spend(cost: CostBreakdown) -> None:
    """Add `cost` to the active meter; a no-op outside a question."""
    meter = _ACTIVE.get()
    if meter is not None:
        meter.add(cost)
