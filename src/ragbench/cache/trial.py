"""Which trial of a repeated run (`evaluation.trials`) the current call belongs to.

Every LLM and decision request made while answering or judging trial `t > 0` carries `t` in its disk-cache key, so trial 2 of a
question is a fresh answer rather than a replay of trial 1. Trial 0 adds nothing to the key: a one-trial run, and the first trial of
any run, hit the entries an older run stored. A ContextVar, not a global: trials of different questions run on different threads at
the same time (`runtime.parallel.ordered_parallel_map` carries the value into the threads a system starts itself).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_TRIAL: ContextVar[int] = ContextVar("ragbench_trial", default=0)


def current_trial() -> int:
    return _TRIAL.get()


def trial_key() -> dict[str, int]:
    """The cache-key part for the current trial: empty for trial 0."""
    trial = _TRIAL.get()
    return {"trial": trial} if trial else {}


@contextmanager
def trial_scope(trial: int) -> Iterator[int]:
    token = _TRIAL.set(trial)
    try:
        yield trial
    finally:
        _TRIAL.reset(token)
