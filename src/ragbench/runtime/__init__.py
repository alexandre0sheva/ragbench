"""Execution runtime: rate limiting, parallel scheduling, and run-wide execution settings.

The light helpers load eagerly; the executor (which depends on the RAG systems) loads on first use so that model and
system code can import `ragbench.runtime` without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragbench.runtime.context import RuntimeContext, activate_runtime, build_limiters, current_runtime, limiter_for
from ragbench.runtime.limiter import RateLimiter
from ragbench.runtime.parallel import evenly_sample, ordered_parallel_map, warm_up_imports
from ragbench.runtime.progress import ProgressListener, SerializedProgress

if TYPE_CHECKING:
    from ragbench.runtime.executor import ExecutionSettings, ProbeResult, SystemOutcome, SystemRunner

_LAZY = {"ExecutionSettings", "ProbeResult", "SystemOutcome", "SystemRunner"}

__all__ = [
    "ExecutionSettings",
    "ProbeResult",
    "ProgressListener",
    "RateLimiter",
    "RuntimeContext",
    "SerializedProgress",
    "SystemOutcome",
    "SystemRunner",
    "activate_runtime",
    "build_limiters",
    "current_runtime",
    "evenly_sample",
    "limiter_for",
    "ordered_parallel_map",
    "warm_up_imports",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from ragbench.runtime import executor

        return getattr(executor, name)
    raise AttributeError(f"module 'ragbench.runtime' has no attribute {name!r}")
