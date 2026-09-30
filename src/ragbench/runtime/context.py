"""Run-wide execution settings that low-level code (model clients, ingestion) reads at call time.

Like the cache runtime this is a plain module global rather than a ContextVar: worker threads started by the
evaluator must see it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from ragbench.config.schema import LimitsConfig
from ragbench.runtime.limiter import RateLimiter


@dataclass
class RuntimeContext:
    ingest_workers: int = 1
    limiters: dict[str, RateLimiter] = field(default_factory=dict)


_DEFAULT = RuntimeContext()
_ACTIVE: RuntimeContext | None = None


def current_runtime() -> RuntimeContext:
    return _ACTIVE or _DEFAULT


def limiter_for(provider: str) -> RateLimiter | None:
    return current_runtime().limiters.get(provider)


@contextmanager
def activate_runtime(context: RuntimeContext) -> Iterator[RuntimeContext]:
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, context
    try:
        yield context
    finally:
        _ACTIVE = previous


def build_limiters(limits: LimitsConfig) -> dict[str, RateLimiter]:
    """One limiter per provider. Only OpenAI exists today; Task 8 adds per-provider limits."""
    if limits.max_concurrent_requests is None and limits.requests_per_minute is None and limits.tokens_per_minute is None:
        return {}
    return {
        "openai": RateLimiter(
            max_concurrent=limits.max_concurrent_requests,
            requests_per_minute=limits.requests_per_minute,
            tokens_per_minute=limits.tokens_per_minute,
        )
    }
