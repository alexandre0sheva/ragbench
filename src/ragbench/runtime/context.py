"""Run-wide execution settings that low-level code (model clients, ingestion) reads at call time.

Like the cache runtime this is a plain module global rather than a ContextVar: worker threads started by the
evaluator must see it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from ragbench.config.schema import LimitsConfig, ProviderConfig
from ragbench.runtime.limiter import RateLimiter


@dataclass
class RuntimeContext:
    ingest_workers: int = 1
    # Limiter per provider key (`openai`, `anthropic`, `openai_compatible:<endpoint>`); see `limiter_for`.
    limiters: dict[str, RateLimiter] = field(default_factory=dict)
    # `providers:` endpoints, read by `create_llm` / `create_embedding_model` when a system builds its models.
    providers: dict[str, ProviderConfig] = field(default_factory=dict)


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


def _limiter(limits: LimitsConfig) -> RateLimiter:
    return RateLimiter(
        max_concurrent=limits.max_concurrent_requests,
        requests_per_minute=limits.requests_per_minute,
        tokens_per_minute=limits.tokens_per_minute,
    )


def build_limiters(limits: LimitsConfig, providers: dict[str, ProviderConfig] | None = None) -> dict[str, RateLimiter]:
    """One limiter per provider, keyed the way clients look them up with `limiter_for`.

    The top-level `limits:` applies to each hosted API separately (`openai`, `anthropic`: their rate limits are independent,
    so they get independent limiters); every `providers:` endpoint uses its own `limits:` as `openai_compatible:<name>`.
    """
    limiters: dict[str, RateLimiter] = {}
    if limits.is_set:
        limiters.update({"openai": _limiter(limits), "anthropic": _limiter(limits)})
    for name, endpoint in (providers or {}).items():
        if endpoint.limits.is_set:
            limiters[f"openai_compatible:{name}"] = _limiter(endpoint.limits)
    return limiters
