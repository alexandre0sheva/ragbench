"""The cache a benchmark run uses, made visible to the model wrappers.

The active runtime is a plain module global (not a ContextVar): worker threads spawned by the evaluator must see it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragbench.cache.store import DiskCache
from ragbench.config.schema import CacheConfig


@dataclass
class CacheRuntime:
    disk: DiskCache
    config: CacheConfig


_ACTIVE: CacheRuntime | None = None


def open_cache_runtime(config: CacheConfig) -> CacheRuntime:
    """The persistent cache `config` describes (`RAGBENCH_CACHE_DIR` overrides its directory). The caller closes `runtime.disk`."""
    cache_dir = Path(os.environ.get("RAGBENCH_CACHE_DIR") or config.dir)
    return CacheRuntime(disk=DiskCache(cache_dir / "cache.sqlite3", ttl_days=config.ttl_days), config=config)


def active_cache() -> CacheRuntime | None:
    return _ACTIVE


@contextmanager
def activate_cache(runtime: CacheRuntime | None) -> Iterator[CacheRuntime | None]:
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, runtime
    try:
        yield runtime
    finally:
        _ACTIVE = previous


def summarize_cache(
    runtime: CacheRuntime | None,
    *,
    charged_cost_usd: float,
    in_process_saved_usd: float,
    reason: str | None = None,
    extra_spend_usd: float = 0.0,
) -> dict[str, Any]:
    """The `cache` block of `run_summary.json`.

    `charged_cost_usd` is what the run's systems were billed at standalone prices (cache hits included, so systems
    stay comparable); `real_spend_usd` subtracts everything a cache avoided, from both the disk and in-process tiers,
    and adds `extra_spend_usd` that no system is charged for (the latency probe's uncached re-runs).
    """
    avoided = in_process_saved_usd
    if runtime is None:
        return {
            "enabled": False,
            "reason": reason or "disabled",
            "charged_cost_usd": charged_cost_usd,
            "real_spend_usd": max(0.0, charged_cost_usd - avoided) + extra_spend_usd,
        }
    stats = runtime.disk.stats()
    lookups = stats["hits"] + stats["misses"]
    return {
        "enabled": True,
        "path": str(runtime.disk.path),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "hit_rate": stats["hits"] / lookups if lookups else 0.0,
        "saved_cost_usd": stats["saved_cost_usd"],
        "by_namespace": stats["by_namespace"],
        "size_bytes": stats["size_bytes"],
        "charged_cost_usd": charged_cost_usd,
        "real_spend_usd": max(0.0, charged_cost_usd - stats["saved_cost_usd"] - avoided) + extra_spend_usd,
    }
