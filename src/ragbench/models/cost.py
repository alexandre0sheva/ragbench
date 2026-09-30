from __future__ import annotations

import logging
import threading

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ISO date the price table below was last reviewed. Bump it whenever you touch the table.
PRICING_AS_OF = "2026-06-12"

# Approximate USD prices per 1M tokens. Pricing changes; update this registry
# (or override per run with the `pricing:` config section) before using RAGBench
# for budgeting or procurement decisions.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "mock-llm": {"input": 0.0, "output": 0.0},
    "hashing-embedding": {"input": 0.0, "output": 0.0},
}


class CostBreakdown(BaseModel):
    embedding_input_tokens: int = 0
    embedding_cost: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cost: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost: float = 0.0
    rerank_cost: float = 0.0
    query_rewrite_cost: float = 0.0
    tool_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.embedding_cost + self.llm_cost + self.judge_cost + self.rerank_cost + self.query_rewrite_cost + self.tool_cost

    def plus(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            embedding_input_tokens=self.embedding_input_tokens + other.embedding_input_tokens,
            embedding_cost=self.embedding_cost + other.embedding_cost,
            llm_prompt_tokens=self.llm_prompt_tokens + other.llm_prompt_tokens,
            llm_completion_tokens=self.llm_completion_tokens + other.llm_completion_tokens,
            llm_cost=self.llm_cost + other.llm_cost,
            judge_prompt_tokens=self.judge_prompt_tokens + other.judge_prompt_tokens,
            judge_completion_tokens=self.judge_completion_tokens + other.judge_completion_tokens,
            judge_cost=self.judge_cost + other.judge_cost,
            rerank_cost=self.rerank_cost + other.rerank_cost,
            query_rewrite_cost=self.query_rewrite_cost + other.query_rewrite_cost,
            tool_cost=self.tool_cost + other.tool_cost,
        )

    def minus(self, other: CostBreakdown) -> CostBreakdown:
        """Field-wise difference; used to find cost that a trace does not account for (may be negative)."""
        return CostBreakdown(**{name: getattr(self, name) - getattr(other, name) for name in type(self).model_fields})

    def as_dict(self) -> dict[str, float | int]:
        data = self.model_dump()
        data["total_cost"] = self.total_cost
        return data


_lock = threading.Lock()
_pricing_overrides: dict[str, dict[str, float]] = {}
_unknown_priced_models: set[str] = set()


def register_pricing(overrides: dict[str, dict[str, float]]) -> None:
    """Add or replace per-model prices (USD per 1M tokens) for the current process.

    Overrides win over the built-in table. Use `clear_pricing_overrides()` to drop them.
    """
    with _lock:
        for model, price in overrides.items():
            _pricing_overrides[model] = {"input": float(price.get("input", 0.0)), "output": float(price.get("output", 0.0))}


def clear_pricing_overrides() -> None:
    with _lock:
        _pricing_overrides.clear()


def unknown_priced_models() -> list[str]:
    """Models that were billed at $0 because no price is registered (sorted)."""
    with _lock:
        return sorted(_unknown_priced_models)


def reset_unknown_priced_models() -> None:
    with _lock:
        _unknown_priced_models.clear()


def _lookup_pricing(model: str) -> dict[str, float] | None:
    # Exact match first, then the longest registered name that prefixes a dated snapshot
    # (e.g. "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini"). Overrides beat the built-in table.
    for table in (_pricing_overrides, MODEL_PRICING_USD_PER_1M):
        if model in table:
            return table[model]
        candidates = [name for name in table if model.startswith(name + "-")]
        if candidates:
            return table[max(candidates, key=len)]
    return None


def estimate_model_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    with _lock:
        pricing = _lookup_pricing(model)
        first_unknown = pricing is None and model not in _unknown_priced_models
        if pricing is None:
            _unknown_priced_models.add(model)
    if pricing is None:
        if first_unknown:
            logger.warning(
                "No price registered for model %r: its cost is reported as $0. Add it under `pricing:` in your config.", model
            )
        return 0.0
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
