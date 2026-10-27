from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ISO date the price table below was last reviewed. Bump it whenever you touch the table.
PRICING_AS_OF = "2026-10-27"

# A price row: USD per 1M tokens, `{"input": ..., "output": ...}`. A row whose price depends on prompt length also has
# `tiers`: `[{"max_prompt_tokens": N, "input": ..., "output": ...}, ..., {"input": ..., "output": ...}]`, in increasing order.
# Each request is priced on its own at the first tier whose `max_prompt_tokens` is at least its prompt (all of its input
# tokens); a prompt above every limit pays the last tier, which normally has no limit. The flat `input`/`output` of a tiered
# row repeat its first tier, for tables and for code that only needs a rough rate.
PriceRow = dict[str, Any]


def tiered(*tiers: tuple[int | None, float, float]) -> PriceRow:
    """A price row from `(max_prompt_tokens, input, output)` tiers; the last one's limit is `None`."""
    rows = [{"input": input_price, "output": output_price, **({"max_prompt_tokens": limit} if limit is not None else {})} for limit, input_price, output_price in tiers]
    return {"input": rows[0]["input"], "output": rows[0]["output"], "tiers": rows}


# Approximate USD prices per 1M tokens (standard tier). Pricing changes; update this registry
# (or override per run with the `pricing:` config section) before using RAGBench
# for budgeting or procurement decisions.
#
# Sources, fetched 2026-10-27 (re-verify the whole table with docs/release-checklist.md before a release):
#   OpenAI    https://developers.openai.com/api/docs/pricing  and the per-model pages under /api/docs/models/
#   Anthropic https://platform.claude.com/docs/en/about-claude/pricing  and /docs/en/about-claude/models/overview
# OpenRouter models are not in this table: their prices come from the OpenRouter catalog (see `models/openrouter_catalog.py`).
#
# Policy: only currently served models, newest generation per tier (plus the previous generation only while it is still the
# cheaper choice). Dated snapshots are covered by prefix matching, so they need no rows. Cached-input and batch discounts
# are not modelled: every token is billed at the standard input/output rate.
MODEL_PRICING_USD_PER_1M: dict[str, PriceRow] = {
    # OpenAI chat: flagship / balanced / cheap. gpt-6.1-terra ($1/$5) and gpt-6.1-sol ($2/$10) undercut gpt-5.6-terra ($2/$12).
    # Prompts over 272K input tokens pay the long-context rates.
    "gpt-6-astra": tiered((272_000, 10.00, 50.00), (None, 20.00, 75.00)),
    "gpt-6.1-sol": tiered((272_000, 2.00, 10.00), (None, 4.00, 15.00)),
    "gpt-6.1-terra": tiered((272_000, 1.00, 5.00), (None, 2.00, 7.50)),
    "gpt-6-luna": tiered((272_000, 0.10, 0.50), (None, 0.20, 0.75)),
    # OpenAI embeddings (no newer generation has been released).
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    # Anthropic. Claude Haiku 5.5 (2026-10-07) replaces Haiku 4.5 ($1/$5) and is priced by prompt length: over 100,000 tokens, 5x.
    "claude-fable-5-1": {"input": 10.00, "output": 50.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": tiered((100_000, 0.10, 0.50), (None, 0.50, 2.50)),
    "mock-llm": {"input": 0.0, "output": 0.0},
    "hashing-embedding": {"input": 0.0, "output": 0.0},
}

# Models that still have a price row but are deprecated or superseded. Must stay empty: delete such rows instead
# (enforced by tests/test_cost_tracker.py; the set exists to document the rule).
DEPRECATED_MODELS: frozenset[str] = frozenset()

# Refs with these prefixes are not in the table above: local models are free, and an OpenAI-compatible endpoint's prices are
# unknown, so it costs $0 unless `pricing:` has a row for it (keyed by the full ref, `<endpoint>/<model>`, or `<model>`).
FREE_BY_DEFAULT_PREFIXES = ("local:", "openai_compatible:")


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
    # Typed decision calls (`decision_gate`): input tokens sent to the decision model and what they cost.
    decision_input_tokens: int = 0
    decision_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return (
            self.embedding_cost + self.llm_cost + self.judge_cost + self.rerank_cost + self.query_rewrite_cost + self.tool_cost + self.decision_cost
        )

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
            decision_input_tokens=self.decision_input_tokens + other.decision_input_tokens,
            decision_cost=self.decision_cost + other.decision_cost,
        )

    def minus(self, other: CostBreakdown) -> CostBreakdown:
        """Field-wise difference; used to find cost that a trace does not account for (may be negative)."""
        return CostBreakdown(**{name: getattr(self, name) - getattr(other, name) for name in type(self).model_fields})

    def as_dict(self) -> dict[str, float | int]:
        data = self.model_dump()
        data["total_cost"] = self.total_cost
        return data


_lock = threading.Lock()
_pricing_overrides: dict[str, PriceRow] = {}
_unknown_priced_models: set[str] = set()


def _normalise(price: dict[str, Any]) -> PriceRow:
    row: PriceRow = {"input": float(price.get("input") or 0.0), "output": float(price.get("output") or 0.0)}
    if price.get("tiers"):
        row["tiers"] = [_normalise(tier) | ({"max_prompt_tokens": int(tier["max_prompt_tokens"])} if tier.get("max_prompt_tokens") is not None else {}) for tier in price["tiers"]]
    return row


def register_pricing(overrides: dict[str, dict[str, Any]]) -> None:
    """Add or replace per-model prices (USD per 1M tokens, optionally with `tiers`) for the current process.

    Overrides win over the built-in table. Use `clear_pricing_overrides()` to drop them.
    """
    with _lock:
        for model, price in overrides.items():
            _pricing_overrides[model] = _normalise(price)


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


def _match(table: dict[str, PriceRow], model: str) -> PriceRow | None:
    # Exact match first, then the longest registered name that prefixes a dated snapshot
    # (e.g. "gpt-6-luna-2026-05-18" -> "gpt-6-luna").
    if model in table:
        return table[model]
    candidates = [name for name in table if model.startswith(name + "-")]
    return table[max(candidates, key=len)] if candidates else None


def _lookup_pricing(model: str) -> PriceRow | None:
    """Price row for `model` (a bare name or a `provider:model` ref), or None. Overrides beat the built-in table."""
    if model.startswith(FREE_BY_DEFAULT_PREFIXES):
        names = [model]
        if model.startswith("openai_compatible:"):
            endpoint_model = model.split(":", 1)[1]
            names += [endpoint_model, endpoint_model.partition("/")[2]]
        # Only overrides apply: the OpenAI table must not price a proxy's or local server's model of the same name.
        for name in names:
            hit = _match(_pricing_overrides, name)
            if hit is not None:
                return hit
        return {"input": 0.0, "output": 0.0}
    if model.startswith("openrouter:"):
        return _lookup_openrouter(model)
    names = [model]
    provider, _, bare = model.partition(":")
    if bare and provider in ("openai", "anthropic"):
        names.append(bare)
    for table in (_pricing_overrides, MODEL_PRICING_USD_PER_1M):
        for name in names:
            hit = _match(table, name)
            if hit is not None:
                return hit
    return None


def _lookup_openrouter(model: str) -> PriceRow | None:
    """An `openrouter:` ref: a `pricing:` row (full ref or bare model id), else the OpenRouter catalog. Exact names only: on
    OpenRouter a suffix is another model (`deepseek-v4.1-flash-1017`), not a dated snapshot of the same one."""
    from ragbench.models.openrouter_catalog import model_info

    bare = model.split(":", 1)[1]
    for name in (model, bare):
        if name in _pricing_overrides:
            return _pricing_overrides[name]
    info = model_info(bare)
    return info.price if info is not None else None


def price_known(model: str) -> bool:
    """Whether `model` has a price (a built-in row, an override, or a model that is free by default such as `local:`)."""
    with _lock:
        return _lookup_pricing(model) is not None


def rates(row: PriceRow, prompt_tokens: float) -> tuple[float, float]:
    """`(input, output)` USD per 1M tokens for one request with `prompt_tokens` input tokens."""
    tiers = row.get("tiers")
    if not tiers:
        return row["input"], row["output"]
    tier = next((t for t in tiers if t.get("max_prompt_tokens") is None or prompt_tokens <= t["max_prompt_tokens"]), tiers[-1])
    return tier["input"], tier["output"]


def estimate_model_cost(model: str, input_tokens: int = 0, output_tokens: int = 0, *, calls: int = 1, flat: bool = False) -> float:
    """Price `input_tokens` / `output_tokens` at `model`'s rates. When they are the sum of `calls` requests, a tiered price is
    chosen by the mean prompt of a request, since each request is billed on its own. `flat` ignores tiers (the row's base rates)."""
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
    input_rate, output_rate = (pricing["input"], pricing["output"]) if flat else rates(pricing, input_tokens / max(1, calls))
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def estimate_decision_cost(model: str, input_tokens: int) -> float:
    """Cost of one typed-decision request: input tokens only. The Decisions API bills no output, cache or reasoning tokens
    (https://developers.openai.com/api/docs/guides/decisions, 2026-10-07: $0.10 per 1M input tokens for `gpt-6-luna`), so the
    model's price row is used for its `input` rate and its `output` rate is ignored. That page quotes a single input rate, with no
    long-context tier, so a tiered row's base rate applies."""
    return estimate_model_cost(model, input_tokens, 0, flat=True)
