from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from time import perf_counter

from ragbench.cache import active_cache, cache_key
from ragbench.models.cost import CostBreakdown, estimate_decision_cost
from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate

logger = logging.getLogger(__name__)


class CachedDecisionModel(DecisionModel):
    """Serves repeated requests from the run's disk cache instead of the paid API, under the same fairness rules as `CachedLLM`:
    a hit is charged at the current standalone price, the avoided spend is tracked separately, the original latency is replayed, and
    errors are never cached. Without an active cache this is a transparent pass-through."""

    def __init__(self, inner: DecisionModel):
        self.inner = inner
        self.model_name = inner.model_name

    @property
    def provider(self) -> str:  # type: ignore[override]
        return self.inner.provider

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        runtime = active_cache()
        if runtime is None or not runtime.config.llm:
            return self.inner.decide(text, predicates)
        key = cache_key(
            "decision", provider=self.inner.provider, model=self.model_name, text=text, predicates=[[p.name, p.instructions] for p in predicates]
        )
        stored = runtime.disk.get("decision", key)
        if stored is not None:
            try:
                payload = json.loads(stored)
                probabilities = {str(name): (None if value is None else float(value)) for name, value in payload["probabilities"].items()}
                tokens, latency_ms = int(payload["input_tokens"]), float(payload["latency_ms"])
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.warning("Ignoring unreadable decision cache entry: %s", exc)
            else:
                cost = CostBreakdown(decision_input_tokens=tokens, decision_cost=estimate_decision_cost(self.model_name, tokens))
                runtime.disk.record_saved("decision", cost.total_cost)
                return DecisionResult(probabilities, self.model_name, tokens, cost, latency_ms=latency_ms, cached=True)
        start = perf_counter()
        result = self.inner.decide(text, predicates)
        result.latency_ms = (perf_counter() - start) * 1000
        payload = {"probabilities": result.probabilities, "input_tokens": result.input_tokens, "latency_ms": result.latency_ms}
        runtime.disk.put("decision", key, json.dumps(payload).encode("utf-8"), meta={"model": self.model_name})
        return result
