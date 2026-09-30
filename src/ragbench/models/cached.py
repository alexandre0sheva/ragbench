from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from ragbench.cache import active_cache, cache_key
from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.llms import LLM, LLMResult

logger = logging.getLogger(__name__)


class CachedLLM(LLM):
    """Serves repeated temperature-0 requests from the run's disk cache instead of the paid API.

    Fairness rules: a hit is charged at the *current* standalone price (so a system's cost does not depend on
    what was cached before), the avoided spend is tracked separately, and the latency of the original call is
    replayed as the reported latency (`LLMResult.latency_ms`, `cached=True`). Without an active cache runtime
    this is a transparent pass-through. Errors are never cached.
    """

    def __init__(self, inner: LLM):
        self.inner = inner
        self.model_name = inner.model_name

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        runtime = active_cache()
        temperature = kwargs.get("temperature") or 0
        if runtime is None or not runtime.config.llm or (temperature > 0 and not runtime.config.llm_nonzero_temperature):
            return self.inner.generate(messages, **kwargs)

        params = {key: value for key, value in kwargs.items() if value is not None and value is not False}
        params["temperature"] = temperature
        key = cache_key("llm", provider=getattr(self.inner, "provider", "openai"), model=self.model_name, messages=messages, params=params)

        stored = runtime.disk.get("llm", key)
        if stored is not None:
            try:
                payload = json.loads(stored)
                prompt_tokens, completion_tokens = int(payload["prompt_tokens"]), int(payload["completion_tokens"])
                text, latency_ms = str(payload["text"]), float(payload["latency_ms"])
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Ignoring unreadable LLM cache entry: %s", exc)
            else:
                cost = CostBreakdown(
                    llm_prompt_tokens=prompt_tokens,
                    llm_completion_tokens=completion_tokens,
                    llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
                )
                runtime.disk.record_saved("llm", cost.total_cost)
                return LLMResult(text, self.model_name, prompt_tokens, completion_tokens, cost, latency_ms=latency_ms, cached=True)

        start = perf_counter()
        result = self.inner.generate(messages, **kwargs)
        result.latency_ms = (perf_counter() - start) * 1000
        payload = {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
        }
        runtime.disk.put("llm", key, json.dumps(payload).encode("utf-8"), meta={"model": self.model_name})
        return result
