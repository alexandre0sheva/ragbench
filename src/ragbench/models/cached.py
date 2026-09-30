from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from ragbench.cache import active_cache, cache_key
from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.llms import LLM, LLMResult, ToolCall

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

    @property
    def provider(self) -> str:  # type: ignore[override]
        return self.inner.provider

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        runtime = active_cache()
        temperature = temperature or 0
        if runtime is None or not runtime.config.llm or (temperature > 0 and not runtime.config.llm_nonzero_temperature):
            return self.inner.generate(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, tools=tools)

        call = {"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode, "tools": tools}

        params = {name: value for name, value in call.items() if value is not None and value is not False}
        key = cache_key("llm", provider=self.inner.provider, model=self.model_name, messages=messages, params=params)

        stored = runtime.disk.get("llm", key)
        if stored is not None:
            try:
                payload = json.loads(stored)
                prompt_tokens, completion_tokens = int(payload["prompt_tokens"]), int(payload["completion_tokens"])
                text, latency_ms = str(payload["text"]), float(payload["latency_ms"])
                tool_calls = [ToolCall(str(c["id"]), str(c["name"]), dict(c["arguments"])) for c in payload.get("tool_calls", [])]
                finish_reason = payload.get("finish_reason")
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Ignoring unreadable LLM cache entry: %s", exc)
            else:
                cost = CostBreakdown(
                    llm_prompt_tokens=prompt_tokens,
                    llm_completion_tokens=completion_tokens,
                    llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
                )
                runtime.disk.record_saved("llm", cost.total_cost)
                return LLMResult(
                    text, self.model_name, prompt_tokens, completion_tokens, cost, tool_calls, finish_reason, latency_ms=latency_ms, cached=True
                )

        start = perf_counter()
        result = self.inner.generate(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, tools=tools)
        result.latency_ms = (perf_counter() - start) * 1000
        payload = {
            "text": result.text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in result.tool_calls],
            "finish_reason": result.finish_reason,
        }
        runtime.disk.put("llm", key, json.dumps(payload).encode("utf-8"), meta={"model": self.model_name})
        return result
