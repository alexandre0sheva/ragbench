"""A chat model whose replies a test scripts: `respond(prompt)` returns the text, every prompt is kept, each call can be priced."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, LLMResult


class ScriptedLLM(LLM):
    def __init__(self, respond: Callable[[str], str], *, price_per_call: float = 0.0, model_name: str = "scripted-model"):
        self.model_name = model_name
        self.respond = respond
        self.price_per_call = price_per_call
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        prompt = "\n".join(m.get("content") or "" for m in messages)
        with self._lock:
            self.prompts.append(prompt)
        text = self.respond(prompt)
        return LLMResult(text, self.model_name, 10, 10, CostBreakdown(llm_cost=self.price_per_call), finish_reason="stop")
