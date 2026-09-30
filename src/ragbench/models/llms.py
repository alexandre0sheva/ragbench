from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ragbench.models.cost import CostBreakdown
from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
from ragbench.models.errors import ModelInitError
from ragbench.models.refs import parse_model_ref, provider_reachable
from ragbench.utils.text import estimate_tokens, tokenize

logger = logging.getLogger(__name__)

__all__ = ["LLM", "LLMResult", "MockLLM", "ToolCall", "create_llm", "parse_model_ref"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: CostBreakdown
    # Tools the model asked to call. Feed the turn back with `assistant_message()` plus one `role: "tool"` message per call.
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None  # "stop" | "length" | "tool_calls" | provider-specific
    raw: Any | None = None
    # Wall-clock time of the underlying call; for a cache hit, the time the original call took (replayed).
    latency_ms: float | None = None
    cached: bool = False

    def assistant_message(self) -> dict[str, Any]:
        """This turn as a chat message (OpenAI shape: tool-call arguments are a JSON string), ready to append to `messages`."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in self.tool_calls
            ]
        return message


class LLM(ABC):
    model_name: str
    # Identifies the backing service in persistent cache keys (two providers may serve the same model name).
    provider: str = "openai"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """One chat completion.

        `messages` are OpenAI-style chat messages; they may include `{"role": "assistant", "tool_calls": [...]}` and
        `{"role": "tool", "tool_call_id": ..., "content": ...}`. `tools` is a list of OpenAI function schemas
        (`{"type": "function", "function": {"name", "description", "parameters"}}`); providers translate as needed.
        """
        raise NotImplementedError


class MockLLM(LLM):
    def __init__(self):
        self.model_name = "mock-llm"

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        # `tools` is accepted and ignored: the mock never calls one (Task 18 adds a scripted policy).
        prompt = "\n".join(m.get("content") or "" for m in messages)
        if json_mode:
            text = json.dumps({"score": 3, "reasoning": "Mock JSON response."})
        elif "hypothetical passage" in prompt.lower():
            text = self._hypothetical_from_prompt(prompt)
        elif "Rewrite the question" in prompt or "search queries" in prompt:
            question = prompt.strip().splitlines()[-1]
            text = json.dumps({"queries": [question]})
        elif "summary" in prompt.lower() and "key entities" in prompt.lower():
            text = json.dumps({"summary": self._summarize(prompt), "key_entities": [], "hypothetical_questions": []})
        else:
            text = self._answer_from_prompt(prompt)
        prompt_tokens = estimate_tokens(prompt, self.model_name)
        completion_tokens = estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text, model=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=CostBreakdown(), finish_reason="stop"
        )

    @staticmethod
    def _hypothetical_from_prompt(prompt: str) -> str:
        # Echo the question's content words as a pseudo-passage so HyDE's
        # search probe stays on-topic for the hashing embedding model.
        question_match = re.search(r"Question:\s*(.*?)\s*$", prompt, flags=re.S | re.I)
        question = question_match.group(1) if question_match else prompt
        words = [t for t in tokenize(question) if len(t) > 2]
        topic = " ".join(words) or question
        return f"{topic}. This passage describes {topic} in detail. Reference information about {topic}."

    @staticmethod
    def _summarize(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])[:500]

    def _answer_from_prompt(self, prompt: str) -> str:
        context_match = re.search(r"Context:\s*(.*?)\n\nQuestion:", prompt, flags=re.S | re.I)
        question_match = re.search(r"Question:\s*(.*?)\s*$", prompt, flags=re.S | re.I)
        context = context_match.group(1) if context_match else prompt
        question = question_match.group(1) if question_match else ""
        question_tokens = {t for t in tokenize(question) if len(t) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            overlap = len(question_tokens.intersection(tokenize(sentence)))
            if overlap:
                scored.append((overlap, sentence))
        if not scored:
            return "I could not find the answer in the provided documents."
        scored.sort(key=lambda item: item[0], reverse=True)
        answer = " ".join(sentence for _, sentence in scored[:2])
        citations = sorted(set(re.findall(r"\[(doc_[A-Za-z0-9_-]+)\s*\|", answer + "\n" + context)))
        if citations and not re.search(r"\[doc_[A-Za-z0-9_-]+\]", answer):
            answer = f"{answer} " + " ".join(f"[{doc_id}]" for doc_id in citations[:3])
        return answer[:1200]


def create_llm(
    model_name: str | None = None, force_mock: bool = False, strict: bool = True, providers: dict[str, Any] | None = None
) -> LLM:
    """Build the (cache-wrapped) chat model for a model ref such as `gpt-6-luna` or `anthropic:claude-haiku-4-5`.

    `force_mock`, or a hosted provider (OpenAI, Anthropic) whose API key is not set, selects the mock. With credentials
    present a failing client constructor raises `ModelInitError` (`strict=True`, default) instead of silently producing
    mock scores; pass `strict=False` to restore the old fall-back-to-mock behaviour. A missing optional SDK raises
    `ImportError` naming the extra to install. `providers` maps `providers:` endpoint names to their config; when omitted
    the active run's endpoints are used.
    """
    import ragbench.models.providers  # noqa: F401  (registers the built-in providers)
    from ragbench.registry import LLM_PROVIDERS
    from ragbench.runtime.context import current_runtime

    ref = model_name or DEFAULT_GENERATOR_MODEL
    provider, model = parse_model_ref(ref)
    if force_mock or not provider_reachable(provider):
        return MockLLM()
    factory = LLM_PROVIDERS.get(provider)
    endpoints = providers if providers is not None else current_runtime().providers
    try:
        from ragbench.models.cached import CachedLLM

        return CachedLLM(factory(model, providers=endpoints))
    except ImportError:
        raise  # the message already names the extra to install
    except Exception as exc:
        if strict:
            raise ModelInitError(f"Could not create the {provider} client for {ref!r}: {exc}") from exc
        logger.warning("%s client init failed (%s); falling back to the mock LLM.", provider, exc)
        return MockLLM()
