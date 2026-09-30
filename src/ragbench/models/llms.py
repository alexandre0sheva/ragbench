from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.errors import ModelInitError, translate_openai_exception
from ragbench.models.retry import call_with_retry
from ragbench.runtime.context import limiter_for
from ragbench.utils.env import has_openai_key
from ragbench.utils.text import estimate_tokens, tokenize

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: CostBreakdown
    raw: Any | None = None
    # Wall-clock time of the underlying call; for a cache hit, the time the original call took (replayed).
    latency_ms: float | None = None
    cached: bool = False


class LLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        raise NotImplementedError


class MockLLM(LLM):
    def __init__(self):
        self.model_name = "mock-llm"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        prompt = "\n".join(m.get("content", "") for m in messages)
        if kwargs.get("json_mode"):
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
        return LLMResult(text=text, model=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=CostBreakdown())

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


@dataclass(frozen=True)
class ModelCapabilities:
    """Request parameters a model family accepts. Unknown models are assumed to accept everything."""

    temperature: bool = True


# Matched by prefix. Reasoning families reject `temperature`. For models not listed here, a
# rejected `temperature` is detected from the API error and remembered per client instance.
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "o1": ModelCapabilities(temperature=False),
    "o3": ModelCapabilities(temperature=False),
    "o4": ModelCapabilities(temperature=False),
}


def model_capabilities(model: str) -> ModelCapabilities:
    matches = [prefix for prefix in MODEL_CAPABILITIES if model.startswith(prefix)]
    return MODEL_CAPABILITIES[max(matches, key=len)] if matches else ModelCapabilities()


def _error_text(exc: Exception) -> str:
    return f"{getattr(exc, 'message', '')} {exc}".lower()


def _is_endpoint_unsupported(exc: Exception) -> bool:
    """True only when the API says this model cannot be used with Chat Completions."""
    import openai

    if not isinstance(exc, openai.BadRequestError | openai.NotFoundError):
        return False
    text = _error_text(exc)
    return "v1/responses" in text or "not a chat model" in text or "not supported in the v1/chat/completions" in text


def _is_temperature_rejected(exc: Exception) -> bool:
    import openai

    return isinstance(exc, openai.BadRequestError) and "temperature" in _error_text(exc)


class OpenAILLM(LLM):
    """OpenAI chat model. Retries/timeouts are owned here (the SDK's own retries are disabled)."""

    def __init__(self, model_name: str = "gpt-5.4-nano", client: Any | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.model_name = model_name
        self._send_temperature = model_capabilities(model_name).temperature
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=0)
        self.client = client

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        limiter = limiter_for("openai")
        estimated_tokens = 0
        if limiter is not None and limiter.limits_tokens:
            estimated_tokens = sum(estimate_tokens(m.get("content", ""), self.model_name) for m in messages) + int(kwargs.get("max_tokens") or 0)

        def attempt() -> LLMResult:
            # Every attempt, retries included, is one request against the provider's limits.
            with limiter.acquire(estimated_tokens) if limiter is not None else nullcontext():
                return self._attempt(messages, **kwargs)

        return call_with_retry(attempt)

    def _attempt(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        try:
            return self._generate_chat_completions(messages, **kwargs)
        except Exception as exc:
            # Only a model that cannot use Chat Completions may move to the Responses API;
            # any other failure must not trigger a second (billable) request.
            if _is_endpoint_unsupported(exc) and getattr(self.client, "responses", None) is not None:
                try:
                    return self._generate_responses(messages, **kwargs)
                except Exception as fallback_exc:
                    raise translate_openai_exception(fallback_exc) from fallback_exc
            translated = translate_openai_exception(exc)
            if translated is exc:
                raise
            raise translated from exc

    def _create_chat(self, params: dict[str, Any]) -> Any:
        if self._send_temperature and "temperature" in params:
            try:
                return self.client.chat.completions.create(**params)
            except Exception as exc:
                if not _is_temperature_rejected(exc):
                    raise
                self._send_temperature = False
                logger.info("Model %s rejected `temperature`; omitting it from now on.", self.model_name)
        params = {key: value for key, value in params.items() if key != "temperature"}
        return self.client.chat.completions.create(**params)

    def _generate_chat_completions(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        params: dict[str, Any] = {"model": self.model_name, "messages": messages, "temperature": kwargs.get("temperature", 0)}
        if kwargs.get("json_mode"):
            params["response_format"] = {"type": "json_object"}
        if kwargs.get("max_tokens"):
            params["max_completion_tokens"] = kwargs["max_tokens"]
        response = self._create_chat(params)
        text = response.choices[0].message.content or ""
        prompt_tokens = int(response.usage.prompt_tokens) if response.usage else sum(estimate_tokens(m["content"], self.model_name) for m in messages)
        completion_tokens = int(response.usage.completion_tokens) if response.usage else estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=CostBreakdown(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
            ),
            raw=response,
        )

    def _generate_responses(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
        input_messages = [m for m in messages if m.get("role") != "system"]
        if kwargs.get("json_mode"):
            system_messages.append("Return JSON only.")
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": input_messages,
            "instructions": "\n\n".join(system_messages) or None,
        }
        if self._send_temperature:
            params["temperature"] = kwargs.get("temperature", 0)
        if kwargs.get("max_tokens"):
            params["max_output_tokens"] = kwargs["max_tokens"]
        response = self.client.responses.create(**params)
        text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        if not prompt_tokens:
            prompt_tokens = sum(estimate_tokens(m.get("content", ""), self.model_name) for m in messages)
        if not completion_tokens:
            completion_tokens = estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=CostBreakdown(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
            ),
            raw=response,
        )


def create_llm(model_name: str | None = None, force_mock: bool = False, strict: bool = True) -> LLM:
    """Build the generator LLM.

    No API key (or `force_mock`) selects the mock. With a key present a failing client
    constructor raises `ModelInitError` (`strict=True`, default) instead of silently
    producing mock scores; pass `strict=False` to restore the old fall-back-to-mock behaviour.
    """
    if force_mock or not has_openai_key():
        return MockLLM()
    try:
        from ragbench.models.cached import CachedLLM

        return CachedLLM(OpenAILLM(model_name or "gpt-5.4-nano"))
    except Exception as exc:
        if strict:
            raise ModelInitError(f"Could not create the OpenAI client for {model_name!r} although OPENAI_API_KEY is set: {exc}") from exc
        logger.warning("OpenAI client init failed (%s); falling back to the mock LLM.", exc)
        return MockLLM()
