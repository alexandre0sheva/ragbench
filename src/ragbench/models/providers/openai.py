"""OpenAI chat and embedding models. `openai_compatible` subclasses these with another `base_url`."""

from __future__ import annotations

import functools
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL
from ragbench.models.embeddings import EmbeddingModel, EmbeddingResult
from ragbench.models.errors import PermanentModelError, translate_openai_exception
from ragbench.models.llms import LLM, LLMResult, ToolCall
from ragbench.models.retry import call_with_retry
from ragbench.registry import EMBEDDERS, LLM_PROVIDERS
from ragbench.runtime.context import current_runtime, limiter_for
from ragbench.runtime.parallel import ordered_parallel_map
from ragbench.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class ModelCapabilities:
    """Request parameters a model family needs or rejects. Unknown models are assumed to accept everything."""

    temperature: bool = True
    # Sent with every request when set. GPT-6 models reason by default: hidden thinking would eat the `max_completion_tokens`
    # budget of short calls, and Chat Completions allows function calling only at `none` (developers.openai.com model pages).
    reasoning_effort: str | None = None


# Matched by prefix. Reasoning families reject `temperature`. For models not listed here, a rejected `temperature`
# or `reasoning_effort` is detected from the API error and remembered per client instance.
MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    "o1": ModelCapabilities(temperature=False),
    "o3": ModelCapabilities(temperature=False),
    "o4": ModelCapabilities(temperature=False),
    "gpt-6": ModelCapabilities(reasoning_effort="none"),
}

_DROPPABLE_PARAMS = ("temperature", "reasoning_effort")


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


def _rejected_param(exc: Exception, params: dict[str, Any]) -> str | None:
    """The optional request parameter a 400 response complains about, if it is one we can simply leave out."""
    import openai

    if not isinstance(exc, openai.BadRequestError):
        return None
    text = _error_text(exc)
    return next((name for name in _DROPPABLE_PARAMS if name in params and name in text), None)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw_arguments": raw}  # malformed JSON from the model: hand it to the caller rather than hiding it
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": raw}


def _content(message: dict[str, Any]) -> str:
    return message.get("content") or ""


class OpenAILLM(LLM):
    """OpenAI chat model. Retries/timeouts are owned here (the SDK's own retries are disabled)."""

    # The Chat Completions parameter for the output limit; OpenAI-compatible servers mostly know only the older `max_tokens`.
    max_tokens_param = "max_completion_tokens"

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATOR_MODEL,
        client: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        api_model: str | None = None,
        provider: str = "openai",
        limiter_key: str = "openai",
    ):
        self.model_name = model_name
        self.api_model = api_model or model_name  # the name sent on the wire; differs from `model_name` for endpoint refs
        self.provider = provider
        self.limiter_key = limiter_key
        capabilities = model_capabilities(self.api_model)
        self._unsupported: set[str] = set() if capabilities.temperature else {"temperature"}
        self._reasoning_effort = capabilities.reasoning_effort
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=0)
        self.client = client

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        limiter = limiter_for(self.limiter_key)
        estimated_tokens = 0
        if limiter is not None and limiter.limits_tokens:
            estimated_tokens = sum(estimate_tokens(_content(m), self.api_model) for m in messages) + int(max_tokens or 0)

        def attempt() -> LLMResult:
            # Every attempt, retries included, is one request against the provider's limits.
            with limiter.acquire(estimated_tokens) if limiter is not None else nullcontext():
                return self._attempt(messages, temperature, max_tokens, json_mode, tools)

        return call_with_retry(attempt)

    def _attempt(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int | None, json_mode: bool, tools: list[dict[str, Any]] | None) -> LLMResult:
        try:
            return self._generate_chat_completions(messages, temperature, max_tokens, json_mode, tools)
        except Exception as exc:
            # Only a model that cannot use Chat Completions may move to the Responses API;
            # any other failure must not trigger a second (billable) request.
            if _is_endpoint_unsupported(exc) and getattr(self.client, "responses", None) is not None:
                try:
                    return self._generate_responses(messages, temperature, max_tokens, json_mode, tools)
                except Exception as fallback_exc:
                    raise translate_openai_exception(fallback_exc) from fallback_exc
            translated = translate_openai_exception(exc)
            if translated is exc:
                raise
            raise translated from exc

    def _create_chat(self, params: dict[str, Any]) -> Any:
        while True:
            sendable = {key: value for key, value in params.items() if key not in self._unsupported}
            try:
                return self.client.chat.completions.create(**sendable)
            except Exception as exc:
                rejected = _rejected_param(exc, sendable)
                if rejected is None:
                    raise
                self._unsupported.add(rejected)
                logger.info("Model %s rejected `%s`; omitting it from now on.", self.api_model, rejected)

    def _generate_chat_completions(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int | None, json_mode: bool, tools: list[dict[str, Any]] | None) -> LLMResult:
        params: dict[str, Any] = {"model": self.api_model, "messages": messages, "temperature": temperature}
        if self._reasoning_effort:
            params["reasoning_effort"] = self._reasoning_effort
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if max_tokens:
            params[self.max_tokens_param] = max_tokens
        if tools:
            params["tools"] = tools
        response = self._create_chat(params)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=_parse_arguments(call.function.arguments))
            for call in (getattr(choice.message, "tool_calls", None) or [])
        ]
        prompt_tokens = int(response.usage.prompt_tokens) if response.usage else sum(estimate_tokens(_content(m), self.api_model) for m in messages)
        completion_tokens = int(response.usage.completion_tokens) if response.usage else estimate_tokens(text, self.api_model)
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
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )

    def _generate_responses(self, messages: list[dict[str, Any]], temperature: float, max_tokens: int | None, json_mode: bool, tools: list[dict[str, Any]] | None) -> LLMResult:
        if tools:
            raise PermanentModelError(f"{self.api_model} only supports the Responses API, where tool calling is not implemented")
        system_messages = [_content(m) for m in messages if m.get("role") == "system"]
        input_messages = [m for m in messages if m.get("role") != "system"]
        if json_mode:
            system_messages.append("Return JSON only.")
        params: dict[str, Any] = {
            "model": self.api_model,
            "input": input_messages,
            "instructions": "\n\n".join(system_messages) or None,
        }
        if "temperature" not in self._unsupported:
            params["temperature"] = temperature
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        response = self.client.responses.create(**params)
        text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        if not prompt_tokens:
            prompt_tokens = sum(estimate_tokens(_content(m), self.api_model) for m in messages)
        if not completion_tokens:
            completion_tokens = estimate_tokens(text, self.api_model)
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
            finish_reason="stop",
            raw=response,
        )


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 96,
        client: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        api_model: str | None = None,
        provider: str = "openai",
        limiter_key: str = "openai",
    ):
        self.model_name = model_name
        self.api_model = api_model or model_name
        self.batch_size = batch_size
        self.provider = provider
        self.limiter_key = limiter_key
        if client is None:
            from openai import OpenAI

            # Retries are owned by `call_with_retry`, so the SDK's own are disabled.
            client = OpenAI(timeout=timeout, max_retries=0)
        self.client = client

    def _embed_batch(self, batch: list[str]) -> Any:
        limiter = limiter_for(self.limiter_key)
        tokens = sum(estimate_tokens(text, self.api_model) for text in batch) if limiter is not None and limiter.limits_tokens else 0
        try:
            with limiter.acquire(tokens) if limiter is not None else nullcontext():
                return self.client.embeddings.create(model=self.api_model, input=batch)
        except Exception as exc:
            translated = translate_openai_exception(exc)
            if translated is exc:
                raise
            raise translated from exc

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), model=self.model_name, input_tokens=0, cost=CostBreakdown())
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        # Batches are independent requests; `evaluation.ingest_workers` of them may be in flight at once.
        responses = ordered_parallel_map(
            lambda batch: call_with_retry(functools.partial(self._embed_batch, batch)),
            batches,
            workers=current_runtime().ingest_workers,
            thread_name_prefix="ragbench-embed",
        )
        vectors: list[list[float]] = []
        input_tokens = 0
        for batch, response in zip(batches, responses, strict=True):
            vectors.extend([item.embedding for item in response.data])
            if getattr(response, "usage", None):
                input_tokens += int(response.usage.prompt_tokens)
            else:
                input_tokens += sum(estimate_tokens(text, self.api_model) for text in batch)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = np.divide(array, np.maximum(norms, 1e-12))
        cost = estimate_model_cost(self.model_name, input_tokens=input_tokens)
        return EmbeddingResult(
            vectors=array,
            model=self.model_name,
            input_tokens=input_tokens,
            cost=CostBreakdown(embedding_input_tokens=input_tokens, embedding_cost=cost),
        )


@LLM_PROVIDERS.register("openai")
def _openai_llm(model: str, *, providers: dict[str, Any]) -> LLM:
    return OpenAILLM(model)


@EMBEDDERS.register("openai")
def _openai_embedder(model: str, *, providers: dict[str, Any]) -> EmbeddingModel:
    return OpenAIEmbeddingModel(model)
