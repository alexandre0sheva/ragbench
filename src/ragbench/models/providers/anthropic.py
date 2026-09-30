"""Claude models through the native Messages API (optional extra: `pip install 'ragbench[anthropic]'`).

RAGBench speaks OpenAI-style chat messages and tool schemas everywhere; this module translates them to and from Anthropic's
content blocks. The translation is plain functions over dicts so it is testable without the SDK.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.errors import MissingExtraError, PermanentModelError, translate_anthropic_exception
from ragbench.models.llms import LLM, LLMResult, ToolCall
from ragbench.models.retry import call_with_retry
from ragbench.registry import LLM_PROVIDERS
from ragbench.runtime.context import limiter_for
from ragbench.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 4096  # the Messages API requires `max_tokens`
# Thinking tokens count toward `max_tokens` even when they are not returned, so models that think by default get headroom
# on top of the caller's limit (platform.claude.com/docs/en/build-with-claude/effort).
THINKING_HEADROOM = 1024

JSON_INSTRUCTION = "Respond with a single valid JSON object and nothing else: no markdown fences, no commentary."

_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls", "max_tokens": "length"}


@dataclass(frozen=True)
class AnthropicCapabilities:
    effort: bool = False  # accepts `output_config.effort`; such models also think by default
    sampling: bool = False  # still honours `temperature` (current models take no sampling parameters; the SDK dropped the argument)


# Matched by prefix, longest first. Unknown models get neither parameter, which every model accepts.
ANTHROPIC_CAPABILITIES: dict[str, AnthropicCapabilities] = {
    "claude-fable-5": AnthropicCapabilities(effort=True),
    "claude-mythos-5": AnthropicCapabilities(effort=True),
    "claude-opus-5": AnthropicCapabilities(effort=True),
    "claude-sonnet-5": AnthropicCapabilities(effort=True),
    "claude-opus-4-6": AnthropicCapabilities(effort=True),
    "claude-opus-4-7": AnthropicCapabilities(effort=True),
    "claude-opus-4-8": AnthropicCapabilities(effort=True),
    "claude-sonnet-4-6": AnthropicCapabilities(effort=True),
    "claude-haiku-4-5": AnthropicCapabilities(sampling=True),
}


def anthropic_capabilities(model: str) -> AnthropicCapabilities:
    matches = [prefix for prefix in ANTHROPIC_CAPABILITIES if model.startswith(prefix)]
    return ANTHROPIC_CAPABILITIES[max(matches, key=len)] if matches else AnthropicCapabilities()


# --- translation -------------------------------------------------------------------------------------


def _loads_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool call arguments are not valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"tool call arguments must be a JSON object, got {raw!r}")
    return parsed


def to_anthropic_request(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]] | None]:
    """OpenAI-style `messages` / `tools` -> `(system, messages, tools)` for `messages.create`.

    System messages are joined into the top-level `system`. An assistant turn's `tool_calls` become `tool_use` blocks and
    each `role: "tool"` message a `tool_result` block. The API wants alternating roles, so adjacent messages of one role
    (all the results of a parallel tool call, plus any user text after them) are merged into one turn.
    """
    system_parts: list[str] = []
    turns: list[tuple[str, list[dict[str, Any]]]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if turns and turns[-1][0] == role:
            turns[-1][1].extend(blocks)
        else:
            turns.append((role, blocks))

    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if content:
                system_parts.append(str(content))
        elif role == "user":
            if content:
                push("user", [{"type": "text", "text": str(content)}])
        elif role == "assistant":
            blocks: list[dict[str, Any]] = [{"type": "text", "text": str(content)}] if content else []
            for call in message.get("tool_calls") or []:
                function = call["function"]
                blocks.append({"type": "tool_use", "id": call["id"], "name": function["name"], "input": _loads_arguments(function.get("arguments"))})
            if blocks:
                push("assistant", blocks)
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": str(content or "")}])
        else:
            raise ValueError(f"unsupported message role {role!r}")

    translated = [{"role": role, "content": blocks[0]["text"] if len(blocks) == 1 and blocks[0]["type"] == "text" else blocks} for role, blocks in turns]
    anthropic_tools = [_translate_tool(tool) for tool in tools] if tools else None
    return "\n\n".join(system_parts) or None, translated, anthropic_tools


def _translate_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    translated: dict[str, Any] = {"name": function["name"]}
    if function.get("description"):
        translated["description"] = function["description"]
    translated["input_schema"] = function.get("parameters") or {"type": "object", "properties": {}}
    return translated


def _extract_json(text: str) -> str:
    """Claude has no JSON mode: drop markdown fences / surrounding prose so `json.loads` sees just the object."""
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1)
    try:
        json.loads(stripped)
        return stripped
    except ValueError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    return stripped[start : end + 1] if 0 <= start < end else text


def build_client(timeout: float) -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise MissingExtraError("Claude models need the optional `anthropic` package. Install it with: pip install 'ragbench[anthropic]'") from exc
    # Retries are owned by `call_with_retry`, so the SDK's own are disabled. Reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL.
    return anthropic.Anthropic(timeout=timeout, max_retries=0)


class AnthropicLLM(LLM):
    provider = "anthropic"
    limiter_key = "anthropic"

    def __init__(self, model_name: str, client: Any | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.model_name = model_name
        self.api_model = model_name
        self._capabilities = anthropic_capabilities(model_name)
        self._temperature_rejected = False
        self.client = client if client is not None else build_client(timeout)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        system, turns, anthropic_tools = to_anthropic_request(messages, tools)
        if json_mode:
            system = f"{system}\n\n{JSON_INSTRUCTION}" if system else JSON_INSTRUCTION
        params: dict[str, Any] = {
            "model": self.api_model,
            "max_tokens": (max_tokens or DEFAULT_MAX_TOKENS) + (THINKING_HEADROOM if self._capabilities.effort else 0),
            "messages": turns,
        }
        if system:
            params["system"] = system
        if anthropic_tools:
            params["tools"] = anthropic_tools
        if self._capabilities.effort:
            params["output_config"] = {"effort": "low"}  # RAG calls are short and cheap; deep thinking would skew cost and latency
        limiter = limiter_for(self.limiter_key)
        estimated_tokens = 0
        if limiter is not None and limiter.limits_tokens:
            estimated_tokens = sum(estimate_tokens(str(m.get("content") or ""), self.api_model) for m in messages) + int(max_tokens or 0)

        def attempt() -> LLMResult:
            with limiter.acquire(estimated_tokens) if limiter is not None else nullcontext():
                return self._create(params, temperature, json_mode)

        return call_with_retry(attempt)

    def _create(self, params: dict[str, Any], temperature: float, json_mode: bool) -> LLMResult:
        while True:
            sendable = dict(params)
            sends_temperature = self._capabilities.sampling and not self._temperature_rejected
            if sends_temperature:
                sendable["extra_body"] = {"temperature": float(temperature)}
            try:
                response = self.client.messages.create(**sendable)
            except Exception as exc:
                translated = translate_anthropic_exception(exc)
                if sends_temperature and isinstance(translated, PermanentModelError) and "temperature" in str(translated).lower():
                    self._temperature_rejected = True
                    logger.info("Model %s rejected `temperature`; omitting it from now on.", self.api_model)
                    continue
                if translated is exc:
                    raise
                raise translated from exc
            return self._to_result(response, json_mode)

    def _to_result(self, response: Any, json_mode: bool) -> LLMResult:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
            # thinking / redacted_thinking blocks are not part of the answer
        text = "".join(text_parts)
        if json_mode:
            text = _extract_json(text)
        prompt_tokens, completion_tokens = int(response.usage.input_tokens), int(response.usage.output_tokens)
        stop_reason = getattr(response, "stop_reason", None)
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            # Cached-input discounts are not modelled (see the note above the price table): everything is billed at base rates.
            cost=CostBreakdown(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
            ),
            tool_calls=tool_calls,
            finish_reason=_STOP_REASONS.get(stop_reason, stop_reason) if stop_reason else None,
            raw=response,
        )


@LLM_PROVIDERS.register("anthropic")
def _anthropic_llm(model: str, *, providers: dict[str, Any]) -> LLM:
    return AnthropicLLM(model)
