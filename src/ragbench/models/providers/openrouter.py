"""Any model on OpenRouter: `openrouter:<vendor>/<model>`, e.g. `openrouter:deepseek/deepseek-v4.1-flash` (needs `OPENROUTER_API_KEY`).

Built on the OpenAI client (OpenRouter speaks Chat Completions) with what a fair benchmark needs on top:

- **Billed cost.** Every response carries `usage.cost`, what OpenRouter charged; that is the call's cost. The price table
  (the OpenRouter catalog, see `models/openrouter_catalog.py`) is only the estimate, and the fallback if a response has none.
- **Reasoning off by default.** Recent open-weight models reason by default and thinking tokens bill as output, so every request
  sends `reasoning: {effort: none}`, or the lowest effort a model that must reason accepts; the `openrouter:` config section
  overrides it per model. The settings are recorded in the run manifest.
- **Provider pinning.** `openrouter.provider` (`order`, `allow_fallbacks`, `data_collection`) goes out as OpenRouter's
  `provider` parameter, so a run is served by the same upstream every time.
- **Capabilities from the catalog.** A model without `tools` in its `supported_parameters` reports `supports_tools = False`
  (tool-using agents then switch to `react_json`); one without JSON output is asked for JSON in the prompt instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ragbench.config.schema import OpenRouterConfig, OpenRouterModelSettings
from ragbench.models.errors import ModelInitError
from ragbench.models.llms import LLM
from ragbench.models.openrouter_catalog import OpenRouterModel, model_info
from ragbench.models.providers.openai import DEFAULT_TIMEOUT_S, OpenAILLM
from ragbench.registry import LLM_PROVIDERS
from ragbench.utils.env import has_api_key

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL_ENV = "OPENROUTER_BASE_URL"  # for tests and proxies
JSON_INSTRUCTION = "Respond with a single valid JSON object and nothing else: no markdown fences, no commentary."


def request_settings(model: str, settings: OpenRouterModelSettings, info: OpenRouterModel | None) -> dict[str, Any]:
    """The `reasoning` and `provider` parameters sent with every request for `model` (also what the run manifest records)."""
    chosen = settings.reasoning
    reasoning: dict[str, Any] = {}
    if chosen is not None and chosen.max_tokens is not None:
        reasoning["max_tokens"] = chosen.max_tokens
    else:
        effort = chosen.effort if chosen is not None and chosen.effort is not None else (info.lowest_effort() if info is not None else "none")
        if chosen is not None and chosen.effort is not None and info is not None and info.supported_efforts and effort not in info.supported_efforts:
            raise ModelInitError(f"openrouter:{model} does not accept reasoning effort {effort!r}; it accepts {', '.join(info.supported_efforts)}")
        if effort is not None:
            reasoning["effort"] = effort
    reasoning["exclude"] = chosen.exclude if chosen is not None else True
    out: dict[str, Any] = {"reasoning": reasoning}
    if settings.provider is not None:
        routing = settings.provider.model_dump(exclude_none=True)
        if routing.get("order") and "allow_fallbacks" not in routing:
            routing["allow_fallbacks"] = False  # an explicit order pins the provider unless fallbacks are asked for
        if routing:
            out["provider"] = routing
    return out


def manifest_settings(model_refs: list[str], config: OpenRouterConfig) -> dict[str, dict[str, Any]]:
    """`{ref: request settings}` for the `openrouter:` refs among `model_refs`, for `run_manifest.json`."""
    found: dict[str, dict[str, Any]] = {}
    for ref in model_refs:
        if ref.startswith("openrouter:"):
            model = ref.split(":", 1)[1]
            try:
                found[ref] = request_settings(model, config.for_model(model), model_info(model))
            except ModelInitError as exc:
                found[ref] = {"error": str(exc)}
    return found


def _client() -> Any:
    from openai import OpenAI

    if not has_api_key(API_KEY_ENV):
        raise ModelInitError(f"{API_KEY_ENV} is not set (it holds the API key for {BASE_URL})")
    # Retries are owned by `call_with_retry`, so the SDK's own are disabled.
    return OpenAI(base_url=os.environ.get(BASE_URL_ENV) or BASE_URL, api_key=os.environ[API_KEY_ENV], timeout=DEFAULT_TIMEOUT_S, max_retries=0)


class OpenRouterLLM(OpenAILLM):
    max_tokens_param = "max_tokens"
    bills_from_response = True

    def __init__(self, model: str, settings: OpenRouterModelSettings | None = None, client: Any | None = None, info: OpenRouterModel | None = None):
        self.info = info if info is not None else model_info(model)
        self.request_settings = request_settings(model, settings or OpenRouterModelSettings(), self.info)
        # The settings change the answers, so they are part of the provider id that keys the persistent cache.
        digest = hashlib.sha256(json.dumps(self.request_settings, sort_keys=True).encode()).hexdigest()[:12]
        super().__init__(f"openrouter:{model}", client or _client(), api_model=model, provider=f"openrouter@{digest}", limiter_key="openrouter")
        self._reasoning_effort = None  # reasoning goes out as OpenRouter's own `reasoning` object
        self.supports_tools = self.info.supports_tools if self.info is not None else True
        self.supports_json = self.info.supports_json if self.info is not None else True

    def _adapt_params(self, params: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(params)
        if "response_format" in adapted and not self.supports_json:
            del adapted["response_format"]
            adapted["messages"] = [{"role": "system", "content": JSON_INSTRUCTION}, *adapted["messages"]]
        adapted["extra_body"] = {**self.request_settings, "usage": {"include": True}}
        return adapted

    def _call_cost(self, response: Any, prompt_tokens: int, completion_tokens: int) -> float:
        billed = getattr(getattr(response, "usage", None), "cost", None)
        if isinstance(billed, int | float) and not isinstance(billed, bool):
            return float(billed)
        return super()._call_cost(response, prompt_tokens, completion_tokens)


@LLM_PROVIDERS.register("openrouter")
def _openrouter_llm(model: str, *, providers: dict[str, Any]) -> LLM:
    from ragbench.runtime.context import current_runtime

    if "/" not in model:
        raise ModelInitError(f"openrouter:{model} must name a model as <vendor>/<model>, e.g. openrouter:deepseek/deepseek-v4.1-flash")
    return OpenRouterLLM(model, current_runtime().openrouter.for_model(model))
