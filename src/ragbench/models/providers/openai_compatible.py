"""Any server that speaks the OpenAI API (Ollama, vLLM, LM Studio, OpenRouter, Together, Gemini's compatible endpoint, ...).

Refs look like `openai_compatible:<endpoint>/<model>`, where `<endpoint>` names an entry of the top-level `providers:` config.
Such models cost $0 unless `pricing:` says otherwise (see `models/cost.py`).
"""

from __future__ import annotations

import os
from typing import Any

from ragbench.config.schema import ProviderConfig
from ragbench.models.embeddings import EmbeddingModel
from ragbench.models.errors import ModelInitError
from ragbench.models.llms import LLM
from ragbench.models.providers.openai import DEFAULT_TIMEOUT_S, OpenAIEmbeddingModel, OpenAILLM
from ragbench.models.refs import split_endpoint
from ragbench.registry import EMBEDDERS, LLM_PROVIDERS
from ragbench.utils.env import has_api_key


def _resolve(model: str, providers: dict[str, ProviderConfig]) -> tuple[str, ProviderConfig, str]:
    endpoint, api_model = split_endpoint(model)
    config = providers.get(endpoint)
    if config is None:
        defined = ", ".join(sorted(providers)) or "none"
        raise ModelInitError(f"openai_compatible endpoint {endpoint!r} is not defined under `providers:` (defined: {defined})")
    return endpoint, config, api_model


def _client(config: ProviderConfig) -> Any:
    from openai import OpenAI

    api_key = None
    if config.api_key_env:
        if not has_api_key(config.api_key_env):
            raise ModelInitError(f"{config.api_key_env} is not set (it holds the API key for {config.base_url})")
        api_key = os.environ[config.api_key_env]
    # The SDK insists on a key; local servers ignore whatever they are sent.
    return OpenAI(base_url=config.base_url, api_key=api_key or "no-key-needed", timeout=DEFAULT_TIMEOUT_S, max_retries=0)


def _provider_id(endpoint: str, config: ProviderConfig) -> str:
    # Keys the persistent cache; the base URL is included so repointing an endpoint never serves stale answers.
    return f"openai_compatible:{endpoint}@{config.base_url}"


def _limiter_key(endpoint: str) -> str:
    return f"openai_compatible:{endpoint}"  # selects the endpoint's own `limits:`


class OpenAICompatibleLLM(OpenAILLM):
    max_tokens_param = "max_tokens"

    def __init__(self, endpoint: str, config: ProviderConfig, api_model: str, client: Any | None = None):
        super().__init__(
            f"openai_compatible:{endpoint}/{api_model}",
            client or _client(config),
            api_model=api_model,
            provider=_provider_id(endpoint, config),
            limiter_key=_limiter_key(endpoint),
        )


class OpenAICompatibleEmbeddingModel(OpenAIEmbeddingModel):
    def __init__(self, endpoint: str, config: ProviderConfig, api_model: str, client: Any | None = None):
        super().__init__(
            f"openai_compatible:{endpoint}/{api_model}",
            client=client or _client(config),
            api_model=api_model,
            provider=_provider_id(endpoint, config),
            limiter_key=_limiter_key(endpoint),
        )


@LLM_PROVIDERS.register("openai_compatible")
def _llm(model: str, *, providers: dict[str, ProviderConfig]) -> LLM:
    endpoint, config, api_model = _resolve(model, providers)
    return OpenAICompatibleLLM(endpoint, config, api_model)


@EMBEDDERS.register("openai_compatible")
def _embedder(model: str, *, providers: dict[str, ProviderConfig]) -> EmbeddingModel:
    endpoint, config, api_model = _resolve(model, providers)
    return OpenAICompatibleEmbeddingModel(endpoint, config, api_model)
