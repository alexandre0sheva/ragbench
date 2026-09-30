"""Model refs (`provider:model`), which models a config uses, and whether a run can reach them.

A ref without a recognised provider prefix is an OpenAI model name (`gpt-6-luna` == `openai:gpt-6-luna`), so every
config written before multi-provider support keeps working.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL
from ragbench.registry import EMBEDDERS, LLM_PROVIDERS, UnknownComponentError
from ragbench.utils.env import has_api_key

if TYPE_CHECKING:
    from ragbench.config.schema import ExperimentConfig, ProviderConfig

ModelKind = Literal["llm", "embedding"]

# Hosted APIs are live only when their key is set; endpoints and local models need none.
HOSTED_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# Optional SDKs a provider imports lazily; warmed up on the main thread before workers start (see `warm_up_imports`).
PROVIDER_MODULES = {"anthropic": ["anthropic"], "local": ["sentence_transformers"]}


def _provider_names() -> set[str]:
    import ragbench.models.providers  # noqa: F401  (registers the built-in providers)

    return set(LLM_PROVIDERS.names()) | set(EMBEDDERS.names())


def parse_model_ref(ref: str) -> tuple[str, str]:
    """Split `provider:model` into `(provider, model)`; a bare name is an OpenAI model.

    Only the first colon separates the provider, so `openai_compatible:ollama/llama3.1:8b` keeps its tag. A prefix that is
    not a provider is part of the model name (`llama3:8b`), unless it is one typo away from one (`anthropics:`).
    """
    providers = _provider_names()
    if ":" in ref:
        prefix, model = ref.split(":", 1)
        if prefix in providers:
            if not model.strip():
                raise ValueError(f"Model ref {ref!r} has an empty model name after '{prefix}:'")
            return prefix, model
        close = difflib.get_close_matches(prefix, sorted(providers), n=1, cutoff=0.75)
        if close:
            raise UnknownComponentError(f"Unknown provider {prefix!r} in model ref {ref!r}. Did you mean {close[0]!r}? Available: {', '.join(sorted(providers))}")
    if not ref.strip():
        raise ValueError("Model ref is empty")
    return "openai", ref


def split_endpoint(model: str) -> tuple[str, str]:
    """`ollama/llama3.1:8b` -> `("ollama", "llama3.1:8b")` for an `openai_compatible:` ref."""
    endpoint, _, name = model.partition("/")
    return endpoint, name


def validate_model_ref(ref: str, kind: ModelKind, providers: dict[str, ProviderConfig]) -> None:
    """Raise `ValueError` when `ref` cannot name a `kind` model given the configured `providers:` endpoints."""
    import ragbench.models.providers  # noqa: F401

    provider, model = parse_model_ref(ref)
    registry, other = (LLM_PROVIDERS, "embedding") if kind == "llm" else (EMBEDDERS, "chat")
    label = "chat" if kind == "llm" else "embedding"
    if provider not in registry:
        raise ValueError(f"`{provider}:` provides {other} models only; it cannot be used for {label} models (got {ref!r})")
    if provider == "openai_compatible":
        endpoint, name = split_endpoint(model)
        if not endpoint or not name:
            raise ValueError(f"{ref!r} must look like openai_compatible:<endpoint>/<model>")
        if endpoint not in providers:
            defined = ", ".join(sorted(providers)) or "none"
            raise ValueError(f"{ref!r} uses unknown endpoint {endpoint!r}; define it under `providers:` (defined: {defined})")


@dataclass
class ModelRefs:
    llm: list[str] = field(default_factory=list)
    embedding: list[str] = field(default_factory=list)


def _uses_embeddings(system_type: str, chunker: dict[str, Any] | None = None) -> bool:
    """Whether a system embeds text: its vector index does, and so does a `semantic` chunker (which embeds sentences)."""
    import ragbench.documents.chunkers  # noqa: F401  (registers the built-in chunkers)
    import ragbench.rag_systems  # noqa: F401
    from ragbench.registry import CHUNKERS, SYSTEMS

    spec = getattr(SYSTEMS.mapping.get(system_type), "spec", None)
    if spec is not None and "vector_store" in spec.options.model_fields:
        return True
    chunker_name = str((chunker or {}).get("type") or "")
    return chunker_name in CHUNKERS and bool(getattr(CHUNKERS.get(chunker_name), "needs_embedder", False))


def collect_model_refs(config: ExperimentConfig) -> ModelRefs:
    """Every model ref the run will actually call: each system's generator, embedder (only systems that embed), and the judge."""
    refs = ModelRefs()
    for system in config.systems:
        _add(refs.llm, str(system.models.get("generator", DEFAULT_GENERATOR_MODEL)))
        if "embedding" in system.models or _uses_embeddings(system.type, system.chunker):
            _add(refs.embedding, str(system.models.get("embedding", DEFAULT_EMBEDDING_MODEL)))
    if config.evaluation.judge_enabled:
        _add(refs.llm, config.evaluation.judge_model)
    return refs


def _add(items: list[str], ref: str) -> None:
    if ref not in items:
        items.append(ref)


def provider_reachable(provider: str) -> bool:
    env_var = HOSTED_KEY_ENV.get(provider)
    return True if env_var is None else has_api_key(env_var)


def _all_refs(config: ExperimentConfig) -> list[str]:
    refs = collect_model_refs(config)
    return [*refs.llm, *refs.embedding]


def resolve_run_mode(config: ExperimentConfig, force_mock: bool) -> Literal["mock", "live"]:
    """`live` when at least one model the run calls can be reached (a key is set for a hosted API, or it is a local/endpoint model)."""
    if force_mock:
        return "mock"
    return "live" if any(provider_reachable(parse_model_ref(ref)[0]) for ref in _all_refs(config)) else "mock"


def missing_credentials(config: ExperimentConfig) -> list[str]:
    """Problems that would make a live run fail on its first request or, worse, quietly mix mock and real models."""
    problems: list[str] = []
    for ref in _all_refs(config):
        provider, model = parse_model_ref(ref)
        env_var = HOSTED_KEY_ENV.get(provider)
        if env_var is not None and not has_api_key(env_var):
            problems.append(f"model {ref!r} needs {env_var}, which is not set")
        elif provider == "openai_compatible":
            endpoint = config.providers.get(split_endpoint(model)[0])
            if endpoint is not None and endpoint.api_key_env and not has_api_key(endpoint.api_key_env):
                problems.append(f"model {ref!r} needs {endpoint.api_key_env}, which is not set")
    return problems


def warm_up_modules(config: ExperimentConfig) -> list[str]:
    """Optional SDKs the run's providers import lazily."""
    modules: list[str] = []
    for ref in _all_refs(config):
        for module in PROVIDER_MODULES.get(parse_model_ref(ref)[0], []):
            if module not in modules:
                modules.append(module)
    return modules
