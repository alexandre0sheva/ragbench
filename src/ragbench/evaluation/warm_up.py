"""Libraries a run imports up front, on the main thread, because worker threads would otherwise import them lazily while live API calls
are in flight (see `runtime.warm_up_imports`)."""

from __future__ import annotations

from ragbench.config.schema import ExperimentConfig
from ragbench.models.refs import warm_up_modules
from ragbench.registry import SYSTEMS, VECTOR_BACKENDS


def modules_to_warm_up(config: ExperimentConfig, mode: str) -> list[str]:
    modules = ["httpx", "openai"]
    for system_config in config.systems:
        spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
        if spec is not None and "vector_store" in spec.options.model_fields:
            backend = system_config.retrieval.get("vector_store", spec.options.model_fields["vector_store"].default)
            modules.extend(module for module in VECTOR_BACKENDS.get(backend).import_modules if module not in modules)
    if mode == "live":  # mock runs never load a reranking model
        modules.extend(_reranker_modules(config))
    return [*modules, *warm_up_modules(config)]


def _reranker_modules(config: ExperimentConfig) -> list[str]:
    """Libraries the configured rerankers import lazily (e.g. sentence-transformers for `cross_encoder`)."""
    import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
    from ragbench.registry import RERANKERS

    modules: list[str] = []
    for system_config in config.systems:
        spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
        if spec is None or "reranker" not in spec.options.model_fields:
            continue
        name = system_config.retrieval.get("reranker", spec.options.model_fields["reranker"].default)
        modules.extend(module for module in getattr(RERANKERS.get(name), "import_modules", ()) if module not in modules)
    return modules
