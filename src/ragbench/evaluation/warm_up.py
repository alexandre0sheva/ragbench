"""Libraries a run imports up front, on the main thread, because worker threads would otherwise import them lazily while live API calls
are in flight (see `runtime.warm_up_imports`)."""

from __future__ import annotations

import sys
from collections.abc import Iterator

from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.models.refs import warm_up_modules
from ragbench.registry import SYSTEMS, VECTOR_BACKENDS


def _with_routes(systems: list[SystemConfig]) -> Iterator[SystemConfig]:
    """Every configured system and, for an `adaptive` one, the pipelines of its routes (each built in the same process)."""
    for system in systems:
        yield system
        routes = system.retrieval.get("routes") if system.type == "adaptive" else None
        if isinstance(routes, dict):
            yield from _with_routes([route if isinstance(route, SystemConfig) else SystemConfig.model_validate(route) for route in routes.values()])


def modules_to_warm_up(config: ExperimentConfig, mode: str) -> list[str]:
    modules = ["httpx", "openai"]
    for system_config in _with_routes(config.systems):
        spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
        if spec is not None and "vector_store" in spec.options.model_fields:
            backend = system_config.retrieval.get("vector_store", spec.options.model_fields["vector_store"].default)
            modules.extend(module for module in VECTOR_BACKENDS.get(backend).import_modules if module not in modules)
    if mode == "live":  # mock runs never load a reranking or token-embedding model
        modules.extend(_reranker_modules(config))
        modules.extend(_system_modules(config))
    return [*modules, *warm_up_modules(config)]


def _system_modules(config: ExperimentConfig) -> list[str]:
    """Libraries a configured system class imports lazily for a model of its own (its `import_modules`, e.g. torch for `late_interaction`)."""
    modules: list[str] = []
    for system_config in _with_routes(config.systems):
        cls = SYSTEMS.mapping.get(system_config.type)
        modules.extend(module for module in getattr(cls, "import_modules", ()) if module not in modules)
    return modules


def _reranker_modules(config: ExperimentConfig) -> list[str]:
    """Libraries the configured rerankers import lazily (e.g. sentence-transformers for `cross_encoder`)."""
    import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
    from ragbench.registry import RERANKERS

    modules: list[str] = []
    for system_config in _with_routes(config.systems):
        spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
        if spec is None or "reranker" not in spec.options.model_fields:
            continue
        name = system_config.retrieval.get("reranker", spec.options.model_fields["reranker"].default)
        modules.extend(module for module in getattr(RERANKERS.get(name), "import_modules", ()) if module not in modules)
    return modules


TORCH_LIBRARIES = ("torch", "transformers", "sentence_transformers")


def _platform() -> str:
    return sys.platform


def openmp_conflict(config: ExperimentConfig, mode: str) -> str | None:
    """Why this run cannot start, when it would load torch and faiss into one macOS process; None when it can.

    Both ship their own OpenMP runtime on macOS: the second one to load aborts the process ("OMP: Error #15"), or, with
    `KMP_DUPLICATE_LIB_OK` set (scikit-learn sets it on import, so every run has it), the process crashes later instead. Mock runs load no
    torch (hashing token vectors, no reranking model, no local embeddings).
    """
    if mode != "live" or _platform() != "darwin":
        return None
    # What each system brings itself (with its routes), beyond what any run of this config loads (its judge and other shared models).
    shared = set(modules_to_warm_up(config.model_copy(update={"systems": []}), mode))
    libraries = {system.resolved_name: set(modules_to_warm_up(config.model_copy(update={"systems": [system]}), mode)) - shared for system in config.systems}
    torch_users = [name for name, modules in libraries.items() if modules & set(TORCH_LIBRARIES)]
    faiss_users = [name for name, modules in libraries.items() if "faiss" in modules]
    if not faiss_users or not (torch_users or shared & set(TORCH_LIBRARIES)):
        return None
    return (
        f"On macOS torch ({', '.join(torch_users) or 'a local model'}) and faiss ({', '.join(faiss_users)}) cannot run in one process: each brings "
        "its own OpenMP runtime, and the run would abort (\"OMP: Error #15\") or crash."
    )


def openmp_hint() -> str:
    """What to do about an `openmp_conflict`."""
    return (
        "Give the faiss systems `vector_store: numpy` (exact, no extra), or run the torch systems and the faiss systems from separate configs."
    )
