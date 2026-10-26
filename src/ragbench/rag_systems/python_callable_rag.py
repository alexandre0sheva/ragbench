from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from ragbench.documents.schema import Document
from ragbench.rag_systems.base import AnswerResult
from ragbench.rag_systems.external_rag import ExternalRAG, ExternalReply, as_float, as_int
from ragbench.rag_systems.options import PythonCallableOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import Step
from ragbench.registry import SYSTEMS

_file_modules: dict[Path, ModuleType] = {}
_lock = threading.Lock()


def _module_from_file(path: Path) -> ModuleType:
    """Load a `.py` file once per process, so `target` and `ingest_target` in the same file share its state."""
    with _lock:
        if path not in _file_modules:
            name = f"ragbench_user_{abs(hash(path))}_{path.stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            _file_modules[path] = module
        return _file_modules[path]


def resolve_target(target: str) -> Callable[..., Any]:
    """`package.module:function` or `path/to/file.py:function` -> the function. Raises ValueError saying what could not be found."""
    where, _, name = target.rpartition(":")
    try:
        module = _module_from_file(Path(where).resolve()) if where.endswith(".py") or "/" in where or "\\" in where else importlib.import_module(where)
    except (ImportError, OSError) as exc:
        raise ValueError(f"python_callable target {target!r}: cannot import {where!r} ({type(exc).__name__}: {exc})") from None
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"python_callable target {target!r}: {where!r} has no function {name!r}")
    return function


@SYSTEMS.register("python_callable")
class PythonCallableRAG(ExternalRAG):
    spec = SystemSpec(
        type="python_callable",
        title="Python callable",
        summary="Your own pipeline, called in-process: a function `(question, top_k)` returning a dict (answer, passages, usage, cost) or an `AnswerResult`",
        best_for="Benchmarking your own code or a framework pipeline against the built-in systems without deploying it",
        cost_profile="low",
        latency_profile="medium",
        requires_llm=False,
        agentic=False,
        options=PythonCallableOptions,
        chunker=None,
        external=True,
    )
    options: PythonCallableOptions

    def __init__(self, config, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self._answer = resolve_target(self.options.target)
        self._ingest = resolve_target(self.options.ingest_target) if self.options.ingest_target else None

    def _ingest_external(self, documents: list[Document]) -> None:
        if self._ingest is not None:
            self._offline_guard(self._ingest, documents)

    def _offline_guard(self, function: Callable[..., Any], *args: Any) -> Any:
        """Call the user's function; in a live run, refuse a result made with RAGBench's offline mock models (the reference pipelines fall
        back to them without `RAGBENCH_REFERENCE_LIVE=1`), which would otherwise be scored and priced as the real pipeline's."""
        from ragbench.adapters import offline

        before = offline.calls()
        result = function(*args)
        if not self.force_mock and offline.calls() != before:
            raise RuntimeError(
                f"{self.name}: {self.options.target} used RAGBench's offline mock models in a live run. "
                "Set RAGBENCH_REFERENCE_LIVE=1 (and OPENAI_API_KEY) to run it for real, or run with --mock."
            )
        return result

    def _call(self, question: str, top_k: int) -> ExternalReply:
        result = self._offline_guard(self._answer, question, top_k)
        if isinstance(result, AnswerResult):
            chunks = result.retrieval_result.chunks
            return ExternalReply(
                answer=result.answer,
                contexts=[{"text": chunk.text, "doc_id": chunk.doc_id or None, "score": chunk.score} for chunk in chunks],
                prompt_tokens=result.cost.llm_prompt_tokens or result.token_usage.get("prompt_tokens"),
                completion_tokens=result.cost.llm_completion_tokens or result.token_usage.get("completion_tokens"),
                cost=result.cost,
                steps=list(result.steps),
            )
        if not isinstance(result, dict):
            raise TypeError(f"{self.options.target} returned {type(result).__name__}; return a dict with an `answer` (and optionally `contexts`, `usage`, `cost_usd`) or an AnswerResult")
        if "answer" not in result:
            raise ValueError(f"{self.options.target} returned a dict without an `answer`")
        usage = result.get("usage") or {}
        contexts = result.get("contexts")
        return ExternalReply(
            answer=str(result["answer"]),
            contexts=None if contexts is None else [item if isinstance(item, dict) else {"text": str(item)} for item in contexts],
            prompt_tokens=as_int(usage.get("prompt_tokens")),
            completion_tokens=as_int(usage.get("completion_tokens")),
            cost_usd=as_float(result.get("cost_usd")),
            steps=[step if isinstance(step, Step) else Step.model_validate(step) for step in result.get("steps") or []],
        )
