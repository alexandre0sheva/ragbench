"""What the framework adapters share: a recorder for the retrieval and generation calls of one answer, and the reply `python_callable` expects.

An adapter turns a LangChain, LlamaIndex or Haystack pipeline into a `python_callable` target. While the pipeline answers one
question, the framework tells the adapter about each retrieval and each model call (callbacks, events, or a wrapped component). The
recorder keeps them per question, so concurrent questions never mix, and turns them into `retrieve` and `generate` steps with the
tokens and price of every model call. RAGBench cannot see the framework's embedding calls, so a step's embedding cost is not counted.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from ragbench.models import cost as pricing
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.trace import PREVIEW_CHARS, Step
from ragbench.utils.text import truncate

# Where a document says which file it came from, most specific first. RAGBench matches the value to a corpus document by id, path, file name or stem.
SOURCE_KEYS = ("doc_id", "file_path", "source", "path", "filename", "file_name", "url", "name", "source_id")


def source_ref(metadata: Mapping[str, Any] | None, *extra: Any) -> str | None:
    """The corpus document a framework document came from: the first of `SOURCE_KEYS` its metadata has, else the first non-empty `extra`."""
    for key in SOURCE_KEYS:
        value = (metadata or {}).get(key)
        if value:
            return str(value)
    return next((str(value) for value in extra if value), None)


def passage(text: str, ref: str | None, score: Any = None) -> dict[str, Any]:
    """One retrieved passage, in the shape of `python_callable`'s `contexts`."""
    return {"text": text, "doc_id": ref, "score": float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None}


@dataclass
class Call:
    kind: Literal["retrieve", "generate"]
    name: str
    latency_ms: float
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_preview: str | None = None
    output_preview: str | None = None
    error: str | None = None


class Recorder:
    """The calls one answer made. `model` is the model to price a call at when the framework does not report one (or reports one without a price)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.calls: list[Call] = []
        self.passages: list[dict[str, Any]] | None = None  # what the last retrieval returned, for pipelines that do not hand back their sources
        self._lock = threading.Lock()

    def add(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    def steps(self) -> list[Step]:
        with self._lock:
            return [self._step(call) for call in self.calls]

    def _step(self, call: Call) -> Step:
        model = next((name for name in (call.model, self.model) if name and pricing.price_known(name)), None)
        cost = CostBreakdown()
        if call.kind == "generate":
            usd = pricing.estimate_model_cost(model, call.prompt_tokens, call.completion_tokens) if model else 0.0
            cost = CostBreakdown(llm_prompt_tokens=call.prompt_tokens, llm_completion_tokens=call.completion_tokens, llm_cost=usd)
        metadata: dict[str, Any] = {"framework_model": call.model} if call.model else {}
        if call.error:
            metadata["error"] = call.error
        return Step(
            kind=call.kind,
            name=call.name,
            latency_ms=call.latency_ms,
            cost=cost,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            input_preview=truncate(call.input_preview, PREVIEW_CHARS) if call.input_preview else None,
            output_preview=truncate(call.output_preview, PREVIEW_CHARS) if call.output_preview else None,
            metadata=metadata,
        )


_CURRENT: ContextVar[Recorder | None] = ContextVar("ragbench_adapter_recorder", default=None)


def current_recorder() -> Recorder | None:
    """The recorder of the answer being produced in this thread, for framework hooks that are registered once and fire for every question."""
    return _CURRENT.get()


@contextmanager
def recording(model: str | None = None) -> Iterator[Recorder]:
    recorder = Recorder(model)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT.reset(token)


def reply(answer: str, contexts: list[dict[str, Any]] | None, recorder: Recorder) -> dict[str, Any]:
    """The dict a `python_callable` returns: answer, passages, and the recorded steps (their tokens add up to `usage`)."""
    steps = recorder.steps()
    usage = {"prompt_tokens": sum(step.prompt_tokens for step in steps), "completion_tokens": sum(step.completion_tokens for step in steps)}
    return {"answer": answer, "contexts": contexts, "steps": steps, "usage": usage}


def first_int(*values: Any) -> int:
    """The first value that is a number (frameworks name token counts differently), else 0."""
    return next((int(value) for value in values if isinstance(value, int | float) and not isinstance(value, bool)), 0)


Target = Callable[[str, int], dict[str, Any]]
