"""Per-question execution traces: an ordered list of steps with latency, tokens, and cost.

`BaseRAGSystem.answer_question` activates a fresh `Tracer` for every question (held in a `ContextVar`, so
concurrent questions, even on the same system instance, never see each other's steps). Systems record what
they do with `self.trace.step(kind, name)`; the evaluator persists the steps and rolls costs up by stage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult
from ragbench.utils.text import truncate, unique_preserve_order

StepKind = Literal["retrieve", "rerank", "llm", "tool", "embed", "route", "grade", "generate"]
UNTRACKED = "untracked"
# Columns of the per-stage rollups: every step kind, plus the residual bucket for cost no step accounts for.
STAGE_KEYS: tuple[str, ...] = ("retrieve", "rerank", "llm", "tool", "embed", "route", "grade", "generate", UNTRACKED)
PREVIEW_CHARS = 300
_EPSILON = 1e-12


class Step(BaseModel):
    """One thing a system did while answering a question."""

    kind: StepKind
    name: str
    latency_ms: float = 0.0
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_preview: str | None = None
    output_preview: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def step_to_dict(step: Step) -> dict[str, Any]:
    """JSON-ready form; the cost block includes `total_cost`."""
    data = step.model_dump(mode="json", exclude={"cost"})
    data["cost"] = step.cost.as_dict()
    return data


class StepHandle:
    """Lets the code inside `with tracer.step(...)` attach cost, tokens, and previews to the step being recorded."""

    def __init__(self) -> None:
        self.cost = CostBreakdown()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.input_preview: str | None = None
        self.output_preview: str | None = None
        self.metadata: dict[str, Any] = {}
        self.latency_override_ms: float | None = None
        self.kind: StepKind | None = None
        self.name: str | None = None

    def relabel(self, kind: StepKind, name: str | None = None) -> None:
        """Record this step under another kind (and name): for work whose role is only known once it finished (a turn that turned out to be the answer)."""
        self.kind, self.name = kind, name or self.name

    def set_cost(self, cost: CostBreakdown) -> None:
        self.cost = cost

    def set_tokens(self, prompt: int, completion: int) -> None:
        self.prompt_tokens, self.completion_tokens = int(prompt), int(completion)

    def set_input(self, text: str) -> None:
        self.input_preview = truncate(text, PREVIEW_CHARS)

    def set_output(self, text: str) -> None:
        self.output_preview = truncate(text, PREVIEW_CHARS)

    def set_meta(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def set_chunks(self, chunks: Iterable[Any], cost: CostBreakdown | None = None) -> None:
        """Record a retrieval outcome: how many chunks, from which documents, and what it cost."""
        chunk_list = list(chunks)
        docs = unique_preserve_order(chunk.doc_id for chunk in chunk_list)
        shown = ", ".join(docs[:6]) + (" …" if len(docs) > 6 else "")
        self.set_output(f"{len(chunk_list)} chunks from {len(docs)} docs: {shown}" if chunk_list else "0 chunks")
        if cost is not None:
            self.cost = cost

    def set_llm(self, result: LLMResult, messages: list[dict[str, Any]] | None = None, cost: CostBreakdown | None = None) -> None:
        """Record an LLM call. `cost` overrides `result.cost` when the system books it under another category."""
        self.cost = cost if cost is not None else result.cost
        self.set_tokens(result.prompt_tokens, result.completion_tokens)
        self.set_output(result.text)
        if result.cached:
            # A cache hit returns instantly; report the original call's latency so re-runs stay comparable.
            self.metadata["cached"] = True
            self.latency_override_ms = result.latency_ms
        if messages:
            self.set_input(str(messages[-1].get("content", "")))


class Tracer:
    """Collects the steps of one question. Not shared between questions or threads."""

    def __init__(self, record: bool = True) -> None:
        self.record = record
        self.steps: list[Step] = []
        # Replayed time (cached LLM calls) that did not really elapse; added to the answer's measured latency.
        self.latency_credit_ms = 0.0

    @contextmanager
    def step(self, kind: StepKind, name: str, **metadata: Any) -> Iterator[StepHandle]:
        handle = StepHandle()
        handle.metadata.update(metadata)
        start = perf_counter()
        try:
            yield handle
        except BaseException as exc:
            handle.metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.record:
                elapsed_ms = (perf_counter() - start) * 1000
                latency_ms = max(elapsed_ms, handle.latency_override_ms or 0.0)
                self.latency_credit_ms += latency_ms - elapsed_ms
                self.steps.append(
                    Step(
                        kind=handle.kind or kind,
                        name=handle.name or name,
                        latency_ms=latency_ms,
                        cost=handle.cost,
                        prompt_tokens=handle.prompt_tokens,
                        completion_tokens=handle.completion_tokens,
                        input_preview=handle.input_preview,
                        output_preview=handle.output_preview,
                        metadata=handle.metadata,
                    )
                )

    def add(self, step: Step) -> None:
        if self.record:
            self.steps.append(step)


# Shared, stateless stand-in used when nothing is tracing (e.g. `fetch_context` called directly).
NULL_TRACER = Tracer(record=False)
_ACTIVE: ContextVar[Tracer | None] = ContextVar("ragbench_active_tracer", default=None)


def current_tracer() -> Tracer:
    return _ACTIVE.get() or NULL_TRACER


@contextmanager
def activate(tracer: Tracer) -> Iterator[Tracer]:
    token = _ACTIVE.set(tracer)
    try:
        yield tracer
    finally:
        _ACTIVE.reset(token)


def sum_costs(costs: Iterable[CostBreakdown]) -> CostBreakdown:
    total = CostBreakdown()
    for cost in costs:
        total = total.plus(cost)
    return total


def reconcile_steps(steps: list[Step], total: CostBreakdown) -> list[Step]:
    """Make the steps add up to `total`.

    Any difference (cost a system did not trace, or cost it counted twice) becomes an explicit `untracked`
    step, so stage rollups always sum to the real cost and instrumentation gaps are visible, not silent.
    """
    residual = total.minus(sum_costs(step.cost for step in steps))
    drift = any(abs(value) > _EPSILON for value in residual.model_dump().values() if isinstance(value, float))
    drift = drift or any(getattr(residual, name) != 0 for name in ("embedding_input_tokens", "llm_prompt_tokens", "llm_completion_tokens"))
    if not drift:
        return steps
    return [*steps, Step(kind="retrieve", name=UNTRACKED, cost=residual)]


def stage_key(step: Step) -> str:
    return UNTRACKED if step.name == UNTRACKED else step.kind


def stage_costs(steps: Iterable[Step]) -> dict[str, float]:
    """Total cost per stage (see `STAGE_KEYS`); every key is present."""
    rolled = dict.fromkeys(STAGE_KEYS, 0.0)
    for step in steps:
        rolled[stage_key(step)] += step.cost.total_cost
    return rolled


def mean_by_stage(per_question: list[dict[str, float]]) -> dict[str, float]:
    """Average per-question stage values (e.g. cost) across questions."""
    if not per_question:
        return dict.fromkeys(STAGE_KEYS, math.nan)
    return {key: sum(row[key] for row in per_question) / len(per_question) for key in STAGE_KEYS}
