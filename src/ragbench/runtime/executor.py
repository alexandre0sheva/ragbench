"""Scheduling for a benchmark run: which systems, questions, and probes run when, and on how many threads.

What is *computed* per question (metrics, judging, failure types) stays in `evaluation/evaluator.py`; this module only
decides how the work is spread over threads and keeps results in deterministic (config / dataset) order.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from ragbench.cache import activate_cache
from ragbench.config.schema import SystemConfig
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem
from ragbench.runtime.parallel import evenly_sample, ordered_parallel_map
from ragbench.runtime.progress import ProgressListener

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExecutionSettings:
    max_workers: int = 4
    system_workers: int = 1
    ingest_workers: int = 4
    latency_probe_questions: int = 5


class QuestionEvaluator(Protocol):
    """What the runner needs from the evaluator (implemented by `BenchmarkEvaluator`)."""

    def create_system(self, system_config: SystemConfig) -> BaseRAGSystem: ...

    def evaluate_single_question(self, system: BaseRAGSystem, system_config: SystemConfig, question: Any) -> dict[str, Any]: ...

    def error_result(self, system: BaseRAGSystem, system_config: SystemConfig, question: Any, error: dict[str, str]) -> dict[str, Any]: ...

    def answer_for_probe(self, system: BaseRAGSystem, question: Any) -> AnswerResult: ...

    def budget_exhausted(self) -> bool:
        """The spending cap is reached: start no more systems."""
        ...

    def charge(self, usd: float) -> None:
        """Count money spent outside a question (ingestion) against the spending cap."""
        ...


@dataclass
class SystemOutcome:
    """Everything one system produced, in the order the evaluator needs it."""

    index: int
    name: str
    system_type: str
    system: BaseRAGSystem | None  # None when the system was never started (the budget ran out first)
    ingestion_row: dict[str, Any] | None
    results: list[dict[str, Any]]  # per question; `{"skipped": True}` for a question the budget stopped before it started
    runtime_row: dict[str, Any]

    @property
    def answered(self) -> int:
        return sum(1 for result in self.results if not result.get("skipped"))

    @property
    def complete(self) -> bool:
        """Every question was either answered or failed on its own: the system's results are comparable with the others'."""
        return self.system is not None and self.answered == len(self.results)


@dataclass
class ProbeResult:
    latencies_ms: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0


def _error_info(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


class SystemRunner:
    """Runs systems (ingest once, then all questions) and the latency probe."""

    def __init__(self, evaluator: QuestionEvaluator, settings: ExecutionSettings, progress: ProgressListener):
        self.evaluator = evaluator
        self.settings = settings
        self.progress = progress

    # -- main pass ------------------------------------------------------------------------------

    def run_all(self, system_configs: Sequence[SystemConfig], documents: list[Any], questions: list[Any]) -> list[SystemOutcome]:
        total = len(system_configs)
        return ordered_parallel_map(
            lambda pair: self.run_system(pair[0], total, pair[1], documents, questions),
            list(enumerate(system_configs, start=1)),
            workers=self.settings.system_workers,
            thread_name_prefix="ragbench-system",
        )

    def run_system(self, index: int, total: int, system_config: SystemConfig, documents: list[Any], questions: list[Any]) -> SystemOutcome:
        name = system_config.resolved_name
        system_start = perf_counter()
        self.progress.system_started(name, index, total)
        if self.evaluator.budget_exhausted():
            logger.warning("Skipping system %s: the spending cap (evaluation.max_cost_usd) was already reached.", name)
            self.progress.system_finished(name, 0.0)
            return SystemOutcome(index, name, system_config.type, None, None, [], {"system": name, "system_type": system_config.type, "num_questions": len(questions)})
        system = self.evaluator.create_system(system_config)
        ingest_start = perf_counter()
        ingest_error: dict[str, str] | None = None
        ingestion = None
        try:
            ingestion = system.ingest(documents)
        except Exception as exc:
            logger.warning("Ingestion failed for system %s: %s", system.name, exc)
            ingest_error = _error_info(exc)
        ingestion_wall_time_ms = (perf_counter() - ingest_start) * 1000
        ingestion_row: dict[str, Any] | None = None
        if ingestion is not None:
            self.evaluator.charge(ingestion.cost.total_cost)
            self.progress.ingestion_finished(system.name, ingestion.num_chunks, ingestion_wall_time_ms)
            ingestion_row = {
                "system": system.name,
                "system_type": system_config.type,
                "stage": "ingestion",
                "num_documents": ingestion.num_documents,
                "num_chunks": ingestion.num_chunks,
                "latency_ms": ingestion.latency_ms,
                **ingestion.cost.as_dict(),
            }
        else:
            self.progress.ingestion_finished(system.name, 0, ingestion_wall_time_ms)
        question_start = perf_counter()
        if ingest_error is not None:
            # A broken system must not take the other systems' (possibly paid-for) results down with it.
            results = [self.evaluator.error_result(system, system_config, q, ingest_error) for q in questions]
            for done in range(1, len(questions) + 1):
                self.progress.question_finished(system.name, done, len(questions))
        else:
            results = ordered_parallel_map(
                lambda question: self.evaluator.evaluate_single_question(system, system_config, question),
                questions,
                workers=self.settings.max_workers,
                on_done=lambda done, count: self.progress.question_finished(system.name, done, count),
                thread_name_prefix=f"ragbench-{system.name}",
            )
        question_wall_time_ms = (perf_counter() - question_start) * 1000
        system_wall_time_ms = (perf_counter() - system_start) * 1000
        self.progress.system_finished(system.name, system_wall_time_ms)
        return SystemOutcome(
            index=index,
            name=system.name,
            system_type=system_config.type,
            system=system,
            ingestion_row=ingestion_row,
            results=results,
            runtime_row={
                "system": system.name,
                "system_type": system_config.type,
                "ingestion_wall_time_ms": ingestion_wall_time_ms,
                "question_wall_time_ms": question_wall_time_ms,
                "system_wall_time_ms": system_wall_time_ms,
                "num_questions": len(questions),
                "max_workers": self.settings.max_workers,
                "system_workers": self.settings.system_workers,
                "ingest_workers": self.settings.ingest_workers,
            },
        )

    # -- latency probe --------------------------------------------------------------------------

    def probe(self, outcomes: list[SystemOutcome], questions: list[Any]) -> dict[str, ProbeResult]:
        """Re-ask a few questions one at a time, with no disk cache and nothing else running, to time each system cleanly.

        Runs after *all* systems finished so other systems' threads cannot disturb the timings. Probe answers are
        discarded except for latency and cost; a failing probe call is skipped, never fatal.
        """
        by_id = {question.id: question for question in questions}
        results: dict[str, ProbeResult] = {}
        with activate_cache(None):
            for outcome in outcomes:
                if outcome.system is None:
                    continue
                ok_ids = [r["per_question"]["question_id"] for r in outcome.results if not r.get("skipped") and r["per_question"]["error"] is None]
                chosen = evenly_sample(ok_ids, self.settings.latency_probe_questions)
                probe = ProbeResult()
                for done, question_id in enumerate(chosen, start=1):
                    try:
                        answer = self.evaluator.answer_for_probe(outcome.system, by_id[question_id])
                    except Exception as exc:
                        logger.warning("Latency probe failed for %s / %s: %s", outcome.name, question_id, exc)
                    else:
                        probe.latencies_ms.append(answer.latency_ms)
                        probe.cost_usd += answer.cost.total_cost
                        probe.calls += 1
                    self.progress.latency_probe(outcome.name, done, len(chosen))
                results[outcome.name] = probe
        return results
