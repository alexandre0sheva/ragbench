"""Scheduling for a benchmark run: which systems, questions, and probes run when, and on how many threads.

What is *computed* per question (metrics, judging, failure types) lives in `evaluation/question_runner.py`; this module only
decides how the work is spread over threads and keeps results in deterministic (config / dataset) order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Protocol

import numpy as np

from ragbench.cache import activate_cache
from ragbench.config.schema import SystemConfig
from ragbench.evaluation.plan import EvaluationPlan
from ragbench.evaluation.stats import QUANTILE_MIN_N, quantile_ci
from ragbench.evaluation.unscored import blank_unscorable_retrieval
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem
from ragbench.runtime.parallel import evenly_sample, ordered_parallel_map
from ragbench.runtime.progress import ProgressListener

logger = logging.getLogger(__name__)

# `latency_probe_questions: auto`: start with the fewest probe calls whose p95 has an upper bound at all, add more in steps until the
# p95's exact 95% interval lies within +-AUTO_PROBE_TOLERANCE of it, and stop at AUTO_PROBE_MAX calls (questions are re-asked when a
# system has fewer).
AUTO_PROBE_START = QUANTILE_MIN_N
AUTO_PROBE_STEP = 24
AUTO_PROBE_MAX = 120
AUTO_PROBE_TOLERANCE = 0.15


@dataclass(frozen=True)
class ExecutionSettings:
    max_workers: int = 4
    system_workers: int = 1
    ingest_workers: int = 4
    latency_probe_questions: int | Literal["auto"] = 5


class QuestionEvaluator(Protocol):
    """What the runner needs from the evaluator (implemented by `BenchmarkEvaluator`)."""

    def create_system(self, system_config: SystemConfig) -> BaseRAGSystem: ...

    def evaluate_single_question(self, system: BaseRAGSystem, system_config: SystemConfig, question: Any, trial: int) -> dict[str, Any]: ...

    def error_result(self, system: BaseRAGSystem, system_config: SystemConfig, question: Any, error: dict[str, str], trial: int) -> dict[str, Any]: ...

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
    results: list[dict[str, Any]]  # per question and trial (question-major); `{"skipped": True}` for one the budget stopped before it started
    runtime_row: dict[str, Any]
    restored: bool = False  # loaded from a checkpoint of an earlier attempt of this run instead of being run now

    @property
    def answered(self) -> int:
        return sum(1 for result in self.results if not result.get("skipped"))

    @property
    def complete(self) -> bool:
        """Every question was either answered or failed on its own: the system's results are comparable with the others'."""
        return (self.restored or self.system is not None) and self.answered == len(self.results)


@dataclass
class ProbeResult:
    latencies_ms: list[float] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0


def answer_key(result: dict[str, Any]) -> tuple[str, int]:
    """Which question and trial an answered result is for."""
    per_question = result["per_question"]
    return str(per_question["question_id"]), int(per_question.get("trial") or 0)


def _error_info(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


class SystemRunner:
    """Runs systems (ingest once, then all questions) and the latency probe."""

    def __init__(
        self,
        evaluator: QuestionEvaluator,
        settings: ExecutionSettings,
        progress: ProgressListener,
        on_system_done: Callable[[SystemOutcome], None] | None = None,
    ):
        self.evaluator = evaluator
        self.settings = settings
        self.progress = progress
        self.on_system_done = on_system_done  # called from the system's worker thread the moment it finishes (checkpointing)

    # -- main pass ------------------------------------------------------------------------------

    def run_all(self, plan: EvaluationPlan, documents: list[Any], earlier: Mapping[str, list[dict[str, Any]]] | None = None) -> list[SystemOutcome]:
        """Every system of `plan` on every question of `plan`, `plan.trials` times each, as outcomes in the plan's system order.

        `earlier` maps a system name to answers an earlier attempt of the run already gave (its partial checkpoint): those are kept and
        only the other questions are asked."""
        total = len(plan.systems)
        earlier = earlier or {}

        def run_one(pair: tuple[int, SystemConfig]) -> SystemOutcome:
            outcome = self.run_system(pair[0], total, pair[1], documents, plan.questions, plan.trials, earlier.get(pair[1].resolved_name))
            if self.on_system_done is not None:
                self.on_system_done(outcome)
            return outcome

        return ordered_parallel_map(
            run_one,
            list(enumerate(plan.systems, start=1)),
            workers=self.settings.system_workers,
            thread_name_prefix="ragbench-system",
        )

    def run_system(
        self,
        index: int,
        total: int,
        system_config: SystemConfig,
        documents: list[Any],
        questions: list[Any],
        trials: int = 1,
        earlier: list[dict[str, Any]] | None = None,
    ) -> SystemOutcome:
        name = system_config.resolved_name
        system_start = perf_counter()
        work = [(question, trial) for question in questions for trial in range(trials)]  # a question's trials stay next to each other
        kept = {answer_key(result): result for result in earlier or []}
        todo = [item for item in work if (str(item[0].id), item[1]) not in kept]
        resumed_at = len(work) - len(todo)
        self.progress.system_started(name, index, total)
        if kept:
            self.progress.system_resumed(name, resumed_at, len(work))
        if self.evaluator.budget_exhausted():
            logger.warning("Skipping system %s: the spending cap (evaluation.max_cost_usd) was already reached.", name)
            self.progress.system_finished(name, 0.0)
            # The earlier attempt's answers stay counted (and in the partial checkpoint) for the next attempt.
            results = [kept.get((str(question.id), trial), {"skipped": True}) for question, trial in work] if kept else []
            return SystemOutcome(index, name, system_config.type, None, None, results, {"system": name, "system_type": system_config.type, "num_questions": len(questions)})
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
            fresh = [self.evaluator.error_result(system, system_config, q, ingest_error, trial) for q, trial in todo]
            for done in range(1, len(todo) + 1):
                self.progress.question_finished(system.name, resumed_at + done, len(work))
        else:
            fresh = ordered_parallel_map(
                lambda item: self.evaluator.evaluate_single_question(system, system_config, item[0], item[1]),
                todo,
                workers=self.settings.max_workers,
                on_done=lambda done, count: self.progress.question_finished(system.name, resumed_at + done, len(work)),
                thread_name_prefix=f"ragbench-{system.name}",
            )
        answers = iter(fresh)
        results = [kept[(str(q.id), trial)] if (str(q.id), trial) in kept else next(answers) for q, trial in work]
        if ingest_error is None:
            results = blank_unscorable_retrieval(results)
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
                ok_ids = list(dict.fromkeys(r["per_question"]["question_id"] for r in outcome.results if not r.get("skipped") and r["per_question"]["error"] is None))
                if self.settings.latency_probe_questions == "auto":
                    results[outcome.name] = self._probe_until_stable(outcome, ok_ids, by_id)
                    continue
                chosen = evenly_sample(ok_ids, self.settings.latency_probe_questions)
                probe = ProbeResult()
                for done, question_id in enumerate(chosen, start=1):
                    if not self._probe_once(outcome, by_id[question_id], probe):
                        break
                    self.progress.latency_probe(outcome.name, done, len(chosen))
                results[outcome.name] = probe
        return results

    def _probe_once(self, outcome: SystemOutcome, question: Any, probe: ProbeResult) -> bool:
        """One probe call, charged to the spending cap like everything else the run pays for; False (nothing asked) once the cap is reached."""
        assert outcome.system is not None
        if self.evaluator.budget_exhausted():
            return False
        try:
            answer = self.evaluator.answer_for_probe(outcome.system, question)
        except Exception as exc:
            logger.warning("Latency probe failed for %s / %s: %s", outcome.name, question.id, exc)
        else:
            probe.latencies_ms.append(answer.latency_ms)
            probe.cost_usd += answer.cost.total_cost
            probe.calls += 1
            self.evaluator.charge(answer.cost.total_cost)
        return True

    def _probe_until_stable(self, outcome: SystemOutcome, ok_ids: list[str], by_id: dict[str, Any]) -> ProbeResult:
        """`latency_probe_questions: auto`: probe in steps until the p95 is known to within +-AUTO_PROBE_TOLERANCE, or AUTO_PROBE_MAX calls."""
        probe = ProbeResult()
        if not ok_ids:
            return probe
        order = [ok_ids[i] for i in np.random.default_rng(0).permutation(len(ok_ids))]  # spread over the dataset, reproducibly
        attempts, target = 0, AUTO_PROBE_START
        while attempts < AUTO_PROBE_MAX:
            while attempts < target:
                if not self._probe_once(outcome, by_id[order[attempts % len(order)]], probe):
                    return probe
                attempts += 1
                self.progress.latency_probe(outcome.name, attempts, AUTO_PROBE_MAX)
            p95, lo, hi = quantile_ci(probe.latencies_ms, 95)
            if p95 is not None and lo is not None and hi is not None and max(p95 - lo, hi - p95) <= AUTO_PROBE_TOLERANCE * p95:
                break
            target = min(AUTO_PROBE_MAX, target + AUTO_PROBE_STEP)
        return probe
