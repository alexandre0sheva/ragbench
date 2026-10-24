"""Run orchestration: load the dataset, plan the run, schedule the systems, honour the spending cap and checkpoints, write the outputs.

What a run covers is the `EvaluationPlan` (evaluation/plan.py); one question is answered and scored in evaluation/question_runner.py,
its rows are built in evaluation/rows.py, per-system aggregates come from evaluation/summary.py, and the files are written by
evaluation/outputs.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ragbench.cache import CacheRuntime, activate_cache, open_cache_runtime
from ragbench.cache.trial import trial_scope
from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.config.sweep import expand_sweeps
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Dataset, has_relevance_labels
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_dataset_documents
from ragbench.evaluation import outputs, question_runner, rows
from ragbench.evaluation.budget import BudgetExceededError, BudgetGuard
from ragbench.evaluation.checkpoint import Checkpoint, CheckpointStore, context_hash
from ragbench.evaluation.judge_panel import JudgePanel
from ragbench.evaluation.manifest import dataset_hash, utc_now_iso
from ragbench.evaluation.pairwise import PairwiseJudge
from ragbench.evaluation.pairwise_run import PairwiseResult, run_pairwise
from ragbench.evaluation.plan import EvaluationPlan
from ragbench.evaluation.rows import RunRows
from ragbench.evaluation.run_stats import cheapest_system
from ragbench.evaluation.summary import failed_systems
from ragbench.evaluation.warm_up import modules_to_warm_up
from ragbench.models import cost as pricing
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.models.errors import ModelInitError
from ragbench.models.refs import missing_credentials, resolve_run_mode, system_model_refs
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem
from ragbench.runtime import (
    ExecutionSettings,
    ProbeResult,
    RuntimeContext,
    SerializedProgress,
    SystemOutcome,
    SystemRunner,
    activate_runtime,
    build_limiters,
    warm_up_imports,
)
from ragbench.runtime.progress import ProgressListener
from ragbench.utils.jsonl import write_jsonl

__all__ = ["BenchmarkEvaluator", "BenchmarkRunError", "ProgressListener", "run_benchmark"]

logger = logging.getLogger(__name__)


class BenchmarkRunError(Exception):
    """Raised after a run finishes (and its partial results are written) when a system exceeded `evaluation.max_error_rate`."""

    def __init__(self, output_dir: Path, failed_systems: dict[str, float], max_error_rate: float):
        self.output_dir = output_dir
        self.failed_systems = failed_systems
        self.max_error_rate = max_error_rate
        details = ", ".join(f"{name} ({rate:.0%} of questions failed)" for name, rate in failed_systems.items())
        super().__init__(f"Error rate above the allowed {max_error_rate:.0%}: {details}. Partial results were written to {output_dir}.")


class BenchmarkEvaluator:
    """Top-level orchestrator: ingests once per system, evaluates every question, writes reports.

    A single `BenchmarkEvaluator` corresponds to a single config + run. Output is
    written to a timestamped subdirectory under `config.run.output_dir` and the
    method `run()` returns that path.

    Per-question evaluation is parallelized across questions within each system
    via a `ThreadPoolExecutor` (size from `evaluation.max_workers`). Order of
    output rows is preserved regardless of completion order.
    """

    def __init__(
        self,
        config_path: Path,
        force_mock: bool = False,
        max_workers: int | None = None,
        progress: ProgressListener | None = None,
        use_cache: bool = True,
        system_workers: int | None = None,
        raw_config: dict[str, Any] | None = None,
        run_dir: Path | None = None,
    ):
        self.config_path = config_path
        self.use_cache = use_cache
        # With `run_dir` the run lives in that directory instead of a new timestamped one, checkpoints each system as it finishes, and picks
        # up the checkpoints of an earlier attempt there (see evaluation/checkpoint.py).
        self.run_dir = run_dir
        # `raw_config` is a config assembled in code (a preset, or a file narrowed with `--systems`) instead of read from `config_path`.
        self._config_given = raw_config is not None
        if raw_config is None:
            self.config: ExperimentConfig = load_config(config_path)
            self.raw_config = load_config_dict(config_path)
        else:
            self.raw_config = expand_sweeps(raw_config)
            self.config = ExperimentConfig.model_validate(self.raw_config)
        self.budget = BudgetGuard(self.config.evaluation.max_cost_usd)
        self.plan: EvaluationPlan | None = None  # set once the dataset is loaded
        self._reported: list[SystemConfig] | None = None  # the systems whose results are comparable; None = all of the plan's
        # Live when any model the config calls can be reached (a key for a hosted API, or a local / endpoint model).
        self.mode = resolve_run_mode(self.config, force_mock)
        self.force_mock = force_mock or self.mode == "mock"
        # A live run must not quietly mix real and mock models (e.g. a Claude generator with no OpenAI key for the judge).
        self.credential_problems = missing_credentials(self.config) if self.mode == "live" else []
        self.models_used: set[str] = set()
        self.started_utc = utc_now_iso()
        self.dataset_digest = ""
        self.dataset_info: dict[str, Any] = {}
        self.dataset_warnings: list[str] = []
        self.document_warnings: list[str] = []  # files skipped or decoded with a fallback while loading the corpus
        self.cache_runtime: CacheRuntime | None = None
        self.cache_reason: str | None = None
        # Systems and questions run on worker threads; serializing the hooks means listeners need no locking of their own.
        self.progress = SerializedProgress(progress or ProgressListener())
        configured_workers = int(max_workers if max_workers is not None else self.config.evaluation.max_workers)
        self.max_workers = max(1, configured_workers)
        self.system_workers = max(1, int(system_workers if system_workers is not None else self.config.evaluation.system_workers))
        self.probe_results: dict[str, ProbeResult] = {}
        self.generator_models: dict[str, tuple[str, str]] = {}  # system name -> (provider, model) of its generator, for the self-preference check
        self._qrels: dict[str, dict[str, int]] = {}
        self.judge: JudgePanel | None = None
        self.pairwise: PairwiseResult | None = None  # the head-to-head stage, when `evaluation.pairwise.enabled`
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.config.run.name}_{timestamp}"
        # Runs can finish within one second of each other (a fully cached re-run is fast): never reuse a directory.
        suffix = 1
        while (self.config.run.output_dir / self.run_id).exists():
            suffix += 1
            self.run_id = f"{self.config.run.name}_{timestamp}_{suffix}"
        self.output_dir = self.config.run.output_dir / self.run_id
        if run_dir is not None:
            self.run_id, self.output_dir = run_dir.name, run_dir
        self.checkpoints = CheckpointStore(run_dir) if run_dir is not None else None
        self.restored_systems: list[str] = []  # systems taken from checkpoints instead of being run in this attempt

    def run(self) -> Path:
        """Execute the configured benchmark and return the output directory path."""
        if self.credential_problems:
            raise ModelInitError("Cannot start a live run: " + "; ".join(self.credential_problems) + ". Set them, pass --mock, or change the model refs.")
        pricing.clear_pricing_overrides()
        pricing.reset_unknown_priced_models()
        pricing.register_pricing({model: price.model_dump() for model, price in self.config.pricing.items()})
        self.cache_runtime = self._open_cache()
        runtime = RuntimeContext(
            ingest_workers=self.config.evaluation.ingest_workers,
            limiters=build_limiters(self.config.limits, self.config.providers),
            providers=self.config.providers,
            openrouter=self.config.openrouter,
            tools=self.config.tools,
            tools_now=self.config.evaluation.resolved_tools_now,
        )
        try:
            with activate_cache(self.cache_runtime), activate_runtime(runtime):
                return self._run()
        finally:
            pricing.clear_pricing_overrides()  # keep per-run price overrides from leaking into the next run
            if self.cache_runtime is not None:
                self.cache_runtime.disk.close()

    def _cache_disabled_reason(self) -> str | None:
        if not self.use_cache:
            return "disabled with --no-cache"
        if not self.config.cache.enabled:
            return "disabled in the config (`cache.enabled: false`)"
        if self.mode == "mock":
            return "mock run: nothing paid to save, so the disk cache is not used"
        return None

    def _open_cache(self) -> CacheRuntime | None:
        """The persistent cache for this run, or None (see `_cache_disabled_reason`)."""
        self.cache_reason = self._cache_disabled_reason()
        if self.cache_reason is not None:
            return None
        return open_cache_runtime(self.config.cache)

    def _plan(self, dataset: Dataset) -> EvaluationPlan:
        """Which systems run on which questions, how many times: the one place a run's scope is decided."""
        return EvaluationPlan.from_config(self.config, dataset.questions)

    def _run(self) -> Path:
        run_start = perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs.write_config_copy(self.output_dir / "config.yaml", self.config_path, self.raw_config, self._config_given)
        EMBEDDING_CACHE.clear()
        EMBEDDING_CACHE.enabled = self.config.evaluation.embedding_cache
        documents = load_dataset_documents(self.config.dataset, warnings=self.document_warnings)
        dataset = load_dataset(self.config.dataset.questions_path, self.config.dataset.qrels_path)
        self.dataset_warnings = validate_dataset(documents, dataset)
        self.dataset_digest = dataset_hash(documents, self.config.dataset.questions_path, self.config.dataset.qrels_path)
        plan = self.plan = self._plan(dataset)
        questions = plan.questions
        self.progress.run_started(len(plan.systems), plan.num_questions)
        judge = JudgePanel.from_config(self.config.evaluation, self.force_mock)
        if judge.enabled:
            self.models_used.update(judge.model_names())

        self._qrels, self.judge = dataset.qrels, judge
        labeled = sum(has_relevance_labels(question, dataset.qrels.get(question.id, {})) for question in questions)
        self.dataset_info = {
            "questions": len(questions),
            "labeled_questions": labeled,
            "label_free": dataset.label_free,
            "synthetic_questions": sum(bool(q.metadata.get("synthetic")) for q in questions),
            "needs_review_questions": sum(bool(q.metadata.get("needs_review")) for q in questions),
            "mock_questions": sum(bool(q.metadata.get("mock")) for q in questions),
        }
        warm_up_imports(self._modules_to_warm_up())

        settings = ExecutionSettings(
            max_workers=self.max_workers,
            system_workers=self.system_workers,
            ingest_workers=self.config.evaluation.ingest_workers,
            latency_probe_questions=self.config.evaluation.latency_probe_questions,
        )
        runner = SystemRunner(self, settings, self.progress, on_system_done=self._checkpoint if self.checkpoints is not None else None)
        restored, restored_probes = self._restore_checkpoints(plan)
        fresh = {outcome.name: outcome for outcome in runner.run_all(plan.without_systems(restored), documents)}
        all_outcomes = [restored[cfg.resolved_name] if cfg.resolved_name in restored else fresh[cfg.resolved_name] for cfg in plan.systems]
        # A system the spending cap stopped partway (or never started) has results over fewer questions than the others: they are
        # set aside rather than compared. Everything below works on the systems that finished.
        outcomes = [outcome for outcome in all_outcomes if outcome.complete]
        unfinished = [outcome for outcome in all_outcomes if not outcome.complete]
        if unfinished:
            done = {outcome.name for outcome in outcomes}
            self._reported = [cfg for cfg in plan.systems if cfg.resolved_name in done]
        # Timings from the parallel pass include queueing behind other threads. A live run therefore re-times a few
        # questions one at a time afterwards; mock runs skip it (their latencies are microseconds of CPU).
        fresh_probes = runner.probe(outcomes, questions) if self._should_probe(questions) and not unfinished else {}
        self.probe_results = {**restored_probes, **fresh_probes}
        if self.checkpoints is not None:
            for name, probe in fresh_probes.items():
                self.checkpoints.save_probe(name, asdict(probe))
        run_rows = RunRows.collect(outcomes)
        if self.config.evaluation.pairwise.enabled and len(outcomes) > 1 and not unfinished:
            self.pairwise = self._run_pairwise(run_rows, questions)

        run_wall_time_ms = (perf_counter() - run_start) * 1000
        if outcomes:
            outputs.write_outputs(self, run_rows, run_wall_time_ms)
        if unfinished:
            raise self._budget_stop(outcomes, unfinished, plan.num_answers, run_wall_time_ms)
        failed = failed_systems(run_rows.per_question, plan.num_answers, self.config.evaluation.max_error_rate)
        if failed:
            raise BenchmarkRunError(self.output_dir, failed, self.config.evaluation.max_error_rate)
        return self.output_dir

    def _run_pairwise(self, run_rows: RunRows, questions: list) -> PairwiseResult:
        """Head-to-head judging of the finished systems' answers (`evaluation.pairwise`), charged to the run's spending cap."""
        assert self.judge is not None
        evaluation = self.config.evaluation
        systems = self.reported_systems
        names = [cfg.resolved_name for cfg in systems]
        costs: dict[str, list[float]] = {}
        for row in run_rows.cost:
            costs.setdefault(row["system"], []).append(float(row["total_cost"]))
        priced = [{"system": name, "avg_cost_per_question": sum(costs[name]) / len(costs[name]) if costs.get(name) else None} for name in names]
        return run_pairwise(
            run_rows.per_question,
            questions,
            names,
            judges={ref: PairwiseJudge.from_answer_judge(judge) for ref, judge in self.judge.judges.items()},
            generators={cfg.resolved_name: system_model_refs(cfg).llm[0] for cfg in systems},
            settings=evaluation.pairwise,
            independent=evaluation.judge.independent,
            baseline=evaluation.stats.baseline or cheapest_system(priced),  # the same default baseline as `significance.csv`
            seed=evaluation.stats.seed,
            workers=self.max_workers,
            budget=self.budget,
        )

    def _context_hash(self, system_config: SystemConfig) -> str:
        return context_hash(system_config, self.config, self.dataset_digest, self.mode)

    def _checkpoint(self, outcome: SystemOutcome) -> None:
        """Save a system that just finished cleanly (every question answered, none failed); anything else is re-run on resume."""
        assert self.checkpoints is not None
        if outcome.system is None or not outcome.complete or any(result["per_question"]["error"] is not None for result in outcome.results):
            return
        system_config = next(cfg for cfg in self.config.systems if cfg.resolved_name == outcome.name)
        try:
            self.checkpoints.save(
                Checkpoint(
                    system=outcome.name,
                    system_type=outcome.system_type,
                    ingestion_row=outcome.ingestion_row,
                    results=outcome.results,
                    runtime_row=outcome.runtime_row,
                    models_used=sorted(_model_names(outcome.system)),
                    generator_model=self.generator_models.get(outcome.name),
                ),
                self._context_hash(system_config),
            )
        except (OSError, TypeError) as exc:  # a checkpoint is a convenience: failing to write one must not fail the run
            logger.warning("Could not checkpoint system %s: %s", outcome.name, exc)

    def _restore_checkpoints(self, plan: EvaluationPlan) -> tuple[dict[str, SystemOutcome], dict[str, ProbeResult]]:
        """The systems an earlier attempt of this run finished in the same context, as outcomes; the rest must be run."""
        restored: dict[str, SystemOutcome] = {}
        probes: dict[str, ProbeResult] = {}
        if self.checkpoints is None:
            return restored, probes
        (self.output_dir / "per_question_partial.jsonl").unlink(missing_ok=True)  # left by an earlier budget stop; rewritten if it happens again
        for position, system_config in enumerate(plan.systems, start=1):
            name = system_config.resolved_name
            checkpoint = self.checkpoints.load(name, self._context_hash(system_config))
            if checkpoint is None or len(checkpoint.results) != plan.num_answers:
                continue
            restored[name] = SystemOutcome(position, name, checkpoint.system_type, None, checkpoint.ingestion_row, checkpoint.results, checkpoint.runtime_row, restored=True)
            self.models_used.update(checkpoint.models_used)
            if checkpoint.generator_model is not None:
                self.generator_models[name] = checkpoint.generator_model
            if checkpoint.probe:
                probes[name] = ProbeResult(**checkpoint.probe)
            spent = float((checkpoint.ingestion_row or {}).get("total_cost", 0.0)) + sum(float(r["per_question"]["cost"]["total_cost"]) for r in checkpoint.results)
            self.budget.charge(spent)  # the spending cap is a cap on the whole run, so what the earlier attempt charged still counts
            self.restored_systems.append(name)
            self.progress.system_restored(name, len(checkpoint.results))
        return restored, probes

    def budget_summary(self, completed: list[str], incomplete: dict[str, dict[str, int]]) -> dict[str, Any]:
        return {
            "max_cost_usd": self.config.evaluation.max_cost_usd,
            "spent_usd": self.budget.spent,
            "stopped": bool(incomplete),
            "completed_systems": completed,
            "incomplete_systems": incomplete,
        }

    def _budget_stop(self, outcomes: list[SystemOutcome], unfinished: list[SystemOutcome], num_questions: int, run_wall_time_ms: float) -> BudgetExceededError:
        """Write what the stopped run still has to say (the unfinished systems' answers, and a run summary when nothing finished) and build the error."""
        incomplete = {outcome.name: {"answered": outcome.answered, "total": num_questions} for outcome in unfinished}
        completed = [outcome.name for outcome in outcomes]
        partial = [result["per_question"] for outcome in unfinished for result in outcome.results if not result.get("skipped")]
        if partial:
            write_jsonl(self.output_dir / "per_question_partial.jsonl", partial)
        budget = self.budget_summary(completed, incomplete)
        if not outcomes:  # `write_outputs` did not run: leave the run summary behind so the directory explains itself
            (self.output_dir / "run_summary.json").write_text(
                json.dumps({"run_id": self.run_id, "mode": self.mode, "run_wall_time_ms": run_wall_time_ms, "num_systems": 0, "budget": budget}, indent=2), encoding="utf-8"
            )
        else:
            summary_path = self.output_dir / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["budget"] = budget
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            outputs.write_report(self.output_dir, self.config.report.max_embedded_mb)  # now that the summary says the run stopped at its budget
        cap = self.config.evaluation.max_cost_usd or 0.0
        return BudgetExceededError(self.output_dir, cap, self.budget.spent, completed, incomplete)

    @property
    def reported_systems(self) -> list[SystemConfig]:
        """The systems the outputs describe: all of the plan's, unless the spending cap left some unfinished."""
        if self._reported is not None:
            return self._reported
        return self.plan.systems if self.plan is not None else self.config.systems

    def _modules_to_warm_up(self) -> list[str]:
        return modules_to_warm_up(self.config, self.mode)

    def _should_probe(self, questions: list) -> bool:
        size = self.config.evaluation.latency_probe_questions
        return self.mode == "live" and (size == "auto" or size > 0) and bool(questions)

    # -- QuestionEvaluator protocol (used by runtime.executor.SystemRunner) -----------------------

    def budget_exhausted(self) -> bool:
        return self.budget.exhausted

    def charge(self, usd: float) -> None:
        self.budget.charge(usd)

    def create_system(self, system_config: SystemConfig) -> BaseRAGSystem:
        system = create_rag_system(system_config, force_mock=self.force_mock)
        self.models_used.update(_model_names(system))
        self.generator_models[system.name] = (system.llm.provider, system.llm.model_name)
        return system

    def evaluate_single_question(self, system: BaseRAGSystem, system_config: SystemConfig, question, trial: int = 0) -> dict[str, Any]:
        assert self.judge is not None
        if self.budget.exhausted:  # the cap is reached: nothing new is started (questions already running finish)
            return {"skipped": True}
        with trial_scope(trial):  # every model call of this answer, and of its judging, is keyed to the trial
            result = question_runner.evaluate_question(system, system_config, question, self._qrels.get(question.id, {}), self.judge, self.config.evaluation)
        self.budget.charge(float(result["per_question"]["cost"]["total_cost"]))
        return self._tagged(result, trial)

    def answer_for_probe(self, system: BaseRAGSystem, question) -> AnswerResult:
        """Answer `question` exactly as the main pass does (same depth and context size), without scoring it."""
        return question_runner.answer(system, question, self.config.evaluation)

    def error_result(self, system: BaseRAGSystem, system_config: SystemConfig, question, error: dict[str, str], trial: int = 0) -> dict[str, Any]:
        return self._tagged(rows.error_result(system, system_config, question, error), trial)

    def _tagged(self, result: dict[str, Any], trial: int) -> dict[str, Any]:
        return rows.tag_trial(result, trial) if self.plan is not None and self.plan.trials > 1 else result


def _model_names(system: BaseRAGSystem) -> set[str]:
    names = {system.llm.model_name}
    embedding_model = getattr(system, "embedding_model", None)
    if embedding_model is not None:
        names.add(embedding_model.model_name)
    chunker_embedder = getattr(getattr(system, "chunker", None), "embedder", None)  # a `semantic` chunker embeds sentences
    if chunker_embedder is not None:
        names.add(chunker_embedder.model_name)
    return names


def run_benchmark(
    config_path: Path,
    force_mock: bool = False,
    max_workers: int | None = None,
    progress: ProgressListener | None = None,
    use_cache: bool = True,
    system_workers: int | None = None,
    raw_config: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Convenience wrapper: build a `BenchmarkEvaluator` and run it.

    Returns the timestamped output directory containing leaderboard, reports,
    and per-question results. With `run_dir` the run uses that directory and can be resumed there.
    """
    return BenchmarkEvaluator(
        config_path,
        force_mock=force_mock,
        max_workers=max_workers,
        progress=progress,
        use_cache=use_cache,
        system_workers=system_workers,
        raw_config=raw_config,
        run_dir=run_dir,
    ).run()
