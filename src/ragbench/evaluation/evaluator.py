from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ragbench.cache import CacheRuntime, activate_cache, open_cache_runtime, summarize_cache
from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.config.sweep import expand_sweeps
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import has_relevance_labels
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_dataset_documents
from ragbench.evaluation.answer_judge import AnswerJudge
from ragbench.evaluation.answer_metrics import answer_metrics, is_refusal, refusal_metrics
from ragbench.evaluation.budget import BudgetExceededError, BudgetGuard
from ragbench.evaluation.checkpoint import Checkpoint, CheckpointStore, context_hash
from ragbench.evaluation.context_metrics import context_precision, context_recall_doc
from ragbench.evaluation.failure_analysis import classify_failure
from ragbench.evaluation.manifest import build_manifest, dataset_hash, utc_now_iso
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics, unmeasured_retrieval_metrics
from ragbench.evaluation.route_metrics import route_accuracy, route_rows
from ragbench.evaluation.run_stats import SIGNIFICANCE_COLUMNS, RunStats, compute_run_stats
from ragbench.evaluation.stats import latency_percentiles
from ragbench.evaluation.tool_metrics import summary_fields as tool_summary_fields
from ragbench.evaluation.tool_metrics import tool_steps
from ragbench.evaluation.tool_metrics import usage_rows as tool_usage_rows
from ragbench.models import cost as pricing
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.models.errors import ModelInitError
from ragbench.models.refs import missing_credentials, resolve_run_mode, warm_up_modules
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem
from ragbench.rag_systems.trace import STAGE_KEYS, UNTRACKED, mean_by_stage, stage_costs, step_to_dict
from ragbench.registry import SYSTEMS, VECTOR_BACKENDS
from ragbench.reporting.html_report import write_report
from ragbench.reporting.markdown_report import write_failures, write_leaderboard, write_qrels_audit
from ragbench.reporting.notices import build_notices
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
from ragbench.selection.recommend import recommend, write_recommendation
from ragbench.utils.jsonl import write_jsonl
from ragbench.utils.text import truncate, unique_preserve_order

logger = logging.getLogger(__name__)


class BenchmarkRunError(Exception):
    """Raised after a run finishes (and its partial results are written) when a system exceeded `evaluation.max_error_rate`."""

    def __init__(self, output_dir: Path, failed_systems: dict[str, float], max_error_rate: float):
        self.output_dir = output_dir
        self.failed_systems = failed_systems
        self.max_error_rate = max_error_rate
        details = ", ".join(f"{name} ({rate:.0%} of questions failed)" for name, rate in failed_systems.items())
        super().__init__(f"Error rate above the allowed {max_error_rate:.0%}: {details}. Partial results were written to {output_dir}.")


def _error_info(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


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
        self._reported: list[SystemConfig] | None = None  # the systems whose results are comparable; None = all of them
        # Live when any model the config calls can be reached (a key for a hosted API, or a local / endpoint model).
        self.mode = resolve_run_mode(self.config, force_mock)
        self.force_mock = force_mock or self.mode == "mock"
        # A live run must not quietly mix real and mock models (e.g. a Claude generator with no OpenAI key for the judge).
        self.credential_problems = missing_credentials(self.config) if self.mode == "live" else []
        self.models_used: set[str] = set()
        self.started_utc = utc_now_iso()
        self.dataset_digest = ""
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
        self._judge: AnswerJudge | None = None
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

    def _run(self) -> Path:
        run_start = perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_config_copy()
        EMBEDDING_CACHE.clear()
        EMBEDDING_CACHE.enabled = self.config.evaluation.embedding_cache
        documents = load_dataset_documents(self.config.dataset, warnings=self.document_warnings)
        dataset = load_dataset(self.config.dataset.questions_path, self.config.dataset.qrels_path)
        self.dataset_warnings = validate_dataset(documents, dataset)
        self.dataset_digest = dataset_hash(documents, self.config.dataset.questions_path, self.config.dataset.qrels_path)
        questions = dataset.questions[: self.config.evaluation.max_questions] if self.config.evaluation.max_questions else dataset.questions
        self.progress.run_started(len(self.config.systems), len(questions))
        judge = AnswerJudge(
            model_name=self.config.evaluation.judge_model,
            enabled=self.config.evaluation.judge_enabled,
            force_mock=self.force_mock,
            samples=self.config.evaluation.judge.samples,
            temperature=self.config.evaluation.judge.temperature,
        )
        if judge.enabled:
            self.models_used.add(judge.llm.model_name)

        self._qrels, self._judge = dataset.qrels, judge
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
        per_question_rows: list[dict[str, Any]] = []
        retrieval_rows: list[dict[str, Any]] = []
        answer_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        ingestion_rows: list[dict[str, Any]] = []
        system_runtime_rows: list[dict[str, Any]] = []

        settings = ExecutionSettings(
            max_workers=self.max_workers,
            system_workers=self.system_workers,
            ingest_workers=self.config.evaluation.ingest_workers,
            latency_probe_questions=self.config.evaluation.latency_probe_questions,
        )
        runner = SystemRunner(self, settings, self.progress, on_system_done=self._checkpoint if self.checkpoints is not None else None)
        restored, restored_probes = self._restore_checkpoints(questions)
        fresh = {outcome.name: outcome for outcome in runner.run_all([cfg for cfg in self.config.systems if cfg.resolved_name not in restored], documents, questions)}
        all_outcomes = [restored[cfg.resolved_name] if cfg.resolved_name in restored else fresh[cfg.resolved_name] for cfg in self.config.systems]
        # A system the spending cap stopped partway (or never started) has results over fewer questions than the others: they are
        # set aside rather than compared. Everything below works on the systems that finished.
        outcomes = [outcome for outcome in all_outcomes if outcome.complete]
        unfinished = [outcome for outcome in all_outcomes if not outcome.complete]
        if unfinished:
            done = {outcome.name for outcome in outcomes}
            self._reported = [cfg for cfg in self.config.systems if cfg.resolved_name in done]
        # Timings from the parallel pass include queueing behind other threads. A live run therefore re-times a few
        # questions one at a time afterwards; mock runs skip it (their latencies are microseconds of CPU).
        fresh_probes = runner.probe(outcomes, questions) if self._should_probe(questions) and not unfinished else {}
        self.probe_results = {**restored_probes, **fresh_probes}
        if self.checkpoints is not None:
            for name, probe in fresh_probes.items():
                self.checkpoints.save_probe(name, asdict(probe))
        self._collect(outcomes, per_question_rows, retrieval_rows, answer_rows, cost_rows, ingestion_rows, system_runtime_rows)

        run_wall_time_ms = (perf_counter() - run_start) * 1000
        if outcomes:
            self._write_outputs(per_question_rows, retrieval_rows, answer_rows, cost_rows, ingestion_rows, system_runtime_rows, run_wall_time_ms)
        if unfinished:
            raise self._budget_stop(outcomes, unfinished, len(questions), run_wall_time_ms)
        failed = self._failed_systems(per_question_rows, len(questions))
        if failed:
            raise BenchmarkRunError(self.output_dir, failed, self.config.evaluation.max_error_rate)
        return self.output_dir

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

    def _restore_checkpoints(self, questions: list) -> tuple[dict[str, SystemOutcome], dict[str, ProbeResult]]:
        """The systems an earlier attempt of this run finished in the same context, as outcomes; the rest must be run."""
        restored: dict[str, SystemOutcome] = {}
        probes: dict[str, ProbeResult] = {}
        if self.checkpoints is None:
            return restored, probes
        (self.output_dir / "per_question_partial.jsonl").unlink(missing_ok=True)  # left by an earlier budget stop; rewritten if it happens again
        for position, system_config in enumerate(self.config.systems, start=1):
            name = system_config.resolved_name
            checkpoint = self.checkpoints.load(name, self._context_hash(system_config))
            if checkpoint is None or len(checkpoint.results) != len(questions):
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

    def _write_config_copy(self) -> None:
        """`config.yaml` in the run directory: the file as written when nothing changed (comments and all), else the config as it ran
        (sweeps expanded, a preset's systems, a `--systems` subset), so the directory always says exactly what was benchmarked."""
        target = self.output_dir / "config.yaml"
        if not self._config_given and self.config_path.exists():
            text = self.config_path.read_text(encoding="utf-8")
            if (yaml.safe_load(text) or {}) == self.raw_config:
                target.write_text(text, encoding="utf-8")
                return
        target.write_text(yaml.safe_dump(self.raw_config, sort_keys=False), encoding="utf-8")

    def _budget_summary(self, completed: list[str], incomplete: dict[str, dict[str, int]]) -> dict[str, Any]:
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
        budget = self._budget_summary(completed, incomplete)
        if not outcomes:  # `_write_outputs` did not run: leave the run summary behind so the directory explains itself
            (self.output_dir / "run_summary.json").write_text(
                json.dumps({"run_id": self.run_id, "mode": self.mode, "run_wall_time_ms": run_wall_time_ms, "num_systems": 0, "budget": budget}, indent=2), encoding="utf-8"
            )
        else:
            summary_path = self.output_dir / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["budget"] = budget
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            self._write_report()  # now that the summary says the run stopped at its budget
        cap = self.config.evaluation.max_cost_usd or 0.0
        return BudgetExceededError(self.output_dir, cap, self.budget.spent, completed, incomplete)

    @property
    def reported_systems(self) -> list[SystemConfig]:
        """The systems the outputs describe: all of them, unless the spending cap left some unfinished."""
        return self._reported if self._reported is not None else self.config.systems

    # -- spending cap (QuestionEvaluator protocol) ----------------------------------------------

    def budget_exhausted(self) -> bool:
        return self.budget.exhausted

    def charge(self, usd: float) -> None:
        self.budget.charge(usd)

    def _modules_to_warm_up(self) -> list[str]:
        """Libraries that worker threads would otherwise import lazily, racing with live API calls (see `warm_up_imports`)."""
        modules = ["httpx", "openai"]
        for system_config in self.config.systems:
            spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
            if spec is not None and "vector_store" in spec.options.model_fields:
                backend = system_config.retrieval.get("vector_store", spec.options.model_fields["vector_store"].default)
                modules.extend(module for module in VECTOR_BACKENDS.get(backend).import_modules if module not in modules)
        if self.mode == "live":  # mock runs never load a reranking model
            modules.extend(self._reranker_modules())
        return [*modules, *warm_up_modules(self.config)]

    def _reranker_modules(self) -> list[str]:
        """Libraries the configured rerankers import lazily (e.g. sentence-transformers for `cross_encoder`)."""
        import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
        from ragbench.registry import RERANKERS

        modules: list[str] = []
        for system_config in self.config.systems:
            spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
            if spec is None or "reranker" not in spec.options.model_fields:
                continue
            name = system_config.retrieval.get("reranker", spec.options.model_fields["reranker"].default)
            modules.extend(module for module in getattr(RERANKERS.get(name), "import_modules", ()) if module not in modules)
        return modules

    def _should_probe(self, questions: list) -> bool:
        return self.mode == "live" and self.config.evaluation.latency_probe_questions > 0 and bool(questions)

    @staticmethod
    def _collect(
        outcomes: list[SystemOutcome],
        per_question_rows: list[dict[str, Any]],
        retrieval_rows: list[dict[str, Any]],
        answer_rows: list[dict[str, Any]],
        cost_rows: list[dict[str, Any]],
        ingestion_rows: list[dict[str, Any]],
        system_runtime_rows: list[dict[str, Any]],
    ) -> None:
        """Flatten per-system outcomes into the run's row lists, in config order."""
        for outcome in outcomes:
            if outcome.ingestion_row is not None:
                ingestion_rows.append(outcome.ingestion_row)
            for result in outcome.results:
                per_question_rows.append(result["per_question"])
                if result["per_question"]["error"] is None:  # failed questions carry no scores and stay out of every mean
                    retrieval_rows.append(result["retrieval_row"])
                    answer_rows.append(result["answer_row"])
                    cost_rows.append(result["cost_row"])
            system_runtime_rows.append(outcome.runtime_row)

    # -- QuestionEvaluator protocol (used by runtime.executor.SystemRunner) -----------------------

    def create_system(self, system_config: SystemConfig) -> BaseRAGSystem:
        system = create_rag_system(system_config, force_mock=self.force_mock)
        self.models_used.update(_model_names(system))
        self.generator_models[system.name] = (system.llm.provider, system.llm.model_name)
        return system

    def evaluate_single_question(self, system: BaseRAGSystem, system_config: SystemConfig, question) -> dict[str, Any]:
        assert self._judge is not None
        if self.budget.exhausted:  # the cap is reached: nothing new is started (questions already running finish)
            return {"skipped": True}
        result = self._evaluate_single_question(system, system_config, question, self._qrels.get(question.id, {}), self._judge)
        self.budget.charge(float(result["per_question"]["cost"]["total_cost"]))
        return result

    def answer_for_probe(self, system: BaseRAGSystem, question):
        """Answer `question` exactly as the main pass does (same depth and context size), without scoring it."""
        return self._answer(system, question)

    def _errors_by_system(self, per_question_rows: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in per_question_rows:
            if row["error"] is not None:
                counts[row["system"]] = counts.get(row["system"], 0) + 1
        return counts

    def _failed_systems(self, per_question_rows: list[dict[str, Any]], num_questions: int) -> dict[str, float]:
        if not num_questions:
            return {}
        rates = {name: count / num_questions for name, count in self._errors_by_system(per_question_rows).items()}
        return {name: rate for name, rate in rates.items() if rate > self.config.evaluation.max_error_rate}

    def _evaluate_single_question(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        question,
        qrels: dict[str, int],
        judge: AnswerJudge,
    ) -> dict[str, Any]:
        try:
            return self._evaluate_question(system, system_config, question, qrels, judge)
        except Exception as exc:
            logger.warning("Question %s failed for system %s: %s: %s", question.id, system.name, type(exc).__name__, exc)
            return self.error_result(system, system_config, question, _error_info(exc))

    def _evaluate_question(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        question,
        qrels: dict[str, int],
        judge: AnswerJudge,
    ) -> dict[str, Any]:
        evaluation = self.config.evaluation
        answer = self._answer(system, question)
        # Retrieval is scored on the whole ranking; the judge sees only what the generator was given.
        context_ids = answer.metadata.get("context_chunk_ids")
        ranked = answer.retrieval_result.chunks
        in_context = {chunk.chunk_id for chunk in ranked} if context_ids is None else set(context_ids)
        spec = system.spec
        # A system that retrieves nothing has no retrieval metrics (an empty dict); its rows get blanks, never zeros.
        retrieves = spec is None or spec.retrieves
        # ... and so does a question that has an answer but no labels to score the ranking against (label-free mode); an unanswerable one still scores 0.
        scorable = retrieves and (has_relevance_labels(question, qrels) or not question.is_answerable)
        retrieval_metrics = compute_retrieval_metrics(ranked, question.relevant_doc_ids, qrels=qrels, k_values=evaluation.k_values) if scorable else {}
        context_chunks = [chunk for chunk in ranked if chunk.chunk_id in in_context]
        # Same notion of "relevant" as the retrieval metrics: the question's own list, else the positive qrels.
        relevant = question.relevant_doc_ids or [doc_id for doc_id, grade in qrels.items() if grade > 0]
        context_metrics = {
            "context_recall": context_recall_doc(context_chunks, relevant) if retrieves else None,
            "context_precision": context_precision(context_chunks, relevant) if retrieves else None,
        }
        judge_result = judge.judge(question, answer.answer, context_chunks)
        failure_type = classify_failure(
            question,
            answer.answer,
            retrieval_metrics,
            judge_result,
            primary_k=evaluation.resolved_primary_k,
            context_recall=context_metrics["context_recall"],
        )
        return self._build_question_rows(
            system, system_config, question, answer, retrieval_metrics, judge_result, failure_type, in_context, context_metrics
        )

    def _answer(self, system: BaseRAGSystem, question) -> AnswerResult:
        evaluation = self.config.evaluation
        context_k = system.configured_context_k() or evaluation.context_k
        depth = max(evaluation.resolved_retrieval_depth, context_k)
        return system.answer_question(question.question, top_k=depth, context_k=context_k)

    def error_result(self, system: BaseRAGSystem, system_config: SystemConfig, question, error: dict[str, str]) -> dict[str, Any]:
        zero_judge = {
            "correctness": 0.0,
            "faithfulness": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "citation_quality": 0.0,
            "answer_score": 0.0,
            "is_supported_by_context": False,
            "is_hallucinated": False,
            "reasoning": "",
            "metadata": {},
        }
        per_question = {
            "system": system.name,
            "system_type": system_config.type,
            "question_id": question.id,
            "question": question.question,
            "category": question.category,
            "difficulty": question.difficulty,
            "answer_type": question.answer_type,
            "reference_answer": question.reference_answer,
            "relevant_doc_ids": question.relevant_doc_ids,
            "answer": "",
            "retrieved_contexts": [],
            "retrieval_metrics": {},
            "answer_judge": zero_judge,
            "cost": CostBreakdown().as_dict(),
            "latency_ms": 0.0,
            "failure_type": "run_error",
            "error": error,
            "answerable": question.is_answerable,
            "refused": False,
            "answer_metrics": {},
            "context_metrics": {},
            "steps": [],
            "requires_tools": list(question.requires_tools),
            "route": None,
            "routing_hint": question.routing_hint,
            "agent": None,
        }
        return {"per_question": per_question, "retrieval_row": None, "answer_row": None, "cost_row": None}

    def _build_question_rows(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        question,
        answer,
        retrieval_metrics: dict[str, Any],
        judge_result,
        failure_type: str,
        in_context: set[str],
        context_metrics: dict[str, float | None],
    ) -> dict[str, Any]:
        refused = is_refusal(answer.answer)
        deterministic = answer_metrics(answer.answer, question.reference_answer, question.expected_keywords, question.is_answerable)
        total_cost = answer.cost.plus(judge_result.cost)
        retrieved_contexts = [
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "in_context": chunk.chunk_id in in_context,
                "text_preview": truncate(chunk.text, 260),
            }
            for chunk in answer.retrieval_result.chunks
        ]
        per_question = {
            "system": system.name,
            "system_type": system_config.type,
            "question_id": question.id,
            "question": question.question,
            "category": question.category,
            "difficulty": question.difficulty,
            "answer_type": question.answer_type,
            "reference_answer": question.reference_answer,
            "relevant_doc_ids": question.relevant_doc_ids,
            "answer": answer.answer,
            "retrieved_contexts": retrieved_contexts,
            "retrieval_metrics": retrieval_metrics,
            "answer_judge": {
                "correctness": judge_result.correctness,
                "faithfulness": judge_result.faithfulness,
                "completeness": judge_result.completeness,
                "relevance": judge_result.relevance,
                "citation_quality": judge_result.citation_quality,
                "answer_score": judge_result.answer_score,
                "is_supported_by_context": judge_result.is_supported_by_context,
                "is_hallucinated": judge_result.is_hallucinated,
                "reasoning": judge_result.reasoning,
                "metadata": judge_result.metadata,
            },
            "cost": total_cost.as_dict(),
            "latency_ms": answer.latency_ms,
            "failure_type": failure_type,
            "error": None,
            "answerable": question.is_answerable,
            "refused": refused,
            "answer_metrics": deterministic,
            "context_metrics": context_metrics,
            "steps": [step_to_dict(step) for step in answer.steps],
            "requires_tools": list(question.requires_tools),
            "route": answer.metadata.get("route"),  # the pipeline an `adaptive` system chose; None for every other system
            "routing_hint": question.routing_hint,
            "agent": answer.metadata.get("agent"),  # steps / hops / termination / budget_exhausted / llm_calls; None unless the system is agentic
        }
        return {
            "per_question": per_question,
            "retrieval_row": {
                "system": system.name,
                "question_id": question.id,
                "category": question.category,
                **(retrieval_metrics or unmeasured_retrieval_metrics(self.config.evaluation.k_values)),
            },
            "answer_row": {
                "system": system.name,
                "question_id": question.id,
                "category": question.category,
                "correctness": judge_result.correctness,
                "faithfulness": judge_result.faithfulness,
                "completeness": judge_result.completeness,
                "relevance": judge_result.relevance,
                "citation_quality": judge_result.citation_quality,
                "answer_score": judge_result.answer_score,
                **deterministic,
                **context_metrics,
                "answerable": question.is_answerable,
                "refused": refused,
                "failure_type": failure_type,
            },
            "cost_row": {
                "system": system.name,
                "system_type": system_config.type,
                "stage": "question",
                "question_id": question.id,
                "latency_ms": answer.latency_ms,
                **total_cost.as_dict(),
                **{f"stage_{key}_cost": value for key, value in stage_costs(answer.steps).items()},
            },
        }

    def _write_outputs(
        self,
        per_question_rows: list[dict[str, Any]],
        retrieval_rows: list[dict[str, Any]],
        answer_rows: list[dict[str, Any]],
        cost_rows: list[dict[str, Any]],
        ingestion_rows: list[dict[str, Any]],
        system_runtime_rows: list[dict[str, Any]],
        run_wall_time_ms: float,
    ) -> None:
        write_jsonl(self.output_dir / "per_question_results.jsonl", per_question_rows)
        retrieval_df = pd.DataFrame(retrieval_rows)
        answer_df = pd.DataFrame(answer_rows)
        cost_df = pd.DataFrame([*ingestion_rows, *cost_rows])
        runtime_df = pd.DataFrame(system_runtime_rows)
        retrieval_df.to_csv(self.output_dir / "retrieval_metrics.csv", index=False)
        answer_df.to_csv(self.output_dir / "answer_metrics.csv", index=False)
        cost_df.to_csv(self.output_dir / "cost_breakdown.csv", index=False)
        runtime_df.to_csv(self.output_dir / "system_runtime.csv", index=False)

        summary_rows = self._build_summary(retrieval_df, answer_df, pd.DataFrame(cost_rows), per_question_rows, runtime_df)
        run_stats = self._compute_stats(per_question_rows, summary_rows)
        for summary_row in summary_rows:
            summary_row.update(run_stats.summary_fields[summary_row["system"]])
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(run_stats.significance_rows, columns=SIGNIFICANCE_COLUMNS).to_csv(self.output_dir / "significance.csv", index=False)
        (self.output_dir / "stats.json").write_text(json.dumps(run_stats.stats, indent=2), encoding="utf-8")
        (self.output_dir / "pareto.json").write_text(json.dumps(run_stats.pareto, indent=2), encoding="utf-8")
        if routes := self._build_routes(per_question_rows):
            pd.DataFrame(routes).to_csv(self.output_dir / "routes.csv", index=False)
        if tool_usage := self._build_tool_usage(per_question_rows):
            pd.DataFrame(tool_usage).to_csv(self.output_dir / "tool_usage.csv", index=False)
        unknown_priced = pricing.unknown_priced_models()
        charged = float(sum(row.get("total_cost", 0.0) for row in [*ingestion_rows, *cost_rows]))
        in_process = EMBEDDING_CACHE.stats()
        probe_cost = sum(probe.cost_usd for probe in self.probe_results.values())
        probe_calls = sum(probe.calls for probe in self.probe_results.values())
        # Live runs without a probe report latency measured under concurrency: say so next to the numbers.
        concurrent_latency = self.mode == "live" and not probe_calls and self.max_workers * self.system_workers > 1
        cache_summary = summarize_cache(
            self.cache_runtime,
            charged_cost_usd=charged,
            in_process_saved_usd=float(in_process["saved_cost_usd"]),
            reason=self.cache_reason,
            extra_spend_usd=probe_cost,
        )
        notices = build_notices(self.mode, unknown_priced, concurrent_latency=concurrent_latency, **self._judge_notices(summary_rows))
        write_leaderboard(
            self.output_dir / "leaderboard.md",
            summary_rows,
            notices=notices,
            primary_k=self.config.evaluation.resolved_primary_k,
            stage_rows=self._build_stage_summary(per_question_rows),
            significance_rows=run_stats.significance_rows,
            stats_info=run_stats.stats,
        )
        write_failures(self.output_dir / "failures.md", per_question_rows)
        qrels_audit_rows = self._build_qrels_audit(per_question_rows)
        pd.DataFrame(qrels_audit_rows).to_csv(self.output_dir / "qrels_audit.csv", index=False)
        write_qrels_audit(self.output_dir / "qrels_audit.md", qrels_audit_rows, primary_k=self.config.evaluation.resolved_primary_k)
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "models_used": sorted(self.models_used),
                    "unknown_priced_models": unknown_priced,
                    "pricing_as_of": pricing.PRICING_AS_OF,
                    "run_wall_time_ms": run_wall_time_ms,
                    "max_workers": self.max_workers,
                    "execution": {
                        "max_workers": self.max_workers,
                        "system_workers": self.system_workers,
                        "ingest_workers": self.config.evaluation.ingest_workers,
                        "latency_probe_questions": self.config.evaluation.latency_probe_questions,
                        "latency_source": "probe" if probe_calls else "concurrent",
                        "probe_calls": probe_calls,
                        "probe_cost_usd": probe_cost,
                    },
                    "num_systems": len(self.reported_systems),
                    "budget": self._budget_summary([cfg.resolved_name for cfg in self.reported_systems], {}),
                    "num_question_rows": len(per_question_rows),
                    "num_errors": sum(self._errors_by_system(per_question_rows).values()),
                    "errors_by_system": self._errors_by_system(per_question_rows),
                    "k_values": self.config.evaluation.k_values,
                    "retrieval_depth": self.config.evaluation.resolved_retrieval_depth,
                    "context_k": self.config.evaluation.context_k,
                    "primary_k": self.config.evaluation.resolved_primary_k,
                    "embedding_cache": EMBEDDING_CACHE.stats(),
                    "cache": cache_summary,
                    "dataset": getattr(self, "dataset_info", {}),
                    "notices": notices,
                    "dataset_warnings": getattr(self, "dataset_warnings", []),
                    "document_warnings": self.document_warnings,
                    "outputs": {
                        "leaderboard": "leaderboard.md",
                        "report": "report.html",
                        "qrels_audit": "qrels_audit.md",
                        "stats": "stats.json",
                        "significance": "significance.csv",
                        "pareto": "pareto.json",
                        "recommendation": "recommendation.md",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest = build_manifest(
            config_path=self.config_path,
            raw_config=self.raw_config,
            dataset_digest=self.dataset_digest,
            mode=self.mode,
            models_used=sorted(self.models_used),
            started_utc=self.started_utc,
            finished_utc=utc_now_iso(),
        )
        (self.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._write_recommendation()
        self._write_report()  # last: it reads the recommendation, the manifest and every other file above

    def _write_report(self) -> None:
        """`report.html` and `report_data.json`, built from the files this run just wrote (so the same call can rebuild a report later)."""
        try:
            write_report(self.output_dir)
        except Exception as exc:  # the results are already on disk; a page that cannot be drawn must not turn the run into an error
            logger.warning("Could not write the HTML report: %s: %s", type(exc).__name__, exc)

    def _write_recommendation(self) -> None:
        """Which system to deploy (`recommendation.json` / `.md`, and `winner.yaml` when there is one), from this run's results."""
        selection = self.config.selection
        try:
            write_recommendation(self.output_dir, recommend(self.output_dir, constraints=selection.constraints, weights=selection.weights, profile=selection.profile))
        except Exception as exc:  # the results are already on disk; a recommendation that cannot be made must not turn the run into an error
            logger.warning("Could not write the recommendation: %s: %s", type(exc).__name__, exc)

    def _compute_stats(self, per_question_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> RunStats:
        settings = self.config.evaluation.stats
        return compute_run_stats(
            per_question_rows,
            summary_rows,
            k_values=self.config.evaluation.k_values,
            primary_k=self.config.evaluation.resolved_primary_k,
            n_boot=settings.n_boot,
            seed=settings.seed,
            baseline=settings.baseline,
        )

    def _build_stage_summary(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mean per-question cost and latency by stage for each system (failed questions excluded)."""

        def rollup(steps: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> dict[str, float]:
            rolled = dict.fromkeys(STAGE_KEYS, 0.0)
            for step in steps:
                rolled[UNTRACKED if step["name"] == UNTRACKED else step["kind"]] += value(step)
            return rolled

        rows: list[dict[str, Any]] = []
        for cfg in self.reported_systems:
            ok = [row for row in per_question_rows if row["system"] == cfg.resolved_name and row["error"] is None]
            if not ok:
                continue
            rows.append(
                {
                    "system": cfg.resolved_name,
                    "cost": mean_by_stage([rollup(row["steps"], lambda step: step["cost"]["total_cost"]) for row in ok]),
                    "latency_ms": mean_by_stage([rollup(row["steps"], lambda step: step["latency_ms"]) for row in ok]),
                }
            )
        return rows

    def _build_qrels_audit(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        primary_k = self.config.evaluation.resolved_primary_k
        audit_rows: list[dict[str, Any]] = []
        for row in per_question_rows:
            if row["error"] is not None:
                continue
            metrics = row["retrieval_metrics"]
            judge = row["answer_judge"]
            relevant_doc_ids = row.get("relevant_doc_ids", [])
            if not relevant_doc_ids:
                continue
            if metrics.get(f"recall@{primary_k}", 0.0) >= 1.0:
                continue
            if judge.get("answer_score", 0.0) < 4.0 or judge.get("faithfulness", 0.0) < 4.0:
                continue
            # Only documents the generator actually saw can explain a well-supported answer.
            retrieved_doc_ids = unique_preserve_order(context["doc_id"] for context in row.get("retrieved_contexts", []) if context.get("in_context", True))
            unlabeled = [doc_id for doc_id in retrieved_doc_ids if doc_id not in set(relevant_doc_ids)]
            if not unlabeled:
                continue
            audit_rows.append(
                {
                    "system": row["system"],
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "severity": "high" if metrics.get(f"hit@{primary_k}", 0.0) == 0.0 else "medium",
                    "primary_k": primary_k,
                    "recall": metrics.get(f"recall@{primary_k}", 0.0),
                    "answer_score": judge.get("answer_score", 0.0),
                    "faithfulness": judge.get("faithfulness", 0.0),
                    "labeled_relevant_doc_ids": ", ".join(relevant_doc_ids),
                    "unlabeled_retrieved_doc_ids": ", ".join(unlabeled),
                }
            )
        return audit_rows

    def _tools_offered(self) -> dict[str, list[str]]:
        """The tool names each system is configured with (empty for a system that has none)."""
        return {cfg.resolved_name: SYSTEMS.get(cfg.type).offered_tools(cfg) for cfg in self.reported_systems}

    def _build_routes(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Per-route usage of every system that routes, in the order the systems are configured."""
        rows: list[dict[str, Any]] = []
        for cfg in self.reported_systems:
            ok = [r for r in per_question_rows if r["system"] == cfg.resolved_name and r["error"] is None]
            rows.extend(route_rows(cfg.resolved_name, ok))
        return rows

    def _build_tool_usage(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        offered = self._tools_offered()
        rows: list[dict[str, Any]] = []
        for system_name, configured in offered.items():
            system_rows = [r for r in per_question_rows if r["system"] == system_name and r["error"] is None]
            if configured or any(tool_steps(r) for r in system_rows):
                rows.extend(tool_usage_rows(system_name, configured, system_rows))
        return rows

    def _judge_fallback_rate(self, rows: list[dict[str, Any]]) -> float | None:
        """Share of questions the heuristic had to score because the LLM judge failed; None when no LLM judge was in use."""
        if self._judge is None or not self._judge.uses_llm or not rows:
            return None
        return sum(1 for row in rows if row["answer_judge"]["metadata"].get("fallback")) / len(rows)

    def _judge_notices(self, summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Facts about the judge that `build_notices` turns into report warnings: self-preference and heuristic fallbacks."""
        judge = self._judge
        if judge is None or not judge.uses_llm:
            return {}
        judge_model = (judge.llm.provider, judge.llm.model_name)
        same_model = [name for name, model in self.generator_models.items() if model == judge_model] if self.config.evaluation.judge.independent else []
        fallbacks = {row["system"]: row["judge_fallback_rate"] for row in summary_rows if (row.get("judge_fallback_rate") or 0) > 0}
        return {"judge_model": judge.llm.model_name, "judge_shares_model_with": same_model, "judge_fallbacks": fallbacks}

    @staticmethod
    def _agent_fields(agents: list[dict[str, Any]]) -> dict[str, float]:
        """Mean loop iterations, mean LLM calls and the share of questions that hit the budget; empty for a system that is not agentic."""
        if not agents:
            return {}
        return {
            "avg_steps": sum(a["steps"] for a in agents) / len(agents),
            "avg_llm_calls": sum(a["llm_calls"] for a in agents) / len(agents),
            "budget_exhausted_rate": sum(1 for a in agents if a["budget_exhausted"]) / len(agents),
        }

    @staticmethod
    def _latency_fields(concurrent_ms: list[float], probe: ProbeResult | None) -> dict[str, Any]:
        """Latency columns of one system.

        The headline latency comes from the one-at-a-time probe when there is one (clean: no queueing behind other
        threads, no cache); otherwise from the parallel pass, labeled as such. The parallel-pass mean is always kept.
        """
        concurrent_mean = float(np.mean(concurrent_ms)) if concurrent_ms else None
        clean = probe.latencies_ms if probe is not None and probe.latencies_ms else None
        values = clean if clean is not None else concurrent_ms
        return {
            "avg_latency_ms": float(np.mean(values)) if values else None,
            **{f"latency_ms_{name}": value for name, value in latency_percentiles(values).items()},
            "latency_source": "probe" if clean is not None else "concurrent",
            "avg_latency_concurrent_ms": concurrent_mean,
        }

    def _build_summary(
        self,
        retrieval_df: pd.DataFrame,
        answer_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        per_question_rows: list[dict[str, Any]],
        runtime_df: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        system_types = {cfg.resolved_name: cfg.type for cfg in self.reported_systems}
        tools_offered = self._tools_offered()
        ok_rows = [row for row in per_question_rows if row["error"] is None]
        latency = pd.DataFrame([{"system": row["system"], "latency_ms": row["latency_ms"]} for row in ok_rows], columns=["system", "latency_ms"])
        errors = self._errors_by_system(per_question_rows)
        rows: list[dict[str, Any]] = []
        for system_name in system_types:
            r = retrieval_df[retrieval_df["system"] == system_name] if not retrieval_df.empty else retrieval_df
            a = answer_df[answer_df["system"] == system_name] if not answer_df.empty else answer_df
            c = cost_df[cost_df["system"] == system_name] if not cost_df.empty else cost_df
            latency_rows = latency[latency["system"] == system_name]
            n_ok = sum(1 for row in ok_rows if row["system"] == system_name)
            row: dict[str, Any] = {"system": system_name, "system_type": system_types[system_name], "n_ok": n_ok, "n_error": errors.get(system_name, 0)}
            # A system with no successful question has no measurements: leave them empty (shown as an em dash), never 0.
            for col in r.columns:
                if col not in {"system", "question_id", "category"}:
                    mean = r[col].mean() if not r.empty else None
                    row[f"retrieval_{col}"] = None if mean is None or pd.isna(mean) else float(mean)
            for col in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality", "answer_score"]:
                row[col] = float(a[col].mean()) if not a.empty else None
            # Deterministic and context metrics are missing for a question that cannot have them (no reference, no keywords, no
            # retrieval); the mean covers the questions that do, and a system with none gets an empty cell.
            for col in ["exact_match", "token_f1", "keyword_recall", "context_recall", "context_precision"]:
                mean = pd.to_numeric(a[col], errors="coerce").mean() if col in a else float("nan")
                row[col] = None if pd.isna(mean) else float(mean)
            row["avg_cost_per_question"] = float(c["total_cost"].mean()) if not c.empty else None
            ok_system_rows = [r for r in ok_rows if r["system"] == system_name]
            row.update(refusal_metrics(ok_system_rows))
            row["judge_fallback_rate"] = self._judge_fallback_rate(ok_system_rows)
            row.update(tool_summary_fields(ok_system_rows, tooled=bool(tools_offered[system_name]) or any(tool_steps(r) for r in ok_system_rows)))
            if (accuracy := route_accuracy(ok_system_rows)) is not None:
                row["route_accuracy"] = accuracy
            row.update(self._agent_fields([r["agent"] for r in ok_rows if r["system"] == system_name and r.get("agent")]))
            row.update(self._latency_fields(latency_rows["latency_ms"].tolist(), self.probe_results.get(system_name)))
            if runtime_df is not None and not runtime_df.empty:
                runtime = runtime_df[runtime_df["system"] == system_name]
                row["system_wall_time_ms"] = float(runtime["system_wall_time_ms"].iloc[0]) if not runtime.empty else 0.0
            rows.append(row)
        return rows


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
