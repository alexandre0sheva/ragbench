from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ragbench.cache import CacheRuntime, DiskCache, activate_cache, summarize_cache
from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents
from ragbench.evaluation.answer_judge import AnswerJudge
from ragbench.evaluation.failure_analysis import classify_failure
from ragbench.evaluation.manifest import build_manifest, dataset_hash, utc_now_iso
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics
from ragbench.models import cost as pricing
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem
from ragbench.rag_systems.trace import STAGE_KEYS, UNTRACKED, mean_by_stage, stage_costs, step_to_dict
from ragbench.registry import SYSTEMS
from ragbench.reporting.html_report import write_html_report
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
from ragbench.utils.env import has_openai_key
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
    ):
        self.config_path = config_path
        self.use_cache = use_cache
        self.config: ExperimentConfig = load_config(config_path)
        self.raw_config = load_config_dict(config_path)
        self.force_mock = force_mock
        self.mode = "mock" if force_mock or not has_openai_key() else "live"
        self.models_used: set[str] = set()
        self.started_utc = utc_now_iso()
        self.dataset_digest = ""
        self.cache_runtime: CacheRuntime | None = None
        self.cache_reason: str | None = None
        # Systems and questions run on worker threads; serializing the hooks means listeners need no locking of their own.
        self.progress = SerializedProgress(progress or ProgressListener())
        configured_workers = int(max_workers if max_workers is not None else self.config.evaluation.max_workers)
        self.max_workers = max(1, configured_workers)
        self.system_workers = max(1, int(system_workers if system_workers is not None else self.config.evaluation.system_workers))
        self.probe_results: dict[str, ProbeResult] = {}
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

    def run(self) -> Path:
        """Execute the configured benchmark and return the output directory path."""
        pricing.clear_pricing_overrides()
        pricing.reset_unknown_priced_models()
        pricing.register_pricing({model: price.model_dump() for model, price in self.config.pricing.items()})
        self.cache_runtime = self._open_cache()
        runtime = RuntimeContext(ingest_workers=self.config.evaluation.ingest_workers, limiters=build_limiters(self.config.limits))
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
        cache_dir = Path(os.environ.get("RAGBENCH_CACHE_DIR") or self.config.cache.dir)
        disk = DiskCache(cache_dir / "cache.sqlite3", ttl_days=self.config.cache.ttl_days)
        return CacheRuntime(disk=disk, config=self.config.cache)

    def _run(self) -> Path:
        run_start = perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.config_path, self.output_dir / "config.yaml")
        EMBEDDING_CACHE.clear()
        EMBEDDING_CACHE.enabled = self.config.evaluation.embedding_cache
        documents = load_documents(self.config.dataset.documents_path)
        dataset = load_dataset(self.config.dataset.questions_path, self.config.dataset.qrels_path)
        self.dataset_warnings = validate_dataset(documents, dataset)
        self.dataset_digest = dataset_hash(documents, self.config.dataset.questions_path, self.config.dataset.qrels_path)
        questions = dataset.questions[: self.config.evaluation.max_questions] if self.config.evaluation.max_questions else dataset.questions
        self.progress.run_started(len(self.config.systems), len(questions))
        judge = AnswerJudge(
            model_name=self.config.evaluation.judge_model,
            enabled=self.config.evaluation.judge_enabled,
            force_mock=self.force_mock,
        )
        if judge.enabled:
            self.models_used.add(judge.llm.model_name)

        self._qrels, self._judge = dataset.qrels, judge
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
        runner = SystemRunner(self, settings, self.progress)
        outcomes = runner.run_all(self.config.systems, documents, questions)
        # Timings from the parallel pass include queueing behind other threads. A live run therefore re-times a few
        # questions one at a time afterwards; mock runs skip it (their latencies are microseconds of CPU).
        self.probe_results = runner.probe(outcomes, questions) if self._should_probe(questions) else {}
        self._collect(outcomes, per_question_rows, retrieval_rows, answer_rows, cost_rows, ingestion_rows, system_runtime_rows)

        run_wall_time_ms = (perf_counter() - run_start) * 1000
        self._write_outputs(per_question_rows, retrieval_rows, answer_rows, cost_rows, ingestion_rows, system_runtime_rows, run_wall_time_ms)
        failed = self._failed_systems(per_question_rows, len(questions))
        if failed:
            raise BenchmarkRunError(self.output_dir, failed, self.config.evaluation.max_error_rate)
        return self.output_dir

    def _modules_to_warm_up(self) -> list[str]:
        """Libraries that worker threads would otherwise import lazily, racing with live API calls (see `warm_up_imports`)."""
        modules = ["httpx", "openai"]
        for system_config in self.config.systems:
            spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
            if spec is not None and "vector_store" in spec.options.model_fields:
                if system_config.retrieval.get("vector_store", spec.options.model_fields["vector_store"].default) == "chroma":
                    modules.append("chromadb")
                    break
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
        return system

    def evaluate_single_question(self, system: BaseRAGSystem, system_config: SystemConfig, question) -> dict[str, Any]:
        assert self._judge is not None
        return self._evaluate_single_question(system, system_config, question, self._qrels.get(question.id, {}), self._judge)

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
        retrieval_metrics = compute_retrieval_metrics(ranked, question.relevant_doc_ids, qrels=qrels, k_values=evaluation.k_values)
        judge_result = judge.judge(question, answer.answer, [chunk for chunk in ranked if chunk.chunk_id in in_context])
        failure_type = classify_failure(question, answer.answer, retrieval_metrics, judge_result, primary_k=evaluation.resolved_primary_k)
        return self._build_question_rows(system, system_config, question, answer, retrieval_metrics, judge_result, failure_type, in_context)

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
            "steps": [],
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
    ) -> dict[str, Any]:
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
            "steps": [step_to_dict(step) for step in answer.steps],
        }
        return {
            "per_question": per_question,
            "retrieval_row": {"system": system.name, "question_id": question.id, "category": question.category, **retrieval_metrics},
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
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / "metrics_summary.csv", index=False)
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
        write_leaderboard(
            self.output_dir / "leaderboard.md",
            summary_rows,
            notices=build_notices(self.mode, unknown_priced, concurrent_latency=concurrent_latency),
            primary_k=self.config.evaluation.resolved_primary_k,
            stage_rows=self._build_stage_summary(per_question_rows),
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
                    "num_systems": len(self.config.systems),
                    "num_question_rows": len(per_question_rows),
                    "num_errors": sum(self._errors_by_system(per_question_rows).values()),
                    "errors_by_system": self._errors_by_system(per_question_rows),
                    "k_values": self.config.evaluation.k_values,
                    "retrieval_depth": self.config.evaluation.resolved_retrieval_depth,
                    "context_k": self.config.evaluation.context_k,
                    "primary_k": self.config.evaluation.resolved_primary_k,
                    "embedding_cache": EMBEDDING_CACHE.stats(),
                    "cache": cache_summary,
                    "dataset_warnings": getattr(self, "dataset_warnings", []),
                    "outputs": {
                        "leaderboard": "leaderboard.md",
                        "report": "report.html",
                        "qrels_audit": "qrels_audit.md",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        category_df = answer_df.groupby(["system", "category"], as_index=False)[["answer_score", "faithfulness"]].mean() if not answer_df.empty else pd.DataFrame()
        per_question_df = pd.DataFrame(per_question_rows)
        if not per_question_df.empty:
            classified_failures_df = per_question_df[per_question_df["failure_type"] != "no_failure"]
            failures_df = classified_failures_df.groupby(["system", "failure_type"], as_index=False).size().rename(columns={"size": "count"})
        else:
            failures_df = pd.DataFrame()
        write_html_report(
            self.output_dir / "report.html",
            run_id=self.run_id,
            summary_rows=summary_rows,
            category_rows=category_df.to_dict("records") if not category_df.empty else [],
            cost_rows=self._build_cost_summary(ingestion_rows, cost_rows),
            failure_rows=failures_df.to_dict("records") if not failures_df.empty else [],
            config_text=yaml.safe_dump(self.raw_config, sort_keys=False),
            run_meta={
                "mode": self.mode,
                "unknown_priced_models": unknown_priced,
                "primary_k": self.config.evaluation.resolved_primary_k,
                "num_systems": len(self.config.systems),
                "num_questions": len({row["question_id"] for row in per_question_rows}) if per_question_rows else 0,
                "run_wall_time_ms": run_wall_time_ms,
                "cache_hits": in_process["hits"] + cache_summary.get("hits", 0),
                "cache_saved_usd": in_process["saved_cost_usd"] + cache_summary.get("saved_cost_usd", 0.0),
            },
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

    @staticmethod
    def _build_cost_summary(ingestion_rows: list[dict[str, Any]], cost_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary: dict[str, dict[str, float]] = {}
        for row in ingestion_rows:
            entry = summary.setdefault(row["system"], {"ingestion_cost": 0.0, "query_cost": 0.0, "judge_cost": 0.0, "total_cost": 0.0})
            entry["ingestion_cost"] += float(row.get("total_cost", 0.0))
            entry["total_cost"] += float(row.get("total_cost", 0.0))
        for row in cost_rows:
            entry = summary.setdefault(row["system"], {"ingestion_cost": 0.0, "query_cost": 0.0, "judge_cost": 0.0, "total_cost": 0.0})
            judge = float(row.get("judge_cost", 0.0))
            total = float(row.get("total_cost", 0.0))
            entry["judge_cost"] += judge
            entry["query_cost"] += total - judge
            entry["total_cost"] += total
        return [{"system": system, **costs} for system, costs in summary.items()]

    def _build_stage_summary(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mean per-question cost and latency by stage for each system (failed questions excluded)."""

        def rollup(steps: list[dict[str, Any]], value: Callable[[dict[str, Any]], float]) -> dict[str, float]:
            rolled = dict.fromkeys(STAGE_KEYS, 0.0)
            for step in steps:
                rolled[UNTRACKED if step["name"] == UNTRACKED else step["kind"]] += value(step)
            return rolled

        rows: list[dict[str, Any]] = []
        for cfg in self.config.systems:
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
            "latency_ms_p50": float(np.percentile(values, 50)) if values else None,
            "latency_ms_p95": float(np.percentile(values, 95)) if values else None,
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
        system_types = {cfg.resolved_name: cfg.type for cfg in self.config.systems}
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
                    row[f"retrieval_{col}"] = float(r[col].mean()) if not r.empty else None
            for col in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality", "answer_score"]:
                row[col] = float(a[col].mean()) if not a.empty else None
            row["avg_cost_per_question"] = float(c["total_cost"].mean()) if not c.empty else None
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
    return names


def run_benchmark(
    config_path: Path,
    force_mock: bool = False,
    max_workers: int | None = None,
    progress: ProgressListener | None = None,
    use_cache: bool = True,
    system_workers: int | None = None,
) -> Path:
    """Convenience wrapper: build a `BenchmarkEvaluator` and run it.

    Returns the timestamped output directory containing leaderboard, reports,
    and per-question results.
    """
    return BenchmarkEvaluator(
        config_path, force_mock=force_mock, max_workers=max_workers, progress=progress, use_cache=use_cache, system_workers=system_workers
    ).run()
