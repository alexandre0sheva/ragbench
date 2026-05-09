from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.loader import load_dataset
from ragbench.documents.loaders import load_documents
from ragbench.evaluation.answer_judge import AnswerJudge
from ragbench.evaluation.failure_analysis import classify_failure
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.base import BaseRAGSystem
from ragbench.reporting.html_report import write_html_report
from ragbench.reporting.markdown_report import write_failures, write_leaderboard, write_qrels_audit
from ragbench.utils.jsonl import write_jsonl
from ragbench.utils.text import truncate, unique_preserve_order


class BenchmarkEvaluator:
    """Top-level orchestrator: ingests once per system, evaluates every question, writes reports.

    A single `BenchmarkEvaluator` corresponds to a single config + run. Output is
    written to a timestamped subdirectory under `config.run.output_dir` and the
    method `run()` returns that path.

    Per-question evaluation is parallelized across questions within each system
    via a `ThreadPoolExecutor` (size from `evaluation.max_workers`). Order of
    output rows is preserved regardless of completion order.
    """

    def __init__(self, config_path: Path, force_mock: bool = False, max_workers: int | None = None):
        self.config_path = config_path
        self.config: ExperimentConfig = load_config(config_path)
        self.raw_config = load_config_dict(config_path)
        self.force_mock = force_mock
        configured_workers = int(max_workers if max_workers is not None else self.config.evaluation.max_workers)
        self.max_workers = max(1, configured_workers)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.config.run.name}_{timestamp}"
        self.output_dir = self.config.run.output_dir / self.run_id

    def run(self) -> Path:
        """Execute the configured benchmark and return the output directory path."""
        run_start = perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.config_path, self.output_dir / "config.yaml")
        documents = load_documents(self.config.dataset.documents_path)
        dataset = load_dataset(self.config.dataset.questions_path, self.config.dataset.qrels_path)
        questions = dataset.questions[: self.config.evaluation.max_questions] if self.config.evaluation.max_questions else dataset.questions
        judge = AnswerJudge(
            model_name=self.config.evaluation.judge_model,
            enabled=self.config.evaluation.judge_enabled,
            force_mock=self.force_mock,
        )

        per_question_rows: list[dict[str, Any]] = []
        retrieval_rows: list[dict[str, Any]] = []
        answer_rows: list[dict[str, Any]] = []
        cost_rows: list[dict[str, Any]] = []
        ingestion_rows: list[dict[str, Any]] = []
        system_runtime_rows: list[dict[str, Any]] = []

        for system_config in self.config.systems:
            system_start = perf_counter()
            system = create_rag_system(system_config, force_mock=self.force_mock)
            ingest_start = perf_counter()
            ingestion = system.ingest(documents)
            ingestion_wall_time_ms = (perf_counter() - ingest_start) * 1000
            ingestion_rows.append(
                {
                    "system": system.name,
                    "system_type": system_config.type,
                    "stage": "ingestion",
                    "num_documents": ingestion.num_documents,
                    "num_chunks": ingestion.num_chunks,
                    "latency_ms": ingestion.latency_ms,
                    **ingestion.cost.as_dict(),
                }
            )
            question_start = perf_counter()
            system_results = self._evaluate_questions_for_system(system, system_config, questions, dataset.qrels, judge)
            question_wall_time_ms = (perf_counter() - question_start) * 1000
            for result in system_results:
                per_question_rows.append(result["per_question"])
                retrieval_rows.append(result["retrieval_row"])
                answer_rows.append(result["answer_row"])
                cost_rows.append(result["cost_row"])
            system_runtime_rows.append(
                {
                    "system": system.name,
                    "system_type": system_config.type,
                    "ingestion_wall_time_ms": ingestion_wall_time_ms,
                    "question_wall_time_ms": question_wall_time_ms,
                    "system_wall_time_ms": (perf_counter() - system_start) * 1000,
                    "num_questions": len(questions),
                    "max_workers": self.max_workers,
                }
            )

        run_wall_time_ms = (perf_counter() - run_start) * 1000
        self._write_outputs(per_question_rows, retrieval_rows, answer_rows, cost_rows, ingestion_rows, system_runtime_rows, run_wall_time_ms)
        return self.output_dir

    def _evaluate_questions_for_system(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        questions: list,
        qrels_by_question: dict[str, dict[str, int]],
        judge: AnswerJudge,
    ) -> list[dict[str, Any]]:
        if self.max_workers == 1 or len(questions) <= 1:
            return [
                self._evaluate_single_question(system, system_config, question, qrels_by_question.get(question.id, {}), judge)
                for question in questions
            ]

        ordered_results: list[dict[str, Any] | None] = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"ragbench-{system.name}") as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_question,
                    system,
                    system_config,
                    question,
                    qrels_by_question.get(question.id, {}),
                    judge,
                ): idx
                for idx, question in enumerate(questions)
            }
            for future in as_completed(futures):
                ordered_results[futures[future]] = future.result()
        return [result for result in ordered_results if result is not None]

    def _evaluate_single_question(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        question,
        qrels: dict[str, int],
        judge: AnswerJudge,
    ) -> dict[str, Any]:
        answer = system.answer_question(question.question)
        retrieval_metrics = compute_retrieval_metrics(
            answer.retrieval_result.chunks,
            question.relevant_doc_ids,
            qrels=qrels,
            k_values=self.config.evaluation.k_values,
        )
        judge_result = judge.judge(question, answer.answer, answer.retrieval_result.chunks)
        failure_type = classify_failure(question, answer.answer, retrieval_metrics, judge_result)
        return self._build_question_rows(system, system_config, question, answer, retrieval_metrics, judge_result, failure_type)

    def _build_question_rows(
        self,
        system: BaseRAGSystem,
        system_config: SystemConfig,
        question,
        answer,
        retrieval_metrics: dict[str, Any],
        judge_result,
        failure_type: str,
    ) -> dict[str, Any]:
        total_cost = answer.cost.plus(judge_result.cost)
        retrieved_contexts = [
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
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
        write_leaderboard(self.output_dir / "leaderboard.md", summary_rows)
        write_failures(self.output_dir / "failures.md", per_question_rows)
        qrels_audit_rows = self._build_qrels_audit(per_question_rows)
        pd.DataFrame(qrels_audit_rows).to_csv(self.output_dir / "qrels_audit.csv", index=False)
        write_qrels_audit(self.output_dir / "qrels_audit.md", qrels_audit_rows)
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "run_wall_time_ms": run_wall_time_ms,
                    "max_workers": self.max_workers,
                    "num_systems": len(self.config.systems),
                    "num_question_rows": len(per_question_rows),
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
            failures_df = classified_failures_df.groupby(["system", "failure_type"], as_index=False).size()
        else:
            failures_df = pd.DataFrame()
        write_html_report(
            self.output_dir / "report.html",
            run_id=self.run_id,
            leaderboard_html=summary_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
            category_html=category_df.to_html(index=False, float_format=lambda x: f"{x:.3f}") if not category_df.empty else "<p>No category data.</p>",
            cost_html=cost_df.to_html(index=False, float_format=lambda x: f"{x:.6f}"),
            failures_html=failures_df.to_html(index=False) if not failures_df.empty else "<p>No failures.</p>",
            config_text=yaml.safe_dump(self.raw_config, sort_keys=False),
        )

    def _build_qrels_audit(self, per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        audit_rows: list[dict[str, Any]] = []
        for row in per_question_rows:
            metrics = row["retrieval_metrics"]
            judge = row["answer_judge"]
            relevant_doc_ids = row.get("relevant_doc_ids", [])
            if not relevant_doc_ids:
                continue
            if metrics.get("recall@5", 0.0) >= 1.0:
                continue
            if judge.get("answer_score", 0.0) < 4.0 or judge.get("faithfulness", 0.0) < 4.0:
                continue
            retrieved_doc_ids = unique_preserve_order(context["doc_id"] for context in row.get("retrieved_contexts", []))
            unlabeled = [doc_id for doc_id in retrieved_doc_ids if doc_id not in set(relevant_doc_ids)]
            if not unlabeled:
                continue
            audit_rows.append(
                {
                    "system": row["system"],
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "severity": "high" if metrics.get("hit@5", 0.0) == 0.0 else "medium",
                    "recall@5": metrics.get("recall@5", 0.0),
                    "answer_score": judge.get("answer_score", 0.0),
                    "faithfulness": judge.get("faithfulness", 0.0),
                    "labeled_relevant_doc_ids": ", ".join(relevant_doc_ids),
                    "unlabeled_retrieved_doc_ids": ", ".join(unlabeled),
                }
            )
        return audit_rows

    def _build_summary(
        self,
        retrieval_df: pd.DataFrame,
        answer_df: pd.DataFrame,
        cost_df: pd.DataFrame,
        per_question_rows: list[dict[str, Any]],
        runtime_df: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        system_types = {cfg.resolved_name: cfg.type for cfg in self.config.systems}
        latency = pd.DataFrame([{"system": row["system"], "latency_ms": row["latency_ms"]} for row in per_question_rows])
        rows: list[dict[str, Any]] = []
        for system_name in system_types:
            r = retrieval_df[retrieval_df["system"] == system_name]
            a = answer_df[answer_df["system"] == system_name]
            c = cost_df[cost_df["system"] == system_name]
            latency_rows = latency[latency["system"] == system_name]
            row: dict[str, Any] = {"system": system_name, "system_type": system_types[system_name]}
            for col in r.columns:
                if col not in {"system", "question_id", "category"}:
                    row[f"retrieval_{col}"] = float(r[col].mean()) if not r.empty else 0.0
            for col in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality", "answer_score"]:
                row[col] = float(a[col].mean()) if not a.empty else 0.0
            row["avg_cost_per_question"] = float(c["total_cost"].mean()) if not c.empty else 0.0
            row["avg_latency_ms"] = float(latency_rows["latency_ms"].mean()) if not latency_rows.empty else 0.0
            if runtime_df is not None and not runtime_df.empty:
                runtime = runtime_df[runtime_df["system"] == system_name]
                row["system_wall_time_ms"] = float(runtime["system_wall_time_ms"].iloc[0]) if not runtime.empty else 0.0
            rows.append(row)
        return rows


def run_benchmark(config_path: Path, force_mock: bool = False, max_workers: int | None = None) -> Path:
    """Convenience wrapper: build a `BenchmarkEvaluator` and run it.

    Returns the timestamped output directory containing leaderboard, reports,
    and per-question results.
    """
    return BenchmarkEvaluator(config_path, force_mock=force_mock, max_workers=max_workers).run()
