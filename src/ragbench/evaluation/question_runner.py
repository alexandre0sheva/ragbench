"""One question through one system: answer it, score the retrieval, the context and the answer, classify the failure.

Every system is asked the same way (same retrieval depth, same context size), and the judge sees only the chunks the generator was given.
"""

from __future__ import annotations

import logging
from typing import Any

from ragbench.config.schema import EvaluationConfig, SystemConfig
from ragbench.datasets.schema import has_relevance_labels
from ragbench.evaluation.answer_checks import AnswerChecks
from ragbench.evaluation.context_metrics import context_precision, context_recall_doc
from ragbench.evaluation.failure_analysis import classify_failure
from ragbench.evaluation.judge_panel import JudgePanel
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics
from ragbench.evaluation.rows import build_question_rows, error_result
from ragbench.models.refs import system_model_refs
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem

logger = logging.getLogger(__name__)


def _error_info(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def answer(system: BaseRAGSystem, question, evaluation: EvaluationConfig) -> AnswerResult:
    context_k = system.configured_context_k() or evaluation.context_k
    depth = max(evaluation.resolved_retrieval_depth, context_k)
    return system.answer_question(question.question, top_k=depth, context_k=context_k)


def evaluate_question(
    system: BaseRAGSystem,
    system_config: SystemConfig,
    question,
    qrels: dict[str, int],
    judge: JudgePanel,
    evaluation: EvaluationConfig,
    checks: AnswerChecks | None = None,
) -> dict[str, Any]:
    """The question's rows; a question that raises becomes an error result instead of failing the run."""
    try:
        return _score_question(system, system_config, question, qrels, judge, evaluation, checks or AnswerChecks())
    except Exception as exc:
        logger.warning("Question %s failed for system %s: %s: %s", question.id, system.name, type(exc).__name__, exc)
        return error_result(system, system_config, question, _error_info(exc))


def _score_question(
    system: BaseRAGSystem,
    system_config: SystemConfig,
    question,
    qrels: dict[str, int],
    judge: JudgePanel,
    evaluation: EvaluationConfig,
    checks: AnswerChecks,
) -> dict[str, Any]:
    result = answer(system, question, evaluation)
    # Retrieval is scored on the whole ranking; the judge sees only what the generator was given.
    context_ids = result.metadata.get("context_chunk_ids")
    ranked = result.retrieval_result.chunks
    in_context = {chunk.chunk_id for chunk in ranked} if context_ids is None else set(context_ids)
    spec = system.spec
    # A system that retrieves nothing has no retrieval metrics (an empty dict); its rows get blanks, never zeros.
    retrieves = spec is None or spec.retrieves
    # ... and so does a question that has an answer but no labels to score the ranking against (label-free mode); an unanswerable one still scores 0.
    scorable = retrieves and (has_relevance_labels(question, qrels) or not question.is_answerable)
    retrieval_metrics = compute_retrieval_metrics(ranked, question.relevant_doc_ids, qrels=qrels, k_values=evaluation.k_values, evidence=question.evidence) if scorable else {}
    context_chunks = [chunk for chunk in ranked if chunk.chunk_id in in_context]
    # Same notion of "relevant" as the retrieval metrics: the question's own list, else the positive qrels.
    relevant = question.relevant_doc_ids or [doc_id for doc_id, grade in qrels.items() if grade > 0]
    context_metrics = {
        "context_recall": context_recall_doc(context_chunks, relevant) if retrieves else None,
        "context_precision": context_precision(context_chunks, relevant) if retrieves else None,
    }
    judge_result = judge.judge(question, result.answer, context_chunks, generator=system_model_refs(system_config).llm[0])
    refusal = checks.refusal.detect(question.question, result.answer)
    claims = checks.claims.check(question.question, result.answer, context_chunks) if checks.claims is not None else None
    failure_type = classify_failure(
        question,
        result.answer,
        retrieval_metrics,
        judge_result,
        primary_k=evaluation.resolved_primary_k,
        context_recall=context_metrics["context_recall"],
        refused=refusal.refused,
    )
    return build_question_rows(
        system,
        system_config,
        question,
        result,
        retrieval_metrics,
        judge_result,
        failure_type,
        in_context,
        context_metrics,
        evaluation.k_values,
        refusal=refusal,
        claims=claims,
    )
