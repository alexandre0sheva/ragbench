from __future__ import annotations

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudgeResult
from ragbench.evaluation.answer_metrics import is_refusal, keyword_recall

FAILURE_TYPES = {
    "no_failure",
    "possible_qrels_gap",
    "retrieval_miss",
    "bad_reranking",
    "insufficient_context",
    "answer_hallucination",
    "partial_answer",
    "wrong_entity",
    "wrong_date",
    "over_refusal",
    "format_error",
    "run_error",
}


DATE_ANSWER_TYPES = frozenset({"date"})
ENTITY_ANSWER_TYPES = frozenset({"entity", "person", "name", "organization"})


def classify_failure(
    question: Question,
    answer: str,
    retrieval_metrics: dict[str, float],
    judge: AnswerJudgeResult,
    primary_k: int = 5,
    context_recall: float | None = None,
) -> str:
    """Why a question went wrong, from structured signals only (no regexes over the answer text).

    `context_recall` is the share of relevant documents the generator was *given* (None when unmeasured), `retrieval_metrics`
    describe the whole ranking (an empty dict means the system retrieves nothing, so no hit/miss verdict applies), and the
    judge supplies scores and flags. The wrong value of a wrong answer is typed by `answer_type` (`date` -> `wrong_date`,
    `entity` / `person` / `name` / `organization` -> `wrong_entity`) when its expected keywords are missing from the answer.
    """
    if not answer.strip():
        return "format_error"
    refused = is_refusal(answer)
    if not question.is_answerable:
        return "no_failure" if refused else "answer_hallucination"
    hit = retrieval_metrics.get(f"hit@{primary_k}", 0.0) if retrieval_metrics else None
    # What reached the generator is what counts; the ranking cut-off only stands in for it when that is unmeasured.
    evidence_missing = context_recall == 0.0 if context_recall is not None else hit == 0.0
    if evidence_missing:
        if judge.answer_score >= 4 and judge.faithfulness >= 4 and not refused:
            return "possible_qrels_gap"
        found_deeper = any(value > 0 for key, value in retrieval_metrics.items() if key.startswith("hit@"))
        return "bad_reranking" if found_deeper else "retrieval_miss"  # ranked, but below what the generator reads, or not found at all
    partial_evidence = context_recall is not None and context_recall < 1.0
    if refused:
        return "insufficient_context" if partial_evidence else "over_refusal"
    if judge.is_hallucinated:
        return "answer_hallucination"
    if judge.correctness >= 3.5 and judge.faithfulness >= 3:
        return "no_failure"
    if partial_evidence:
        return "insufficient_context"
    value_missing = not question.expected_keywords or keyword_recall(answer, question.expected_keywords) < 0.5
    if value_missing and question.answer_type in DATE_ANSWER_TYPES:
        return "wrong_date"
    if value_missing and question.answer_type in ENTITY_ANSWER_TYPES:
        return "wrong_entity"
    if judge.correctness > 1.5:
        return "partial_answer"
    if context_recall is None:
        return "insufficient_context"  # nothing is known about what the generator saw
    return "partial_answer" if judge.is_supported_by_context else "answer_hallucination"
