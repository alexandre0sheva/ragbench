from __future__ import annotations

import re

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudgeResult

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
}


def classify_failure(question: Question, answer: str, retrieval_metrics: dict[str, float], judge: AnswerJudgeResult) -> str:
    refused = "could not find the answer" in answer.lower() or "not available" in answer.lower()
    if not question.is_answerable:
        return "no_failure" if refused else "answer_hallucination"
    if retrieval_metrics.get("hit@5", 0.0) == 0.0 and judge.answer_score >= 4 and judge.faithfulness >= 4 and not refused:
        return "possible_qrels_gap"
    if retrieval_metrics.get("hit@5", 0.0) == 0.0:
        return "retrieval_miss"
    if refused:
        return "over_refusal"
    if judge.is_hallucinated:
        return "answer_hallucination"
    if judge.correctness >= 3.5 and judge.faithfulness >= 3:
        return "no_failure"
    answer_years = set(re.findall(r"\b20\d{2}\b", answer))
    ref_years = set(re.findall(r"\b20\d{2}\b", question.reference_answer or ""))
    if ref_years and answer_years and not ref_years.intersection(answer_years):
        return "wrong_date"
    expected_names = [kw for kw in question.expected_keywords if any(part[:1].isupper() for part in kw.split())]
    if expected_names and not any(name.lower() in answer.lower() for name in expected_names):
        return "wrong_entity"
    if judge.correctness > 1.5:
        return "partial_answer"
    return "insufficient_context"
