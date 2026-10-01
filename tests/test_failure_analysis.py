from __future__ import annotations

import pytest

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudgeResult
from ragbench.evaluation.failure_analysis import FAILURE_TYPES, classify_failure

REFUSAL = "I could not find the answer in the provided documents."
GOOD = AnswerJudgeResult(correctness=5, faithfulness=5, completeness=5, relevance=5, citation_quality=5, is_supported_by_context=True)
WRONG = AnswerJudgeResult(correctness=1, faithfulness=4, completeness=1, relevance=3, citation_quality=2, is_supported_by_context=True)
PARTIAL = AnswerJudgeResult(correctness=2.5, faithfulness=4, completeness=2, relevance=4, citation_quality=3, is_supported_by_context=True)
HALLUCINATED = AnswerJudgeResult(correctness=1, faithfulness=1, completeness=1, relevance=3, citation_quality=0, is_hallucinated=True)
SEEN_NOTHING = {"hit@5": 0.0, "hit@10": 0.0}
RANKED_TOO_LOW = {"hit@5": 0.0, "hit@10": 1.0}
FOUND = {"hit@5": 1.0, "hit@10": 1.0}


def _question(answer_type: str = "single_fact", answerable: bool = True) -> Question:
    return Question(
        id="q",
        question="When did Atlas ship?",
        reference_answer="Atlas shipped in 2023." if answerable else None,
        expected_keywords=["2023"] if answerable else [],
        relevant_doc_ids=["doc_001"] if answerable else [],
        answer_type=answer_type,
    )


def classify(answer: str, judge: AnswerJudgeResult, metrics: dict | None = None, context_recall: float | None = None, **question) -> str:
    return classify_failure(_question(**question), answer, FOUND if metrics is None else metrics, judge, primary_k=5, context_recall=context_recall)


def test_failure_type_names_are_stable():
    assert {"no_failure", "possible_qrels_gap", "retrieval_miss", "bad_reranking", "insufficient_context", "answer_hallucination", "partial_answer"} <= FAILURE_TYPES
    assert {"wrong_entity", "wrong_date", "over_refusal", "format_error", "run_error"} <= FAILURE_TYPES


def test_unanswerable_questions_are_judged_on_the_refusal_alone():
    assert classify(REFUSAL, GOOD, {}, answerable=False) == "no_failure"
    assert classify("Atlas shipped in 2023.", HALLUCINATED, {}, answerable=False) == "answer_hallucination"


def test_an_empty_answer_is_a_format_error():
    assert classify("   ", WRONG) == "format_error"


def test_what_the_generator_saw_decides_between_retrieval_and_ranking_failures():
    assert classify("It shipped in 2020.", WRONG, SEEN_NOTHING, context_recall=0.0) == "retrieval_miss"
    assert classify("It shipped in 2020.", WRONG, RANKED_TOO_LOW, context_recall=0.0) == "bad_reranking"
    assert classify(REFUSAL, WRONG, RANKED_TOO_LOW, context_recall=0.0) == "bad_reranking"


def test_a_well_supported_answer_without_labeled_evidence_points_at_the_qrels():
    assert classify("Atlas shipped in 2023.", GOOD, SEEN_NOTHING, context_recall=0.0) == "possible_qrels_gap"


def test_context_recall_wins_over_the_ranking_cutoff():
    # hit@5 says miss, but the generator read ten chunks and the evidence was among them.
    assert classify("Atlas shipped in 2023.", GOOD, SEEN_NOTHING, context_recall=1.0) == "no_failure"


def test_refusal_with_the_evidence_in_hand_is_over_refusal_and_with_part_of_it_is_insufficient_context():
    assert classify(REFUSAL, WRONG, FOUND, context_recall=1.0) == "over_refusal"
    assert classify(REFUSAL, WRONG, FOUND, context_recall=0.5) == "insufficient_context"
    assert classify(REFUSAL, WRONG, FOUND) == "over_refusal"  # recall not measured: the legacy behaviour


def test_a_wrong_answer_from_partial_evidence_is_insufficient_context():
    assert classify("Atlas shipped in 2020.", WRONG, FOUND, context_recall=0.5) == "insufficient_context"


def test_judge_flags_and_scores_classify_answers_that_had_the_evidence():
    assert classify("It is pink.", HALLUCINATED, FOUND, context_recall=1.0) == "answer_hallucination"
    assert classify("Atlas shipped in 2023.", GOOD, FOUND, context_recall=1.0) == "no_failure"
    assert classify("Sometime in 2023, maybe.", PARTIAL, FOUND, context_recall=1.0) == "partial_answer"


@pytest.mark.parametrize(
    ("answer_type", "expected"),
    [("date", "wrong_date"), ("entity", "wrong_entity"), ("single_fact", "answer_hallucination"), ("unknown", "answer_hallucination")],
)
def test_wrong_value_is_typed_by_the_questions_answer_type_not_by_regexes(answer_type, expected):
    unsupported = AnswerJudgeResult(correctness=1, faithfulness=2, completeness=1, relevance=3, citation_quality=0, is_supported_by_context=False)
    assert classify("It shipped last spring.", unsupported, FOUND, context_recall=1.0, answer_type=answer_type) == expected


def test_a_wrong_year_alone_no_longer_makes_a_wrong_date():
    assert classify("Atlas shipped in 2024.", PARTIAL, FOUND, context_recall=1.0) == "partial_answer"


def test_systems_that_retrieve_nothing_are_never_retrieval_misses():
    assert classify("It is pink.", WRONG, {}) == "insufficient_context"  # nothing is known about the evidence: the legacy fallback
