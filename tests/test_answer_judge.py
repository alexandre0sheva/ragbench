from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudge, heuristic_judge
from ragbench.rag_systems.base import RetrievedChunk


def _chunk(doc_id: str, text: str, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}::chunk::0",
        doc_id=doc_id,
        text=text,
        score=1.0,
        rank=rank,
    )


def test_heuristic_judge_rewards_keyword_overlap():
    question = Question(
        id="q1",
        question="Who won the IIOTY award in 2023?",
        reference_answer="Maxine Thompson won the Insurance Innovator of the Year award in 2023.",
        expected_keywords=["Maxine Thompson", "Insurance Innovator", "2023"],
        relevant_doc_ids=["doc_005"],
    )
    contexts = [_chunk("doc_005", "Maxine Thompson received the Insurance Innovator of the Year award in 2023.")]
    answer = "Maxine Thompson won the Insurance Innovator of the Year award in 2023."

    result = heuristic_judge(question, answer, contexts)

    assert result.correctness >= 4.0
    assert result.faithfulness >= 4.0
    assert result.is_supported_by_context is True


def test_heuristic_judge_penalizes_unsupported_answer():
    question = Question(
        id="q2",
        question="Where is HarborShield deployed?",
        reference_answer="HarborShield ships in North America.",
        expected_keywords=["HarborShield", "North America"],
        relevant_doc_ids=["doc_001"],
    )
    contexts = [_chunk("doc_999", "Unrelated content about claims triage.")]
    answer = "HarborShield ships exclusively in Antarctica."

    result = heuristic_judge(question, answer, contexts)

    assert result.faithfulness < 3.0
    assert result.is_hallucinated is True


def test_heuristic_judge_rewards_refusal_for_unanswerable():
    question = Question(
        id="q3",
        question="What color is HarborShield?",
        reference_answer=None,
        expected_keywords=[],
        relevant_doc_ids=[],  # empty = unanswerable
    )
    contexts = [_chunk("doc_001", "Some context.")]
    refusal = "I could not find the answer in the provided documents."

    result = heuristic_judge(question, refusal, contexts)

    assert result.correctness == 5.0
    assert result.is_supported_by_context is True
    assert result.is_hallucinated is False


def test_heuristic_judge_punishes_hallucination_on_unanswerable():
    question = Question(
        id="q4",
        question="What color is HarborShield?",
        reference_answer=None,
        expected_keywords=[],
        relevant_doc_ids=[],
    )
    contexts = [_chunk("doc_001", "Some context.")]
    invented = "HarborShield is bright pink."

    result = heuristic_judge(question, invented, contexts)

    assert result.correctness <= 1.0
    assert result.is_hallucinated is True


def test_heuristic_judge_rewards_correct_citation():
    question = Question(
        id="q5",
        question="Who won the award?",
        reference_answer="Maxine Thompson.",
        expected_keywords=["Maxine Thompson"],
        relevant_doc_ids=["doc_005"],
    )
    contexts = [_chunk("doc_005", "Maxine Thompson won.")]
    cited = "Maxine Thompson won the award [doc_005]."

    result = heuristic_judge(question, cited, contexts)

    assert result.citation_quality == 5.0


def test_answer_judge_in_mock_mode_uses_heuristic_path(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = AnswerJudge(force_mock=True)
    question = Question(
        id="q6",
        question="Who won?",
        reference_answer="Alice.",
        expected_keywords=["Alice"],
        relevant_doc_ids=["doc_001"],
    )
    result = judge.judge(question, "Alice won.", [_chunk("doc_001", "Alice won the trophy.")])
    assert result.metadata.get("judge") == "heuristic_fallback"
    assert result.cost.total_cost == 0.0
