import json

import pytest

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudge, heuristic_judge
from ragbench.evaluation.judge_prompts import JUDGE_PROMPT_VERSION, build_messages
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, LLMResult
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
    assert result.metadata.get("judge") == "heuristic" and result.metadata.get("fallback") is False
    assert result.cost.total_cost == 0.0


class ScriptedLLM(LLM):
    """Replies with the scripted texts in order (the last one repeats) and charges a fixed cost per call."""

    model_name = "scripted-judge"

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        self.calls.append({"messages": messages, "temperature": temperature, "json_mode": json_mode})
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return LLMResult(text=text, model=self.model_name, prompt_tokens=100, completion_tokens=20, cost=CostBreakdown(llm_cost=0.01))


def _reply(score: float, **flags) -> str:
    axes = {axis: score for axis in ("correctness", "faithfulness", "completeness", "relevance", "citation_quality")}
    return json.dumps({**axes, "is_supported_by_context": True, "is_hallucinated": False, "reasoning": f"scored {score}", **flags})


QUESTION = Question(id="q", question="Who won?", reference_answer="Alice.", expected_keywords=["Alice"], relevant_doc_ids=["doc_001"])
CONTEXT = [_chunk("doc_001", "Alice won the trophy.")]


def _live_judge(llm: LLM, **kwargs) -> AnswerJudge:
    return AnswerJudge(llm=llm, **kwargs)


def test_judge_with_one_sample_records_model_prompt_version_and_no_error():
    llm = ScriptedLLM(_reply(4))
    result = _live_judge(llm).judge(QUESTION, "Alice won.", CONTEXT)

    assert result.answer_score == 4.0 and result.reasoning == "scored 4"
    assert result.metadata == {
        "judge": "scripted-judge",
        "prompt_version": JUDGE_PROMPT_VERSION,
        "parse_error": False,
        "fallback": False,
        "retries": 0,
        "samples": 1,
    }
    assert result.cost.judge_cost == pytest.approx(0.01) and result.cost.judge_prompt_tokens == 100


def test_judge_with_several_samples_averages_them_and_records_the_variance():
    llm = ScriptedLLM(_reply(2), _reply(4), _reply(3))
    result = _live_judge(llm, samples=3, temperature=0.7).judge(QUESTION, "Alice won.", CONTEXT)

    assert len(llm.calls) == 3 and all(call["temperature"] == 0.7 for call in llm.calls)
    assert result.correctness == pytest.approx(3.0) and result.answer_score == pytest.approx(3.0)
    assert result.metadata["samples"] == 3
    assert result.metadata["answer_score_variance"] == pytest.approx(2 / 3)  # population variance of 2, 4, 3
    assert result.cost.judge_cost == pytest.approx(0.03)  # every sample is paid for


def test_judge_flags_use_a_majority_vote_across_samples():
    llm = ScriptedLLM(_reply(4, is_hallucinated=True), _reply(4, is_hallucinated=True), _reply(4, is_supported_by_context=False))
    result = _live_judge(llm, samples=3, temperature=0.5).judge(QUESTION, "Alice won.", CONTEXT)
    assert result.is_hallucinated is True and result.is_supported_by_context is True


def test_unparseable_reply_is_retried_and_counted():
    llm = ScriptedLLM("Sure! Here you go: not json", _reply(5))
    result = _live_judge(llm).judge(QUESTION, "Alice won.", CONTEXT)

    assert result.answer_score == 5.0
    assert result.metadata["judge"] == "scripted-judge" and result.metadata["retries"] == 1
    assert result.metadata["parse_error"] is True and result.metadata["fallback"] is False
    assert result.cost.judge_cost == pytest.approx(0.02)  # the failed attempt cost money too


def test_judge_that_never_parses_falls_back_to_the_heuristic_for_that_row_only_and_says_so():
    llm = ScriptedLLM("nope", "still nope", "never")
    result = _live_judge(llm).judge(QUESTION, "Alice won.", CONTEXT)

    assert result.metadata["judge"] == "heuristic"
    assert result.metadata["fallback"] is True and result.metadata["parse_error"] is True
    assert result.metadata["retries"] == 1 and len(llm.calls) == 2  # one retry per sample, then give up
    assert result.raw_output == "still nope"
    assert result.cost.judge_cost == pytest.approx(0.02)


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"correctness": 4, "faithfulness": 4}),  # missing axes must not silently become zeros
        json.dumps({"correctness": "high", "faithfulness": 4, "completeness": 4, "relevance": 4, "citation_quality": 4}),
        json.dumps([1, 2, 3]),
        "",
    ],
)
def test_incomplete_or_malformed_judgments_are_parse_errors(reply):
    result = _live_judge(ScriptedLLM(reply)).judge(QUESTION, "Alice won.", CONTEXT)
    assert result.metadata["fallback"] is True and result.metadata["judge"] == "heuristic"


def test_judgment_wrapped_in_a_code_fence_is_accepted():
    result = _live_judge(ScriptedLLM(f"```json\n{_reply(3)}\n```")).judge(QUESTION, "Alice won.", CONTEXT)
    assert result.answer_score == 3.0 and result.metadata["fallback"] is False


def test_out_of_range_scores_are_clamped():
    result = _live_judge(ScriptedLLM(_reply(9))).judge(QUESTION, "Alice won.", CONTEXT)
    assert result.correctness == 5.0


def test_disabled_and_mock_judges_use_the_heuristic_by_design_not_as_a_fallback():
    for judge in (AnswerJudge(force_mock=True), AnswerJudge(enabled=False)):
        assert judge.uses_llm is False
        assert judge.judge(QUESTION, "Alice won.", CONTEXT).metadata["fallback"] is False
    assert _live_judge(ScriptedLLM(_reply(4))).uses_llm is True


def test_prompt_v2_has_rubric_anchors_grounding_rules_and_the_json_schema():
    system, user = build_messages(QUESTION, "Alice won.", CONTEXT)
    assert system["role"] == "system" and user["role"] == "user"
    text = system["content"]
    assert "evaluation judge" in text  # the offline fake server recognizes judge calls by this phrase
    for anchor in ("0 =", "3 =", "5 ="):
        assert anchor in text
    assert "only" in text.lower() and "reference" in text.lower()
    for key in ("correctness", "faithfulness", "completeness", "relevance", "citation_quality", "is_supported_by_context", "is_hallucinated", "reasoning"):
        assert key in text
    payload = json.loads(user["content"])
    assert payload["model_answer"] == "Alice won." and payload["reference_answer"] == "Alice."
    assert "[doc_001 | doc_001::chunk::0]" in payload["retrieved_context"]


def test_the_retry_is_a_different_request_that_shows_the_model_its_bad_reply():
    llm = ScriptedLLM("Sure! not json", _reply(5))
    _live_judge(llm).judge(QUESTION, "Alice won.", CONTEXT)

    first, second = (call["messages"] for call in llm.calls)
    assert second[: len(first)] == first and len(second) == len(first) + 2  # an identical request would be replayed by the cache
    assert second[-2] == {"role": "assistant", "content": "Sure! not json"}
    assert "JSON" in second[-1]["content"]
