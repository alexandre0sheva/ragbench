from __future__ import annotations

import json
import re

import pytest

from ragbench.datasets.schema import Question
from ragbench.evaluation.pairwise import PAIRWISE_PROMPT_VERSION, Candidate, PairwiseJudge, pairwise_messages, select_best
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, LLMResult, MockLLM

QUESTION = Question(id="q1", question="Who won the 2019 award?", reference_answer="Maxine Thompson.", relevant_doc_ids=["doc_005"])
LABEL_FREE = Question(id="q2", question="What does the policy say about refunds?", answerable=True)


def quality(text: str) -> int:
    return int(re.search(r"quality=(\d+)", text).group(1))


class ScriptedPairJudge(LLM):
    """Reads the two anonymous answers from the request and replies with `pick(quality_1, quality_2)`: "1", "2", "tie" or junk."""

    model_name = "scripted-pair-judge"

    def __init__(self, pick):
        self.pick = pick
        self.requests: list[dict] = []

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        payload = json.loads(messages[1]["content"])
        self.requests.append(payload)
        winner = self.pick(quality(payload["answer_1"]), quality(payload["answer_2"]))
        text = winner if winner == "junk" else json.dumps({"winner": winner, "reason": "scripted"})
        return LLMResult(text=text, model=self.model_name, prompt_tokens=50, completion_tokens=10, cost=CostBreakdown(llm_cost=0.001))


def prefers_quality(a: int, b: int) -> str:
    return "1" if a > b else "2" if b > a else "tie"


def always_first(a: int, b: int) -> str:
    return "1"


def _judge(pick) -> tuple[PairwiseJudge, ScriptedPairJudge]:
    llm = ScriptedPairJudge(pick)
    return PairwiseJudge(llm, uses_llm=True), llm


def _answer(q: int) -> Candidate:
    return Candidate(f"An answer, quality={q}.")


# -- the prompt -----------------------------------------------------------------------------------


def test_the_prompt_shows_two_anonymous_answers_and_the_reference_when_there_is_one():
    system, user = pairwise_messages(QUESTION, "first text", "second text")
    payload = json.loads(user["content"])
    assert payload == {"question": QUESTION.question, "answerable": True, "reference_answer": "Maxine Thompson.", "answer_1": "first text", "answer_2": "second text"}
    assert "pairwise comparison judge" in system["content"] and "evaluation judge" not in system["content"]  # the fake server keys on that phrase
    assert "position" in system["content"]  # told not to prefer an answer for where it stands
    assert json.loads(pairwise_messages(LABEL_FREE, "a", "b")[1]["content"])["reference_answer"] is None
    assert PAIRWISE_PROMPT_VERSION == "p1"


# -- one verdict ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("reply", "verdict"), [('{"winner": "1"}', "first"), ('{"winner": 2}', "second"), ('```json\n{"winner": "tie"}\n```', "tie")])
def test_a_reply_names_the_position_that_won(reply, verdict):
    class Fixed(LLM):
        model_name = "fixed"

        def generate(self, messages, **kwargs):
            return LLMResult(text=reply, model="fixed", prompt_tokens=1, completion_tokens=1, cost=CostBreakdown())

    assert PairwiseJudge(Fixed(), uses_llm=True).verdict(QUESTION, _answer(1), _answer(2))[0] == verdict


def test_an_unusable_reply_is_retried_once_then_gives_no_verdict():
    judge, llm = _judge(lambda a, b: "junk")
    verdict, cost = judge.verdict(QUESTION, _answer(1), _answer(2))
    assert verdict is None and len(llm.requests) == 2 and cost.judge_cost == pytest.approx(0.002)


def test_without_a_model_the_heuristic_prefers_the_higher_pointwise_score():
    judge = PairwiseJudge(MockLLM(), uses_llm=False)
    assert judge.verdict(QUESTION, Candidate("x", score=4.0), Candidate("y", score=2.0))[0] == "first"
    assert judge.verdict(QUESTION, Candidate("x", score=2.0), Candidate("y", score=4.0))[0] == "second"
    assert judge.verdict(QUESTION, Candidate("x", score=3.0), Candidate("y", score=3.1))[0] == "tie"
    # With no score it scores each answer with the heuristic judge first.
    assert judge.verdict(QUESTION, Candidate("Maxine Thompson won it."), Candidate("I could not find that."))[0] == "first"


# -- both orders ----------------------------------------------------------------------------------


def test_a_pair_is_judged_in_both_orders():
    judge, llm = _judge(prefers_quality)
    comparison, cost = judge.compare(QUESTION, "good", _answer(5), "poor", _answer(1))
    assert comparison.x == "good" and comparison.verdicts == ("first", "second") and comparison.score == 1.0
    assert [quality(r["answer_1"]) for r in llm.requests] == [5, 1] and cost.judge_cost == pytest.approx(0.002)


def test_a_judge_that_always_picks_the_first_answer_is_cancelled_into_a_tie():
    judge, _ = _judge(always_first)
    comparison, _ = judge.compare(QUESTION, "good", _answer(5), "poor", _answer(1))
    assert comparison.flipped and comparison.score == 0.5


# -- select_best ----------------------------------------------------------------------------------


def test_select_best_finds_the_best_candidate_wherever_it_stands():
    for best in range(4):
        judge, _ = _judge(prefers_quality)
        candidates = [f"candidate {i}, quality={5 if i == best else i % 3}" for i in range(4)]
        selection = select_best(candidates, QUESTION, judge, seed=best)
        assert selection.choice == best and selection.cost.judge_cost > 0


def test_select_best_shows_candidates_anonymously_in_a_seeded_shuffled_order():
    candidates = [f"candidate {i}, quality={i}" for i in range(4)]
    firsts = set()
    for seed in range(6):
        judge, llm = _judge(prefers_quality)
        select_best(candidates, QUESTION, judge, seed=seed)
        firsts.add(llm.requests[0]["answer_1"])
        assert all("candidate" in r["answer_1"] and set(r) == {"question", "answerable", "reference_answer", "answer_1", "answer_2"} for r in llm.requests)
    assert len(firsts) > 1  # not always the base candidate first
    judge_a, llm_a = _judge(prefers_quality)
    judge_b, llm_b = _judge(prefers_quality)
    select_best(candidates, QUESTION, judge_a, seed=3)
    select_best(candidates, QUESTION, judge_b, seed=3)
    assert llm_a.requests == llm_b.requests


def test_select_best_keeps_the_base_when_the_judge_only_follows_position_or_fails():
    candidates = [f"candidate {i}, quality={i}" for i in range(3)]
    assert select_best(candidates, QUESTION, _judge(always_first)[0]).choice == 0  # every pair cancels to a tie: the base stays
    failed = select_best(candidates, QUESTION, _judge(lambda a, b: "junk")[0])
    assert failed.choice is None  # no usable verdict: the caller falls back to its base


def test_select_best_of_one_or_none():
    judge, llm = _judge(prefers_quality)
    assert select_best(["only, quality=1"], QUESTION, judge).choice == 0 and not llm.requests
    assert select_best([], QUESTION, judge).choice is None
