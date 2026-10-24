"""Pairwise judging: which of two anonymous answers to a question is better, asked in both orders so position bias cancels.

The judge sees the question, the reference answer when there is one, and the two answers as "Answer 1" and "Answer 2", never which
system wrote them. It needs no reference, so it ranks systems on label-free datasets too. With no model to ask (mock runs, or the
judge switched off) a heuristic stands in: the answer with the higher pointwise score wins.

`select_best` picks one of several candidates the same way (Harness²-style selection without labels): candidates in a seeded shuffled
order, every pair judged in both orders, the most points wins, ties and failures keep the first (base) candidate.

Bump `PAIRWISE_PROMPT_VERSION` whenever the wording changes. Keep "evaluation judge" out of the prompt: `tests/fake_openai_server.py`
treats requests with that phrase as pointwise judge calls.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import dataclass, field
from itertools import combinations

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import AnswerJudge, heuristic_judge
from ragbench.evaluation.arena import Comparison, Verdict
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM

logger = logging.getLogger(__name__)

PAIRWISE_PROMPT_VERSION = "p1"
HEURISTIC_TIE = 0.25  # pointwise scores closer than this are a tie for the heuristic stand-in

PAIRWISE_SYSTEM_PROMPT = """You are an impartial pairwise comparison judge for question answering. You get a question, the reference answer when one exists, and two anonymous answers to it, "answer_1" and "answer_2".

Decide which answer serves the person who asked better: correct (it agrees with the reference answer when one is given), complete, on point, and free of claims it cannot back up. A clear "I could not find that" beats a confident wrong answer, and when "answerable" is false the answer that says the question cannot be answered wins. Without a reference answer, prefer the answer that is specific, internally consistent and cautious about what it does not know.

The answers are in random order: do not prefer one for its position, its length or its style. Reply "tie" only when neither is better.

Return JSON only: {"winner": "1" | "2" | "tie", "reason": "one sentence"}"""

RETRY_PROMPT = 'That reply was not a valid verdict. Reply again with one JSON object and nothing else: {"winner": "1" | "2" | "tie", "reason": "..."}'
_WINNERS: dict[str, Verdict] = {"1": "first", "2": "second", "tie": "tie"}


def pairwise_messages(question: Question, first: str, second: str) -> list[dict[str, str]]:
    payload = {
        "question": question.question,
        "answerable": question.is_answerable,
        "reference_answer": question.reference_answer,
        "answer_1": first,
        "answer_2": second,
    }
    return [{"role": "system", "content": PAIRWISE_SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def _parse(text: str) -> Verdict:
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    winner = parsed.get("winner") if isinstance(parsed, dict) else None
    return _WINNERS.get(str(winner).strip().lower()) if winner is not None and not isinstance(winner, bool) else None


@dataclass(frozen=True)
class Candidate:
    """An answer to compare; `score` is its pointwise answer score when known (the heuristic stand-in uses it)."""

    text: str
    score: float | None = None


class PairwiseJudge:
    MAX_ATTEMPTS = 2  # the first try and one retry

    def __init__(self, llm: LLM, *, uses_llm: bool, temperature: float = 0.0):
        self.llm = llm
        self.uses_llm = uses_llm
        self.temperature = temperature
        self.calls = 0  # model requests so far, retries included
        self._lock = threading.Lock()

    @classmethod
    def from_answer_judge(cls, judge: AnswerJudge) -> PairwiseJudge:
        """The same model (or the same mock) as a pointwise judge, asked pairwise."""
        return cls(judge.llm, uses_llm=judge.uses_llm, temperature=judge.temperature)

    @property
    def model_name(self) -> str:
        return self.llm.model_name if self.uses_llm else "heuristic"

    def verdict(self, question: Question, first: Candidate, second: Candidate) -> tuple[Verdict, CostBreakdown]:
        """Which position won (`first`, `second`, `tie`), or None when the judge gave no usable verdict; and what asking cost."""
        if not self.uses_llm:
            a, b = (c.score if c.score is not None else heuristic_judge(question, c.text, []).answer_score for c in (first, second))
            return ("tie" if abs(a - b) < HEURISTIC_TIE else "first" if a > b else "second"), CostBreakdown()
        messages = request = pairwise_messages(question, first.text, second.text)
        cost = CostBreakdown()
        for _ in range(self.MAX_ATTEMPTS):
            result = self.llm.generate(request, json_mode=True, temperature=self.temperature)
            with self._lock:
                self.calls += 1
            cost = cost.plus(CostBreakdown(judge_prompt_tokens=result.prompt_tokens, judge_completion_tokens=result.completion_tokens, judge_cost=result.cost.total_cost))
            verdict = _parse(result.text)
            if verdict is not None:
                return verdict, cost
            request = [*messages, {"role": "assistant", "content": result.text}, {"role": "user", "content": RETRY_PROMPT}]
        logger.warning("Pairwise judge reply for question %r (model=%s) is not a valid verdict: %.120r", question.id, self.llm.model_name, result.text)
        return None, cost

    def compare(self, question: Question, x: str, x_answer: Candidate, y: str, y_answer: Candidate) -> tuple[Comparison, CostBreakdown]:
        """`x` against `y`, judged with x shown first and then with y shown first."""
        shown_first, cost_first = self.verdict(question, x_answer, y_answer)
        shown_second, cost_second = self.verdict(question, y_answer, x_answer)
        return Comparison(question.id, x, y, (shown_first, shown_second)), cost_first.plus(cost_second)


@dataclass
class Selection:
    """What `select_best` chose: an index into the candidates, or None when no comparison gave a verdict (fall back to the base)."""

    choice: int | None
    points: list[float] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    cost: CostBreakdown = field(default_factory=CostBreakdown)


def select_best(candidates: list[str], question: Question, judge: PairwiseJudge, *, seed: int = 0) -> Selection:
    """The best of `candidates` for `question` by pairwise judgment, without labels.

    Candidates are shown anonymously in a seeded shuffled order; every pair is judged in both orders and a comparison gives its two
    candidates the points they won (0 to 1). The candidate with the most points wins; on a tie the earlier candidate in `candidates`
    does, so the first one (the base) is kept unless another is shown to be better.
    """
    if len(candidates) <= 1:
        return Selection(0 if candidates else None, [0.0] * len(candidates))
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)
    labels = [str(index) for index in range(len(candidates))]
    points = [0.0] * len(candidates)
    comparisons: list[Comparison] = []
    cost = CostBreakdown()
    for a, b in combinations(order, 2):
        comparison, spent = judge.compare(question, labels[a], Candidate(candidates[a]), labels[b], Candidate(candidates[b]))
        comparisons.append(comparison)
        cost = cost.plus(spent)
        if comparison.score is not None:
            points[a] += comparison.score
            points[b] += 1 - comparison.score
    if not any(c.complete for c in comparisons):
        return Selection(None, points, comparisons, cost)
    best = max(range(len(candidates)), key=lambda index: (points[index], -index))
    return Selection(best, points, comparisons, cost)
