from __future__ import annotations

import json
import logging
import math
import re
import statistics
from typing import Any

from pydantic import BaseModel, Field

from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_metrics import is_refusal
from ragbench.evaluation.judge_prompts import JUDGE_PROMPT_VERSION, build_messages, retry_messages
from ragbench.models.cost import CostBreakdown
from ragbench.models.defaults import DEFAULT_JUDGE_MODEL
from ragbench.models.llms import LLM, MockLLM, create_llm
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.text import normalize_text, tokenize

logger = logging.getLogger(__name__)


class AnswerJudgeResult(BaseModel):
    """Per-question judgment of answer quality on a 0-5 scale across five axes.

    `answer_score` is the unweighted mean used by the leaderboard.
    """

    correctness: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    citation_quality: float = 0.0
    is_supported_by_context: bool = False
    is_hallucinated: bool = False
    reasoning: str = ""
    raw_output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)

    @property
    def answer_score(self) -> float:
        return (self.correctness + self.faithfulness + self.completeness + self.relevance + self.citation_quality) / 5


class AnswerJudge:
    """LLM-as-a-judge with a deterministic heuristic stand-in.

    When disabled, or in mock mode, the heuristic judge scores every question by design (`metadata["judge"] == "heuristic"`,
    `fallback` False). A live judge asks for `samples` independent judgments per question and averages them; a reply that
    is not a complete, well-formed judgment is retried once, and if that fails too the heuristic scores that one question
    with `fallback: True`, so the run counts and reports it (`judge_fallback_rate`) instead of silently mixing the two.
    """

    MAX_ATTEMPTS = 2  # per sample: the first try and one retry

    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE_MODEL,
        enabled: bool = True,
        force_mock: bool = False,
        *,
        samples: int = 1,
        temperature: float = 0.0,
        llm: LLM | None = None,
    ):
        self.enabled = enabled
        self.samples = samples
        self.temperature = temperature
        # A disabled judge never calls a model, so it must not need credentials for one either.
        self.llm: LLM = llm if llm is not None else create_llm(model_name, force_mock=force_mock or not enabled)
        # create_llm falls back to the mock when a hosted provider has no key; the heuristic judge then stands in for it.
        self.force_mock = force_mock or isinstance(self.llm, MockLLM)

    @property
    def uses_llm(self) -> bool:
        """Whether judgments come from a model (False: the heuristic judge scores everything, by design)."""
        return self.enabled and not self.force_mock

    def judge(self, question: Question, answer: str, contexts: list[RetrievedChunk]) -> AnswerJudgeResult:
        """Score `answer` against the reference and the context the generator saw.

        Returns a populated `AnswerJudgeResult` with all five quality axes, hallucination/support flags, the cost of every
        judge call (retries and extra samples included) and `metadata` describing how it was produced.
        """
        if not self.uses_llm:
            result = heuristic_judge(question, answer, contexts)
            result.metadata.update({"judge": "heuristic", "parse_error": False, "fallback": False})
            return result
        messages = build_messages(question, answer, contexts)
        cost = CostBreakdown()
        judgments: list[AnswerJudgeResult] = []
        retries = parse_errors = 0
        raw_output: str | None = None
        for _ in range(self.samples):
            request = messages
            for attempt in range(self.MAX_ATTEMPTS):
                llm_result = self.llm.generate(request, json_mode=True, temperature=self.temperature)
                cost = cost.plus(
                    CostBreakdown(
                        judge_prompt_tokens=llm_result.prompt_tokens,
                        judge_completion_tokens=llm_result.completion_tokens,
                        judge_cost=llm_result.cost.total_cost,
                    )
                )
                raw_output = llm_result.text
                judgment = _parse_judgment(llm_result.text)
                if judgment is not None:
                    judgments.append(judgment)
                    break
                parse_errors += 1
                retries += attempt + 1 < self.MAX_ATTEMPTS
                request = retry_messages(messages, llm_result.text)
                logger.warning("Judge reply for question %r (model=%s) is not a valid judgment: %.120r", question.id, self.llm.model_name, llm_result.text)
        bookkeeping = {"prompt_version": JUDGE_PROMPT_VERSION, "parse_error": parse_errors > 0, "retries": retries}
        if not judgments:
            result = heuristic_judge(question, answer, contexts)
            result.raw_output = raw_output
            result.cost = cost
            result.metadata.update({"judge": "heuristic", "fallback": True, "samples": 0, "attempted_judge": self.llm.model_name, **bookkeeping})
            return result
        result = _combine(judgments)
        result.cost = cost
        result.metadata.update({"judge": self.llm.model_name, "fallback": False, "samples": len(judgments), **bookkeeping})
        if len(judgments) > 1:
            result.metadata["answer_score_variance"] = statistics.pvariance([judgment.answer_score for judgment in judgments])
        return result


def heuristic_judge(question: Question, answer: str, contexts: list[RetrievedChunk]) -> AnswerJudgeResult:
    answer_norm = normalize_text(answer).lower()
    context_norm = normalize_text(" ".join(c.text for c in contexts)).lower()
    refused = is_refusal(answer)
    if not question.is_answerable:
        score = 5.0 if refused else 1.0
        return AnswerJudgeResult(
            correctness=score,
            faithfulness=score,
            completeness=score,
            relevance=score,
            citation_quality=5.0 if refused else 0.0,
            is_supported_by_context=refused,
            is_hallucinated=not refused,
            reasoning="Heuristic judge: unanswerable question should be refused.",
        )

    keyword_hits = 0
    for keyword in question.expected_keywords:
        if keyword.lower() in answer_norm:
            keyword_hits += 1
    keyword_score = keyword_hits / max(1, len(question.expected_keywords))
    ref_tokens = set(tokenize(question.reference_answer or ""))
    answer_tokens = set(tokenize(answer))
    overlap_score = len(ref_tokens.intersection(answer_tokens)) / max(1, len(ref_tokens))
    correctness = min(5.0, 5.0 * max(keyword_score, overlap_score))
    content_words = [t for t in tokenize(answer) if len(t) > 3 and t not in {"could", "provided", "documents", "answer"}]
    supported_words = sum(1 for t in content_words if t in context_norm)
    support_ratio = supported_words / max(1, len(content_words))
    faithfulness = 5.0 * min(1.0, support_ratio)
    if not question.reference_answer and not question.expected_keywords:
        correctness = faithfulness  # nothing to compare with (label-free question): the context is the only yardstick
    cited_doc_ids = set(re.findall(r"\[(doc_[A-Za-z0-9_-]+)\]", answer))
    citation_quality = 5.0 if cited_doc_ids.intersection(question.relevant_doc_ids) else (2.0 if cited_doc_ids else 0.0)
    if refused:
        correctness = min(correctness, 1.0)
        faithfulness = 4.0
    return _clamp_scores(
        AnswerJudgeResult(
            correctness=correctness,
            faithfulness=faithfulness,
            completeness=correctness,
            relevance=correctness,
            citation_quality=citation_quality,
            is_supported_by_context=faithfulness >= 3,
            is_hallucinated=faithfulness < 2 and not refused,
            reasoning="Heuristic judge: keyword/reference overlap plus simple context support check.",
        )
    )


def _clamp_scores(result: AnswerJudgeResult) -> AnswerJudgeResult:
    for field in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality"]:
        value = getattr(result, field)
        setattr(result, field, max(0.0, min(5.0, float(value))))
    return result


SCORE_AXES = ("correctness", "faithfulness", "completeness", "relevance", "citation_quality")


def _parse_judgment(text: str) -> AnswerJudgeResult | None:
    """The judgment in a judge reply, or None unless it is a JSON object with all five numeric scores (a bare fence is tolerated)."""
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    scores: dict[str, float] = {}
    for axis in SCORE_AXES:
        value = parsed.get(axis)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            return None  # a missing axis must not silently count as 0
        scores[axis] = float(value)
    return _clamp_scores(
        AnswerJudgeResult(
            **scores,
            is_supported_by_context=_as_bool(parsed.get("is_supported_by_context")),
            is_hallucinated=_as_bool(parsed.get("is_hallucinated")),
            reasoning=str(parsed.get("reasoning", ""))[:800],
            raw_output=text,
        )
    )


def _as_bool(value: Any) -> bool:
    return value.strip().lower() == "true" if isinstance(value, str) else bool(value)


def _combine(judgments: list[AnswerJudgeResult]) -> AnswerJudgeResult:
    """One judgment from several samples: mean of each axis; a flag is set when most samples set it (a hallucination flag at half)."""
    if len(judgments) == 1:
        return judgments[0]
    count = len(judgments)
    return AnswerJudgeResult(
        **{axis: sum(getattr(judgment, axis) for judgment in judgments) / count for axis in SCORE_AXES},
        is_supported_by_context=sum(judgment.is_supported_by_context for judgment in judgments) * 2 > count,
        is_hallucinated=sum(judgment.is_hallucinated for judgment in judgments) * 2 >= count,
        reasoning=judgments[0].reasoning,
        raw_output=judgments[0].raw_output,
    )
