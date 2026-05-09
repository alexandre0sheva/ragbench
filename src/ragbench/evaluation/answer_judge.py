from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from ragbench.datasets.schema import Question
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, create_llm
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.env import has_openai_key
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
    """LLM-as-a-judge with a deterministic heuristic fallback.

    When disabled, when running in mock mode, or when the LLM call returns
    unparseable JSON, falls back to `heuristic_judge` so a benchmark run
    always produces scores. The fallback is recorded in `metadata["judge"]`.
    """

    def __init__(self, model_name: str = "gpt-5.4-nano", enabled: bool = True, force_mock: bool = False):
        self.enabled = enabled
        self.force_mock = force_mock or not has_openai_key()
        self.llm: LLM = create_llm(model_name, force_mock=self.force_mock)

    def judge(self, question: Question, answer: str, contexts: list[RetrievedChunk]) -> AnswerJudgeResult:
        """Score `answer` against the reference and retrieved context.

        Returns a populated `AnswerJudgeResult` with all five quality axes,
        hallucination/support flags, and the cost incurred by the judge call.
        """
        if not self.enabled or self.force_mock:
            result = heuristic_judge(question, answer, contexts)
            result.metadata["judge"] = "heuristic_fallback"
            return result
        context_text = "\n\n".join(f"[{c.doc_id} | {c.chunk_id}]\n{c.text}" for c in contexts)
        prompt = {
            "question": question.question,
            "reference_answer": question.reference_answer,
            "model_answer": answer,
            "retrieved_context": context_text[:12000],
            "expected_keywords": question.expected_keywords,
            "category": question.category,
            "answer_type": question.answer_type,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluation judge. Return JSON only with numeric scores 0-5 for "
                    "correctness, faithfulness, completeness, relevance, citation_quality, plus booleans "
                    "is_supported_by_context and is_hallucinated, and a short reasoning string."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        llm_result = self.llm.generate(messages, json_mode=True, temperature=0)
        try:
            parsed = json.loads(llm_result.text)
            result = AnswerJudgeResult(
                correctness=float(parsed.get("correctness", 0)),
                faithfulness=float(parsed.get("faithfulness", 0)),
                completeness=float(parsed.get("completeness", 0)),
                relevance=float(parsed.get("relevance", 0)),
                citation_quality=float(parsed.get("citation_quality", 0)),
                is_supported_by_context=bool(parsed.get("is_supported_by_context", False)),
                is_hallucinated=bool(parsed.get("is_hallucinated", False)),
                reasoning=str(parsed.get("reasoning", ""))[:800],
                raw_output=llm_result.text,
                metadata={"judge": self.llm.model_name},
                cost=CostBreakdown(
                    judge_prompt_tokens=llm_result.prompt_tokens,
                    judge_completion_tokens=llm_result.completion_tokens,
                    judge_cost=llm_result.cost.total_cost,
                ),
            )
            return _clamp_scores(result)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Judge JSON parse failed for question %r (model=%s): %s. "
                "Falling back to heuristic judge.",
                question.id,
                self.llm.model_name,
                exc,
            )
            result = heuristic_judge(question, answer, contexts)
            result.raw_output = llm_result.text
            result.metadata["judge_parse_error"] = True
            result.metadata["judge_parse_error_message"] = str(exc)
            result.cost = CostBreakdown(
                judge_prompt_tokens=llm_result.prompt_tokens,
                judge_completion_tokens=llm_result.completion_tokens,
                judge_cost=llm_result.cost.total_cost,
            )
            return result


def heuristic_judge(question: Question, answer: str, contexts: list[RetrievedChunk]) -> AnswerJudgeResult:
    answer_norm = normalize_text(answer).lower()
    context_norm = normalize_text(" ".join(c.text for c in contexts)).lower()
    refused = "could not find the answer" in answer_norm or "not available" in answer_norm
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
