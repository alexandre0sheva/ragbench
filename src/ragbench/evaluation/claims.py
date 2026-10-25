"""Claim-level faithfulness: split an answer into atomic claims, then ask a decision model, claim by claim, whether the context the
generator was given supports it. The judge's `faithfulness` is one number for the whole answer; this says which statements are unsupported.

A claim the model gave no usable answer for is left out of the rate (missing is not unsupported), an answer that asserts nothing (a
refusal) has no rate, and spend is charged as judge cost. Offline (`ClaimChecker.mock`) the answer is split into sentences and a claim
is supported when enough of its content words occur in the context: no meaning is read, but it is deterministic and free.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragbench.evaluation.answer_metrics import CITATION, is_refusal
from ragbench.evaluation.judge_prompts import MAX_CONTEXT_CHARS
from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import DecisionModel, MockDecisionModel, Predicate, create_decision_model, formats
from ragbench.models.llms import LLM, MockLLM, create_llm
from ragbench.rag_systems.base import RetrievedChunk

if TYPE_CHECKING:
    from ragbench.config.schema import EvaluationConfig

logger = logging.getLogger(__name__)

SUPPORT_THRESHOLD = 0.5  # P(supported) at or above this is a supported claim
SUPPORTED = Predicate(formats.SINGLE, formats.SUPPORTED_INSTRUCTIONS)
SPLIT_SYSTEM_PROMPT = """You split a model's answer into atomic claims.

An atomic claim is one self-contained statement of fact that can be true or false on its own: replace pronouns with what they refer to, keep numbers and names exactly, and leave out citations such as [doc_001]. Include only what the answer itself states; never add facts, and do not check whether they are true. If the answer declines to answer or states no facts, there are no claims.

Return JSON only: {"claims": ["first claim", "second claim"]}"""
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Claim:
    text: str
    supported: bool | None  # None: the model gave no usable answer
    probability: float | None = None


@dataclass
class ClaimResult:
    claims: list[Claim] = field(default_factory=list)
    split_failed: bool = False  # the splitter's reply could not be read: nothing was checked
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def unsupported(self) -> list[str]:
        return [claim.text for claim in self.claims if claim.supported is False]

    @property
    def unreadable(self) -> int:
        return sum(claim.supported is None for claim in self.claims)

    @property
    def support_rate(self) -> float | None:
        """Share of the checked claims that are supported; None when there is none to check."""
        checked = [claim for claim in self.claims if claim.supported is not None]
        return sum(claim.supported is True for claim in checked) / len(checked) if checked else None

    def as_dict(self) -> dict[str, Any]:
        """The `claims` block of a question's row."""
        return {
            "total": len(self.claims),
            "unsupported": self.unsupported,
            "unreadable": self.unreadable,
            "support_rate": self.support_rate,
            "split_failed": self.split_failed,
            "claims": [{"text": claim.text, "supported": claim.supported, "probability": claim.probability} for claim in self.claims],
        }


def _sentences(answer: str) -> list[str]:
    """The mock splitter: the answer's sentences without citations, minus those that decline to answer."""
    cleaned = CITATION.sub("", answer)
    parts = (" ".join(part.split()) for part in _SENTENCE_END.split(cleaned))
    return [part for part in parts if part and not is_refusal(part)]


def parse_claims(text: str) -> list[str] | None:
    """The claims in a splitter reply (a bare fence is tolerated), or None when it is not a JSON object with a `claims` list."""
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(claims, list):
        return None
    return [claim.strip() for claim in claims if isinstance(claim, str) and claim.strip()]


class ClaimChecker:
    """`llm` splits answers into claims (None: split into sentences, for mock runs); `decider` says whether the context supports each."""

    def __init__(self, llm: LLM | None, decider: DecisionModel, *, max_claims: int = 12):
        self.llm = llm
        self.decider = decider
        self.max_claims = max_claims

    @classmethod
    def mock(cls, max_claims: int = 12) -> ClaimChecker:
        return cls(None, MockDecisionModel(), max_claims=max_claims)

    @classmethod
    def from_config(cls, evaluation: EvaluationConfig, force_mock: bool) -> ClaimChecker | None:
        """The checker for a run, or None when `evaluation.claims` is off. Mock runs, and a model with no key, use the offline stand-in."""
        if not evaluation.claims.enabled:
            return None
        llm = create_llm(evaluation.claims_model, force_mock=force_mock)
        if isinstance(llm, MockLLM):
            return cls.mock(evaluation.claims.max_claims)
        return cls(llm, create_decision_model("llm_json", evaluation.claims_model), max_claims=evaluation.claims.max_claims)

    def check(self, question: str, answer: str, contexts: Sequence[RetrievedChunk]) -> ClaimResult:
        result = ClaimResult()
        texts = self._split(question, answer, result)
        if texts is None:
            result.split_failed = True
            return result
        passage = "\n\n".join(f"[{chunk.doc_id} | {chunk.chunk_id}]\n{chunk.text}" for chunk in contexts)[:MAX_CONTEXT_CHARS]
        for text in texts[: self.max_claims]:
            decision = self.decider.decide(formats.claim_input(text, passage), [SUPPORTED])
            result.cost = result.cost.plus(CostBreakdown(judge_prompt_tokens=decision.input_tokens, judge_cost=decision.cost.total_cost))
            probability = decision.probabilities.get(SUPPORTED.name)
            result.claims.append(Claim(text, None if probability is None else probability >= SUPPORT_THRESHOLD, probability))
        return result

    def _split(self, question: str, answer: str, result: ClaimResult) -> list[str] | None:
        if self.llm is None:
            return _sentences(answer)
        messages = [
            {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"question": question, "answer": answer}, ensure_ascii=False)},
        ]
        reply = self.llm.generate(messages, json_mode=True, temperature=0.0)
        result.cost = result.cost.plus(
            CostBreakdown(
                judge_prompt_tokens=reply.prompt_tokens, judge_completion_tokens=reply.completion_tokens, judge_cost=reply.cost.total_cost
            )
        )
        claims = parse_claims(reply.text)
        if claims is None:
            logger.warning("Claim splitter reply (model=%s) is not a list of claims: %.120r", self.llm.model_name, reply.text)
        return claims
