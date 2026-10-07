"""The offline decision model: deterministic probabilities from word overlap, so `--mock` runs need no key and repeat exactly."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import formats
from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate
from ragbench.models.usage import note_decision
from ragbench.utils.text import estimate_tokens, tokenize

MOCK_MODEL = "mock-decisions"


def _content_words(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 3}


class MockDecisionModel(DecisionModel):
    """P(true) is the share of the question's content words found in the text a predicate is about, scaled so that covering about two
    thirds of them already reads as certain. It cannot read meaning (no model can offline) but it is monotone: a passage that shares
    more of the question's words never scores lower, and an unrelated one scores 0."""

    provider = "mock"

    def __init__(self, model_name: str = MOCK_MODEL):
        self.model_name = model_name

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        question = _content_words(formats.parse_question(text))
        about = formats.parse_passages(text)
        probabilities: dict[str, float | None] = {}
        for predicate in predicates:
            words = _content_words(about.get(predicate.name, ""))
            coverage = len(question & words) / len(question) if question else 0.0
            probabilities[predicate.name] = round(min(1.0, coverage * 1.5), 3)
        tokens = estimate_tokens(text + " ".join(p.instructions for p in predicates), self.model_name)
        note_decision(tokens)  # a no-op unless `ragbench estimate` is measuring
        return DecisionResult(probabilities=probabilities, model=self.model_name, input_tokens=tokens, cost=CostBreakdown(decision_input_tokens=tokens))
