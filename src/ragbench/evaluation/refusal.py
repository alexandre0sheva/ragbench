"""Whether an answer declines to answer: the wording heuristic (`is_refusal`, the default and what mock runs use), or a decision model
that reads the question and the answer (`evaluation.refusal_detector: model`), for the refusals the wording does not anticipate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ragbench.evaluation.answer_metrics import is_refusal
from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import DecisionModel, MockDecisionModel, Predicate, create_decision_model, formats

if TYPE_CHECKING:
    from ragbench.config.schema import EvaluationConfig

REFUSAL_THRESHOLD = 0.5  # P(declines) at or above this is a refusal
DECLINES = Predicate(formats.SINGLE, formats.DECLINES_INSTRUCTIONS)


@dataclass
class RefusalVerdict:
    """`source` is `model` when a decision model's probability decided it, else `heuristic` (by design, or because the model gave no usable answer)."""

    refused: bool
    source: Literal["heuristic", "model"] = "heuristic"
    probability: float | None = None
    cost: CostBreakdown = field(default_factory=CostBreakdown)


class RefusalDetector:
    def __init__(self, decider: DecisionModel | None = None):
        self.decider = decider

    @classmethod
    def from_config(cls, evaluation: EvaluationConfig, force_mock: bool) -> RefusalDetector:
        """The detector for a run. The model one reads with the judge model; with no key, or in a mock run, the heuristic stands in."""
        if evaluation.refusal_detector != "model" or force_mock:
            return cls()
        return cls(create_decision_model("llm_json", evaluation.judge_model))

    @property
    def uses_model(self) -> bool:
        return self.decider is not None and not isinstance(self.decider, MockDecisionModel)  # the mock cannot read meaning

    def detect(self, question: str, answer: str) -> RefusalVerdict:
        if not self.uses_model or not answer.strip():
            return RefusalVerdict(is_refusal(answer))
        assert self.decider is not None
        result = self.decider.decide(formats.single_input(question, answer), [DECLINES])
        cost = CostBreakdown(
            judge_prompt_tokens=result.input_tokens, judge_cost=result.cost.total_cost
        )  # spend on checking answers is judge spend
        probability = result.probabilities.get(DECLINES.name)
        if probability is None:  # no usable reply: the wording decides, rather than a guess
            return RefusalVerdict(is_refusal(answer), cost=cost)
        return RefusalVerdict(probability >= REFUSAL_THRESHOLD, "model", probability, cost)
