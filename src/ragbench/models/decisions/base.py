"""The typed-decision interface: yes/no questions about a text, answered with a probability each.

A decision model does not write text. It reads one input and a list of named predicates ("Does this passage state the
answer?") and returns, for each, the probability that it is true. `decision_gate` uses that to keep or drop evidence before
any generator sees it. Backends differ in how they get the numbers (OpenAI's Decisions API, a prompted chat model, a fake for
offline runs) but all return the same `DecisionResult`, so the gate, its cost accounting and its tests do not care which.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragbench.models.cost import CostBreakdown


@dataclass(frozen=True)
class Predicate:
    """One yes/no question. `name` is unique within a request; `instructions` say what "true" means."""

    name: str
    instructions: str


@dataclass
class DecisionResult:
    """Answers to one request.

    `probabilities` maps every predicate name to P(true) in [0, 1], or None when the model gave no usable answer for it (a refusal, an
    unparseable reply). `cost` books the spend as `decision_cost` / `decision_input_tokens`. For a cache hit `latency_ms` is the
    original call's latency (replayed) and `cached` is true.
    """

    probabilities: dict[str, float | None]
    model: str
    input_tokens: int = 0
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    latency_ms: float | None = None
    cached: bool = False


class DecisionModel(ABC):
    model_name: str
    # Identifies the backing service in cache keys (two backends may serve the same model name).
    provider: str = "decision"

    @abstractmethod
    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        """Answer every predicate about `text` in one request. Predicate names must be unique."""
        raise NotImplementedError


def clamp_probability(value: object) -> float | None:
    """`value` as a probability in [0, 1], or None when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return None if number != number else min(1.0, max(0.0, number))
