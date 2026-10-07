"""Jev (TypeSafe AI), a decision-only model with the same shape of API: reserved, not implemented.

The backend is registered so a config can name it and fail with a clear message, and so the interface is designed for it: Jev's
primitives (yes/no probability, choice, score) map onto `Predicate` and `DecisionResult`. What is missing is the client: its API is
not documented publicly enough to code against, and nothing here may guess at it. To add it, implement `decide` for it (see
`openai_decisions.py` for a complete backend) and register the class below instead of this stub.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate
from ragbench.models.errors import ModelInitError


class JevDecisionModel(DecisionModel):
    provider = "jev"

    def __init__(self, model_name: str = "jev"):
        raise ModelInitError(
            "The `jev` decision backend is not implemented yet: its API is not public enough to build a client against. "
            "Use `openai_decisions` (needs OPENAI_API_KEY), `llm_json` or `mock`; the interface is in ragbench/models/decisions/base.py."
        )

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:  # pragma: no cover - never constructed
        raise NotImplementedError
