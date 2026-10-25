"""The optional checks on an answer beyond the judge's verdict: whether it refuses, and (with `evaluation.claims`) which of its claims the context does not support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ragbench.evaluation.claims import ClaimChecker
from ragbench.evaluation.refusal import RefusalDetector

if TYPE_CHECKING:
    from ragbench.config.schema import EvaluationConfig


@dataclass
class AnswerChecks:
    refusal: RefusalDetector = field(default_factory=RefusalDetector)
    claims: ClaimChecker | None = None

    @classmethod
    def from_config(cls, evaluation: EvaluationConfig, force_mock: bool) -> AnswerChecks:
        return cls(RefusalDetector.from_config(evaluation, force_mock), ClaimChecker.from_config(evaluation, force_mock))
