"""Hard requirements a system must meet to be recommended at all."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class Constraints(BaseModel):
    """`selection.constraints:` section. A system that breaks any set constraint is never recommended; unset ones are ignored."""

    model_config = ConfigDict(extra="forbid")

    max_cost_per_question: float | None = Field(default=None, ge=0, description="Highest acceptable mean `$/Q` (answering plus judging).")
    max_latency_ms_p95: float | None = Field(default=None, ge=0, description="Highest acceptable 95th-percentile latency in milliseconds.")
    min_faithfulness: float | None = Field(default=None, ge=0, le=5, description="Lowest acceptable mean faithfulness (0-5).")
    min_answer_score: float | None = Field(default=None, ge=0, le=5, description="Lowest acceptable mean answer score (0-5).")
    max_ingestion_cost: float | None = Field(default=None, ge=0, description="Highest acceptable one-off cost of indexing the corpus, in dollars.")
    require_local_models: bool = Field(
        default=False, description="Every model the system calls must run on your machine: a `local:` embedder or an `openai_compatible:` endpoint on localhost."
    )
    require_no_network: bool = Field(
        default=False, description="No data may leave the machine: local models only (as above) and no tool that uses the network."
    )

    @property
    def is_empty(self) -> bool:
        return not any(value not in (None, False) for value in self.model_dump().values())


@dataclass(frozen=True)
class SystemFacts:
    """What a run measured about one system, as far as constraints are concerned. None = not measured / could not be determined."""

    system: str
    cost_per_question: float | None
    latency_ms_p95: float | None
    faithfulness: float | None
    answer_score: float | None
    ingestion_cost: float | None
    non_local_models: list[str] | None = None  # models that do not run on this machine
    network_tools: list[str] | None = None  # tools with network side effects


@dataclass(frozen=True)
class Violation:
    constraint: str
    message: str
    severity: float  # how far past the limit, as a ratio (1.2 = 20% over); infinite when the value is missing or the limit is 0


def _ratio(value: float, limit: float, *, upper: bool) -> float:
    numerator, denominator = (value, limit) if upper else (limit, value)
    if denominator == 0:
        return math.inf if numerator > 0 else 1.0
    return numerator / denominator


def check_constraints(constraints: Constraints, facts: SystemFacts) -> list[Violation]:
    """Every constraint `facts` breaks, in a fixed order. A metric the run did not measure cannot be shown to meet its limit, so it breaks it."""
    violations: list[Violation] = []

    def upper(name: str, value: float | None, limit: float | None, label: str, fmt: str) -> None:
        if limit is None:
            return
        if value is None:
            violations.append(Violation(name, f"{label} was not measured, so it cannot be checked against the maximum {fmt.format(limit)}", math.inf))
        elif value > limit:
            violations.append(Violation(name, f"{label} {fmt.format(value)} is above the maximum {fmt.format(limit)}", _ratio(value, limit, upper=True)))

    def lower(name: str, value: float | None, limit: float | None, label: str, fmt: str) -> None:
        if limit is None:
            return
        if value is None:
            violations.append(Violation(name, f"{label} was not measured, so it cannot be checked against the minimum {fmt.format(limit)}", math.inf))
        elif value < limit:
            violations.append(Violation(name, f"{label} {fmt.format(value)} is below the minimum {fmt.format(limit)}", _ratio(value, limit, upper=False)))

    upper("max_cost_per_question", facts.cost_per_question, constraints.max_cost_per_question, "cost per question", "${:.5f}")
    upper("max_latency_ms_p95", facts.latency_ms_p95, constraints.max_latency_ms_p95, "p95 latency", "{:.0f} ms")
    lower("min_faithfulness", facts.faithfulness, constraints.min_faithfulness, "faithfulness", "{:.2f}")
    lower("min_answer_score", facts.answer_score, constraints.min_answer_score, "answer score", "{:.2f}")
    upper("max_ingestion_cost", facts.ingestion_cost, constraints.max_ingestion_cost, "ingestion cost", "${:.5f}")
    if constraints.require_local_models or constraints.require_no_network:
        if facts.non_local_models is None:
            violations.append(Violation("require_local_models", "the models it uses could not be determined from the run's config", math.inf))
        elif facts.non_local_models:
            violations.append(Violation("require_local_models", f"uses models that do not run locally: {', '.join(facts.non_local_models)}", math.inf))
    if constraints.require_no_network:
        if facts.network_tools is None:
            violations.append(Violation("require_no_network", "its tools could not be inspected for network use", math.inf))
        elif facts.network_tools:
            violations.append(Violation("require_no_network", f"uses tools that reach the network: {', '.join(facts.network_tools)}", math.inf))
    return violations
