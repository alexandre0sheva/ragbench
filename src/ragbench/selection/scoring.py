"""Weighted scoring of the feasible systems, and the named profiles that bundle weights with a tie policy."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Weights(BaseModel):
    """`selection.weights:` section: how much quality, cost and latency matter. Only the proportions count (they need not sum to 1)."""

    model_config = ConfigDict(extra="forbid")

    quality: float = Field(default=0.6, ge=0, description="Weight of answer quality (answer score; token F1 or recall when there is none).")
    cost: float = Field(default=0.2, ge=0, description="Weight of `$/Q` (lower is better).")
    latency: float = Field(default=0.2, ge=0, description="Weight of p95 latency (lower is better).")

    @model_validator(mode="after")
    def _some_weight(self) -> Weights:
        if self.quality + self.cost + self.latency <= 0:
            raise ValueError("selection weights must not all be zero")
        return self


@dataclass(frozen=True)
class Profile:
    """A named way of deciding. `merge_ties` False means a statistically insignificant quality edge still wins (no tie-breaking on cost)."""

    weights: Weights
    merge_ties: bool
    description: str


PROFILES: dict[str, Profile] = {
    "balanced": Profile(Weights(quality=0.6, cost=0.2, latency=0.2), True, "Quality first, then the best mix of cost and speed among systems that are equally good."),
    "max_quality": Profile(Weights(quality=1.0, cost=0.0, latency=0.0), False, "The highest answer quality, whatever it costs (no tie-breaking: a lead within the noise still wins)."),
    "cheapest_acceptable": Profile(Weights(quality=0.2, cost=0.8, latency=0.0), True, "The cheapest system whose quality is statistically indistinguishable from the best."),
    "lowest_latency": Profile(Weights(quality=0.2, cost=0.0, latency=0.8), True, "The fastest system whose quality is statistically indistinguishable from the best."),
}


def resolve_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        import difflib

        close = difflib.get_close_matches(name, PROFILES, n=1)
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise ValueError(f"Unknown selection profile '{name}'.{hint} Available: {', '.join(PROFILES)}") from None


def normalize(values: dict[str, float | None], *, higher_is_better: bool) -> dict[str, float | None]:
    """Min-max scale to 0..1 (1 = best) across the systems that have a value; all-equal values are all 1."""
    present = {system: value for system, value in values.items() if value is not None}
    if not present:
        return dict.fromkeys(values)
    low, high = min(present.values()), max(present.values())
    span = high - low

    def scaled(value: float) -> float:
        if span == 0:
            return 1.0
        return (value - low) / span if higher_is_better else (high - value) / span

    return {system: None if value is None else scaled(value) for system, value in values.items()}


def weighted_scores(quality: dict[str, float | None], cost: dict[str, float | None], latency: dict[str, float | None], weights: Weights) -> dict[str, float]:
    """Weighted mean of the normalized components each system has (a missing component drops out and the rest are re-weighted)."""
    parts = [
        (weights.quality, normalize(quality, higher_is_better=True)),
        (weights.cost, normalize(cost, higher_is_better=False)),
        (weights.latency, normalize(latency, higher_is_better=False)),
    ]
    scores: dict[str, float] = {}
    for system in quality:
        total = sum(weight for weight, scaled in parts if scaled[system] is not None)
        scores[system] = sum(weight * (scaled[system] or 0.0) for weight, scaled in parts) / total if total > 0 else 0.0
    return scores


def efficiency_penalties(cost: dict[str, float | None], latency: dict[str, float | None], weights: Weights) -> dict[str, float]:
    """How much worse each system is on cost and latency than the best of the group (0 = best), weighted by the profile.

    Used to choose between systems whose quality cannot be told apart. With no weight on cost or latency, cost decides.
    """
    cost_weight, latency_weight = (weights.cost, weights.latency) if weights.cost + weights.latency > 0 else (1.0, 0.0)
    scaled_cost, scaled_latency = normalize(cost, higher_is_better=False), normalize(latency, higher_is_better=False)
    penalties: dict[str, float] = {}
    for system in cost:
        parts = [(cost_weight, scaled_cost[system]), (latency_weight, scaled_latency[system])]
        total = sum(weight for weight, value in parts if value is not None)
        penalties[system] = sum(weight * (1 - value) for weight, value in parts if value is not None) / total if total > 0 else 0.0
    return penalties
