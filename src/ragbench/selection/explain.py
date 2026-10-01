"""Plain-language explanations of a recommendation, generated from the data (never canned): the rationale bullets and `recommendation.md`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragbench.selection.constraints import Constraints
from ragbench.selection.scoring import Profile, Weights

if TYPE_CHECKING:
    from ragbench.selection.recommend import Recommendation, ScoredSystem

MOCK_CAVEAT = (
    "Mock run: these scores come from a scripted model and a heuristic judge, so this recommendation only demonstrates the process. "
    "It says nothing about which system is really best."
)
FEW_QUESTIONS = 30


@dataclass(frozen=True)
class TieTest:
    """The best system compared with another on the same questions."""

    mean_diff: float  # best minus this system (positive: the best scored higher)
    p_holm: float  # Holm-adjusted two-sided p-value (NaN when there were no shared questions)
    tied: bool  # not significantly worse than the best


@dataclass(frozen=True)
class Comparison:
    """The winner against another system: quality difference with its interval, and how much more or less the winner costs."""

    winner: str
    other: str
    quality_diff: float
    ci_lo: float
    ci_hi: float
    significant: bool
    cost_ratio: float | None
    n: int


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.5f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _names(names: list[str]) -> str:
    quoted = [f"`{name}`" for name in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _quality(value: float | None, label: str) -> str:
    return "n/a" if value is None else f"{value:.2f}" if label == "answer score" else f"{value:.3f}"


def _tie_basis(winner: ScoredSystem, others: list[ScoredSystem], weights: Weights) -> str:
    """Why the winner is the one picked from systems whose quality cannot be told apart. Only differences that exist are mentioned."""
    cost_matters = weights.cost > 0 or weights.latency == 0  # with no weight on either, cost decides
    latency_matters = weights.latency > 0
    priciest = max((o.cost_per_question for o in others if o.cost_per_question is not None), default=None)
    slowest = max((o.latency_ms_p95 for o in others if o.latency_ms_p95 is not None), default=None)
    cost_differs = cost_matters and priciest is not None and winner.cost_per_question is not None and priciest > winner.cost_per_question
    latency_differs = latency_matters and slowest is not None and winner.latency_ms_p95 is not None and slowest > winner.latency_ms_p95
    parts: list[str] = []
    if cost_differs:
        parts.append(f"{_money(winner.cost_per_question)} per question against up to {_money(priciest)}")
    if latency_differs:
        parts.append(f"p95 latency {_ms(winner.latency_ms_p95)} against up to {_ms(slowest)}")
    if parts:
        goal = "cost and latency" if cost_differs and latency_differs else "cost" if cost_differs else "latency"
        return f"so it wins on {goal} ({'; '.join(parts)})"
    return "so, as they cost and run alike, the simplest (fewest model calls and steps) wins"


def explain_winner(
    *,
    ranking: list[ScoredSystem],
    leader: str,
    ties: dict[str, TieTest],
    tied: list[str],
    comparisons: list[Comparison],
    label: str,
    profile_name: str,
    profile: Profile,
    weights: Weights,
    infeasible: dict[str, list[str]],
    mode: str,
    n_questions: int,
    constraints: Constraints,
    alpha: float,
) -> list[str]:
    """The reasons for the choice, most important first. `ranking[0]` is the winner."""
    by_name = {row.system: row for row in ranking}
    winner = ranking[0]
    lines: list[str] = [MOCK_CAVEAT] if mode == "mock" else []
    headline = (
        f"`{winner.system}` is recommended under the `{profile_name}` profile ({profile.description[0].lower()}{profile.description[1:].rstrip('.')}): "
        f"{label} {_quality(winner.quality, label)}, {_money(winner.cost_per_question)} per question, p95 latency {_ms(winner.latency_ms_p95)}."
    )
    lines.append(headline)
    others_tied = [by_name[name] for name in tied if name != winner.system]
    if len(ranking) == 1:
        lines.append(f"`{winner.system}` is the only system {'that meets your constraints' if infeasible else 'in the run'}, so there is nothing to choose between.")
    elif not profile.merge_ties:
        lines.append(f"The `{profile_name}` profile does not treat statistical ties as ties: `{winner.system}` has the highest {label}, whatever it costs.")
    elif others_tied:
        lines.append(
            f"`{winner.system}` ties with {_names([o.system for o in others_tied])} on {label}: they are statistically indistinguishable from the best "
            f"(`{leader}`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ {alpha:g}). Quality does not separate them, "
            f"{_tie_basis(winner, others_tied, weights)}."
        )
    elif all(name in ties and not math.isnan(ties[name].p_holm) for name in (row.system for row in ranking[1:])):
        lines.append(
            f"`{winner.system}` has the best {label} and is significantly better than every other feasible system "
            f"(paired bootstrap on the same questions, Holm-adjusted p < {alpha:g} against each)."
        )
    else:
        lines.append(f"`{winner.system}` has the best {label}; no other feasible system is statistically tied with it, though some could not be compared on shared questions.")
    cheapest = min((r for r in ranking if r.cost_per_question is not None), key=lambda r: r.cost_per_question or 0.0, default=None)
    for comparison in comparisons:
        roles = [role for role, name in (("the best " + label, leader), ("the cheapest feasible system", cheapest.system if cheapest else None)) if name == comparison.other]
        role = f" ({' and '.join(roles)})" if roles else ""
        if not comparison.n or math.isnan(comparison.quality_diff):
            lines.append(f"{comparison.winner} vs {comparison.other}{role}: no question was measured on both, so they cannot be compared.")
            continue
        verdict = "significant" if comparison.significant else "not significant"
        cost = f", {comparison.cost_ratio:.1f}× cost" if comparison.cost_ratio is not None else ""
        lines.append(
            f"{comparison.winner} vs {comparison.other}{role}: {comparison.quality_diff:+.2f} {label} "
            f"(95% CI {comparison.ci_lo:+.2f} to {comparison.ci_hi:+.2f}, {verdict}){cost}"
        )
    if infeasible:
        shown = [f"`{name}` ({reasons[0]}{f' and {len(reasons) - 1} more' if len(reasons) > 1 else ''})" for name, reasons in list(infeasible.items())[:4]]
        more = f" and {len(infeasible) - 4} more" if len(infeasible) > 4 else ""
        lines.append(f"Excluded by your constraints: {'; '.join(shown)}{more}.")
    if 0 < n_questions < FEW_QUESTIONS:
        lines.append(
            f"Only {n_questions} questions were answered, so most differences cannot be told apart from noise. Treat this as a shortlist and add questions before relying on it."
        )
    return lines


def explain_no_winner(infeasible: dict[str, list[str]], closest_miss: str | None, mode: str, label: str) -> list[str]:
    lines: list[str] = [MOCK_CAVEAT] if mode == "mock" else []
    head = "No system meets all of your constraints, so nothing is recommended."
    if closest_miss is not None:
        reasons = infeasible[closest_miss]
        head += f" The closest is `{closest_miss}`, which {'is only held back by' if len(reasons) == 1 else 'is held back by'}: {'; '.join(reasons)}."
    lines.append(head)
    lines.append("Loosen a constraint, or add a system that fits it, and run `ragbench recommend` again on this run.")
    return lines


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(headers), _row(["---"] * len(headers)), *(_row(row) for row in rows)]


def render_markdown(rec: Recommendation) -> str:
    """`recommendation.md`: the decision, the reasons, the ranking, per-category winners, and what was excluded."""
    label = {"answer_score": "answer score", "token_f1": "token F1"}.get(rec.quality_metric, rec.quality_metric)
    lines = ["# Recommendation", ""]
    if rec.winner is None:
        lines += ["**No system meets your constraints.**", ""]
    else:
        lines += [f"**Deploy `{rec.winner}`** (profile `{rec.profile}`, run `{rec.run_id}`). Its config is in `winner.yaml`.", ""]
    lines += ["## Why", "", *(f"- {line}" for line in rec.rationale), ""]
    if rec.ranking:
        rows = [
            [
                str(rank),
                f"`{row.system}`",
                f"{row.score:.2f}",
                _quality(row.quality, label),
                _money(row.cost_per_question),
                _ms(row.latency_ms_p95),
                _money(row.ingestion_cost),
                f"{row.llm_calls:.1f}",
                "tie" if row.tied else "",
                "✓" if row.pareto else "",
            ]
            for rank, row in enumerate(rec.ranking, start=1)
        ]
        lines += [
            "## Ranking",
            "",
            "The winner first, then the systems tied with it (best choice first), then the rest by weighted score. "
            "*Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; "
            "*tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.",
            "",
            *_table(["#", "System", "Score", label.capitalize(), "$/Q", "p95", "Ingestion", "Calls/Q", "Tie", "Pareto"], rows),
            "",
        ]
    if rec.by_category:
        systems = [row.system for row in rec.ranking]
        rows = [
            [category, f"`{winner}`", *[_quality(rec.category_scores[category].get(system), label) for system in systems]]
            for category, winner in rec.by_category.items()
        ]
        lines += [
            "## Winner by category",
            "",
            "The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.",
            "",
            *_table(["Category", "Winner", *[f"`{system}`" for system in systems]], rows),
            "",
        ]
    if rec.infeasible:
        lines += ["## Excluded by your constraints", ""]
        lines += [f"- `{name}`: {'; '.join(reasons)}" for name, reasons in rec.infeasible.items()]
        lines.append("")
    constraints = {key: value for key, value in rec.constraints.model_dump().items() if value not in (None, False)}
    lines += [
        "## How this was chosen",
        "",
        f"- Quality is the mean **{label}** over {rec.n_questions} questions. Weights: quality {rec.weights.quality:g}, cost {rec.weights.cost:g}, latency {rec.weights.latency:g}.",
        f"- Constraints: {', '.join(f'{key} = {value}' for key, value in constraints.items()) if constraints else 'none'}.",
        "- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. "
        "See docs/methodology.md#selection.",
        "",
    ]
    return "\n".join(lines)
