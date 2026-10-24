"""Turns a finished run's per-question rows into the statistics reports show (see docs/methodology.md#statistics).

Produces, per system, confidence intervals for the headline metrics, `cost_per_correct` and `pareto_optimal`
(merged into `metrics_summary.csv`), plus the contents of `stats.json`, `significance.csv` and `pareto.json`. With repeated trials a
question's score is the mean of its trials, intervals and comparisons resample trials as well as questions, and each system also gets
`answer_score_trial_sd`, `pass_at_k`, `pass_hat_k` and `rank_stability`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragbench.evaluation.stats import (
    ALPHA,
    CORRECT_THRESHOLD,
    cost_per_correct,
    grouped_bootstrap_ci,
    grouped_paired_bootstrap,
    holm_adjust,
    pareto_dominators,
    pareto_front,
    pass_rates,
)

PARETO_MAXIMIZE = ["answer_score"]
PARETO_MINIMIZE = ["avg_cost_per_question", "avg_latency_ms"]
SIGNIFICANCE_COLUMNS = [
    "metric",
    "system",
    "baseline",
    "n",
    "mean_diff",
    "ci_lo",
    "ci_hi",
    "p_value",
    "p_holm",
    "significant",
    "verdict",
    "wins",
    "ties",
    "losses",
]

Extractor = Callable[[dict[str, Any]], float | None]


@dataclass
class RunStats:
    summary_fields: dict[str, dict[str, Any]] = field(default_factory=dict)  # system -> columns to merge into its summary row
    significance_rows: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)  # stats.json
    pareto: dict[str, Any] = field(default_factory=dict)  # pareto.json
    baseline: str | None = None


def _none_if_nan(value: float) -> float | None:
    return None if math.isnan(value) else value


def headline_metrics(k_values: list[int], primary_k: int) -> dict[str, Extractor]:
    """Summary column -> how to read that metric off one `per_question_results.jsonl` row. These get intervals."""
    deepest = max(k_values)

    def retrieval(name: str) -> Extractor:
        return lambda row: row["retrieval_metrics"].get(name)  # empty for a system that retrieves nothing: missing, not 0

    return {
        f"retrieval_recall@{primary_k}": retrieval(f"recall@{primary_k}"),
        f"retrieval_mrr@{deepest}": retrieval(f"mrr@{deepest}"),
        f"retrieval_ndcg@{deepest}": retrieval(f"ndcg@{deepest}"),
        "answer_score": lambda row: row["answer_judge"]["answer_score"],
        "faithfulness": lambda row: row["answer_judge"]["faithfulness"],
    }


def cheapest_system(summary_rows: list[dict[str, Any]]) -> str | None:
    """The system with the lowest `$/Q` (the first one on a tie, and the first of all when no cost is known)."""
    priced = [row for row in summary_rows if row.get("avg_cost_per_question") is not None]
    if priced:
        return min(priced, key=lambda row: row["avg_cost_per_question"])["system"]
    return summary_rows[0]["system"] if summary_rows else None


def compute_run_stats(
    per_question_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    k_values: list[int],
    primary_k: int,
    n_boot: int,
    seed: int,
    baseline: str | None,
    trials: int = 1,
) -> RunStats:
    ok_rows = [row for row in per_question_rows if row["error"] is None]
    by_system: dict[str, dict[str, list[dict[str, Any]]]] = {row["system"]: {} for row in summary_rows}  # system -> question -> its trials
    for row in ok_rows:
        by_system[row["system"]].setdefault(row["question_id"], []).append(row)
    metrics = headline_metrics(k_values, primary_k)

    def values(system: str, extract: Extractor) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for question_id, rows in by_system[system].items():
            scores = [float(value) for row in rows if (value := extract(row)) is not None and not math.isnan(float(value))]
            if scores:
                found[question_id] = scores
        return found

    result = RunStats()
    per_correct = cost_per_correct(ok_rows)
    points = [
        {"system": row["system"], **{axis: row.get(axis) for axis in (*PARETO_MAXIMIZE, *PARETO_MINIMIZE)}}
        for row in summary_rows
    ]
    front = pareto_front(points, maximize=PARETO_MAXIMIZE, minimize=PARETO_MINIMIZE)
    system_stats: dict[str, Any] = {}
    for row in summary_rows:
        system = row["system"]
        fields: dict[str, Any] = {"cost_per_correct": per_correct.get(system), "pareto_optimal": system in front}
        intervals: dict[str, Any] = {}
        for key, extract in metrics.items():
            mean, lo, hi = grouped_bootstrap_ci(list(values(system, extract).values()), n_boot=n_boot, seed=seed)
            fields[f"{key}_ci_lo"], fields[f"{key}_ci_hi"] = _none_if_nan(lo), _none_if_nan(hi)
            intervals[key] = {"mean": _none_if_nan(mean), "lo": _none_if_nan(lo), "hi": _none_if_nan(hi)}
        if trials > 1:
            fields.update(_trial_fields(system, per_question_rows, values(system, metrics["answer_score"])))
        result.summary_fields[system] = fields
        system_stats[system] = {"n": len(by_system[system]), "metrics": intervals, "cost_per_correct": fields["cost_per_correct"]}
    if trials > 1:
        stability = rank_stability(ok_rows, [row["system"] for row in summary_rows], trials)
        for system, share in stability.items():
            result.summary_fields[system]["rank_stability"] = share

    result.baseline = baseline if baseline is not None else cheapest_system(summary_rows)
    comparisons = {key: metrics[key] for key in ("answer_score", f"retrieval_recall@{primary_k}")}
    result.significance_rows = _significance(by_system, comparisons, values, result.baseline, [row["system"] for row in summary_rows], n_boot, seed)
    method = "percentile bootstrap over questions; paired bootstrap against the baseline with Holm-Bonferroni adjustment per metric"
    if trials > 1:
        method = method.replace("over questions", "over questions, then over each question's trials") + f"; {trials} trials per question"
    result.stats = {
        "method": method,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": ALPHA,
        "baseline": result.baseline,
        "baseline_source": "config" if baseline is not None else "cheapest system",
        "correct_threshold": CORRECT_THRESHOLD,
        "metrics": list(metrics),
        "systems": system_stats,
        **({"trials": trials} if trials > 1 else {}),
    }
    result.pareto = {
        "maximize": PARETO_MAXIMIZE,
        "minimize": PARETO_MINIMIZE,
        "front": front,
        "dominated_by": pareto_dominators(points, maximize=PARETO_MAXIMIZE, minimize=PARETO_MINIMIZE),
        "points": points,
    }
    return result


def _trial_fields(system: str, per_question_rows: list[dict[str, Any]], scores: dict[str, list[float]]) -> dict[str, Any]:
    """How much a system's answers move between trials: the mean within-question SD of the answer score, and pass@k / pass^k. A trial
    that failed with an error counts as not correct (the user got no answer)."""
    # Identical trials are exactly 0: np.std would report the rounding of their mean (about 1e-16).
    spreads = [float(np.std(trial_scores, ddof=1)) if np.ptp(trial_scores) else 0.0 for trial_scores in scores.values() if len(trial_scores) > 1]
    correct: dict[str, list[bool]] = {}
    for row in per_question_rows:
        if row["system"] == system:
            right = row["error"] is None and float(row["answer_judge"]["answer_score"]) >= CORRECT_THRESHOLD
            correct.setdefault(row["question_id"], []).append(right)
    pass_at_k, pass_hat_k = pass_rates(list(correct.values()))
    return {"answer_score_trial_sd": float(np.mean(spreads)) if spreads else None, "pass_at_k": pass_at_k, "pass_hat_k": pass_hat_k}


def rank_stability(ok_rows: list[dict[str, Any]], systems: list[str], trials: int) -> dict[str, float | None]:
    """Per system, the share of trials on which its rank by mean answer score (over that trial's answers) is its rank over all trials.
    Ranks count the systems strictly ahead, so ties share the better rank; a system with no answer on a trial does not hold its rank."""
    by_trial: dict[int, dict[str, list[float]]] = {}
    overall: dict[str, dict[str, list[float]]] = {system: {} for system in systems}
    for row in ok_rows:
        if row["system"] not in overall:
            continue
        score = float(row["answer_judge"]["answer_score"])
        by_trial.setdefault(int(row.get("trial", 0)), {}).setdefault(row["system"], []).append(score)
        overall[row["system"]].setdefault(row["question_id"], []).append(score)

    def ranks(means: dict[str, float]) -> dict[str, int]:
        return {system: sum(1 for other in means.values() if other > mean) for system, mean in means.items()}

    final = ranks({system: float(np.mean([np.mean(s) for s in qs.values()])) for system, qs in overall.items() if qs})
    held = {system: 0 for system in systems}
    for trial in range(trials):
        per_trial = ranks({system: float(np.mean(scores)) for system, scores in by_trial.get(trial, {}).items()})
        for system, rank in per_trial.items():
            held[system] += rank == final.get(system)
    return {system: held[system] / trials if system in final else None for system in systems}


def _significance(
    by_system: dict[str, dict[str, list[dict[str, Any]]]],
    comparisons: dict[str, Extractor],
    values: Callable[[str, Extractor], dict[str, list[float]]],
    baseline: str | None,
    systems: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Every system against the baseline on the same questions; Holm-adjusted within each metric's family of comparisons."""
    if baseline is None:
        return []
    rows: list[dict[str, Any]] = []
    for metric, extract in comparisons.items():
        base_values = values(baseline, extract)
        family: list[dict[str, Any]] = []
        for system in systems:
            if system == baseline:
                continue
            mine = values(system, extract)
            shared = [question_id for question_id in by_system[baseline] if question_id in mine and question_id in base_values]
            if not shared:
                continue  # nothing measured on both (e.g. a system that retrieves nothing, for a retrieval metric)
            paired = grouped_paired_bootstrap([mine[q] for q in shared], [base_values[q] for q in shared], n_boot=n_boot, seed=seed)
            family.append(
                {
                    "metric": metric,
                    "system": system,
                    "baseline": baseline,
                    "n": paired.n,
                    "mean_diff": paired.mean_diff,
                    "ci_lo": paired.ci_lo,
                    "ci_hi": paired.ci_hi,
                    "p_value": paired.p_two_sided,
                    "wins": paired.wins,
                    "ties": paired.ties,
                    "losses": paired.losses,
                }
            )
        for entry, adjusted in zip(family, holm_adjust([entry["p_value"] for entry in family]), strict=True):
            significant = adjusted < ALPHA
            entry["p_holm"] = adjusted
            entry["significant"] = significant
            entry["verdict"] = ("better" if entry["mean_diff"] > 0 else "worse") if significant else "no clear difference"
        rows.extend({column: entry[column] for column in SIGNIFICANCE_COLUMNS} for entry in family)
    return rows
