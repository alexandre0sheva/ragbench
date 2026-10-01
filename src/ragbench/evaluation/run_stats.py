"""Turns a finished run's per-question rows into the statistics reports show (see docs/methodology.md#statistics).

Produces, per system, confidence intervals for the headline metrics, `cost_per_correct` and `pareto_optimal`
(merged into `metrics_summary.csv`), plus the contents of `stats.json`, `significance.csv` and `pareto.json`.
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
    bootstrap_ci,
    cost_per_correct,
    holm_adjust,
    paired_bootstrap,
    pareto_dominators,
    pareto_front,
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
) -> RunStats:
    ok_rows = [row for row in per_question_rows if row["error"] is None]
    by_system: dict[str, dict[str, dict[str, Any]]] = {row["system"]: {} for row in summary_rows}
    for row in ok_rows:
        by_system[row["system"]][row["question_id"]] = row
    metrics = headline_metrics(k_values, primary_k)

    def values(system: str, extract: Extractor) -> dict[str, float]:
        found = ((question_id, extract(row)) for question_id, row in by_system[system].items())
        return {question_id: float(value) for question_id, value in found if value is not None and not math.isnan(float(value))}

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
            mean, lo, hi = bootstrap_ci(np.array(list(values(system, extract).values())), n_boot=n_boot, seed=seed)
            fields[f"{key}_ci_lo"], fields[f"{key}_ci_hi"] = _none_if_nan(lo), _none_if_nan(hi)
            intervals[key] = {"mean": _none_if_nan(mean), "lo": _none_if_nan(lo), "hi": _none_if_nan(hi)}
        result.summary_fields[system] = fields
        system_stats[system] = {"n": len(by_system[system]), "metrics": intervals, "cost_per_correct": fields["cost_per_correct"]}

    result.baseline = baseline if baseline is not None else cheapest_system(summary_rows)
    comparisons = {key: metrics[key] for key in ("answer_score", f"retrieval_recall@{primary_k}")}
    result.significance_rows = _significance(by_system, comparisons, values, result.baseline, [row["system"] for row in summary_rows], n_boot, seed)
    result.stats = {
        "method": "percentile bootstrap over questions; paired bootstrap against the baseline with Holm-Bonferroni adjustment per metric",
        "n_boot": n_boot,
        "seed": seed,
        "alpha": ALPHA,
        "baseline": result.baseline,
        "baseline_source": "config" if baseline is not None else "cheapest system",
        "correct_threshold": CORRECT_THRESHOLD,
        "metrics": list(metrics),
        "systems": system_stats,
    }
    result.pareto = {
        "maximize": PARETO_MAXIMIZE,
        "minimize": PARETO_MINIMIZE,
        "front": front,
        "dominated_by": pareto_dominators(points, maximize=PARETO_MAXIMIZE, minimize=PARETO_MINIMIZE),
        "points": points,
    }
    return result


def _significance(
    by_system: dict[str, dict[str, dict[str, Any]]],
    comparisons: dict[str, Extractor],
    values: Callable[[str, Extractor], dict[str, float]],
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
            paired = paired_bootstrap(
                np.array([mine[q] for q in shared]), np.array([base_values[q] for q in shared]), n_boot=n_boot, seed=seed
            )
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
