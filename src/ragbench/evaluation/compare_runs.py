"""Compare two finished runs of (mostly) the same systems: how each metric moved, and whether a drop is real.

Where both runs answered the same questions, an answer-score change is judged by the paired bootstrap of `ragbench.evaluation.stats` (the same
test the leaderboard uses against its baseline), Holm-adjusted across the systems compared. Where the questions differ, there is nothing to pair,
so a drop is only flagged when it is larger than `min_drop` and the verdict says it was not tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.evaluation.stats import ALPHA, holm_adjust, paired_bootstrap
from ragbench.reporting.report_data import _read_csv, _read_json, _read_jsonl

MIN_PAIRED_QUESTIONS = 5
DEFAULT_MIN_DROP = 0.25  # answer-score points; only used when the runs share too few questions to test
REGRESSION, IMPROVEMENT, UNCHANGED, UNTESTED_DROP, UNTESTED = "regression", "improvement", "no clear change", "drop (not tested)", "not tested"


@dataclass
class MetricDelta:
    key: str
    header: str
    a: float | None
    b: float | None
    higher_is_better: bool

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a


@dataclass
class SystemComparison:
    system: str
    metrics: list[MetricDelta]
    verdict: str
    paired_questions: int = 0
    paired_diff: float | None = None  # mean answer-score change over the questions both runs answered
    ci: tuple[float, float] | None = None
    p_holm: float | None = None


@dataclass
class RunComparison:
    a: Path
    b: Path
    systems: list[SystemComparison] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[str]:
        return [row.system for row in self.systems if row.verdict in (REGRESSION, UNTESTED_DROP)]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _metric_specs(primary_k: Any, columns: set[str]) -> list[tuple[str, str, bool]]:
    specs = [("answer_score", "Answer", True), ("faithfulness", "Faithful", True)]
    if primary_k is not None and f"recall@{primary_k}" in columns:
        specs.append((f"recall@{primary_k}", f"Recall@{primary_k}", True))
    specs += [("avg_cost_per_question", "$/Q", False), ("avg_latency_ms", "Latency ms", False)]
    return [spec for spec in specs if spec[0] in columns]


def _answer_scores(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    found: dict[str, float] = {}
    for row in rows:
        if row.get("system") == system and row.get("error") is None:
            score = _number((row.get("answer_judge") or {}).get("answer_score"))
            if score is not None:
                found[str(row["question_id"])] = score
    return found


def compare_runs(a_dir: Path, b_dir: Path, *, min_drop: float = DEFAULT_MIN_DROP, n_boot: int = 5000, seed: int = 0) -> RunComparison:
    """How run `b` moved relative to run `a` (deltas are b minus a)."""
    rows_a, rows_b = _read_csv(a_dir / "metrics_summary.csv"), _read_csv(b_dir / "metrics_summary.csv")
    for path, rows in ((a_dir, rows_a), (b_dir, rows_b)):
        if not rows:
            raise ValueError(f"{path} has no metrics_summary.csv with results to compare.")
    by_a, by_b = {str(r["system"]): r for r in rows_a}, {str(r["system"]): r for r in rows_b}
    primary_k = _read_json(b_dir / "run_summary.json").get("primary_k") or _read_json(a_dir / "run_summary.json").get("primary_k")
    specs = _metric_specs(primary_k, set(rows_a[0]) & set(rows_b[0]))
    comparison = RunComparison(a_dir, b_dir, only_a=[n for n in by_a if n not in by_b], only_b=[n for n in by_b if n not in by_a])

    manifest_a, manifest_b = _read_json(a_dir / "run_manifest.json"), _read_json(b_dir / "run_manifest.json")
    if manifest_a.get("mode") != manifest_b.get("mode"):
        comparison.notes.append(f"The runs used different modes ({manifest_a.get('mode')} and {manifest_b.get('mode')}); mock scores say nothing about live ones.")
    elif manifest_b.get("mode") == "mock":
        comparison.notes.append("Both runs are mock runs: the scores only validate the pipeline.")
    if manifest_a.get("dataset_hash") and manifest_b.get("dataset_hash") and manifest_a["dataset_hash"] != manifest_b["dataset_hash"]:
        comparison.notes.append("The runs used different datasets, so a change mixes the systems' and the data's.")

    questions_a, questions_b = _read_jsonl(a_dir / "per_question_results.jsonl"), _read_jsonl(b_dir / "per_question_results.jsonl")
    common = [name for name in by_b if name in by_a]
    p_values: dict[str, float] = {}
    results: dict[str, SystemComparison] = {}
    for name in common:
        metrics = [MetricDelta(key, header, _number(by_a[name].get(key)), _number(by_b[name].get(key)), higher) for key, header, higher in specs]
        scores_a, scores_b = _answer_scores(questions_a, name), _answer_scores(questions_b, name)
        shared = [q for q in scores_a if q in scores_b]
        result = SystemComparison(name, metrics, UNTESTED, paired_questions=len(shared))
        if len(shared) >= MIN_PAIRED_QUESTIONS:
            paired = paired_bootstrap(np.array([scores_b[q] for q in shared]), np.array([scores_a[q] for q in shared]), n_boot=n_boot, seed=seed)
            result.ci = (paired.ci_lo, paired.ci_hi)
            p_values[name] = paired.p_two_sided
            result.p_holm = paired.p_two_sided
            result.paired_diff = paired.mean_diff
            result.verdict = UNCHANGED  # settled below, once the p-values can be adjusted together
        else:
            answer = next((m for m in metrics if m.key == "answer_score"), None)
            if answer is not None and answer.delta is not None and answer.delta < -min_drop:
                result.verdict = UNTESTED_DROP
        results[name] = result
    if p_values:
        adjusted = dict(zip(p_values, holm_adjust(list(p_values.values())), strict=True))
        for name, p in adjusted.items():
            result = results[name]
            result.p_holm = p
            delta = result.paired_diff or 0.0
            if p < ALPHA and delta < 0:
                result.verdict = REGRESSION
            elif p < ALPHA and delta > 0:
                result.verdict = IMPROVEMENT
    comparison.systems = list(results.values())
    if any(row.verdict == UNTESTED for row in comparison.systems):
        comparison.notes.append(f"Fewer than {MIN_PAIRED_QUESTIONS} questions are shared for some systems, so their changes were not tested for significance.")
    return comparison


def comparison_to_dict(comparison: RunComparison) -> dict[str, Any]:
    return {
        "a": str(comparison.a),
        "b": str(comparison.b),
        "regressions": comparison.regressions,
        "only_in_a": comparison.only_a,
        "only_in_b": comparison.only_b,
        "notes": comparison.notes,
        "systems": [
            {
                "system": row.system,
                "verdict": row.verdict,
                "paired_questions": row.paired_questions,
                "answer_score_paired_diff": row.paired_diff,
                "answer_score_delta_ci": list(row.ci) if row.ci else None,
                "p_holm": row.p_holm,
                "metrics": {m.key: {"a": m.a, "b": m.b, "delta": m.delta} for m in row.metrics},
            }
            for row in comparison.systems
        ],
    }
