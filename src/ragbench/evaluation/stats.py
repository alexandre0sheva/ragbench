"""Statistics for turning per-question scores into a decision. numpy only; everything is seeded and reproducible.

With a few dozen questions a gap of a few hundredths between two systems is noise. These helpers say how wide that noise
is (`bootstrap_ci`), whether one system really beats another on the *same* questions (`paired_bootstrap`, `holm_adjust`),
which systems are worth considering at all (`pareto_front`) and what a correct answer costs (`cost_per_correct`).
Missing values (NaN, None) are dropped, never counted as zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ALPHA = 0.05  # significance level and (1 - alpha) interval width used throughout
CORRECT_THRESHOLD = 4.0  # an answer scoring at least this (0-5 judge scale) counts as correct for `cost_per_correct`
_BLOCK_CELLS = 2_000_000  # resample indices drawn at a time: keeps memory flat for large question sets


@dataclass(frozen=True)
class PairedResult:
    """`a` compared with `b` on the same questions: positive `mean_diff` means `a` scored higher."""

    mean_diff: float
    ci_lo: float
    ci_hi: float
    p_two_sided: float
    wins: int  # questions where a > b
    ties: int
    losses: int  # questions where a < b
    n: int  # pairs compared (pairs with a missing side are dropped)


def _finite(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    return array[np.isfinite(array)]


def _resampled_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of `n_boot` bootstrap resamples (same size, with replacement) of `values`."""
    n = len(values)
    means = np.empty(n_boot)
    block = max(1, _BLOCK_CELLS // n)
    for start in range(0, n_boot, block):
        stop = min(n_boot, start + block)
        means[start:stop] = values[rng.integers(0, n, size=(stop - start, n))].mean(axis=1)
    return means


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = ALPHA, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap `(1 - alpha)` confidence interval: `(mean, lo, hi)`. NaN x3 when nothing is measured."""
    data = _finite(values)
    if not len(data):
        return math.nan, math.nan, math.nan
    mean = float(data.mean())
    if np.ptp(data) == 0:
        return mean, mean, mean
    means = _resampled_means(data, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_boot: int = 5000, alpha: float = ALPHA, seed: int = 0) -> PairedResult:
    """Do `a` and `b` differ, judged on the per-question differences? `a[i]` and `b[i]` must be the same question.

    The interval is the bootstrap CI of the mean difference. The two-sided p-value is the share of resampled mean
    differences on the far side of zero (doubled, with a +1 correction so it is never exactly 0), so "the interval
    excludes zero" and "p < alpha" agree.
    """
    a_all, b_all = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_all.shape != b_all.shape:
        raise ValueError(f"paired samples must have the same length, got {a_all.shape} and {b_all.shape}")
    both = np.isfinite(a_all) & np.isfinite(b_all)
    a_ok, b_ok = a_all[both], b_all[both]
    diffs = a_ok - b_ok
    n = len(diffs)
    wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
    if n == 0:
        return PairedResult(math.nan, math.nan, math.nan, math.nan, 0, 0, 0, 0)
    mean_diff = float(diffs.mean())
    if np.ptp(diffs) == 0:
        p = 1.0 if mean_diff == 0 else 2.0 / (n_boot + 1)  # every resample agrees: as extreme as the resolution allows
        return PairedResult(mean_diff, mean_diff, mean_diff, p, wins, n - wins - losses, losses, n)
    means = _resampled_means(diffs, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    below = (int((means <= 0).sum()) + 1) / (n_boot + 1)
    above = (int((means >= 0).sum()) + 1) / (n_boot + 1)
    return PairedResult(mean_diff, float(lo), float(hi), min(1.0, 2 * min(below, above)), wins, n - wins - losses, losses, n)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order: the k-th smallest of m is multiplied by m - k + 1 (and kept
    monotone, capped at 1). Controls the chance of any false "significant" across the whole family of comparisons."""
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _dominators(points: list[dict[str, Any]], maximize: list[str], minimize: list[str]) -> dict[str, list[str]]:
    """For each point that can be placed, the other points that dominate it (empty for a point on the front)."""
    axes = [(name, 1.0) for name in maximize] + [(name, -1.0) for name in minimize]
    if not axes:
        raise ValueError("pareto_front needs at least one axis to maximize or minimize")

    def vector(point: dict[str, Any]) -> tuple[float, ...] | None:
        raw = [point.get(name) for name, _ in axes]
        numbers = [float(value) for value in raw if value is not None]
        if len(numbers) != len(raw) or not all(math.isfinite(number) for number in numbers):
            return None
        return tuple(sign * number for (_, sign), number in zip(axes, numbers, strict=True))  # larger is better on every axis

    def dominates(x: tuple[float, ...], y: tuple[float, ...]) -> bool:
        return all(p >= q for p, q in zip(x, y, strict=True)) and any(p > q for p, q in zip(x, y, strict=True))

    placed = [(str(point["system"]), vec) for point in points if (vec := vector(point)) is not None]
    return {name: [other for other, other_vec in placed if dominates(other_vec, vec)] for name, vec in placed}


def pareto_front(points: list[dict[str, Any]], *, maximize: list[str], minimize: list[str]) -> list[str]:
    """Names (the `system` key) of the points no other point beats on every axis, in input order.

    A point dominates another when it is at least as good on every axis and strictly better on at least one; identical
    points dominate nothing. A point missing a value on any axis cannot be placed, so it is not on the front (and
    dominates nothing).
    """
    return [name for name, beaten_by in _dominators(points, maximize, minimize).items() if not beaten_by]


def pareto_dominators(points: list[dict[str, Any]], *, maximize: list[str], minimize: list[str]) -> dict[str, list[str]]:
    """For every system that is *off* the Pareto front, the systems that dominate it (the reason it is off)."""
    return {name: beaten_by for name, beaten_by in _dominators(points, maximize, minimize).items() if beaten_by}


def cost_per_correct(rows: Iterable[dict[str, Any]], threshold: float = CORRECT_THRESHOLD) -> dict[str, float | None]:
    """Per system: all dollars spent on its (successful) questions ÷ the answers scoring at least `threshold`.

    Rows are `per_question_results.jsonl` rows. Wrong answers cost money too, so they stay in the numerator. A system
    with no correct answer has no finite price per correct answer: None.
    """
    spend: dict[str, float] = defaultdict(float)
    correct: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("error") is not None:
            continue
        spend[row["system"]] += float(row["cost"]["total_cost"])
        correct[row["system"]] += float(row["answer_judge"]["answer_score"]) >= threshold
    return {system: spend[system] / correct[system] if correct[system] else None for system in spend}


def latency_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    """Median and 95th percentile; None for both when there are no measurements."""
    if not len(values):
        return {"p50": None, "p95": None}
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
