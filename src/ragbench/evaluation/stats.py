"""Statistics for turning per-question scores into a decision. numpy only; everything is seeded and reproducible.

With a few dozen questions a gap of a few hundredths between two systems is noise. These helpers say how wide that noise
is (`bootstrap_ci`), whether one system really beats another on the *same* questions (`paired_bootstrap`, `holm_adjust`),
which systems are worth considering at all (`pareto_front`) and what a correct answer costs (`cost_per_correct`).
With repeated trials (`evaluation.trials`) a question has several scores: the `grouped_*` variants resample questions together with
their trials (a cluster bootstrap over per-question means). With one trial each they are exactly the plain functions. Missing values (NaN, None) are dropped, never counted as zero.
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


def _question_means(groups: Sequence[Sequence[float]]) -> list[float | None]:
    """Each question's mean over its finite trial scores; None for a question with none."""
    means: list[float | None] = []
    for group in groups:
        found = _finite(group)
        means.append((float(found[0]) if len(found) == 1 else float(found.mean())) if len(found) else None)
    return means


def grouped_bootstrap_ci(groups: Sequence[Sequence[float]], *, n_boot: int = 2000, alpha: float = ALPHA, seed: int = 0) -> tuple[float, float, float]:
    """`bootstrap_ci` when question i has several scores `groups[i]` (its trials): a cluster bootstrap, resampling questions with all
    their trials, which is the bootstrap of the per-question means. Trial noise is in it as the extra spread it gives those means;
    resampling the trials again inside each question would count it twice. With one score per question this is `bootstrap_ci`."""
    return bootstrap_ci(np.array([mean for mean in _question_means(groups) if mean is not None]), n_boot=n_boot, alpha=alpha, seed=seed)


def grouped_paired_bootstrap(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], *, n_boot: int = 5000, alpha: float = ALPHA, seed: int = 0
) -> PairedResult:
    """`paired_bootstrap` on per-question means, `a[i]` and `b[i]` being the same question's trial scores of two systems (a cluster
    bootstrap, as in `grouped_bootstrap_ci`); questions missing on either side are dropped. With one score each this is `paired_bootstrap`."""
    if len(a) != len(b):
        raise ValueError(f"paired samples must have the same length, got {len(a)} and {len(b)}")
    pairs = [(x, y) for x, y in zip(_question_means(a), _question_means(b), strict=True) if x is not None and y is not None]
    return paired_bootstrap(np.array([x for x, _ in pairs]), np.array([y for _, y in pairs]), n_boot=n_boot, alpha=alpha, seed=seed)


def pass_rates(correct: Sequence[Sequence[bool]]) -> tuple[float | None, float | None]:
    """`(pass_at_k, pass_hat_k)` over questions, each given as its k trials' correct / not correct: the share of questions answered
    correctly on at least one trial, and on every trial (the reliability number for a system that is run once per request)."""
    found = [list(trials) for trials in correct if len(trials)]
    if not found:
        return None, None
    return sum(any(trials) for trials in found) / len(found), sum(all(trials) for trials in found) / len(found)


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



def _binomial_cdf(n: int, p: float) -> np.ndarray:
    """P(B <= j) for j = 0..n, B ~ Binomial(n, p), computed in log space so large n does not underflow."""
    log_pmf = np.array([math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(p) + (n - j) * math.log1p(-p) for j in range(n + 1)])
    return np.cumsum(np.exp(log_pmf))


def quantile_ci(values: Sequence[float], q: float, *, alpha: float = ALPHA) -> tuple[float | None, float | None, float | None]:
    """The `q`-th percentile of `values` and a distribution-free `(1 - alpha)` interval for it: `(value, lo, hi)`.

    The bounds are order statistics chosen with the binomial distribution (the number of values below the true percentile is
    Binomial(n, q/100)), so the coverage is exact whatever the shape of the data. A bootstrap would not do here: it cannot reach above
    the largest value seen, and for a p95 from a few dozen calls it covers far less often than it claims. A bound the data cannot
    support yet is None: the p95 has no upper bound before `QUANTILE_MIN_N` values. None x3 with no data.
    """
    data = np.sort(_finite(values))
    n = len(data)
    if not n:
        return None, None, None
    value = float(np.percentile(data, q))
    cdf = _binomial_cdf(n, q / 100)  # cdf[j] = P(at most j values lie below the true percentile)
    below = [j for j in range(1, n + 1) if cdf[j - 1] <= alpha / 2]  # x_(l) is a lower bound when P(B <= l - 1) <= alpha / 2
    above = [j for j in range(1, n + 1) if cdf[j - 1] >= 1 - alpha / 2]  # x_(u) is an upper bound when P(B <= u - 1) >= 1 - alpha / 2
    lo = float(data[below[-1] - 1]) if below else None
    hi = float(data[above[0] - 1]) if above else None
    return value, lo, hi


def _min_n_for_upper_bound(q: float, alpha: float = ALPHA) -> int:
    """The fewest values whose maximum is a `1 - alpha / 2` upper bound on the `q`-th percentile: `1 - (q/100)**n >= 1 - alpha / 2`."""
    return math.ceil(math.log(alpha / 2) / math.log(q / 100))


QUANTILE_MIN_N = _min_n_for_upper_bound(95)  # 72: before this many measurements a p95 has no upper bound
