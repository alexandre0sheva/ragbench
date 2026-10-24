"""The pairwise arena: Bradley-Terry strengths from head-to-head verdicts, an Elo-scale score with a bootstrap interval, win rates and
the judge's position bias.

Every pair of answers is judged twice, once in each order. A `Comparison` holds both verdicts in position terms (`first`, `second`,
`tie`), so a judge that always prefers one position gives each system one win and one loss: its bias cancels into a tie, and
`position_bias` measures how often that happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Verdict = Literal["first", "second", "tie"] | None
# Every pair of systems gets one virtual drawn comparison: it keeps a system that never wins (or never loses) at a finite strength
# and barely moves the others once there are a few real comparisons.
PRIOR_COMPARISONS = 1.0
ELO_BASE, ELO_SCALE = 1000.0, 400.0  # arena score = 1000 + 400 log10(strength / geometric mean of strengths)
_POINTS = {"first": 1.0, "second": 0.0, "tie": 0.5}
ARENA_SUMMARY_COLUMNS = ("arena_score", "arena_score_ci_lo", "arena_score_ci_hi", "arena_win_rate")  # also added to `metrics_summary.csv`


@dataclass(frozen=True)
class Comparison:
    """`x` against `y` on one question: `verdicts[0]` with `x` shown first, `verdicts[1]` with `y` shown first."""

    question_id: str
    x: str
    y: str
    verdicts: tuple[Verdict, Verdict]

    @property
    def complete(self) -> bool:
        return self.verdicts[0] is not None and self.verdicts[1] is not None

    @property
    def score(self) -> float | None:
        """Points `x` won over both orders, from 0 to 1 (0.5 is a draw); None unless both orders gave a verdict."""
        first, second = self.verdicts
        if first is None or second is None:
            return None
        return (_POINTS[first] + 1.0 - _POINTS[second]) / 2

    @property
    def flipped(self) -> bool:
        """The same position won both times: the order decided the verdict, not the answers."""
        return self.complete and self.verdicts[0] == self.verdicts[1] and self.verdicts[0] != "tie"


def _matrices(comparisons: list[Comparison], index: dict[str, int]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per question: points won `W[q, i, j]` by system i over j and verdicts `N[q, i, j]` between them (two per complete comparison)."""
    questions = sorted({c.question_id for c in comparisons if c.complete})
    position = {question_id: n for n, question_id in enumerate(questions)}
    size = len(index)
    wins = np.zeros((len(questions), size, size))
    counts = np.zeros((len(questions), size, size))
    for c in comparisons:
        score = c.score
        if score is None or c.x not in index or c.y not in index:
            continue
        q, i, j = position[c.question_id], index[c.x], index[c.y]
        wins[q, i, j] += 2 * score
        wins[q, j, i] += 2 * (1 - score)
        counts[q, i, j] += 2
        counts[q, j, i] += 2
    return questions, wins, counts


def _fit(wins: np.ndarray, counts: np.ndarray, iterations: int = 5000, tol: float = 1e-10) -> np.ndarray:
    """Bradley-Terry strengths by Hunter's MM algorithm, for a batch: `wins`/`counts` are (B, S, S), the result (B, S) with geometric mean 1."""
    size = wins.shape[-1]
    off = 1.0 - np.eye(size)
    wins = wins + off * PRIOR_COMPARISONS / 2
    counts = counts + off * PRIOR_COMPARISONS
    total = wins.sum(axis=2)
    strength = np.ones(wins.shape[:2])
    for _ in range(iterations):
        pair = strength[:, :, None] + strength[:, None, :]
        updated = total / (counts / pair).sum(axis=2)
        updated /= np.exp(np.log(updated).mean(axis=1, keepdims=True))
        done = np.max(np.abs(updated - strength)) < tol
        strength = updated
        if done:
            break
    return strength


def _scored(comparisons: list[Comparison], systems: list[str]) -> list[str]:
    met = {name for c in comparisons if c.complete for name in (c.x, c.y)}
    return [name for name in systems if name in met]


def bradley_terry(comparisons: list[Comparison], systems: list[str]) -> dict[str, float]:
    """Each system's strength (geometric mean 1): `strength[x] / (strength[x] + strength[y])` is the chance x beats y.
    Systems without a complete comparison are left out."""
    names = _scored(comparisons, systems)
    if not names:
        return {}
    _, wins, counts = _matrices(comparisons, {name: n for n, name in enumerate(names)})
    strength = _fit(wins.sum(axis=0)[None], counts.sum(axis=0)[None])[0]
    return dict(zip(names, strength.tolist(), strict=True))


def _elo(strength: np.ndarray) -> np.ndarray:
    return ELO_BASE + ELO_SCALE * np.log10(strength)


def arena_table(comparisons: list[Comparison], systems: list[str], *, n_boot: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """One row per system (in `systems` order): arena score with a 95% interval from resampling questions, and its record (a comparison
    is a win above 0.5 points, a loss below, a tie at 0.5). A system without a complete comparison has no score."""
    names = _scored(comparisons, systems)
    index = {name: n for n, name in enumerate(names)}
    point: dict[str, float] = {}
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    if names:
        _, wins, counts = _matrices(comparisons, index)
        scores = _elo(_fit(wins.sum(axis=0)[None], counts.sum(axis=0)[None])[0])
        rng = np.random.default_rng(seed)
        n_questions = wins.shape[0]
        weights = rng.multinomial(n_questions, np.full(n_questions, 1 / n_questions), size=n_boot).astype(float)
        boot = _elo(_fit(np.einsum("bq,qij->bij", weights, wins), np.einsum("bq,qij->bij", weights, counts)))
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for name, n in index.items():
            point[name], low[name], high[name] = float(scores[n]), float(lo[n]), float(hi[n])
    rows = []
    for name in systems:
        scores_of = [c.score if c.x == name else 1 - c.score for c in comparisons if c.score is not None and name in (c.x, c.y)]
        rows.append(
            {
                "system": name,
                "arena_score": point.get(name),
                "arena_score_ci_lo": low.get(name),
                "arena_score_ci_hi": high.get(name),
                "arena_wins": sum(s > 0.5 for s in scores_of),
                "arena_ties": sum(s == 0.5 for s in scores_of),
                "arena_losses": sum(s < 0.5 for s in scores_of),
                "arena_comparisons": len(scores_of),
                "arena_win_rate": sum(scores_of) / len(scores_of) if scores_of else None,
            }
        )
    return rows


def win_rates(comparisons: list[Comparison], systems: list[str]) -> dict[str, dict[str, float | None]]:
    """`rates[x][y]`: the share of points x won against y over their comparisons (None when they never met)."""
    points: dict[tuple[str, str], list[float]] = {}
    for c in comparisons:
        if c.score is not None:
            points.setdefault((c.x, c.y), []).append(c.score)
            points.setdefault((c.y, c.x), []).append(1 - c.score)
    return {x: {y: (sum(v) / len(v)) if (v := points.get((x, y))) else None for y in systems} for x in systems}


def position_bias(comparisons: list[Comparison]) -> dict[str, Any]:
    """How much the order of the answers swayed the judge, over complete comparisons: `flip_rate` is the share whose verdict followed
    the position (the same position won both times), `first_position_rate` the share of decisive verdicts that went to the answer shown first."""
    complete = [c for c in comparisons if c.complete]
    decisive = [v for c in complete for v in c.verdicts if v in ("first", "second")]
    return {
        "pairs": len(complete),
        "flip_rate": sum(c.flipped for c in complete) / len(complete) if complete else None,
        "first_position_rate": decisive.count("first") / len(decisive) if decisive else None,
    }
