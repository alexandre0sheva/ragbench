"""The pairwise stage of a run: which answers meet on which question, who judges each pair, and the arena it adds up to.

It runs after every system has answered, on each system's first-trial answer to each question (a failed answer takes no part).
`pairs` decides the schedule: `all` pairs, every system `vs_baseline`, or `adaptive` (questions in four rounds, each later round
spending its pairs where the arena so far is least sure). Every pair is judged in both orders and its first order is a coin flip, so
neither the config order nor a judge's taste for a position decides anything.

With `independent: true` a pair is judged by the first judge of the panel whose model family wrote neither answer; when every judge
shares a family with one side, all judges take it, so their leanings offset (counted as `same_family`).
"""

from __future__ import annotations

import logging
import math
import random
import threading
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

from ragbench.datasets.schema import Question
from ragbench.evaluation.arena import Comparison, arena_table, bradley_terry, position_bias, win_rates
from ragbench.evaluation.budget import BudgetGuard
from ragbench.evaluation.pairwise import PAIRWISE_PROMPT_VERSION, Candidate, PairwiseJudge
from ragbench.models.cost import CostBreakdown
from ragbench.models.family import model_family
from ragbench.runtime.parallel import ordered_parallel_map

if TYPE_CHECKING:
    from ragbench.config.schema import PairwiseConfig

logger = logging.getLogger(__name__)

ADAPTIVE_ROUNDS = 4
Pair = tuple[str, str]


@dataclass
class PairwiseResult:
    comparisons: list[Comparison] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)  # one per comparison, for `pairwise.jsonl`
    calls: int = 0  # judge requests, retries included
    failed: int = 0  # comparisons without a verdict in both orders (left out of the arena)
    skipped: int = 0  # comparisons not started because the spending cap was reached
    same_family: int = 0  # comparisons judged by a judge of one side's model family (no neutral judge on the panel)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    judges: list[str] = field(default_factory=list)
    pairs: str = "all"
    baseline: str | None = None  # the system every other one met (`pairs: vs_baseline` only)


def _oriented(pairs: list[Pair], rng: random.Random) -> list[Pair]:
    return [(y, x) if rng.random() < 0.5 else (x, y) for x, y in pairs]


def schedule(
    available: dict[str, list[str]],
    mode: str,
    max_pairs: int | None,
    *,
    baseline: str | None,
    seed: int,
    strengths: dict[str, float] | None = None,
    met: dict[frozenset[str], int] | None = None,
) -> dict[str, list[Pair]]:
    """The pairs each question's answers meet in, `(x, y)` with x shown first in the first order.

    `available` lists, per question, the systems with an answer to compare. `adaptive` with no `strengths` yet pairs each question's
    systems around a shuffled ring; with them, it takes the pairs whose outcome is least predictable (p(1 - p) under the strengths so
    far, discounted by how often the pair already met).
    """
    plan: dict[str, list[Pair]] = {}
    for question_id, systems in available.items():
        rng = random.Random(f"{seed}/{question_id}")
        if mode == "vs_baseline":
            pairs = [(baseline, s) for s in systems if s != baseline] if baseline in systems else []
        elif mode == "adaptive" and len(systems) > 1:
            k = max_pairs or len(systems)
            if strengths is None:
                ring = rng.sample(systems, len(systems))
                pairs = [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))][: k if len(ring) > 2 else 1]
            else:
                pairs = sorted(combinations(systems, 2), key=lambda pair: (-_information(pair, strengths, met or {}), rng.random()))[:k]
        else:
            pairs = list(combinations(systems, 2))
        if mode != "adaptive" and max_pairs is not None and len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)
        plan[question_id] = _oriented(pairs, rng)
    return plan


def _information(pair: Pair, strengths: dict[str, float], met: dict[frozenset[str], int]) -> float:
    x, y = (strengths.get(name, 1.0) for name in pair)
    p = x / (x + y)
    return p * (1 - p) / math.sqrt(1 + met.get(frozenset(pair), 0))


class _Counter:
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def add(self, n: int = 1) -> None:
        with self._lock:
            self.value += n


def run_pairwise(
    rows: list[dict[str, Any]],
    questions: list[Question],
    systems: list[str],
    *,
    judges: dict[str, PairwiseJudge],
    generators: dict[str, str | None],
    settings: PairwiseConfig,
    independent: bool,
    baseline: str | None,
    seed: int,
    workers: int = 1,
    budget: BudgetGuard | None = None,
) -> PairwiseResult:
    """Judge the scheduled pairs of the run's answers (`rows`: per-question rows) and collect the comparisons."""
    answers: dict[tuple[str, str], Candidate] = {}
    for row in rows:
        if row.get("error") is None and not row.get("trial") and row["system"] in systems:
            answers[(row["system"], row["question_id"])] = Candidate(str(row.get("answer") or ""), (row.get("answer_judge") or {}).get("answer_score"))
    by_id = {question.id: question for question in questions}
    available = {q.id: [s for s in systems if (s, q.id) in answers] for q in questions}
    refs = list(judges)
    calls_before = {ref: judge.calls for ref, judge in judges.items()}
    result = PairwiseResult(judges=refs, pairs=settings.pairs, baseline=baseline if settings.pairs == "vs_baseline" else None)
    lock = threading.Lock()
    skipped = _Counter()

    def judges_for(x: str, y: str) -> list[str]:
        if not independent:
            return refs[:1]
        families = {model_family(generators[x]), model_family(generators[y])}
        neutral = [ref for ref in refs if model_family(ref) not in families]
        return neutral[:1] or refs

    def judge_one(task: tuple[str, str, str, str]) -> tuple[Comparison, str, CostBreakdown] | None:
        question_id, x, y, ref = task
        if budget is not None and budget.exhausted:
            skipped.add()
            return None
        question = by_id[question_id]
        try:
            comparison, cost = judges[ref].compare(question, x, answers[(x, question_id)], y, answers[(y, question_id)])
        except Exception as exc:  # noqa: BLE001 (one failed comparison is left out, like an unusable verdict)
            logger.warning("Pairwise comparison of %s and %s on %s failed: %s: %s", x, y, question_id, type(exc).__name__, exc)
            comparison, cost = Comparison(question_id, x, y, (None, None)), CostBreakdown()
        if budget is not None:
            budget.charge(cost.total_cost)
        return comparison, ref, cost

    order = [q.id for q in questions]
    rounds = [order] if settings.pairs != "adaptive" else [order[i :: ADAPTIVE_ROUNDS] for i in range(ADAPTIVE_ROUNDS)]
    met: dict[frozenset[str], int] = {}
    for number, round_ids in enumerate(rounds):
        strengths = bradley_terry(result.comparisons, systems) if settings.pairs == "adaptive" and number > 0 else None
        plan = schedule({qid: available[qid] for qid in round_ids}, settings.pairs, settings.max_pairs_per_question, baseline=baseline, seed=seed, strengths=strengths, met=met)
        tasks = [(qid, x, y, ref) for qid in round_ids for x, y in plan[qid] for ref in judges_for(x, y)]
        for done in ordered_parallel_map(judge_one, tasks, workers=workers, thread_name_prefix="ragbench-pairwise"):
            if done is None:
                continue
            comparison, ref, cost = done
            with lock:
                result.comparisons.append(comparison)
                result.cost = result.cost.plus(cost)
                result.failed += not comparison.complete
                own = model_family(ref) in {model_family(generators[comparison.x]), model_family(generators[comparison.y])}
                result.same_family += own
                met[frozenset((comparison.x, comparison.y))] = met.get(frozenset((comparison.x, comparison.y)), 0) + comparison.complete
                result.records.append(
                    {
                        "question_id": comparison.question_id,
                        "systems": [comparison.x, comparison.y],
                        "judge": ref,
                        "verdicts": list(comparison.verdicts),
                        "score": comparison.score,
                        "flipped": comparison.flipped,
                        "same_family": own,
                        "prompt_version": PAIRWISE_PROMPT_VERSION,
                    }
                )
    result.skipped = skipped.value
    result.calls = sum(judge.calls - calls_before[ref] for ref, judge in judges.items())
    return result


def summarize(result: PairwiseResult, systems: list[str], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The arena table, the win-rate matrix and the run-summary block of a pairwise stage."""
    return {
        "arena": arena_table(result.comparisons, systems, n_boot=n_boot, seed=seed),
        "win_rates": win_rates(result.comparisons, systems),
        "summary": {
            "prompt_version": PAIRWISE_PROMPT_VERSION,
            "pairs": result.pairs,
            "baseline": result.baseline,
            "judges": result.judges,
            "comparisons": len(result.comparisons),
            "calls": result.calls,
            "failed": result.failed,
            "skipped_by_budget": result.skipped,
            "same_family": result.same_family,
            "cost_usd": result.cost.total_cost,
            "position_bias": position_bias(result.comparisons),
        },
    }
