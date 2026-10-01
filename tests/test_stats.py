from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.stats import (
    bootstrap_ci,
    cost_per_correct,
    holm_adjust,
    latency_percentiles,
    paired_bootstrap,
    pareto_front,
)

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def test_ci_of_a_constant_array_is_the_point():
    assert bootstrap_ci(np.full(30, 3.5)) == (3.5, 3.5, 3.5)
    assert bootstrap_ci(np.array([2.0])) == (2.0, 2.0, 2.0)


def test_ci_ignores_missing_values_and_is_nan_without_any():
    mean, lo, hi = bootstrap_ci(np.array([1.0, np.nan, 3.0, np.nan]))
    assert mean == 2.0 and lo <= mean <= hi
    assert all(math.isnan(value) for value in bootstrap_ci(np.array([np.nan, np.nan])))
    assert all(math.isnan(value) for value in bootstrap_ci(np.array([])))


def test_ci_is_reproducible_for_a_seed_and_moves_with_it():
    values = np.random.default_rng(1).normal(size=60)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_ci_brackets_the_mean_with_roughly_the_textbook_width():
    values = np.random.default_rng(7).normal(loc=0.0, scale=1.0, size=200)
    mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=0)
    assert lo < mean < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 / math.sqrt(200), rel=0.2)  # width of a 95% interval for a standard-normal mean


def test_ci_coverage_on_a_known_distribution():
    """Of 200 seeded samples from N(0, 1), the 95% intervals around the mean cover the true mean 0 almost always."""
    rng = np.random.default_rng(11)
    covered = 0
    for index in range(200):
        _, lo, hi = bootstrap_ci(rng.normal(size=50), n_boot=400, seed=index)
        covered += lo <= 0.0 <= hi
    assert 0.90 <= covered / 200 <= 0.99


def test_alpha_controls_the_interval_width():
    values = np.random.default_rng(2).normal(size=80)
    _, lo95, hi95 = bootstrap_ci(values, alpha=0.05)
    _, lo80, hi80 = bootstrap_ci(values, alpha=0.20)
    assert (hi80 - lo80) < (hi95 - lo95)


def _per_question_difficulty(n: int = 60) -> np.ndarray:
    return np.random.default_rng(5).uniform(1.0, 4.0, size=n)  # questions differ a lot; systems differ little


def test_paired_bootstrap_detects_a_planted_shift():
    base = _per_question_difficulty()
    noise = np.random.default_rng(6)
    better = base + 0.3 + noise.normal(scale=0.15, size=base.size)
    result = paired_bootstrap(better, base, seed=0)

    assert result.mean_diff == pytest.approx(0.3, abs=0.1)
    assert result.ci_lo > 0 and result.p_two_sided < 0.01
    assert result.wins > result.losses and result.wins + result.ties + result.losses == base.size == result.n


def test_paired_bootstrap_does_not_flag_pure_noise():
    base = _per_question_difficulty()
    noise = np.random.default_rng(8)
    a = base + noise.normal(scale=0.3, size=base.size)
    b = base + noise.normal(scale=0.3, size=base.size)
    result = paired_bootstrap(a, b, seed=0)

    assert result.ci_lo < 0 < result.ci_hi
    assert result.p_two_sided > 0.05


def test_paired_bootstrap_counts_wins_ties_and_losses_and_handles_identical_systems():
    result = paired_bootstrap(np.array([3.0, 2.0, 5.0, 1.0]), np.array([2.0, 2.0, 5.0, 4.0]))
    assert (result.wins, result.ties, result.losses) == (1, 2, 1)
    same = paired_bootstrap(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert same.mean_diff == 0 and same.p_two_sided == 1.0 and (same.ci_lo, same.ci_hi) == (0.0, 0.0)


def test_paired_bootstrap_drops_pairs_with_a_missing_side_and_is_reproducible():
    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([0.0, 5.0, np.nan, 3.0])
    result = paired_bootstrap(a, b, seed=2)
    assert result.n == 2 and result.mean_diff == 1.0
    assert paired_bootstrap(a, b, seed=2) == result
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap(np.ones(3), np.ones(4))


def test_holm_adjusts_in_step_down_order_and_stays_monotone():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm_adjust([0.001]) == [0.001]
    assert holm_adjust([]) == []
    adjusted = holm_adjust([0.2, 0.001, 0.04, 0.01])
    assert adjusted == pytest.approx([0.2, 0.004, 0.08, 0.03])


def test_pareto_front_on_a_hand_built_set():
    points = [
        {"system": "cheap_weak", "quality": 0.5, "cost": 1.0, "latency": 10.0},
        {"system": "mid", "quality": 0.7, "cost": 2.0, "latency": 20.0},
        {"system": "best", "quality": 0.9, "cost": 5.0, "latency": 30.0},
        {"system": "dominated", "quality": 0.6, "cost": 3.0, "latency": 25.0},  # mid is better on every axis
        {"system": "fast_pricey", "quality": 0.7, "cost": 6.0, "latency": 5.0},  # nothing beats its latency
    ]
    front = pareto_front(points, maximize=["quality"], minimize=["cost", "latency"])
    assert front == ["cheap_weak", "mid", "best", "fast_pricey"]  # input order, dominated one removed


def test_pareto_front_keeps_identical_points_and_skips_points_with_a_missing_axis():
    points = [
        {"system": "a", "quality": 0.5, "cost": 1.0},
        {"system": "b", "quality": 0.5, "cost": 1.0},  # a tie dominates nothing
        {"system": "c", "quality": None, "cost": 0.1},  # unmeasured: cannot be placed, so it is not on the front
        {"system": "d", "quality": 0.4, "cost": 2.0},
    ]
    assert pareto_front(points, maximize=["quality"], minimize=["cost"]) == ["a", "b"]
    assert pareto_front([], maximize=["quality"], minimize=[]) == []
    with pytest.raises(ValueError, match="at least one"):
        pareto_front(points, maximize=[], minimize=[])


def _row(system: str, score: float, cost: float, error: dict | None = None) -> dict:
    return {"system": system, "error": error, "answer_judge": {"answer_score": score}, "cost": {"total_cost": cost}}


def test_cost_per_correct_divides_total_spend_by_correct_answers():
    rows = [_row("a", 5.0, 0.01), _row("a", 4.0, 0.01), _row("a", 3.9, 0.02), _row("b", 2.0, 0.5), _row("b", 1.0, 0.5)]
    result = cost_per_correct(rows)
    assert result["a"] == pytest.approx(0.04 / 2)  # every question's cost is paid, only two answers count
    assert result["b"] is None  # nothing correct: no finite price per correct answer
    assert cost_per_correct(rows, threshold=3.0)["a"] == pytest.approx(0.04 / 3)


def test_cost_per_correct_ignores_failed_questions():
    rows = [_row("a", 5.0, 0.01), _row("a", 0.0, 0.0, error={"type": "X", "message": "boom"})]
    assert cost_per_correct(rows) == {"a": pytest.approx(0.01)}


def test_latency_percentiles():
    assert latency_percentiles([10.0, 20.0, 30.0, 40.0, 50.0]) == {"p50": 30.0, "p95": pytest.approx(48.0)}
    assert latency_percentiles([]) == {"p50": None, "p95": None}


# -- end to end ---------------------------------------------------------------------------------


def _run(tmp_path: Path, **evaluation) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"name": "stats", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [
            {"type": "bm25", "name": "bm25"},
            {"type": "vector", "name": "vector", "retrieval": {"vector_store": "numpy"}},
            {"type": "no_retrieval", "name": "floor"},
        ],
        "evaluation": {"max_workers": 1, "max_questions": 60, **evaluation},
    }
    path = tmp_path / "stats.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(path, force_mock=True)


def test_run_writes_intervals_significance_and_pareto_outputs(tmp_path):
    out = _run(tmp_path)

    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    for metric in ("answer_score", "faithfulness", "retrieval_recall@5", "retrieval_mrr@10", "retrieval_ndcg@10"):
        measured = summary[metric].notna()
        assert (summary.loc[measured, f"{metric}_ci_lo"] <= summary.loc[measured, metric]).all()
        assert (summary.loc[measured, metric] <= summary.loc[measured, f"{metric}_ci_hi"]).all()
    assert summary.loc["floor", ["retrieval_recall@5_ci_lo", "retrieval_recall@5_ci_hi"]].isna().all()  # nothing retrieved, nothing to bracket
    assert summary["pareto_optimal"].dtype == bool and summary["pareto_optimal"].any()
    assert "cost_per_correct" in summary.columns

    stats = json.loads((out / "stats.json").read_text())
    assert stats["n_boot"] == 2000 and stats["seed"] == 0 and stats["baseline"] in summary.index
    assert stats["systems"]["bm25"]["metrics"]["answer_score"]["lo"] == pytest.approx(summary.loc["bm25", "answer_score_ci_lo"])
    pareto = json.loads((out / "pareto.json").read_text())
    assert set(pareto["front"]) == set(summary.index[summary["pareto_optimal"]])
    assert pareto["maximize"] == ["answer_score"] and set(pareto["minimize"]) == {"avg_cost_per_question", "avg_latency_ms"}

    significance = pd.read_csv(out / "significance.csv")
    assert set(significance["baseline"]) == {stats["baseline"]} and stats["baseline"] not in set(significance["system"])
    assert {"answer_score", "retrieval_recall@5"} == set(significance["metric"])
    assert (significance["ci_lo"] <= significance["mean_diff"]).all() and (significance["mean_diff"] <= significance["ci_hi"]).all()
    assert (significance["p_holm"] >= significance["p_value"] - 1e-12).all() and (significance["p_holm"] <= 1).all()
    assert (significance["significant"] == (significance["p_holm"] < 0.05)).all()
    assert not ((significance["metric"] == "retrieval_recall@5") & (significance["system"] == "floor")).any(), "no retrieval metric to compare"

    leaderboard = (out / "leaderboard.md").read_text()
    bm25_row = next(line for line in leaderboard.splitlines() if line.startswith("| bm25 |"))
    assert "[" in bm25_row, "headline metrics show mean [lo, hi]"
    assert "Significance" in leaderboard and "bootstrap" in leaderboard


def test_statistics_are_reproducible_and_follow_the_config(tmp_path):
    first = pd.read_csv(_run(tmp_path / "a", stats={"seed": 3, "baseline": "vector"}) / "significance.csv")
    again = pd.read_csv(_run(tmp_path / "b", stats={"seed": 3, "baseline": "vector"}) / "significance.csv")
    other = pd.read_csv(_run(tmp_path / "c", stats={"seed": 4, "baseline": "vector"}) / "significance.csv")

    assert set(first["baseline"]) == {"vector"}
    pd.testing.assert_frame_equal(first, again)
    assert not first["ci_lo"].equals(other["ci_lo"])


def test_unknown_baseline_is_a_config_error(tmp_path):
    with pytest.raises(ValueError, match=r"stats\.baseline.*vectr.*Did you mean 'vector'"):
        _run(tmp_path, stats={"baseline": "vectr"})


def test_stats_section_is_validated():
    from ragbench.config.schema import EvaluationConfig

    assert EvaluationConfig().stats.n_boot == 2000 and EvaluationConfig().stats.baseline is None
    with pytest.raises(ValueError, match="n_boot"):
        EvaluationConfig(stats={"n_boot": 10})
    with pytest.raises(ValueError, match="sead"):
        EvaluationConfig(stats={"sead": 1})
