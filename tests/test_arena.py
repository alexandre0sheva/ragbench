from __future__ import annotations

import math

import pytest

from ragbench.evaluation.arena import Comparison, arena_table, bradley_terry, position_bias, win_rates


def _comparisons(strength: dict[str, float], questions: int = 30, seed: int = 0) -> list[Comparison]:
    """Every pair on every question, both orders, decided by a Bradley-Terry draw with the given strengths."""
    import random

    rng = random.Random(seed)
    names = list(strength)
    out = []
    for q in range(questions):
        for i, x in enumerate(names):
            for y in names[i + 1 :]:
                p = strength[x] / (strength[x] + strength[y])
                shown_first = "first" if rng.random() < p else "second"  # x shown first
                shown_second = "second" if rng.random() < p else "first"  # y shown first
                out.append(Comparison(f"q{q}", x, y, (shown_first, shown_second)))
    return out


def test_a_comparison_scores_both_orders_and_a_flip_counts_as_a_tie():
    assert Comparison("q", "a", "b", ("first", "second")).score == 1.0  # a won as first and as second
    assert Comparison("q", "a", "b", ("second", "first")).score == 0.0
    assert Comparison("q", "a", "b", ("tie", "second")).score == 0.75
    flip = Comparison("q", "a", "b", ("first", "first"))  # whoever was shown first won: the order decided, not the answers
    assert flip.score == 0.5 and flip.flipped
    assert Comparison("q", "a", "b", ("first", None)).score is None  # one order failed: the pair does not count


def test_bradley_terry_recovers_a_known_order():
    strength = {"weak": 1.0, "fair": 2.0, "good": 4.0, "best": 8.0}
    fitted = bradley_terry(_comparisons(strength, questions=60), list(strength))
    assert sorted(fitted, key=fitted.get) == ["weak", "fair", "good", "best"]
    # Strength ratios come back roughly: log2(best/weak) = 3.
    assert math.log2(fitted["best"] / fitted["weak"]) == pytest.approx(3.0, abs=0.6)


def test_bradley_terry_stays_finite_for_a_system_that_never_wins():
    comparisons = [Comparison(f"q{i}", "a", "b", ("first", "second")) for i in range(10)]
    fitted = bradley_terry(comparisons, ["a", "b"])
    assert all(math.isfinite(v) and v > 0 for v in fitted.values()) and fitted["a"] > fitted["b"]


def test_the_arena_table_has_elo_scale_scores_with_intervals_and_records():
    strength = {"weak": 1.0, "best": 6.0, "mid": 2.5}
    table = {row["system"]: row for row in arena_table(_comparisons(strength, questions=40), list(strength), n_boot=300, seed=1)}
    assert [row for row in table] == ["weak", "best", "mid"]  # the systems' own order
    assert table["best"]["arena_score"] > table["mid"]["arena_score"] > table["weak"]["arena_score"]
    assert sum(row["arena_score"] for row in table.values()) / 3 == pytest.approx(1000, abs=1e-6)  # centred on 1000
    for row in table.values():
        assert row["arena_score_ci_lo"] < row["arena_score"] < row["arena_score_ci_hi"]
        assert row["arena_wins"] + row["arena_losses"] + row["arena_ties"] == row["arena_comparisons"] == 80
    assert table["weak"]["arena_score_ci_hi"] < table["best"]["arena_score_ci_lo"]


def test_the_interval_comes_from_resampling_questions_and_is_seeded():
    comparisons = _comparisons({"a": 1.0, "b": 1.5}, questions=20)
    once = arena_table(comparisons, ["a", "b"], n_boot=200, seed=7)
    assert once == arena_table(comparisons, ["a", "b"], n_boot=200, seed=7)
    narrow = arena_table(_comparisons({"a": 1.0, "b": 1.5}, questions=200), ["a", "b"], n_boot=200, seed=7)
    width = lambda rows: rows[0]["arena_score_ci_hi"] - rows[0]["arena_score_ci_lo"]  # noqa: E731
    assert width(narrow) < width(once)


def test_a_system_with_no_usable_comparison_has_no_score():
    rows = {row["system"]: row for row in arena_table([Comparison("q", "a", "b", ("first", "second"))], ["a", "b", "c"], n_boot=50)}
    assert rows["c"]["arena_score"] is None and rows["c"]["arena_comparisons"] == 0 and rows["a"]["arena_score"] is not None


def test_win_rates_are_points_won_against_each_opponent():
    comparisons = [Comparison("q1", "a", "b", ("first", "second")), Comparison("q2", "b", "a", ("tie", "tie"))]
    rates = win_rates(comparisons, ["a", "b", "c"])
    assert rates["a"]["b"] == pytest.approx(0.75) and rates["b"]["a"] == pytest.approx(0.25)
    assert rates["a"]["c"] is None and rates["a"]["a"] is None


def test_position_bias_is_measured_on_the_raw_verdicts():
    biased = [Comparison(f"q{i}", "a", "b", ("first", "first")) for i in range(6)] + [Comparison("q9", "a", "b", ("first", "second"))]
    bias = position_bias(biased)
    assert bias["flip_rate"] == pytest.approx(6 / 7) and bias["first_position_rate"] == pytest.approx(13 / 14)
    fair = position_bias([Comparison("q", "a", "b", ("first", "second")), Comparison("r", "a", "b", ("second", "first"))])
    assert fair == {"pairs": 2, "flip_rate": 0.0, "first_position_rate": 0.5}
