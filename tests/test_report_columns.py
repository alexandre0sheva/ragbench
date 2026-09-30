from __future__ import annotations

import math

from ragbench.reporting.columns import MISSING, format_value, leaderboard_columns, metric_ks

KEYS_DEFAULT = ["retrieval_recall@1", "retrieval_recall@3", "retrieval_recall@5", "retrieval_recall@10", "answer_score"]


def test_default_k_values_keep_the_familiar_headers():
    headers = [column.header for column in leaderboard_columns(KEYS_DEFAULT)]
    assert headers[:3] == ["Recall@5", "MRR@10", "nDCG@10"]
    assert headers[3:] == ["Answer", "Faithful", "$/Q", "Latency"]


def test_columns_follow_the_measured_ks_and_explicit_primary_k():
    keys = ["retrieval_recall@1", "retrieval_recall@3"]
    assert [c.header for c in leaderboard_columns(keys)][:3] == ["Recall@3", "MRR@3", "nDCG@3"]
    keys = ["retrieval_recall@2", "retrieval_recall@4", "retrieval_recall@8"]
    assert leaderboard_columns(keys)[0].header == "Recall@4"  # median when 5 is not measured
    assert leaderboard_columns(KEYS_DEFAULT, primary_k=3)[0].header == "Recall@3"
    assert leaderboard_columns(KEYS_DEFAULT, primary_k=7)[0].header == "Recall@5"  # unmeasured primary_k is ignored
    assert metric_ks(["retrieval_recall@10", "retrieval_recall@1", "retrieval_mrr@3"]) == [1, 10]


def test_no_retrieval_columns_when_nothing_was_measured():
    assert [c.header for c in leaderboard_columns(["answer_score"])][0] == "Answer"


def test_missing_values_render_as_a_dash_not_zero():
    column = leaderboard_columns(KEYS_DEFAULT)[0]
    assert format_value(column, None) == MISSING
    assert format_value(column, float("nan")) == MISSING
    assert format_value(column, math.inf) != MISSING
    assert format_value(column, 0.5) == "0.500"
    assert format_value(column, 0.0) == "0.000"  # a real zero is still a zero
