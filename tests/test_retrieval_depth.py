from __future__ import annotations

import json

import pandas as pd
import pytest
from fake_systems import RankedFakeSystem, question, register_fakes, write_experiment

from ragbench.config.schema import EvaluationConfig, SystemConfig
from ragbench.evaluation.evaluator import run_benchmark


def _system(**retrieval) -> RankedFakeSystem:
    return RankedFakeSystem(SystemConfig(type="fake_ranked", name="fake", retrieval=retrieval), force_mock=True)


def test_generator_sees_only_context_k_chunks_but_full_ranking_is_returned():
    RankedFakeSystem.generated_with = []

    result = _system().answer_question("What is topic 1?", top_k=10, context_k=3)

    assert len(result.retrieval_result.chunks) == 10
    assert RankedFakeSystem.generated_with == [3]
    assert result.metadata["context_chunk_ids"] == [f"doc_{i:03d}::chunk::0" for i in (1, 2, 3)]


def test_default_call_keeps_legacy_behaviour_and_uses_configured_top_k_as_context():
    RankedFakeSystem.generated_with = []

    result = _system(top_k=4).answer_question("What is topic 1?")

    assert len(result.retrieval_result.chunks) == 4
    assert RankedFakeSystem.generated_with == [4]


def test_evaluation_config_resolves_depth_and_primary_k():
    cfg = EvaluationConfig(k_values=[1, 3, 5, 10])
    assert (cfg.resolved_retrieval_depth, cfg.resolved_primary_k, cfg.context_k) == (10, 5, 5)
    assert EvaluationConfig(k_values=[1, 3]).resolved_primary_k == 3
    assert EvaluationConfig(k_values=[2, 4, 8]).resolved_primary_k == 4  # median when 5 is absent
    with pytest.raises(ValueError):
        EvaluationConfig(k_values=[1, 5], retrieval_depth=3)  # depth below max(k) would make metrics meaningless
    with pytest.raises(ValueError):
        EvaluationConfig(k_values=[1, 5], primary_k=3)
    with pytest.raises(ValueError):
        EvaluationConfig(k_values=[])


def test_metrics_use_full_ranking_while_generator_uses_context_k(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    # The relevant document sits at rank 7: invisible to an @5 cut-off, found by @10.
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 7?", ["doc_007"])],
        [{"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 5}}],
        {"k_values": [1, 5, 10]},
    )

    out = run_benchmark(config, force_mock=True)

    retrieval = pd.read_csv(out / "retrieval_metrics.csv").iloc[0]
    assert retrieval["recall@5"] == 0.0 and retrieval["recall@10"] == 1.0
    assert retrieval["mrr@10"] == pytest.approx(1 / 7)
    assert RankedFakeSystem.generated_with == [5]
    row = json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])
    assert len(row["retrieved_contexts"]) == 10
    assert [c["in_context"] for c in row["retrieved_contexts"]] == [True] * 5 + [False] * 5
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["retrieval_depth"] == 10 and summary["primary_k"] == 5 and summary["k_values"] == [1, 5, 10]


def test_k_values_without_5_or_10_get_real_columns_not_zeros(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 2?", ["doc_002"])],
        [{"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 3}}],
        {"k_values": [1, 3]},
    )

    out = run_benchmark(config, force_mock=True)

    leaderboard = (out / "leaderboard.md").read_text()
    assert "Recall@3" in leaderboard and "MRR@3" in leaderboard
    assert "Recall@5" not in leaderboard and "nDCG@10" not in leaderboard
    html = (out / "report.html").read_text()
    assert "Recall@3" in html and "nDCG@10" not in html
    assert json.loads((out / "run_summary.json").read_text())["primary_k"] == 3


def test_failure_classification_uses_primary_k_not_a_hardcoded_hit_at_5(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 2?", ["doc_002"])],
        [{"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 3}}],
        {"k_values": [1, 3]},
    )

    out = run_benchmark(config, force_mock=True)

    row = json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])
    assert row["failure_type"] != "retrieval_miss"  # doc_002 is in the top 3; there is no `hit@5` key here


def test_every_real_system_reports_monotonic_recall_and_some_differ_at_10(tmp_path):
    import shutil
    from pathlib import Path

    demo = Path(__file__).resolve().parents[1] / "data" / "demo"
    shutil.copytree(demo, tmp_path / "demo")
    raw = (Path(__file__).resolve().parents[1] / "configs" / "all.yaml").read_text()
    cfg_path = tmp_path / "all.yaml"
    cfg_path.write_text(
        raw.replace("data/demo", str(tmp_path / "demo")).replace("output_dir: results", f"output_dir: {tmp_path / 'results'}"),
        encoding="utf-8",
    )

    out = run_benchmark(cfg_path, force_mock=True, max_workers=4)

    summary = pd.read_csv(out / "metrics_summary.csv")
    assert (summary["retrieval_recall@10"] >= summary["retrieval_recall@5"] - 1e-12).all()
    assert (summary["retrieval_recall@10"] > summary["retrieval_recall@5"]).any(), "@10 must be a real measurement, not a copy of @5"
