from __future__ import annotations

import json

import pandas as pd
import pytest
from fake_systems import question, register_fakes, write_experiment

from ragbench.evaluation.evaluator import BenchmarkRunError, run_benchmark

FAKE = {"type": "fake_ranked", "name": "healthy", "retrieval": {"top_k": 3}}
EXPLODING = {"type": "fake_exploding", "name": "flaky", "retrieval": {"top_k": 3}}


def _questions(n_bad: int, n_good: int):
    good = [question(f"g{i}", f"What is topic {i + 1}?", [f"doc_{i + 1:03d}"]) for i in range(n_good)]
    bad = [question(f"b{i}", f"Please explode on topic {i}", [f"doc_{i + 1:03d}"]) for i in range(n_bad)]
    return [*good[:1], *bad, *good[1:]]  # the failure is not the last question


@pytest.mark.parametrize("workers", [1, 4])
def test_one_failing_question_does_not_kill_the_run(tmp_path, monkeypatch, workers):
    register_fakes(monkeypatch)
    config = write_experiment(tmp_path, _questions(1, 9), [EXPLODING], {"max_workers": workers})

    out = run_benchmark(config, force_mock=True)

    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    failed = [row for row in rows if row["error"]]
    assert [row["question_id"] for row in failed] == ["b0"]
    assert failed[0]["failure_type"] == "run_error"
    assert failed[0]["error"] == {"type": "RuntimeError", "message": "boom from fake retriever"}
    # Error rows carry no scores and are excluded from every mean.
    answer_csv = pd.read_csv(out / "answer_metrics.csv")
    assert len(answer_csv) == 9 and "b0" not in set(answer_csv["question_id"])
    summary = pd.read_csv(out / "metrics_summary.csv").iloc[0]
    assert (summary["n_ok"], summary["n_error"]) == (9, 1)
    run_summary = json.loads((out / "run_summary.json").read_text())
    assert run_summary["num_errors"] == 1 and run_summary["errors_by_system"] == {"flaky": 1}
    assert run_summary["num_question_rows"] == 10
    assert "run_error" in (out / "failures.md").read_text()


def test_run_fails_loudly_but_keeps_results_when_error_rate_is_exceeded(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(tmp_path, _questions(3, 3), [FAKE, EXPLODING])  # flaky fails 50% > default 20%

    with pytest.raises(BenchmarkRunError) as excinfo:
        run_benchmark(config, force_mock=True)

    err = excinfo.value
    assert err.failed_systems == {"flaky": pytest.approx(0.5)}
    assert (err.output_dir / "per_question_results.jsonl").exists()
    assert (err.output_dir / "report.html").exists(), "partial results must still be written"
    healthy = pd.read_csv(err.output_dir / "metrics_summary.csv").set_index("system").loc["healthy"]
    assert healthy["n_error"] == 0


def test_error_rate_threshold_is_configurable(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(tmp_path, _questions(3, 3), [EXPLODING], {"max_error_rate": 0.6})

    out = run_benchmark(config, force_mock=True)

    assert json.loads((out / "run_summary.json").read_text())["num_errors"] == 3


def test_ingestion_failure_marks_the_system_failed_but_other_systems_finish(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    broken = {"type": "fake_broken_ingest", "name": "broken", "retrieval": {"top_k": 3}}
    config = write_experiment(tmp_path, _questions(0, 4), [broken, FAKE])

    with pytest.raises(BenchmarkRunError) as excinfo:
        run_benchmark(config, force_mock=True)

    out = excinfo.value.output_dir
    assert excinfo.value.failed_systems == {"broken": 1.0}
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    broken_rows = [row for row in rows if row["system"] == "broken"]
    assert len(broken_rows) == 4 and all(row["error"]["message"] == "ingestion exploded" for row in broken_rows)
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert summary.loc["healthy", "n_ok"] == 4 and summary.loc["broken", "n_error"] == 4


def test_system_without_successful_questions_shows_dashes_not_zeros(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    broken = {"type": "fake_broken_ingest", "name": "broken", "retrieval": {"top_k": 3}}
    config = write_experiment(tmp_path, _questions(0, 2), [broken, FAKE])

    with pytest.raises(BenchmarkRunError) as excinfo:
        run_benchmark(config, force_mock=True)

    out = excinfo.value.output_dir
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert summary.loc["broken"].isna()["answer_score"] and not summary.loc["healthy"].isna()["answer_score"]
    broken_line = next(line for line in (out / "leaderboard.md").read_text().splitlines() if line.startswith("| broken"))
    assert "—" in broken_line and "0.000" not in broken_line
    assert "broken" in (out / "report.html").read_text()  # the HTML report renders the failed system too


def test_cli_exits_nonzero_but_prints_partial_results_when_errors_exceed_threshold(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ragbench.cli import app

    register_fakes(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = write_experiment(tmp_path, _questions(3, 3), [EXPLODING])

    result = CliRunner().invoke(app, ["run", "--config", str(config), "--mock"])

    output = " ".join(result.output.split())  # Rich wraps long lines
    assert result.exit_code == 1
    assert "allowed 20%" in output and "Partial results were written" in output
