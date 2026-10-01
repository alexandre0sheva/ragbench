from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from paid_fakes import EMBEDDING, GENERATOR, JUDGE, PRICING
from paid_fakes import install as install_paid_fakes
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.cli import run as cli
from ragbench.config.loader import load_config
from ragbench.config.schema import EvaluationConfig
from ragbench.evaluation.budget import BudgetExceededError, BudgetGuard
from ragbench.evaluation.estimate import (
    STALE_PRICES_AFTER_DAYS,
    Estimate,
    SystemEstimate,
    confirmation_decision,
    estimate_run,
)
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models import cost as pricing
from ragbench.models.embeddings import EMBEDDING_CACHE

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
# `ragbench estimate` must land within this factor of what a run is charged (here: a mock run priced as if it were live).
# The measurement is exact for prompts and call counts; what it cannot know is how long a real model's answers are.
TOLERANCE = 0.35


def _config(tmp_path: Path, systems: list[dict[str, Any]] | None = None, **evaluation: Any) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    models = {"generator": GENERATOR, "embedding": EMBEDDING}
    systems = systems or [
        {"type": "bm25", "name": "bm25", "models": models},
        {"type": "vector", "name": "vector", "models": models, "retrieval": {"vector_store": "numpy"}},
        {"type": "hyde", "name": "hyde", "models": models, "retrieval": {"vector_store": "numpy"}},
        {"type": "contextual", "name": "contextual", "models": models, "retrieval": {"vector_store": "numpy"}},
        {"type": "agent_search", "name": "agent", "models": models, "retrieval": {"vector_store": "numpy"}, "tools": ["calculator"]},
    ]
    config = {
        "run": {"name": "est", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": systems,
        "evaluation": {"max_workers": 1, "max_questions": 12, "judge_model": JUDGE, "latency_probe_questions": 0, **evaluation},
        "pricing": PRICING,
        "cache": {"enabled": False},
    }
    path = tmp_path / "est.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _charged(out: Path) -> dict[str, float]:
    costs = pd.read_csv(out / "cost_breakdown.csv")
    return costs.groupby("system")["total_cost"].sum().to_dict()


# -- accuracy -------------------------------------------------------------------------------------


def test_the_estimate_is_within_tolerance_of_a_mock_run_priced_as_live(tmp_path, monkeypatch):
    install_paid_fakes(monkeypatch)
    path = _config(tmp_path)
    estimate = estimate_run(load_config(path))
    actual = _charged(run_benchmark(path, max_workers=1))

    assert {s.system for s in estimate.systems} == set(actual) and all(s.error is None for s in estimate.systems)
    for system in estimate.systems:
        assert actual[system.system] > 0
        assert system.total_usd == pytest.approx(actual[system.system], rel=TOLERANCE), f"{system.system}: {system.total_usd} vs {actual[system.system]}"
    assert estimate.total_usd == pytest.approx(sum(actual.values()), rel=TOLERANCE)
    # Judging and LLM-heavy ingestion are separate lines, and both are real money here.
    by_name = {s.system: s for s in estimate.systems}
    assert by_name["contextual"].ingestion_usd > by_name["vector"].ingestion_usd > by_name["bm25"].ingestion_usd == 0
    assert all(s.judge_usd_per_question > 0 for s in estimate.systems)
    assert by_name["agent"].question_seconds > by_name["bm25"].question_seconds  # the loop makes more model calls


def test_without_a_judge_the_estimate_has_no_judge_line(tmp_path, monkeypatch):
    install_paid_fakes(monkeypatch)
    path = _config(tmp_path, judge_enabled=False)
    estimate = estimate_run(load_config(path))
    actual = _charged(run_benchmark(path, max_workers=1))
    assert all(s.judge_usd_per_question == 0 for s in estimate.systems)
    assert estimate.total_usd == pytest.approx(sum(actual.values()), rel=TOLERANCE)


def test_estimating_never_calls_a_live_model_and_leaves_no_state_behind(tmp_path, monkeypatch):
    install_paid_fakes(monkeypatch)
    import ragbench.models.providers.openai as openai_provider

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("estimate must not build a live client")

    monkeypatch.setattr(openai_provider, "OpenAILLM", forbidden)
    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingModel", forbidden)
    EMBEDDING_CACHE.enabled = True
    estimate = estimate_run(load_config(_config(tmp_path)))

    assert estimate.total_usd > 0
    assert EMBEDDING_CACHE.enabled is True, "the in-process embedding cache is switched off while measuring, then restored"
    assert GENERATOR not in pricing._pricing_overrides, "the run's price overrides do not leak out of the estimate"


def test_the_estimate_scales_with_the_question_count_not_the_sample(tmp_path):
    small = estimate_run(load_config(_config(tmp_path / "a", max_questions=6)))
    large = estimate_run(load_config(_config(tmp_path / "b", max_questions=24)))
    assert small.n_questions == 6 and large.n_questions == 24
    for little, big in zip(small.systems, large.systems, strict=True):
        assert big.ingestion_usd == pytest.approx(little.ingestion_usd), "indexing the corpus costs the same however many questions follow"
        assert big.total_usd - big.ingestion_usd == pytest.approx((little.total_usd - little.ingestion_usd) * 4, rel=0.5)  # a different sample, so not exact


def test_wall_time_shrinks_with_more_workers(tmp_path):
    slow = estimate_run(load_config(_config(tmp_path / "a", max_workers=1, system_workers=1)))
    fast = estimate_run(load_config(_config(tmp_path / "b", max_workers=4, system_workers=2)))
    assert slow.wall_seconds > 0 and fast.wall_seconds < slow.wall_seconds


# -- warnings -------------------------------------------------------------------------------------


def test_stale_price_tables_and_unpriced_models_are_called_out(tmp_path):
    config = load_config(_config(tmp_path))
    fresh = estimate_run(config, today=date.fromisoformat(pricing.PRICING_AS_OF))
    assert fresh.pricing_as_of == pricing.PRICING_AS_OF and not any("older" in w for w in fresh.warnings)

    stale = estimate_run(config, today=date.fromisoformat(pricing.PRICING_AS_OF) + timedelta(days=STALE_PRICES_AFTER_DAYS + 1))
    assert any("older than 90 days" in w and pricing.PRICING_AS_OF in w for w in stale.warnings)

    unpriced = load_config(_config(tmp_path / "u", [{"type": "bm25", "name": "bm25", "models": {"generator": "mystery-model-9"}}], judge_enabled=False))
    result = estimate_run(unpriced)
    assert result.unpriced_models == ["mystery-model-9"] and any("mystery-model-9" in w and "no price" in w for w in result.warnings)
    assert result.total_usd == 0


def test_agentic_systems_are_flagged_as_approximate(tmp_path):
    estimate = estimate_run(load_config(_config(tmp_path)))
    assert any("agentic" in w and "agent" in w for w in estimate.warnings)
    assert {s.system for s in estimate.systems if s.agentic} == {"agent"}


def test_a_system_that_cannot_be_measured_is_reported_and_left_out_of_the_total(tmp_path):
    config = load_config(_config(tmp_path, [{"type": "bm25", "name": "ok"}, {"type": "grep_agent", "name": "broken"}]))
    config.systems[1] = config.systems[1].model_copy(update={"type": "no_such_type"})
    estimate = estimate_run(config)
    broken = next(s for s in estimate.systems if s.system == "broken")
    assert broken.error and "no_such_type" in broken.error
    assert estimate.total_usd == sum(s.total_usd for s in estimate.systems if s.error is None)
    assert any("broken" in w and "could not be estimated" in w for w in estimate.warnings)


# -- confirmation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "threshold", "yes", "interactive", "ci", "expected"),
    [
        (0.50, 1.00, False, False, False, "proceed"),  # under the threshold: no questions asked
        (1.00, 1.00, False, False, False, "proceed"),  # exactly at it is still fine
        (5.00, 1.00, True, False, False, "proceed"),  # --yes
        (5.00, 1.00, True, True, True, "proceed"),  # --yes beats CI
        (5.00, 1.00, False, True, False, "ask"),  # a person at a terminal
        (5.00, 1.00, False, False, False, "refuse"),  # nobody to ask
        (5.00, 1.00, False, True, True, "refuse"),  # CI=1: never prompt
        (0.00, 0.00, False, False, False, "proceed"),
    ],
)
def test_confirmation_decision(total, threshold, yes, interactive, ci, expected):
    assert confirmation_decision(total, threshold, yes=yes, interactive=interactive, ci=ci) == expected


# -- the budget guard -------------------------------------------------------------------------------


def test_budget_guard_counts_charges_across_threads_and_trips_at_the_cap():
    import threading

    guard = BudgetGuard(1.0)
    assert not guard.exhausted
    threads = [threading.Thread(target=lambda: [guard.charge(0.001) for _ in range(100)]) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert guard.spent == pytest.approx(0.8) and not guard.exhausted
    guard.charge(0.2)
    assert guard.exhausted
    assert not BudgetGuard(None).exhausted and BudgetGuard(None).charge(1e9) is None


def test_the_budget_options_are_validated():
    assert EvaluationConfig().max_cost_usd is None and EvaluationConfig().cost_confirm_threshold_usd == 1.0
    assert EvaluationConfig(max_cost_usd=5).max_cost_usd == 5
    with pytest.raises(ValueError, match="max_cost_usd"):
        EvaluationConfig(max_cost_usd=0)
    with pytest.raises(ValueError, match="cost_confirm_threshold_usd"):
        EvaluationConfig(cost_confirm_threshold_usd=-1)


def _uncapped_costs(tmp_path: Path, monkeypatch: Any) -> dict[str, float]:
    install_paid_fakes(monkeypatch)
    return _charged(run_benchmark(_config(tmp_path / "uncapped", _three()), max_workers=1))


def _three() -> list[dict[str, Any]]:
    models = {"generator": GENERATOR, "embedding": EMBEDDING}
    return [
        {"type": "bm25", "name": "first", "models": models},
        {"type": "vector", "name": "second", "models": models, "retrieval": {"vector_store": "numpy"}},
        {"type": "hyde", "name": "third", "models": models, "retrieval": {"vector_store": "numpy"}},
    ]


def test_hitting_the_cap_stops_the_run_with_partial_results_for_the_completed_systems(tmp_path, monkeypatch):
    costs = _uncapped_costs(tmp_path, monkeypatch)
    cap = costs["first"] + 0.4 * costs["second"]  # the first system fits, the second runs out partway
    path = _config(tmp_path / "capped", _three(), max_cost_usd=cap)

    with pytest.raises(BudgetExceededError) as caught:
        run_benchmark(path, max_workers=1)
    error = caught.value
    out = error.output_dir

    assert error.completed == ["first"] and set(error.incomplete) == {"second", "third"}
    assert 0 < error.incomplete["second"]["answered"] < error.incomplete["second"]["total"] == 12
    assert error.incomplete["third"]["answered"] == 0
    assert cap <= error.spent_usd < cap + 0.5 * costs["second"], "stops promptly: only the question in flight overshoots"
    message = str(error)
    assert "max_cost_usd" in message and "budget" in message
    assert "first" in message and "second" in message and "per_question_partial.jsonl" in message

    # Outputs exist for what completed, and only for that.
    assert pd.read_csv(out / "metrics_summary.csv")["system"].tolist() == ["first"]
    assert "| second |" not in (out / "leaderboard.md").read_text() and "| first |" in (out / "leaderboard.md").read_text()
    run_summary = json.loads((out / "run_summary.json").read_text())
    assert run_summary["budget"]["stopped"] is True and run_summary["budget"]["max_cost_usd"] == cap
    assert run_summary["budget"]["completed_systems"] == ["first"] and run_summary["budget"]["spent_usd"] == pytest.approx(error.spent_usd)
    assert json.loads((out / "recommendation.json").read_text())["winner"] == "first"
    partial = [json.loads(line) for line in (out / "per_question_partial.jsonl").read_text().splitlines()]
    assert {row["system"] for row in partial} == {"second"} and len(partial) == error.incomplete["second"]["answered"]


def test_a_cap_above_the_cost_changes_nothing(tmp_path, monkeypatch):
    costs = _uncapped_costs(tmp_path, monkeypatch)
    out = run_benchmark(_config(tmp_path / "roomy", _three(), max_cost_usd=sum(costs.values()) * 3), max_workers=1)
    budget = json.loads((out / "run_summary.json").read_text())["budget"]
    assert budget["stopped"] is False and budget["incomplete_systems"] == {}
    assert budget["spent_usd"] == pytest.approx(sum(costs.values()), rel=0.02)
    assert not (out / "per_question_partial.jsonl").exists()


def test_a_cap_smaller_than_ingestion_stops_before_any_question(tmp_path, monkeypatch):
    install_paid_fakes(monkeypatch)
    with pytest.raises(BudgetExceededError) as caught:
        run_benchmark(_config(tmp_path, _three(), max_cost_usd=1e-9), max_workers=1)
    error = caught.value
    assert error.completed == [] and set(error.incomplete) == {"first", "second", "third"}
    assert error.incomplete["first"]["answered"] == 1 and error.incomplete["second"]["answered"] == 0, "the first question ends the budget; nothing else starts"
    assert json.loads((error.output_dir / "run_summary.json").read_text())["budget"]["stopped"] is True
    assert not (error.output_dir / "metrics_summary.csv").exists(), "no system finished, so there is nothing to compare"


def test_mock_runs_cost_nothing_so_a_cap_never_stops_them(tmp_path):
    path = _config(tmp_path, [{"type": "bm25", "name": "bm25"}], max_cost_usd=0.01)
    out = run_benchmark(path, force_mock=True)
    assert json.loads((out / "run_summary.json").read_text())["budget"] == {
        "max_cost_usd": 0.01,
        "spent_usd": 0.0,
        "stopped": False,
        "completed_systems": ["bm25"],
        "incomplete_systems": {},
    }


# -- the CLI --------------------------------------------------------------------------------------


def test_estimate_command_prints_a_table_for_the_shipped_config():
    result = CliRunner().invoke(app, ["estimate", "--config", str(ROOT / "configs" / "all.yaml")], env={"COLUMNS": "160"})
    assert result.exit_code == 0, result.output
    for text in ("Estimate", "bm25_default", "contextual_default", "agent_search_tools", "Total", "Ingestion", "price", pricing.PRICING_AS_OF):
        assert text in result.output
    assert "163 questions" in result.output.replace("\n", " ")


def test_estimate_command_accepts_a_preset_and_a_system_subset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["estimate", "--preset", "standard", "--docs", str(DEMO / "docs"), "--questions", str(DEMO / "questions.jsonl"), "--systems", "bm25,hyde"],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0, result.output
    assert "bm25" in result.output and "hyde" in result.output and "contextual" not in result.output


class _FakeRun:
    calls = 0


@pytest.fixture
def live_cli(tmp_path, monkeypatch):
    """The `compare` command in live mode, with the run itself and the estimate replaced by stubs."""
    path = _config(tmp_path, [{"type": "bm25", "name": "bm25"}])
    out = tmp_path / "results" / "stub"
    out.mkdir(parents=True)
    _FakeRun.calls = 0

    def stub_run(*_args: Any, **_kwargs: Any) -> Path:
        _FakeRun.calls += 1
        return out

    monkeypatch.setattr(cli, "run_benchmark", stub_run)
    monkeypatch.setattr(cli, "resolve_run_mode", lambda config, force_mock: "live")
    monkeypatch.setattr(cli, "print_run_summary", lambda output_dir: None)
    monkeypatch.delenv("CI", raising=False)

    def with_estimate(total: float) -> Path:
        monkeypatch.setattr(
            cli, "estimate_run", lambda config, **_kw: Estimate([SystemEstimate("bm25", "bm25", False, 0.0, total / 12, 0.0, 0.0, 1.0, 12)], 12, 60, 1000, total, 30.0, "2026-10-01", [], [])
        )
        return path

    return with_estimate


def test_a_live_run_over_the_threshold_is_refused_without_yes_when_nobody_can_answer(live_cli):
    path = live_cli(5.0)
    result = CliRunner().invoke(app, ["compare", "--config", str(path)])
    assert result.exit_code == 2 and "--yes" in result.output and "$5.00" in result.output and _FakeRun.calls == 0


def test_yes_ci_and_the_threshold_decide_whether_a_live_run_proceeds(live_cli, monkeypatch):
    path = live_cli(5.0)
    assert CliRunner().invoke(app, ["compare", "--config", str(path), "--yes"]).exit_code == 0 and _FakeRun.calls == 1
    monkeypatch.setenv("CI", "1")
    assert CliRunner().invoke(app, ["compare", "--config", str(path)]).exit_code == 2 and _FakeRun.calls == 1
    assert CliRunner().invoke(app, ["compare", "--config", str(path), "--yes"]).exit_code == 0 and _FakeRun.calls == 2
    cheap = live_cli(0.40)
    assert CliRunner().invoke(app, ["compare", "--config", str(cheap)]).exit_code == 0 and _FakeRun.calls == 3, "under the threshold nothing is asked"


def test_a_person_at_a_terminal_is_asked(live_cli, monkeypatch):
    path = live_cli(5.0)
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    declined = CliRunner().invoke(app, ["compare", "--config", str(path)], input="n\n")
    assert declined.exit_code == 1 and "Cancelled" in declined.output and _FakeRun.calls == 0
    accepted = CliRunner().invoke(app, ["compare", "--config", str(path)], input="y\n")
    assert accepted.exit_code == 0 and _FakeRun.calls == 1


def test_mock_runs_are_never_estimated_or_confirmed(live_cli, monkeypatch):
    path = live_cli(5.0)
    monkeypatch.setattr(cli, "resolve_run_mode", lambda config, force_mock: "mock")
    monkeypatch.setattr(cli, "estimate_run", lambda *a, **k: pytest.fail("a mock run has nothing to confirm"))
    assert CliRunner().invoke(app, ["compare", "--config", str(path), "--mock"]).exit_code == 0 and _FakeRun.calls == 1


def test_an_estimate_that_fails_does_not_block_a_run(live_cli, monkeypatch):
    path = live_cli(5.0)

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "estimate_run", broken)
    result = CliRunner().invoke(app, ["compare", "--config", str(path)])
    assert result.exit_code == 0 and "Could not estimate" in result.output and _FakeRun.calls == 1


def test_compare_reports_a_budget_stop_and_exits_nonzero(tmp_path, monkeypatch):
    costs = _uncapped_costs(tmp_path, monkeypatch)
    path = _config(tmp_path / "capped", _three(), max_cost_usd=costs["first"] + 0.4 * costs["second"], cost_confirm_threshold_usd=1e6)
    result = CliRunner().invoke(app, ["compare", "--config", str(path), "--max-workers", "1"])
    assert result.exit_code == 1 and "budget" in result.output.lower() and "first" in result.output and "second" in result.output
