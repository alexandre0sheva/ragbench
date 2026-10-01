from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.loader import load_config
from ragbench.config.schema import SelectionConfig
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.selection import PROFILES, Constraints, Weights
from ragbench.selection.recommend import recommend, write_recommendation

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
N = 60  # questions in a synthetic run: enough for paired tests to separate a planted gap from noise


def _base() -> np.ndarray:
    return np.random.default_rng(5).uniform(1.0, 4.0, size=N)  # questions differ a lot in difficulty, systems differ little


def noisy(shift: float, seed: int, half: tuple[float, float] | None = None) -> np.ndarray:
    """Per-question answer scores: shared difficulty + a system effect + noise. `half` gives separate shifts for the two categories."""
    shifts = np.full(N, shift) if half is None else np.where(np.arange(N) < N // 2, half[0], half[1])
    return np.clip(_base() + shifts + np.random.default_rng(seed).normal(scale=0.3, size=N), 0, 5)


def make_run(tmp_path: Path, specs: dict[str, dict[str, Any]], *, mode: str = "live", extra_config: dict[str, Any] | None = None) -> Path:
    """Write the files of a finished run (what `recommend` reads) from per-system specs, without running anything."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    categories = ["cat_a" if i < N // 2 else "cat_b" for i in range(N)]
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for name, spec in specs.items():
        scores = spec["scores"]
        llm_calls = spec.get("llm_calls", 1)
        steps = [{"kind": "llm", "name": "generate", "prompt_tokens": 10, "completion_tokens": 5} for _ in range(llm_calls)]
        steps.append({"kind": "retrieve", "name": "search", "prompt_tokens": 0, "completion_tokens": 0})
        for i in range(N):
            rows.append(
                {
                    "system": name,
                    "system_type": spec.get("type", "bm25"),
                    "question_id": f"q{i}",
                    "category": categories[i],
                    "error": None,
                    "answer_judge": {"answer_score": float(scores[i]), "faithfulness": spec.get("faithfulness", 4.5)},
                    "answer_metrics": {},
                    "retrieval_metrics": {},
                    "cost": {"total_cost": spec["cost"]},
                    "steps": steps,
                    "latency_ms": spec.get("latency", 100.0),
                }
            )
        summary.append(
            {
                "system": name,
                "system_type": spec.get("type", "bm25"),
                "n_ok": N,
                "answer_score": float(np.mean(scores)),
                "faithfulness": spec.get("faithfulness", 4.5),
                "avg_cost_per_question": spec["cost"],
                "avg_latency_ms": spec.get("latency", 100.0),
                "latency_ms_p95": spec.get("p95", spec.get("latency", 100.0) * 1.5),
            }
        )
        cost_rows.append({"system": name, "stage": "ingestion", "total_cost": spec.get("ingestion", 0.0)})
    (run_dir / "per_question_results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    pd.DataFrame(summary).to_csv(run_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(run_dir / "cost_breakdown.csv", index=False)
    (run_dir / "run_summary.json").write_text(json.dumps({"mode": mode, "primary_k": 5, "k_values": [1, 3, 5, 10]}), encoding="utf-8")
    (run_dir / "stats.json").write_text(json.dumps({"n_boot": 1000, "seed": 0, "alpha": 0.05}), encoding="utf-8")
    systems = []
    for name, spec in specs.items():
        system: dict[str, Any] = {"type": spec.get("type", "bm25"), "name": name}
        if system["type"] in {"vector", "agent_search"}:
            system["retrieval"] = {"vector_store": "numpy"}
        if "models" in spec:
            system["models"] = spec["models"]
        if "tools" in spec:
            system["tools"] = spec["tools"]
        systems.append(system)
    config = {
        "run": {"name": "synthetic", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": systems,
        "evaluation": {"max_workers": 1, "max_questions": 4, "stats": {"baseline": next(iter(specs))}},
        **(extra_config or {}),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_dir


def ping(host: str) -> str:
    """Ping a host (a custom tool that declares network side effects; never called)."""
    return host


def tied_pair(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """`pricey` and `thrifty` are equally good (the same difficulty, independent noise); `weak` is clearly worse."""
    specs = {
        "pricey": {"scores": noisy(0.0, 1), "cost": 0.004, "latency": 300.0},
        "thrifty": {"scores": noisy(0.0, 2), "cost": 0.001, "latency": 900.0},
        "weak": {"scores": noisy(-1.0, 3), "cost": 0.0005, "latency": 200.0},
    }
    for name, changes in overrides.items():
        specs[name] = {**specs[name], **changes}
    return specs


# -- constraints --------------------------------------------------------------------------------


def test_constraints_filter_systems_and_name_each_violation(tmp_path):
    specs = {
        "cheap": {"scores": noisy(0, 1), "cost": 0.001, "p95": 900.0, "faithfulness": 4.7, "ingestion": 0.01},
        "costly": {"scores": noisy(0, 2), "cost": 0.004, "p95": 900.0, "faithfulness": 4.7},
        "slow": {"scores": noisy(0, 3), "cost": 0.001, "p95": 4000.0, "faithfulness": 4.7},
        "unfaithful": {"scores": noisy(0, 4), "cost": 0.001, "p95": 900.0, "faithfulness": 3.9},
        "indexing_heavy": {"scores": noisy(0, 5), "cost": 0.001, "p95": 900.0, "faithfulness": 4.7, "ingestion": 2.0},
    }
    constraints = Constraints(max_cost_per_question=0.002, max_latency_ms_p95=2500, min_faithfulness=4.5, max_ingestion_cost=1.0)
    result = recommend(make_run(tmp_path, specs), constraints=constraints)

    assert result.winner == "cheap" and [s.system for s in result.ranking] == ["cheap"]
    assert set(result.infeasible) == {"costly", "slow", "unfaithful", "indexing_heavy"}
    assert "cost per question" in result.infeasible["costly"][0] and "$0.00400" in result.infeasible["costly"][0]
    assert "p95 latency" in result.infeasible["slow"][0]
    assert "faithfulness" in result.infeasible["unfaithful"][0]
    assert "ingestion cost" in result.infeasible["indexing_heavy"][0]
    assert any("Excluded by your constraints" in line for line in result.rationale)


def test_answer_score_floor_and_unmeasured_metrics(tmp_path):
    run_dir = make_run(tmp_path, {"good": {"scores": noisy(0.5, 1), "cost": 0.001}, "poor": {"scores": noisy(-1.5, 2), "cost": 0.001}})
    result = recommend(run_dir, constraints=Constraints(min_answer_score=2.0))
    assert result.winner == "good" and "answer score" in result.infeasible["poor"][0]

    summary = pd.read_csv(run_dir / "metrics_summary.csv")
    summary.loc[summary["system"] == "poor", "latency_ms_p95"] = np.nan
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)
    unmeasured = recommend(run_dir, constraints=Constraints(max_latency_ms_p95=5000))
    assert "not measured" in unmeasured.infeasible["poor"][0]  # a limit cannot be shown to hold for a number nobody measured


def test_when_nothing_is_feasible_there_is_no_winner_and_the_closest_miss_is_named(tmp_path):
    specs = {
        "near": {"scores": noisy(0, 1), "cost": 0.0022, "faithfulness": 4.8},  # over the cost cap by 10%
        "far": {"scores": noisy(0, 2), "cost": 0.02, "faithfulness": 3.0},  # breaks two constraints, one of them badly
    }
    result = recommend(make_run(tmp_path, specs), constraints=Constraints(max_cost_per_question=0.002, min_faithfulness=4.5))

    assert result.winner is None and result.ranking == [] and result.tied_with_winner == []
    assert set(result.infeasible) == {"near", "far"} and len(result.infeasible["far"]) == 2
    assert result.closest_miss == "near"
    assert "No system meets" in result.rationale[0] and "near" in result.rationale[0] and "$0.00220" in result.rationale[0]


def test_local_model_and_network_constraints_read_the_runs_config(tmp_path):
    local_models = {"generator": "openai_compatible:box/llama3", "embedding": "local:BAAI/bge-small-en-v1.5"}
    specs = {
        "hosted": {"type": "vector", "scores": noisy(0.3, 1), "cost": 0.001},
        "on_prem": {"type": "vector", "scores": noisy(0.0, 2), "cost": 0.0, "models": local_models},
        "on_prem_with_tools": {"type": "agent_search", "scores": noisy(0.0, 3), "cost": 0.0, "models": local_models, "tools": [{"name": "ping", "path": "test_recommend:ping", "side_effects": "network"}]},
    }
    extra = {"providers": {"box": {"base_url": "http://localhost:11434/v1"}}, "tools": {"allow": ["ping"], "allow_network": True}}
    run_dir = make_run(tmp_path, specs, extra_config=extra)

    local = recommend(run_dir, constraints=Constraints(require_local_models=True))
    assert local.winner in {"on_prem", "on_prem_with_tools"} and set(local.infeasible) == {"hosted"}
    assert "do not run locally" in local.infeasible["hosted"][0] and "gpt" in local.infeasible["hosted"][0]

    offline = recommend(run_dir, constraints=Constraints(require_no_network=True))
    assert offline.winner == "on_prem" and set(offline.infeasible) == {"hosted", "on_prem_with_tools"}
    assert "reach the network" in offline.infeasible["on_prem_with_tools"][0]


def test_a_remote_endpoint_is_not_a_local_model(tmp_path):
    specs = {"remote": {"type": "vector", "scores": noisy(0, 1), "cost": 0.001, "models": {"generator": "openai_compatible:cloud/llama3", "embedding": "local:x"}}}
    run_dir = make_run(tmp_path, specs, extra_config={"providers": {"cloud": {"base_url": "https://api.example.com/v1"}}})
    result = recommend(run_dir, constraints=Constraints(require_local_models=True))
    assert result.winner is None and "openai_compatible:cloud/llama3" in result.infeasible["remote"][0]


# -- the selection rule -------------------------------------------------------------------------


def test_statistically_tied_systems_are_separated_by_cost_not_by_noise(tmp_path):
    result = recommend(make_run(tmp_path, tied_pair(), mode="live"))

    assert result.winner == "thrifty"  # equally good as `pricey`, four times cheaper (and `weak` is clearly worse, so not in the tie)
    assert result.tied_with_winner == ["pricey"]
    assert [s.system for s in result.ranking][:2] == ["thrifty", "pricey"] and result.ranking[-1].system == "weak"
    assert any("indistinguishable" in line and "pricey" in line for line in result.rationale)


def test_a_significantly_better_system_wins_even_when_it_costs_more(tmp_path):
    specs = {"strong": {"scores": noisy(0.7, 1), "cost": 0.01}, "cheap": {"scores": noisy(0.0, 2), "cost": 0.0005}}
    result = recommend(make_run(tmp_path, specs))

    assert result.winner == "strong" and result.tied_with_winner == []
    joined = " ".join(result.rationale)
    assert "significantly" in joined and "cheap" in joined


def test_tied_systems_that_cost_the_same_are_separated_by_simplicity(tmp_path):
    specs = {
        "agentic": {"scores": noisy(0, 1), "cost": 0.002, "latency": 500.0, "llm_calls": 6},
        "plain": {"scores": noisy(0, 2), "cost": 0.002, "latency": 500.0, "llm_calls": 1},
    }
    assert recommend(make_run(tmp_path, specs)).winner == "plain"


def test_profiles_change_how_ties_are_broken(tmp_path):
    run_dir = make_run(tmp_path, tied_pair())  # pricey: fast and expensive, thrifty: slow and cheap, equally good
    pick = {name: recommend(run_dir, profile=name) for name in PROFILES}

    assert pick["cheapest_acceptable"].winner == "thrifty"
    assert pick["lowest_latency"].winner == "pricey"
    assert pick["balanced"].winner in {"pricey", "thrifty"}
    leader = max(("pricey", "thrifty"), key=lambda name: float(pd.read_csv(run_dir / "metrics_summary.csv").set_index("system").loc[name, "answer_score"]))
    assert pick["max_quality"].winner == leader and pick["max_quality"].tied_with_winner == [], "max_quality does not merge statistical ties"
    assert all(result.profile == name for name, result in pick.items())


def test_explicit_weights_override_the_profile_and_unknown_profiles_are_rejected(tmp_path):
    run_dir = make_run(tmp_path, tied_pair())
    assert recommend(run_dir, weights=Weights(quality=0, cost=0, latency=1)).winner == "pricey"
    assert recommend(run_dir, profile="cheapest_acceptable", weights=Weights(quality=0, cost=0, latency=1)).winner == "pricey"
    with pytest.raises(ValueError, match="Did you mean 'balanced'"):
        recommend(run_dir, profile="balance")


def test_by_category_winners_and_pareto_front(tmp_path):
    specs = {
        "good_at_a": {"scores": noisy(0, 1, half=(0.9, -0.7)), "cost": 0.003, "latency": 400.0},
        "good_at_b": {"scores": noisy(0, 2, half=(-0.9, 0.7)), "cost": 0.002, "latency": 300.0},
        "dominated": {"scores": noisy(0, 3, half=(-1.2, -1.2)), "cost": 0.004, "latency": 500.0},
    }
    result = recommend(make_run(tmp_path, specs))

    assert result.by_category == {"cat_a": "good_at_a", "cat_b": "good_at_b"}
    assert set(result.pareto) == {"good_at_a", "good_at_b"}  # `dominated` is beaten on quality, cost and latency by good_at_b
    assert result.category_scores["cat_a"]["good_at_a"] > result.category_scores["cat_a"]["good_at_b"]


def test_rationale_quotes_numbers_with_intervals_and_cost_ratios(tmp_path):
    result = recommend(make_run(tmp_path, tied_pair()))
    versus = next(line for line in result.rationale if line.startswith("thrifty vs weak") or line.startswith("thrifty vs pricey"))
    assert "95% CI" in versus and "answer score" in versus and "× cost" in versus
    assert any("$0.00100" in line and "thrifty" in line for line in result.rationale)


def test_mock_and_tiny_runs_carry_a_caveat(tmp_path):
    assert "Mock run" in recommend(make_run(tmp_path, tied_pair(), mode="mock")).rationale[0]
    (tmp_path / "live").mkdir()
    live = recommend(make_run(tmp_path / "live", tied_pair(), mode="live"))
    assert not any("Mock run" in line for line in live.rationale)


def test_a_single_system_is_recommended_without_a_tie_story(tmp_path):
    result = recommend(make_run(tmp_path, {"only": {"scores": noisy(0, 1), "cost": 0.001}}))
    assert result.winner == "only" and result.tied_with_winner == [] and result.infeasible == {}


# -- winner.yaml and artifacts ------------------------------------------------------------------


def test_winner_yaml_round_trips_through_load_config_and_runs_in_mock(tmp_path):
    run_dir = make_run(tmp_path, tied_pair())
    result = recommend(run_dir)
    paths = write_recommendation(run_dir, result)

    assert paths["winner"] == run_dir / "winner.yaml" and (run_dir / "recommendation.json").exists() and (run_dir / "recommendation.md").exists()
    assert (run_dir / "winner.yaml").read_text().startswith("#"), "the file says what it is and what to replace"
    config = load_config(run_dir / "winner.yaml")
    assert [system.resolved_name for system in config.systems] == [result.winner]
    assert config.evaluation.stats.baseline is None, "a baseline naming a system that is no longer there would not load"

    out = run_benchmark(run_dir / "winner.yaml", force_mock=True)
    assert pd.read_csv(out / "metrics_summary.csv")["system"].tolist() == [result.winner]


def test_artifacts_describe_the_recommendation_for_humans_and_machines(tmp_path):
    run_dir = make_run(tmp_path, tied_pair())
    result = recommend(run_dir)
    write_recommendation(run_dir, result)

    data = json.loads((run_dir / "recommendation.json").read_text())
    assert data["winner"] == result.winner and data["tied_with_winner"] == ["pricey"] and data["profile"] == "balanced"
    assert [row["system"] for row in data["ranking"]] == [s.system for s in result.ranking]
    assert data["by_category"] == result.by_category and data["rationale"] == result.rationale
    markdown = (run_dir / "recommendation.md").read_text()
    assert markdown.startswith("# Recommendation") and "thrifty" in markdown and "## Ranking" in markdown and "## Winner by category" in markdown


def test_no_winner_means_no_winner_yaml(tmp_path):
    run_dir = make_run(tmp_path, tied_pair())
    paths = write_recommendation(run_dir, recommend(run_dir, constraints=Constraints(max_cost_per_question=0.0)))
    assert "winner" not in paths and not (run_dir / "winner.yaml").exists()
    assert "No system meets" in (run_dir / "recommendation.md").read_text()


# -- config -------------------------------------------------------------------------------------


def test_selection_section_is_validated_with_helpful_errors():
    assert SelectionConfig().profile == "balanced" and SelectionConfig().constraints.is_empty
    config = SelectionConfig(profile="max_quality", constraints={"max_cost_per_question": 0.002}, weights={"quality": 1, "cost": 0, "latency": 0})
    assert config.constraints.max_cost_per_question == 0.002 and config.weights is not None
    with pytest.raises(ValueError, match="Did you mean 'cheapest_acceptable'"):
        SelectionConfig(profile="cheapest_acceptible")
    with pytest.raises(ValueError, match="all be zero"):
        SelectionConfig(weights={"quality": 0, "cost": 0, "latency": 0})
    with pytest.raises(ValueError, match="max_cost"):
        SelectionConfig(constraints={"max_cost": 1})  # a typo is not silently ignored


# -- end to end ---------------------------------------------------------------------------------


def _mock_config(tmp_path: Path, **selection: Any) -> Path:
    config = {
        "run": {"name": "rec", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [{"type": "bm25", "name": "bm25"}, {"type": "vector", "name": "vector", "retrieval": {"vector_store": "numpy"}}, {"type": "no_retrieval", "name": "floor"}],
        "evaluation": {"max_workers": 1, "max_questions": 40},
        "selection": selection,
    }
    path = tmp_path / "rec.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_a_run_writes_its_recommendation_and_a_runnable_winner(tmp_path):
    out = run_benchmark(_mock_config(tmp_path), force_mock=True)

    data = json.loads((out / "recommendation.json").read_text())
    assert data["winner"] in {"bm25", "vector", "floor"} and data["rationale"][0].startswith("Mock run")
    assert (out / "recommendation.md").exists()
    winner = load_config(out / "winner.yaml")
    assert [system.resolved_name for system in winner.systems] == [data["winner"]]
    rerun = run_benchmark(out / "winner.yaml", force_mock=True)
    assert pd.read_csv(rerun / "metrics_summary.csv")["system"].tolist() == [data["winner"]]


def test_the_runs_selection_section_drives_the_automatic_recommendation(tmp_path):
    # Every system in the config uses hosted (OpenAI) models, so demanding local models leaves nothing to recommend.
    out = run_benchmark(_mock_config(tmp_path, profile="max_quality", constraints={"require_local_models": True}), force_mock=True)
    data = json.loads((out / "recommendation.json").read_text())
    assert data["profile"] == "max_quality" and data["winner"] is None and set(data["infeasible"]) == {"bm25", "vector", "floor"}
    assert not (out / "winner.yaml").exists()


def test_recommend_command_works_on_an_existing_run_and_exports_the_winner(tmp_path):
    out = run_benchmark(_mock_config(tmp_path), force_mock=True)
    runner = CliRunner()

    result = runner.invoke(app, ["recommend", "--run", str(out), "--profile", "cheapest_acceptable", "--export", str(tmp_path / "deploy.yaml")])
    assert result.exit_code == 0, result.output
    assert "Recommendation" in result.output and "cheapest_acceptable" in result.output
    assert load_config(tmp_path / "deploy.yaml").systems

    nothing = runner.invoke(app, ["recommend", "--run", str(out), "--max-latency", "0", "--export", str(tmp_path / "none.yaml")])
    assert nothing.exit_code == 1 and "No system meets" in nothing.output and not (tmp_path / "none.yaml").exists()  # nothing to deploy is a failure for scripts

    missing = runner.invoke(app, ["recommend", "--run", str(tmp_path / "nope")])
    assert missing.exit_code == 2


def test_compare_prints_a_recommendation_panel_after_the_leaderboard(tmp_path):
    result = CliRunner().invoke(app, ["compare", "--config", str(_mock_config(tmp_path)), "--mock"])
    assert result.exit_code == 0, result.output
    assert result.output.index("Leaderboard") < result.output.index("Recommendation")
