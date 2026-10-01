from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.presets import PRESET_NAMES, apply_preset, preset_systems
from ragbench.config.schema import ExperimentConfig
from ragbench.config.sweep import MAX_VARIANTS, axis_labels, expand_sweeps, select_systems, split_system_names
from ragbench.evaluation.evaluator import run_benchmark

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def _raw(*systems: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "run": {"name": "sweep", "output_dir": "results"},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": list(systems),
        **extra,
    }


def _names(raw: dict[str, Any]) -> list[str]:
    return [s.get("name") or s["type"] for s in expand_sweeps(raw)["systems"]]


GRID = {"type": "hybrid_rerank", "name": "hr", "sweep": {"chunker.chunk_size": [300, 500], "retrieval.reranker": ["local_relevance", "cross_encoder"]}}


def test_a_two_by_two_sweep_yields_four_systems_with_deterministic_names():
    expanded = expand_sweeps(_raw(GRID))["systems"]

    assert [s["name"] for s in expanded] == [
        "hr[chunk_size=300,reranker=local_relevance]",
        "hr[chunk_size=300,reranker=cross_encoder]",
        "hr[chunk_size=500,reranker=local_relevance]",
        "hr[chunk_size=500,reranker=cross_encoder]",
    ]
    assert [(s["chunker"]["chunk_size"], s["retrieval"]["reranker"]) for s in expanded] == [
        (300, "local_relevance"),
        (300, "cross_encoder"),
        (500, "local_relevance"),
        (500, "cross_encoder"),
    ]
    assert all("sweep" not in s and s["type"] == "hybrid_rerank" for s in expanded)
    assert [s["name"] for s in expand_sweeps(_raw(GRID))["systems"]] == [s["name"] for s in expanded], "same input, same names, same order"


def test_variants_inherit_everything_else_and_do_not_share_state():
    base = {**GRID, "chunker": {"type": "word", "chunk_overlap": 40}, "retrieval": {"final_top_k": 3}, "models": {"generator": "gpt-6-luna"}}
    expanded = expand_sweeps(_raw(base))["systems"]

    assert all(s["chunker"]["type"] == "word" and s["chunker"]["chunk_overlap"] == 40 and s["retrieval"]["final_top_k"] == 3 for s in expanded)
    assert all(s["models"] == {"generator": "gpt-6-luna"} for s in expanded)
    expanded[0]["retrieval"]["final_top_k"] = 99
    assert expanded[1]["retrieval"]["final_top_k"] == 3 and base["retrieval"]["final_top_k"] == 3  # deep copies


def test_the_sweep_is_validated_like_any_other_config_and_the_input_is_untouched():
    raw = _raw(GRID)
    snapshot = copy.deepcopy(raw)
    config = ExperimentConfig.model_validate(raw)  # direct validation expands too
    assert [s.resolved_name for s in config.systems][0] == "hr[chunk_size=300,reranker=local_relevance]" and len(config.systems) == 4
    assert raw == snapshot


def test_tool_sets_are_a_sweepable_axis():
    agent = {"type": "agent_search", "name": "agent", "sweep": {"tools": [[], ["calculator"], ["calculator", {"name": "corpus_grep"}]]}}
    expanded = expand_sweeps(_raw(agent))["systems"]

    assert [s["name"] for s in expanded] == ["agent[tools=none]", "agent[tools=calculator]", "agent[tools=calculator+corpus_grep]"]
    assert [s["tools"] for s in expanded] == [[], ["calculator"], ["calculator", {"name": "corpus_grep"}]]
    assert len(ExperimentConfig.model_validate(_raw(agent)).systems) == 3


def test_systems_without_a_sweep_are_untouched_and_keep_their_place():
    plain = {"type": "bm25", "name": "bm25"}
    nameless = {"type": "vector", "sweep": {"retrieval.top_k": [3, 5]}}
    assert _names(_raw(plain, nameless, {"type": "hybrid"})) == ["bm25", "vector[top_k=3]", "vector[top_k=5]", "hybrid"]
    assert expand_sweeps(_raw(plain)) == _raw(plain)


def test_scalar_axis_values_are_named_compactly():
    system = {"type": "vector", "name": "v", "sweep": {"retrieval.diversity": ["none", "mmr"], "models.generator": ["gpt-6-luna"]}}
    assert _names(_raw(system)) == ["v[diversity=none,generator=gpt-6-luna]", "v[diversity=mmr,generator=gpt-6-luna]"]


def test_axis_labels_fall_back_to_the_full_path_only_when_two_axes_end_alike():
    assert axis_labels(["chunker.chunk_size", "retrieval.reranker"]) == ["chunk_size", "reranker"]
    assert axis_labels(["chunker.size", "retrieval.size", "tools"]) == ["chunker.size", "retrieval.size", "tools"]


def test_a_bad_axis_names_the_valid_ones_and_suggests_the_closest():
    system = {"type": "hybrid_rerank", "name": "hr", "sweep": {"retrieval.rerankr": ["local_relevance"]}}
    with pytest.raises(ValueError) as caught:
        expand_sweeps(_raw(system))
    message = str(caught.value)
    assert "retrieval.rerankr" in message and "Did you mean 'retrieval.reranker'" in message
    assert "Valid axes:" in message and "chunker.chunk_size" in message and "models.generator" in message and "retrieval.final_top_k" in message
    assert "systems[0]" in message and "'hr'" in message


def test_axes_a_system_does_not_have_are_rejected():
    with pytest.raises(ValueError, match=r"'retrieval.reranker' is not valid"):
        expand_sweeps(_raw({"type": "bm25", "sweep": {"retrieval.reranker": ["x"]}}))
    with pytest.raises(ValueError, match="Valid axes:") as no_tools:
        expand_sweeps(_raw({"type": "bm25", "sweep": {"tools": [[]]}}))
    assert "tools" not in str(no_tools.value).split("Valid axes:")[1], "bm25 takes no tools, so `tools` is not offered as an axis"
    with pytest.raises(ValueError, match="nested|not supported"):
        expand_sweeps(_raw({"type": "adaptive", "sweep": {"retrieval.routes.default.retrieval.top_k": [3]}}))
    with pytest.raises(ValueError, match="unknown|Unknown|did you mean|Did you mean"):
        expand_sweeps(_raw({"type": "hybrd_rerank", "sweep": {"tools": [[]]}}))


@pytest.mark.parametrize(
    ("sweep", "problem"),
    [
        ({}, "at least one axis"),
        ({"retrieval.top_k": []}, "at least one value"),
        ({"retrieval.top_k": 5}, "must be a list"),
        ([("retrieval.top_k", [1])], "mapping"),
    ],
)
def test_malformed_sweeps_are_explained(sweep, problem):
    with pytest.raises(ValueError, match=problem):
        expand_sweeps(_raw({"type": "vector", "sweep": sweep}))


def test_a_runaway_sweep_is_refused_before_it_is_expanded():
    sweep = {"retrieval.top_k": list(range(1, 12)), "retrieval.rrf_k": list(range(1, 12))}  # 121 combinations
    with pytest.raises(ValueError, match=rf"{MAX_VARIANTS}"):
        expand_sweeps(_raw({"type": "hybrid", "sweep": sweep}))


def test_a_variant_that_collides_with_another_system_is_a_duplicate_name_error():
    raw = _raw({"type": "vector", "name": "v[top_k=3]"}, {"type": "vector", "name": "v", "sweep": {"retrieval.top_k": [3, 5]}})
    with pytest.raises(ValueError, match="duplicate system name"):
        ExperimentConfig.model_validate(raw)


def test_sweeps_load_from_yaml_and_the_loader_returns_the_expanded_form(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(_raw(GRID)), encoding="utf-8")

    assert len(load_config(path).systems) == 4
    systems = load_config_dict(path)["systems"]
    assert len(systems) == 4 and all("sweep" not in system for system in systems)


# -- selecting systems ---------------------------------------------------------------------------


def test_split_system_names_ignores_commas_inside_variant_brackets():
    assert split_system_names(["bm25,hr[chunk_size=300,reranker=cross_encoder],vector"]) == ["bm25", "hr[chunk_size=300,reranker=cross_encoder]", "vector"]
    assert split_system_names(["a", "b, c"]) == ["a", "b", "c"]


def test_select_systems_by_exact_name_or_by_sweep_base_name():
    raw = expand_sweeps(_raw({"type": "bm25", "name": "bm25"}, GRID, {"type": "vector", "name": "vector"}))
    assert [s["name"] for s in select_systems(raw, ["vector", "bm25"])["systems"]] == ["bm25", "vector"], "config order, not request order"
    assert len(select_systems(raw, ["hr"])["systems"]) == 4, "a sweep's base name selects all its variants"
    assert [s["name"] for s in select_systems(raw, ["hr[chunk_size=500,reranker=cross_encoder]"])["systems"]] == ["hr[chunk_size=500,reranker=cross_encoder]"]
    with pytest.raises(ValueError, match=r"vectr.*Did you mean 'vector'.*Available: "):
        select_systems(raw, ["vectr"])


# -- presets -------------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_every_preset_validates_against_the_system_registry(preset):
    raw = apply_preset(None, preset, docs=DEMO / "docs", questions=DEMO / "questions.jsonl", qrels=DEMO / "qrels.jsonl")
    config = ExperimentConfig.model_validate(raw)
    names = [s.resolved_name for s in config.systems]
    assert len(names) == len(set(names)) and names
    assert preset_systems(preset) == preset_systems(preset), "presets are plain data: the same every time"


def test_preset_contents_follow_the_plan():
    def names(preset: str) -> list[str]:
        return [s.resolved_name for s in ExperimentConfig.model_validate(apply_preset(None, preset, docs=DEMO / "docs", questions=DEMO / "questions.jsonl")).systems]

    quick, standard, thorough, agentic = names("quick"), names("standard"), names("thorough"), names("agentic")
    assert quick == ["bm25", "vector", "hybrid_rerank"]
    assert standard[:3] == quick and {"hybrid", "rerank", "parent_doc", "hyde", "contextual"} <= set(standard)
    assert set(standard) <= set(thorough) and any(name.startswith("hybrid_rerank[chunk_size=") for name in thorough), "thorough adds a chunk-size sweep"
    assert {"llm_heavy", "sentence_window", "hierarchical", "rag_fusion", "decompose"} <= set(thorough)
    assert {"corrective", "iterative", "grep_agent", "adaptive", "hybrid_rerank"} <= set(agentic)
    assert sum(name.startswith("agent_search") for name in agentic) == 2, "agent_search with and without tools"


def test_applying_a_preset_replaces_the_systems_but_keeps_the_rest_of_an_existing_config():
    existing = _raw({"type": "bm25", "name": "mine"}, evaluation={"max_questions": 7, "judge_enabled": False}, pricing={"x": {"input": 1, "output": 2}})
    out = apply_preset(existing, "quick")
    assert out["evaluation"] == {"max_questions": 7, "judge_enabled": False} and out["pricing"] == {"x": {"input": 1, "output": 2}}
    assert [s["name"] for s in out["systems"]] == ["bm25", "vector", "hybrid_rerank"] and existing["systems"][0]["name"] == "mine"
    moved = apply_preset(existing, "quick", docs=Path("elsewhere/docs"), questions=Path("elsewhere/q.jsonl"))
    assert moved["dataset"]["documents_path"] == "elsewhere/docs" and moved["dataset"]["qrels_path"] == existing["dataset"]["qrels_path"]


def test_a_preset_needs_a_dataset_from_flags_or_a_config():
    with pytest.raises(ValueError, match="--docs and --questions"):
        apply_preset(None, "quick")
    with pytest.raises(ValueError, match="Did you mean 'standard'"):
        apply_preset(None, "standrd", docs=DEMO / "docs", questions=DEMO / "questions.jsonl")


# -- through the evaluator and the CLI ---------------------------------------------------------------


def test_a_swept_run_writes_the_expanded_config_so_its_recommendation_and_winner_work(tmp_path):
    raw = _raw(
        {"type": "vector", "name": "v", "retrieval": {"vector_store": "numpy"}, "sweep": {"retrieval.top_k": [3, 5]}},
        evaluation={"max_workers": 1, "max_questions": 8},
        run={"name": "swept", "output_dir": str(tmp_path / "results")},
    )
    path = tmp_path / "swept.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    out = run_benchmark(path, force_mock=True)

    assert pd.read_csv(out / "metrics_summary.csv")["system"].tolist() == ["v[top_k=3]", "v[top_k=5]"]
    written = yaml.safe_load((out / "config.yaml").read_text())
    assert [s["name"] for s in written["systems"]] == ["v[top_k=3]", "v[top_k=5]"] and "sweep" not in (out / "config.yaml").read_text()
    winner = load_config(out / "winner.yaml")
    assert len(winner.systems) == 1 and winner.systems[0].resolved_name in {"v[top_k=3]", "v[top_k=5]"}


def test_an_unswept_config_is_copied_verbatim_comments_included(tmp_path):
    text = "# my notes\n" + yaml.safe_dump(_raw({"type": "bm25", "name": "bm25"}, evaluation={"max_questions": 3}, run={"name": "plain", "output_dir": str(tmp_path / "results")}))
    path = tmp_path / "plain.yaml"
    path.write_text(text, encoding="utf-8")
    out = run_benchmark(path, force_mock=True)
    assert (out / "config.yaml").read_text() == text


def test_cli_runs_a_preset_on_a_dataset_given_by_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a preset's results go to ./results
    result = CliRunner().invoke(
        app,
        ["compare", "--preset", "quick", "--docs", str(DEMO / "docs"), "--questions", str(DEMO / "questions.jsonl"), "--qrels", str(DEMO / "qrels.jsonl"), "--mock"],
    )
    assert result.exit_code == 0, result.output
    assert "Comparison complete" in result.output and all(name in result.output for name in ("bm25", "vector", "hybrid_rerank"))
    out = sorted((tmp_path / "results").iterdir())[-1]
    assert [s["name"] for s in yaml.safe_load((out / "config.yaml").read_text())["systems"]] == ["bm25", "vector", "hybrid_rerank"]


def test_cli_systems_option_selects_a_subset_of_a_config(tmp_path):
    raw = _raw(
        {"type": "bm25", "name": "bm25"},
        {"type": "vector", "name": "v", "retrieval": {"vector_store": "numpy"}, "sweep": {"retrieval.top_k": [3, 5]}},
        evaluation={"max_workers": 1, "max_questions": 6},
        run={"name": "subset", "output_dir": str(tmp_path / "results")},
    )
    path = tmp_path / "subset.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = CliRunner().invoke(app, ["compare", "--config", str(path), "--systems", "bm25,v[top_k=5]", "--mock"])

    assert result.exit_code == 0, result.output
    out = sorted((tmp_path / "results").iterdir())[-1]
    assert pd.read_csv(out / "metrics_summary.csv")["system"].tolist() == ["bm25", "v[top_k=5]"]
    assert [s["name"] for s in yaml.safe_load((out / "config.yaml").read_text())["systems"]] == ["bm25", "v[top_k=5]"], "the run directory records what actually ran"

    bad = CliRunner().invoke(app, ["compare", "--config", str(path), "--systems", "bm26", "--mock"])
    assert bad.exit_code == 2 and "Did you mean 'bm25'" in bad.output


def test_cli_needs_a_config_or_a_preset():
    result = CliRunner().invoke(app, ["compare", "--mock"])
    assert result.exit_code == 2 and "--config" in result.output and "--preset" in result.output
