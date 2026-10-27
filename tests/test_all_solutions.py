"""The published benchmark's configs (docs/benchmarks/multihop-rag.md): every system type is in it, and the comparison is fair."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ragbench.config.loader import load_config
from ragbench.models.family import model_family
from ragbench.models.refs import answering_model
from ragbench.registry import SYSTEMS

ROOT = Path(__file__).resolve().parents[1]
ALL_SOLUTIONS = ROOT / "configs" / "all_solutions.yaml"
MODELS = ROOT / "configs" / "all_solutions_models.yaml"
# `external_http` benchmarks an endpoint the user runs; there is no public one to put in a published benchmark.
NOT_BENCHMARKED = {"external_http"}
HARNESSES = {"claude_code", "codex", "opencode_glm"}
# The framework entries import their framework (an optional extra); without it the entry cannot be built, so a run without the extra leaves it out.
FRAMEWORK_MODULES = {"langchain_default": "langchain_core", "llamaindex_default": "llama_index.core", "haystack_default": "haystack"}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # the parent package of a dotted name is missing
        return False


@pytest.fixture(scope="module")
def config():
    return load_config(ALL_SOLUTIONS)


def test_every_registered_system_type_is_benchmarked(config):
    import ragbench.rag_systems  # noqa: F401  (registers the built-ins)

    missing = set(SYSTEMS.names()) - NOT_BENCHMARKED - {system.type for system in config.systems}
    assert not missing, f"add {sorted(missing)} to configs/all_solutions.yaml (or to NOT_BENCHMARKED, with the reason)"


def test_it_holds_all_of_all_yaml_plus_graph_rag_the_frameworks_and_the_harnesses(config):
    names = [system.resolved_name for system in config.systems]
    every = load_config(ROOT / "configs" / "all.yaml")
    assert len(names) == len(set(names)) == len(every.systems) + 2 + 3 + 3
    assert {s.resolved_name for s in every.systems} - {"full_context_ceiling"} <= set(names) and "full_context_50k" in names
    assert {"graph_rag_local", "graph_rag_global", "langchain_default", "llamaindex_default", "haystack_default", *HARNESSES} <= set(names)


def test_one_generator_writes_every_built_in_and_framework_answer(config):
    generators = {system.resolved_name: answering_model(system) for system in config.systems if system.resolved_name not in HARNESSES}
    # llm_heavy keeps its stronger model, as in all.yaml: that is the system.
    assert {name for name, model in generators.items() if model != "gpt-6-luna"} == {"llm_heavy_default"}


def test_no_answer_is_judged_by_its_own_family_and_each_has_one_judge(config):
    from ragbench.evaluation.judge_panel import scoring_judges

    judge = config.evaluation.judge
    assert judge.scorers == "first" and config.evaluation.judge_models[0] == "anthropic:claude-haiku-5-5"
    for system in config.systems:
        generator = answering_model(system)
        [scorer] = scoring_judges(config.evaluation.judge_models, generator, independent=judge.independent, first=True)
        assert model_family(scorer) != model_family(generator), system.resolved_name
        expected = "gpt-6-luna" if system.resolved_name == "claude_code" else "anthropic:claude-haiku-5-5"
        assert scorer == expected, system.resolved_name


def test_paid_run_settings(config):
    assert config.evaluation.trials == 1
    [full] = [system for system in config.systems if system.type == "full_context"]
    assert full.retrieval["context_token_budget"] <= 50_000  # the judge reads it too, and Haiku 5.5 is 5x dearer above 100k
    # torch and faiss abort in one macOS process, and late_interaction loads torch: no faiss store anywhere.
    assert all(system.retrieval.get("vector_store", "numpy") == "numpy" for system in config.systems)


def test_the_model_comparison_reuses_the_benchmarks_systems_exactly():
    models = load_config(MODELS)
    raw = yaml.safe_load(MODELS.read_text(encoding="utf-8"))
    solutions = {system["name"]: system for system in yaml.safe_load(ALL_SOLUTIONS.read_text(encoding="utf-8"))["systems"]}
    for system in raw["systems"]:
        mine = {key: value for key, value in system.items() if key != "sweep"}
        theirs = solutions[system["name"]]
        assert mine == {**theirs, "models": {"embedding": theirs["models"]["embedding"]}}, system["name"]
    generators = raw["systems"][0]["sweep"]["models.generator"]
    assert len(generators) == 8 and {"gpt-6-luna", "anthropic:claude-haiku-5-5"} <= set(generators)
    assert sum(g.startswith("openrouter:") for g in generators) == 6
    assert len(models.systems) == 3 * 8
    assert models.evaluation.judge.model_dump() == load_config(ALL_SOLUTIONS).evaluation.judge.model_dump()


@pytest.mark.slow
def test_all_solutions_runs_end_to_end_in_mock(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    raw = yaml.safe_load(ALL_SOLUTIONS.read_text(encoding="utf-8"))
    demo = ROOT / "data" / "demo"
    raw["dataset"] = {"documents_path": str(demo / "docs"), "questions_path": str(demo / "questions.jsonl"), "qrels_path": str(demo / "qrels.jsonl")}
    raw["run"]["output_dir"] = str(tmp_path)
    raw["evaluation"].update(max_questions=2, max_workers=2)
    raw["systems"] = [system for system in raw["systems"] if _installed(FRAMEWORK_MODULES.get(system["name"], "sys"))]
    for system in raw["systems"]:
        for key in ("target", "ingest_target"):
            if key in system.get("retrieval", {}):
                system["retrieval"][key] = str(ROOT / system["retrieval"][key])
    path = tmp_path / "all_solutions.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = run_benchmark(path, force_mock=True)
    rows = {row["system"]: row for row in (json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines())}
    assert set(rows) == {system["name"] for system in raw["systems"]}
    assert all(row.get("error") is None for row in rows.values())


def test_the_harness_run_reuses_the_benchmarks_entries_exactly():
    harnesses = yaml.safe_load((ROOT / "configs" / "all_solutions_harnesses.yaml").read_text(encoding="utf-8"))
    solutions = yaml.safe_load(ALL_SOLUTIONS.read_text(encoding="utf-8"))
    entries = {system["name"]: system for system in solutions["systems"]}
    assert {system["name"] for system in harnesses["systems"]} >= HARNESSES
    for system in harnesses["systems"]:
        assert system == entries[system["name"]], system["name"]
    for section in ("openrouter", "limits"):
        assert harnesses[section] == solutions[section]
    same = {key: value for key, value in harnesses["evaluation"].items() if key not in ("max_cost_usd", "stats")}
    assert same == {key: value for key, value in solutions["evaluation"].items() if key != "max_cost_usd"}
    assert harnesses["evaluation"]["stats"] == {"baseline": "hybrid_rerank_default"}  # changes no score, only what significance.csv compares
