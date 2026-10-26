"""On macOS, torch and faiss each bring their own OpenMP runtime, and a process that loads both aborts ("OMP: Error #15"). A live run that would
load both is refused before anything is built or spent; mock runs (which load no torch) and other platforms are left alone."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from paid_fakes import EMBEDDING, GENERATOR, JUDGE
from paid_fakes import install as install_paid_fakes

from ragbench.config.schema import ExperimentConfig
from ragbench.errors import UsageError
from ragbench.evaluation import warm_up
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.warm_up import openmp_conflict

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
MODELS = {"generator": GENERATOR, "embedding": EMBEDDING}


def _config(*systems: dict) -> dict:
    return {
        "run": {"name": "omp"},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": list(systems),
        "evaluation": {"max_questions": 2, "max_workers": 1, "latency_probe_questions": 0, "judge_model": JUDGE},
        "cache": {"enabled": False},
    }


LATE = {"type": "late_interaction", "name": "colbert", "models": MODELS}
FAISS = {"type": "vector", "name": "vector_faiss", "models": MODELS, "retrieval": {"vector_store": "faiss"}}
CROSS = {"type": "hybrid_rerank", "name": "cross", "models": MODELS, "retrieval": {"reranker": "cross_encoder"}}


@pytest.fixture
def macos(monkeypatch):
    monkeypatch.setattr(warm_up, "_platform", lambda: "darwin")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)


def _check(*systems: dict, mode: str = "live") -> str | None:
    return openmp_conflict(ExperimentConfig.model_validate(_config(*systems)), mode)


def test_a_live_macos_run_with_torch_and_faiss_is_explained_naming_both_sides(macos):
    problem = _check(LATE, FAISS, {"type": "bm25", "name": "bm25"})

    assert problem is not None
    assert "colbert" in problem and "vector_faiss" in problem and "bm25" not in problem
    assert "OMP: Error #15" in problem


def test_any_torch_user_counts_not_only_late_interaction(macos):
    assert _check(CROSS, FAISS) is not None


@pytest.mark.parametrize(
    "systems, mode",
    [
        ((LATE, FAISS), "mock"),  # mock runs use hashing token vectors and load no torch
        ((LATE, {**FAISS, "retrieval": {"vector_store": "numpy"}}), "live"),
        ((FAISS, {"type": "bm25", "name": "bm25"}), "live"),
    ],
)
def test_runs_that_do_not_load_both_pass(macos, systems, mode):
    assert _check(*systems, mode=mode) is None


def test_other_platforms_pass(monkeypatch):
    monkeypatch.setattr(warm_up, "_platform", lambda: "linux")

    assert _check(LATE, FAISS) is None


def test_kmp_duplicate_lib_ok_does_not_switch_the_check_off(macos, monkeypatch):
    """scikit-learn sets it on import in every run, and a real run with it set still crashed (exit 139): it is no way past the conflict."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "True")

    assert _check(LATE, FAISS) is not None


def test_the_run_is_refused_before_anything_is_written(macos, monkeypatch, tmp_path):
    install_paid_fakes(monkeypatch)  # a (fake) key, so the run is live
    config = _config(LATE, FAISS)
    config["run"]["output_dir"] = str(tmp_path / "results")
    path = tmp_path / "omp.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(UsageError, match="OpenMP") as raised:
        run_benchmark(path)

    assert "vector_store: numpy" in (raised.value.hint or "")
    assert not (tmp_path / "results").exists()


ADAPTIVE = {
    "type": "adaptive",
    "name": "adaptive",
    "models": MODELS,
    "retrieval": {"routes": {"default": {"type": "late_interaction", "models": MODELS}, "lexical": {"type": "vector", "models": MODELS, "retrieval": {"vector_store": "faiss"}}}},
}


def test_torch_and_faiss_inside_adaptive_routes_are_found(macos):
    problem = _check(ADAPTIVE)

    assert problem is not None and "adaptive" in problem


def test_a_live_run_warms_up_the_libraries_of_adaptive_routes_too():
    from ragbench.evaluation.warm_up import modules_to_warm_up

    modules = modules_to_warm_up(ExperimentConfig.model_validate(_config(ADAPTIVE)), "live")

    assert {"torch", "transformers", "faiss"} <= set(modules)


def test_torch_brought_by_a_model_outside_the_systems_is_not_blamed_on_them(macos, monkeypatch):
    monkeypatch.setattr(warm_up, "warm_up_modules", lambda config: ["sentence_transformers"])  # e.g. a `local:` judge

    problem = _check({"type": "bm25", "name": "bm25"}, FAISS)

    assert problem is not None
    torch_side = problem.split("torch (", 1)[1].split(")", 1)[0]
    assert "bm25" not in torch_side and "vector_faiss" not in torch_side
