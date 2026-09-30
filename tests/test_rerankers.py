"""Rerankers: contract shared by all of them, the cross-encoder (faked: no model download), and their wiring. Offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.errors import MissingExtraError
from ragbench.models.llms import LLM, LLMResult
from ragbench.models.rerankers import RerankResult, create_reranker
from ragbench.registry import RERANKERS

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What is the refund window for damaged goods?"
TEXTS = [
    "Shipping takes five business days within the country.",
    "Refund requests for damaged goods are accepted within thirty days of delivery.",
    "Our offices are closed on public holidays.",
    "The refund window is thirty days; damaged goods get a full refund.",
    "Gift cards never expire.",
    "Contact support to start a refund for damaged goods.",
    "Warranty covers manufacturing defects for two years.",
    "Damaged packaging should be photographed before opening.",
]


def _chunks(n: int = len(TEXTS)) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"c{i}", doc_id=f"doc_{i}", text=TEXTS[i], score=1.0 - i * 0.1, rank=i + 1) for i in range(n)]


class ScriptedLLM(LLM):
    """Returns the chunk ids in a fixed order, as the JSON the LLM reranker asks for."""

    model_name = "scripted"

    def __init__(self, order: list[str]):
        self.order = order

    def generate(self, messages, **kwargs) -> LLMResult:
        return LLMResult(text=json.dumps({"chunk_ids": self.order}), model="scripted", prompt_tokens=10, completion_tokens=5, cost=CostBreakdown(llm_cost=0.5))


class FakeCrossEncoderModule:
    """Stands in for `sentence_transformers`: scores a pair by how many question words the text shares."""

    def __init__(self) -> None:
        self.loads: list[dict[str, Any]] = []
        self.predicts: list[dict[str, Any]] = []
        outer = self

        class CrossEncoder:
            def __init__(self, name: str, **kwargs: Any) -> None:
                outer.loads.append({"name": name, **kwargs})

            def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> np.ndarray:
                outer.predicts.append({"pairs": list(pairs), **kwargs})
                return np.array([len(set(q.lower().split()) & set(t.lower().split())) / 10 for q, t in pairs], dtype=np.float32)

        self.module = types.ModuleType("sentence_transformers")
        self.module.CrossEncoder = CrossEncoder  # type: ignore[attr-defined]


@pytest.fixture()
def fake_cross_encoder(monkeypatch) -> FakeCrossEncoderModule:
    from ragbench.models.rerankers import cross_encoder

    fake = FakeCrossEncoderModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake.module)
    cross_encoder.clear_model_cache()
    yield fake
    cross_encoder.clear_model_cache()


def _make(name: str) -> Any:
    if name == "llm":
        return create_reranker("llm", llm=ScriptedLLM(["c6", "c3", "c1", "c5", "c0"]))
    return create_reranker(name)


ALL = ["simple_keyword_overlap", "local_relevance", "llm", "cross_encoder"]


# --- the shared contract ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_every_reranker_returns_at_most_top_k_ranked_1_to_n_and_remembers_the_original_rank(name, fake_cross_encoder):
    chunks = _chunks()
    original = {c.chunk_id: c.rank for c in chunks}

    result = _make(name).rerank(QUESTION, chunks, top_k=3)

    assert isinstance(result, RerankResult) and 1 <= len(result.chunks) <= 3
    assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))
    assert len({c.chunk_id for c in result.chunks}) == len(result.chunks) and {c.chunk_id for c in result.chunks} <= set(original)
    for chunk in result.chunks:
        assert chunk.metadata["original_rank"] == original[chunk.chunk_id], name
        assert chunk.metadata["reranker"]
    assert result.cost.total_cost >= 0
    assert [c.chunk_id for c in chunks] == [f"c{i}" for i in range(len(TEXTS))], "the input list must not be mutated"


@pytest.mark.parametrize("name", ALL)
def test_fewer_candidates_than_top_k_and_no_candidates(name, fake_cross_encoder):
    few = _make(name).rerank(QUESTION, _chunks(2), top_k=5)
    assert 1 <= len(few.chunks) <= 2 and [c.rank for c in few.chunks] == list(range(1, len(few.chunks) + 1))
    assert _make(name).rerank(QUESTION, [], top_k=5).chunks == []


@pytest.mark.parametrize("name", ["simple_keyword_overlap", "local_relevance", "cross_encoder"])
def test_relevant_chunks_outrank_irrelevant_ones(name, fake_cross_encoder):
    top = _make(name).rerank(QUESTION, _chunks(), top_k=3).chunks

    assert {c.chunk_id for c in top} <= {"c1", "c3", "c5", "c7"}, f"{name} put an unrelated chunk in the top 3"
    assert top[0].score >= top[-1].score, "scores must be non-increasing down the ranking"


def test_llm_reranker_follows_the_models_order_charges_its_cost_and_falls_back_when_it_returns_nonsense():
    result = create_reranker("llm", llm=ScriptedLLM(["c6", "ghost", "c3", "c1"])).rerank(QUESTION, _chunks(), top_k=3)

    assert [c.chunk_id for c in result.chunks] == ["c6", "c3", "c1"]
    assert result.cost.rerank_cost == 0.5
    garbage = create_reranker("llm", llm=ScriptedLLM(["ghost"])).rerank(QUESTION, _chunks(), top_k=3)
    assert len(garbage.chunks) == 3 and all("original_rank" in c.metadata for c in garbage.chunks)
    assert garbage.cost.llm_cost == 0.5, "the failed LLM call was still paid for"


def test_the_llm_reranker_degrades_to_the_free_keyword_reranker_without_an_llm():
    assert create_reranker("llm").name == "simple_keyword_overlap"


# --- cross-encoder ------------------------------------------------------------------------------


def test_cross_encoder_defaults_loads_lazily_and_scores_question_chunk_pairs(fake_cross_encoder):
    reranker = create_reranker("cross_encoder")
    assert fake_cross_encoder.loads == [], "constructing must not download or load weights"

    result = reranker.rerank(QUESTION, _chunks(), top_k=4)

    assert fake_cross_encoder.loads == [{"name": "BAAI/bge-reranker-base", "max_length": 512, "device": None}]  # device=None: library autodetect (cuda/mps/cpu)
    (call,) = fake_cross_encoder.predicts
    assert call["pairs"] == [(QUESTION, text) for text in TEXTS]
    assert call["show_progress_bar"] is False and call["batch_size"] >= 1
    assert result.chunks[0].chunk_id in {"c1", "c3"} and result.cost.total_cost == 0.0
    assert result.chunks[0].metadata["cross_encoder_model"] == "BAAI/bge-reranker-base"
    assert result.chunks[0].score == pytest.approx(max(c.score for c in result.chunks))


def test_cross_encoder_ties_keep_the_retrieval_order(fake_cross_encoder):
    same = [RetrievedChunk(chunk_id=f"t{i}", doc_id="d", text="identical text", score=1.0, rank=i + 1) for i in range(4)]

    assert [c.chunk_id for c in create_reranker("cross_encoder").rerank("question", same, top_k=4).chunks] == ["t0", "t1", "t2", "t3"]


def test_cross_encoder_model_is_configurable_and_loaded_once_per_model_across_reranker_instances_and_threads(fake_cross_encoder):
    default_a, default_b = create_reranker("cross_encoder"), create_reranker("cross_encoder")
    custom = create_reranker("cross_encoder", model="cross-encoder/ms-marco-MiniLM-L6-v2")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda r: r.rerank(QUESTION, _chunks(), top_k=2), [default_a, default_b, custom] * 6))

    assert sorted(load["name"] for load in fake_cross_encoder.loads) == ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L6-v2"]
    assert custom.name == "cross_encoder" and create_reranker("cross_encoder", model=None).model_name == "BAAI/bge-reranker-base"


def test_missing_extra_fails_when_the_reranker_is_built_with_an_actionable_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes the import fail

    with pytest.raises(MissingExtraError, match=r"ragbench\[rerank\]"):
        create_reranker("cross_encoder")
    assert issubclass(MissingExtraError, ImportError)


def test_mock_mode_never_touches_the_model_and_uses_the_tfidf_reranker(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    reranker = create_reranker("cross_encoder", force_mock=True)

    assert reranker.name == "local_relevance" and len(reranker.rerank(QUESTION, _chunks(), top_k=3).chunks) == 3


# --- registry, options, systems, import hygiene -------------------------------------------------


def test_reranker_names_are_the_documented_set_and_config_accepts_the_new_one():
    assert {"simple_keyword_overlap", "local_relevance", "llm", "cross_encoder"} <= set(RERANKERS.names())
    config = SystemConfig(type="rerank", retrieval={"reranker": "cross_encoder", "reranker_model": "BAAI/bge-reranker-v2-m3"})
    assert config.retrieval["reranker_model"] == "BAAI/bge-reranker-v2-m3"
    SystemConfig(type="hybrid_rerank", retrieval={"reranker": "cross_encoder"})
    SystemConfig(type="llm_heavy", retrieval={"reranker": "cross_encoder", "reranker_model": "x/y"})
    with pytest.raises(ValidationError, match="reranker_model.*cross_encoder"):
        SystemConfig(type="rerank", retrieval={"reranker": "local_relevance", "reranker_model": "x/y"})
    with pytest.raises(ValidationError, match="Did you mean 'cross_encoder'"):
        SystemConfig(type="rerank", retrieval={"reranker": "cross_encodr"})


@pytest.mark.parametrize("system_type", ["rerank", "hybrid_rerank", "llm_heavy"])
def test_systems_build_the_configured_cross_encoder_with_its_model_and_degrade_in_mock_mode(system_type, fake_cross_encoder, monkeypatch):
    from ragbench.rag_systems import create_rag_system

    config = SystemConfig(type=system_type, retrieval={"reranker": "cross_encoder", "reranker_model": "my/model", "vector_store": "in_memory"})
    mock_system = create_rag_system(config, force_mock=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")  # a client object is built but nothing is sent
    live_system = create_rag_system(config, force_mock=False)

    assert mock_system.reranker.name == "local_relevance"
    assert live_system.reranker.name == "cross_encoder" and live_system.reranker.model_name == "my/model"
    assert fake_cross_encoder.loads == [], "building a system must not load the model"


def test_importing_the_reranker_package_first_in_a_fresh_interpreter_works():
    """It used to fail with a circular import through `rag_systems.base`."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(
        [sys.executable, "-c", "import ragbench.models.rerankers as r; print(sorted(r.__all__))"], env=env, capture_output=True, text=True, timeout=120
    )

    assert done.returncode == 0, done.stderr[-1500:]
    assert "create_reranker" in done.stdout and "CrossEncoderReranker" in done.stdout


def test_live_runs_warm_up_the_cross_encoder_library_on_the_main_thread(tmp_path, monkeypatch):
    from ragbench.evaluation.evaluator import BenchmarkEvaluator

    path = tmp_path / "c.yaml"
    demo = ROOT / "data" / "demo"
    path.write_text(
        json.dumps(
            {
                "run": {"name": "x", "output_dir": str(tmp_path / "out")},
                "dataset": {"documents_path": str(demo / "docs"), "questions_path": str(demo / "questions.jsonl")},
                "systems": [{"type": "rerank", "retrieval": {"reranker": "cross_encoder", "vector_store": "in_memory"}}, {"type": "bm25"}],
            }
        )
    )
    evaluator = BenchmarkEvaluator(path, force_mock=True)
    assert "sentence_transformers" not in evaluator._modules_to_warm_up(), "mock runs never load the model"
    evaluator.mode = "live"
    assert "sentence_transformers" in evaluator._modules_to_warm_up()
