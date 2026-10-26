"""`raptor`: a tree of LLM summaries over clustered chunks (Sarthi et al. 2024), searched flat (`collapsed`) or top-down (`tree`)."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import SUMMARIZE_CLUSTER_MARKER
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.raptor_rag import cluster_vectors
from ragbench.utils.text import normalize_text

DEMO_DOCS = Path(__file__).resolve().parents[1] / "data" / "demo" / "docs"
CHUNKER = {"type": "word", "chunk_size": 120, "chunk_overlap": 0}


class RecordingLLM(MockLLM):
    def __init__(self, fail_when=None):
        super().__init__()
        self.fail_when = fail_when
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        with self._lock:
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if self.fail_when and self.fail_when(prompt):
            raise RuntimeError("boom")
        return super().generate(messages, **kwargs)


def _demo() -> list[Document]:
    return [Document(doc_id=path.stem, path=path.name, title=path.stem, text=path.read_text(encoding="utf-8")) for path in sorted(DEMO_DOCS.glob("*.md"))]


def _system(llm=None, docs=None, **retrieval):
    system = create_rag_system(SystemConfig(type="raptor", chunker=CHUNKER, retrieval=retrieval), force_mock=True)
    if llm is not None:
        system.llm = llm
    ingestion = system.ingest(docs if docs is not None else _demo())
    return system, ingestion


# --- clustering ---------------------------------------------------------------------------------------------------------------


def test_well_separated_groups_are_separate_clusters():
    rng = np.random.default_rng(0)
    centres = np.eye(3, 16) * 10
    vectors = np.vstack([centre + rng.normal(scale=0.1, size=(12, 16)) for centre in centres])

    clusters = cluster_vectors(vectors, max_clusters=8, pca_dims=5)

    assert sorted(sorted(c) for c in clusters) == [list(range(0, 12)), list(range(12, 24)), list(range(24, 36))]


def test_clustering_is_deterministic():
    vectors = np.random.default_rng(1).normal(size=(40, 12))

    assert cluster_vectors(vectors, max_clusters=10, pca_dims=6) == cluster_vectors(vectors.copy(), max_clusters=10, pca_dims=6)


# --- the tree -------------------------------------------------------------------------------------------------------------------


def test_the_demo_corpus_builds_a_tree_of_several_levels_each_smaller_than_the_last():
    system, ingestion = _system()

    levels = ingestion.metadata["levels"]
    assert len(levels) >= 3, f"leaves plus at least two summary levels, got {levels}"
    assert all(later < earlier for earlier, later in zip(levels, levels[1:], strict=False))
    for node in system.nodes:
        if node.children:
            under = {doc for child in node.children for doc in system.nodes[child].doc_ids}
            assert set(node.doc_ids) == under, "a summary maps to exactly the documents under it"
            assert all(system.nodes[child].level < node.level for child in node.children)


def test_no_summary_prompt_carries_more_than_summary_input_chars_and_members_go_in_whole():
    llm = RecordingLLM()
    system, ingestion = _system(llm, summary_input_chars=2500)

    prompts = [call["prompt"] for call in llm.calls if SUMMARIZE_CLUSTER_MARKER in call["prompt"]]
    assert prompts and ingestion.metadata["summaries"] == len(prompts)
    for prompt in prompts:
        passages = re.findall(r"^\[\d+\] (.*)$", prompt.split("Passages:", 1)[1], flags=re.M)
        assert sum(len(p) for p in passages) <= 2500, "an oversized cluster is split again until its members fit"
    whole = {normalize_text(node.text) for node in system.nodes}
    assert all(p in whole for prompt in prompts for p in re.findall(r"^\[\d+\] (.*)$", prompt.split("Passages:", 1)[1], flags=re.M))


def test_one_summary_call_per_cluster_at_temperature_0():
    llm = RecordingLLM()
    _, ingestion = _system(llm)

    summaries = [call for call in llm.calls if SUMMARIZE_CLUSTER_MARKER in call["prompt"]]
    assert len(summaries) == ingestion.metadata["summaries"] > 0
    assert all(call["kwargs"].get("temperature") == 0 for call in summaries)
    assert ingestion.metadata["summary_failures"] == 0


def test_a_failed_summary_falls_back_to_the_lead_sentences_and_is_counted():
    seen: list[int] = []

    def fail_first(prompt: str) -> bool:
        if SUMMARIZE_CLUSTER_MARKER in prompt and not seen:
            seen.append(1)
            return True
        return False

    system, ingestion = _system(RecordingLLM(fail_first))

    assert ingestion.metadata["summary_failures"] == 1
    assert all(node.text for node in system.nodes)


def test_a_corpus_too_small_to_cluster_is_searched_as_its_chunks():
    docs = [Document(doc_id=f"d{i}", path=f"d{i}.md", title=f"d{i}", text=f"Fact number {i} about pumps.") for i in range(3)]
    system, ingestion = _system(docs=docs)

    assert ingestion.metadata["levels"] == [3] and ingestion.metadata["summaries"] == 0
    assert {chunk.doc_id for chunk in system.fetch_context("pumps", top_k=3).chunks} == {"d0", "d1", "d2"}


# --- search ---------------------------------------------------------------------------------------------------------------------


def test_collapsed_search_ranks_summaries_with_the_leaves_and_maps_them_to_their_lead_document():
    system, _ = _system()
    summary = next(node for node in system.nodes if node.children and node.level == 1)

    result = system.fetch_context(summary.text, top_k=5)

    top = result.chunks[0]
    assert top.chunk_id == summary.chunk_id and top.metadata["level"] == 1
    assert top.doc_id == summary.doc_ids[0] and top.metadata["doc_ids"] == summary.doc_ids
    assert [step.name for step in system.answer_question(summary.text).steps] == ["collapsed_search", "answer"]


def test_tree_search_only_returns_nodes_under_the_branches_it_chose():
    system, _ = _system(mode="tree", branch_k=2)

    result = system.fetch_context("How are severity levels assigned to incidents?", top_k=10)

    chosen = result.metadata["chosen"]  # node ids picked per level, top level first
    assert len(chosen[0]) == 2 and len(chosen) >= 2
    picked = {node_id for level in chosen for node_id in level}
    assert {system.node_index[chunk.chunk_id] for chunk in result.chunks} <= picked
    for upper, lower in zip(chosen, chosen[1:], strict=False):
        assert set(lower) <= {child for node_id in upper for child in system.nodes[node_id].children}


@pytest.mark.parametrize("retrieval", [{"mode": "flat"}, {"max_levels": 0}, {"min_cluster_nodes": 1}])
def test_bad_options_fail_when_the_system_is_built(retrieval):
    with pytest.raises(ValidationError):
        create_rag_system(SystemConfig(type="raptor", retrieval=retrieval), force_mock=True)


def test_near_duplicate_members_that_clustering_cannot_split_are_packed_to_the_budget(monkeypatch):
    import ragbench.rag_systems.raptor_rag as raptor

    real = raptor.cluster_vectors

    def stubborn(vectors, **kwargs):  # what a mixture does on identical vectors: everything in one cluster
        return [list(range(len(vectors)))] if kwargs.get("min_clusters", 2) > 2 or len(vectors) < 12 else real(vectors, **kwargs)

    monkeypatch.setattr(raptor, "cluster_vectors", stubborn)
    text = "The Vela pump runs at forty litres per minute and its housing is painted a deep marine blue. " * 6
    docs = [Document(doc_id=f"d{i:02d}", path=f"d{i}.md", title=f"d{i}", text=text) for i in range(12)]
    llm = RecordingLLM()
    _, ingestion = _system(llm, docs=docs, summary_input_chars=1500, min_cluster_nodes=2)

    prompts = [call["prompt"] for call in llm.calls if SUMMARIZE_CLUSTER_MARKER in call["prompt"]]
    assert ingestion.metadata["summaries"] >= 2
    assert all(sum(len(p) for p in re.findall(r"^\[\d+\] (.*)$", prompt.split("Passages:", 1)[1], flags=re.M)) <= 1500 for prompt in prompts)


def test_the_cluster_count_search_fits_far_fewer_mixtures_than_every_k_and_finds_the_same_one(monkeypatch):
    from sklearn import mixture

    rng = np.random.default_rng(3)
    centres = rng.normal(scale=12, size=(7, 16))
    vectors = np.vstack([centre + rng.normal(scale=0.3, size=(20, 16)) for centre in centres])
    fits: list[int] = []
    real = mixture.GaussianMixture

    class Counted(real):
        def fit(self, X, y=None):
            fits.append(self.n_components)
            return super().fit(X, y)

    monkeypatch.setattr(mixture, "GaussianMixture", Counted)

    clusters = cluster_vectors(vectors, max_clusters=50, pca_dims=10)

    assert len(clusters) == 7
    assert len(fits) <= 16, f"{len(fits)} mixtures fitted for k in 2..50"


def test_a_failed_summary_reads_like_the_mock_summary_of_the_same_members():
    failed: list[str] = []

    def fail_first(prompt: str) -> bool:
        if SUMMARIZE_CLUSTER_MARKER in prompt and not failed:
            failed.append(prompt)
            return True
        return False

    system, _ = _system(RecordingLLM(fail_first))

    expected = MockLLM().generate([{"role": "user", "content": failed[0]}]).text
    assert any(node.children and node.text == expected for node in system.nodes)
