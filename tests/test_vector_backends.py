"""Vector backends: agreement with the exact answer, no silent fallback, optional extras, wiring. Offline.

Backends whose library is not installed are skipped (CI's base job installs neither chromadb nor faiss nor qdrant-client;
its extras job installs chroma and faiss and runs this file).
"""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import TextChunk
from ragbench.models.embeddings import HashingEmbeddingModel
from ragbench.models.errors import MissingExtraError
from ragbench.registry import VECTOR_BACKENDS, UnknownComponentError
from ragbench.stores.index import NumpyIndex
from ragbench.stores.vector_store import VectorStore

EXACT = ["numpy", "faiss", "qdrant"]
APPROXIMATE = ["chroma", "faiss_hnsw"]
MODULE = {"numpy": None, "faiss": "faiss", "faiss_hnsw": "faiss", "chroma": "chromadb", "qdrant": "qdrant_client"}


def _require(backend: str) -> None:
    module = MODULE[backend]
    if module is not None:
        pytest.importorskip(module)


def _corpus(n: int = 400, dim: int = 32, queries: int = 25, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    probes = rng.normal(size=(queries, dim)).astype(np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    return vectors, probes


def _build(backend: str, vectors: np.ndarray, name: str = "idx-test", persist_directory=None) -> Any:
    index = VECTOR_BACKENDS.get(backend)(collection_name=name, persist_directory=persist_directory)
    index.build([f"chunk_{i}" for i in range(len(vectors))], vectors, [{"doc_id": f"doc_{i}", "i": i} for i in range(len(vectors))])
    return index


# --- agreement with the exact answer ------------------------------------------------------------


@pytest.mark.parametrize("backend", EXACT)
def test_exact_backends_return_the_same_top_k_and_scores_as_numpy(backend):
    _require(backend)
    vectors, probes = _corpus()
    reference, index = _build("numpy", vectors), _build(backend, vectors, name=f"idx-{backend}")

    assert index.approximate is False
    for probe in probes:
        expected = reference.search(probe, 10)
        got = index.search(probe, 10)
        assert [row for row, _ in got] == [row for row, _ in expected], backend
        assert [score for _, score in got] == pytest.approx([score for _, score in expected], abs=1e-4)
        assert [score for _, score in got] == sorted((score for _, score in got), reverse=True)


@pytest.mark.parametrize("backend", APPROXIMATE)
def test_approximate_backends_reach_95_percent_recall(backend):
    _require(backend)
    vectors, probes = _corpus()
    exact, index = _build("numpy", vectors), _build(backend, vectors, name=f"idx-{backend}")

    assert index.approximate is True
    recalls = [len({r for r, _ in index.search(p, 10)} & {r for r, _ in exact.search(p, 10)}) / 10 for p in probes]
    assert np.mean(recalls) >= 0.95, (backend, np.mean(recalls))


@pytest.mark.parametrize("backend", [*EXACT, *APPROXIMATE])
def test_asking_for_more_than_exists_returns_everything_once(backend):
    _require(backend)
    vectors, probes = _corpus(n=6, queries=1)

    rows = [row for row, _ in _build(backend, vectors, name=f"small-{backend}").search(probes[0], 50)]

    assert sorted(rows) == list(range(6))


def test_numpy_index_keeps_the_dot_product_ranking_the_golden_snapshot_depends_on():
    vectors, probes = _corpus(n=50)
    scores = vectors @ probes[0]

    got = _build("numpy", vectors).search(probes[0], 5)

    candidates = np.argpartition(scores, -5)[-5:]
    assert [row for row, _ in got] == [int(i) for i in candidates[np.argsort(scores[candidates])[::-1]]]
    assert got[0][1] == float(scores[got[0][0]])


# --- registry, options, defaults ----------------------------------------------------------------


def test_registry_names_aliases_and_did_you_mean():
    assert {"numpy", "chroma", "faiss", "faiss_hnsw", "qdrant"} <= set(VECTOR_BACKENDS.names())
    assert VECTOR_BACKENDS.get("in_memory") is VECTOR_BACKENDS.get("numpy") is NumpyIndex
    with pytest.raises(UnknownComponentError, match="Did you mean 'chroma'"):
        VECTOR_BACKENDS.get("chrome")


def test_the_default_backend_is_exact_numpy_and_config_validates_backend_names():
    from ragbench.rag_systems.options import VectorSystemOptions

    assert VectorSystemOptions().vector_store == "numpy"
    SystemConfig(type="vector", retrieval={"vector_store": "in_memory"})  # deprecated alias still loads
    SystemConfig(type="vector", retrieval={"vector_store": "faiss_hnsw"})
    with pytest.raises(ValidationError, match=r"retrieval\.vector_store.*Did you mean 'chroma'"):
        SystemConfig(type="vector", retrieval={"vector_store": "chrome"})
    with pytest.raises(ValidationError, match="persist_directory.*chroma, qdrant"):
        SystemConfig(type="vector", retrieval={"vector_store": "numpy", "persist_directory": "/tmp/x"})
    SystemConfig(type="vector", retrieval={"vector_store": "chroma", "persist_directory": "/tmp/x"})


def test_using_the_deprecated_alias_is_logged_once_per_name(caplog):
    from ragbench.stores import index as index_package

    index_package._WARNED_ALIASES.clear()
    with caplog.at_level("WARNING"):
        VectorStore(HashingEmbeddingModel(), backend="in_memory")
        VectorStore(HashingEmbeddingModel(), backend="in_memory")

    assert sum("in_memory" in record.message and "numpy" in record.message for record in caplog.records) == 1


# --- no silent fallback; optional extras --------------------------------------------------------


@pytest.mark.parametrize(("backend", "module", "extra"), [("chroma", "chromadb", "chroma"), ("faiss", "faiss", "faiss"), ("faiss_hnsw", "faiss", "faiss"), ("qdrant", "qdrant_client", "qdrant")])
def test_a_missing_extra_is_an_actionable_error_when_the_store_is_built_not_a_silent_numpy_fallback(backend, module, extra, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)  # makes the import fail

    with pytest.raises(MissingExtraError, match=rf"pip install 'ragbench\[{extra}\]'") as raised:
        VectorStore(HashingEmbeddingModel(), backend=backend)

    assert isinstance(raised.value, ImportError)


def test_a_backend_that_fails_while_building_raises_instead_of_falling_back_to_numpy(monkeypatch):
    fake = _FakeChroma()
    fake.fail_add = True
    monkeypatch.setitem(sys.modules, "chromadb", fake.module)
    store = VectorStore(HashingEmbeddingModel(), backend="chroma", collection_name="boom")

    with pytest.raises(RuntimeError, match="chroma exploded"):
        store.build(_chunks(4))


class _FakeChroma:
    """A stand-in `chromadb` module that records calls; search answers by insertion order."""

    def __init__(self, max_batch: int = 3) -> None:
        self.max_batch = max_batch
        self.adds: list[list[str]] = []
        self.fail_add = False
        outer = self

        class Collection:
            def __init__(self) -> None:
                self.ids: list[str] = []

            def add(self, ids, embeddings, metadatas=None, documents=None) -> None:
                if outer.fail_add:
                    raise RuntimeError("chroma exploded")
                assert len(ids) <= outer.max_batch, "a batch above get_max_batch_size() is rejected by the real client"
                outer.adds.append(list(ids))
                self.ids += list(ids)

            def query(self, query_embeddings, n_results, include=None) -> dict:
                ids = self.ids[:n_results]
                return {"ids": [ids], "distances": [[0.1 * (i + 1) for i in range(len(ids))]]}

        class Client:
            def get_max_batch_size(self) -> int:
                return outer.max_batch

            def delete_collection(self, name: str) -> None:
                pass

            def create_collection(self, name: str, metadata: dict | None = None) -> Collection:
                return Collection()

        self.module = types.ModuleType("chromadb")
        self.module.EphemeralClient = lambda: Client()  # type: ignore[attr-defined]
        self.module.PersistentClient = lambda path: Client()  # type: ignore[attr-defined]


def test_chroma_adds_in_batches_no_larger_than_the_clients_limit_and_maps_ids_back_to_rows(monkeypatch):
    fake = _FakeChroma(max_batch=3)
    monkeypatch.setitem(sys.modules, "chromadb", fake.module)

    index = VECTOR_BACKENDS.get("chroma")(collection_name="batches", persist_directory=None)
    vectors, probes = _corpus(n=10, queries=1)
    index.build([f"chunk_{i}" for i in range(10)], vectors, [{} for _ in range(10)])

    assert [len(batch) for batch in fake.adds] == [3, 3, 3, 1]
    assert index.search(probes[0], 2) == [(0, pytest.approx(0.9)), (1, pytest.approx(0.8))]  # score = 1 - cosine distance


# --- VectorStore --------------------------------------------------------------------------------


def _chunks(n: int) -> list[TextChunk]:
    return [TextChunk(chunk_id=f"c{i}", doc_id=f"doc_{i % 3}", text=f"topic {i} alpha beta gamma delta {i}", metadata={"section": f"s{i}"}) for i in range(n)]


def test_vector_store_reports_the_backend_actually_used_and_maps_rows_to_chunks():
    store = VectorStore(HashingEmbeddingModel(), backend="in_memory", collection_name="mapping")
    store.build(_chunks(12))

    result = store.search("topic 7 alpha", top_k=3)

    assert store.backend == "numpy" and result.metadata == {"vector_backend": "numpy"}
    assert [c.rank for c in result.chunks] == [1, 2, 3] and result.chunks[0].chunk_id == "c7"
    assert result.chunks[0].doc_id == "doc_1" and result.chunks[0].metadata == {"section": "s7"} and "topic 7" in result.chunks[0].text
    assert result.chunks[0].score >= result.chunks[-1].score


def test_vector_store_handles_an_unbuilt_and_an_empty_corpus():
    store = VectorStore(HashingEmbeddingModel(), backend="numpy")
    assert store.search("anything").chunks == []
    store.build([])
    assert store.search("anything").chunks == []


@pytest.mark.parametrize("backend", ["faiss", "qdrant", "chroma", "faiss_hnsw"])
def test_every_backend_works_end_to_end_through_the_vector_store(backend, tmp_path):
    _require(backend)
    chunks = _chunks(30)
    reference = VectorStore(HashingEmbeddingModel(), backend="numpy")
    store = VectorStore(HashingEmbeddingModel(), backend=backend, collection_name=f"e2e-{backend}")
    reference.build(chunks)
    store.build(chunks)

    got = store.search("topic 12 alpha beta", top_k=5)

    assert store.backend == backend and got.metadata["vector_backend"] == backend
    assert got.chunks[0].chunk_id == "c12"
    assert got.chunks[0].score == pytest.approx(reference.search("topic 12 alpha beta", 5).chunks[0].score, abs=1e-4)  # near-ties below it may order differently


@pytest.mark.parametrize("backend", ["chroma", "qdrant"])
def test_persistent_backends_write_to_the_given_directory(backend, tmp_path):
    _require(backend)
    directory = tmp_path / "store"
    store = VectorStore(HashingEmbeddingModel(), backend=backend, collection_name="persisted", persist_directory=directory)
    store.build(_chunks(8))

    assert store.search("topic 3", 1).chunks[0].chunk_id == "c3" and any(directory.iterdir())
    if backend == "qdrant":
        store.index.close()  # local Qdrant locks its directory while open


def test_concurrent_chroma_builds_take_turns():
    """Systems may be built on several threads; Chroma's in-process client is shared."""
    pytest.importorskip("chromadb")
    errors: list[BaseException] = []

    def build(i: int) -> str:
        try:
            store = VectorStore(HashingEmbeddingModel(), backend="chroma", collection_name=f"threaded-{i}")
            store.build(_chunks(20))
            return store.search("topic 5", 1).chunks[0].chunk_id
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            return ""

    with ThreadPoolExecutor(6) as pool:
        found = list(pool.map(build, range(6)))

    assert not errors and found == ["c5"] * 6
    assert threading.active_count() < 50


# --- warm-up, manifest, systems -----------------------------------------------------------------


def test_warm_up_imports_the_library_of_each_configured_backend_only(tmp_path, monkeypatch):
    from fake_systems import question, register_fakes, write_experiment

    from ragbench.evaluation.evaluator import BenchmarkEvaluator

    register_fakes(monkeypatch)

    def modules(store: str) -> list[str]:
        workdir = tmp_path / f"w_{store}"
        workdir.mkdir()
        config = write_experiment(workdir, [question("q", "What is topic 1?", ["doc_001"])], [{"type": "vector", "name": "v", "retrieval": {"vector_store": store}}])
        return BenchmarkEvaluator(config, force_mock=True)._modules_to_warm_up()

    assert modules("numpy") == modules("in_memory") == ["httpx", "openai"]
    assert modules("chroma") == ["httpx", "openai", "chromadb"]
    assert modules("faiss_hnsw") == ["httpx", "openai", "faiss"]
    assert modules("qdrant") == ["httpx", "openai", "qdrant_client"]


def test_manifest_tracks_the_optional_backend_versions():
    from ragbench.evaluation.manifest import TRACKED_DEPENDENCIES

    assert {"chromadb", "faiss-cpu", "qdrant-client"} <= set(TRACKED_DEPENDENCIES)


def test_no_shipped_config_needs_an_optional_backend():
    """CI's base job installs only `.[dev]`: every config must run on the default backend."""
    from pathlib import Path

    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert "vector_store: chroma" not in text and "vector_store: faiss" not in text and "vector_store: qdrant" not in text, path.name
