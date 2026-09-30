"""Tiny fake RAG systems and dataset/config builders shared by evaluator-level tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ragbench.documents.schema import Document
from ragbench.rag_systems import SYSTEM_REGISTRY
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk

NUM_DOCS = 10


class RankedFakeSystem(BaseRAGSystem):
    """Always ranks doc_001..doc_010 (one chunk each), honoring `top_k` as the retrieval depth."""

    generated_with: list[int] = []  # number of chunks the generator saw, per question

    def ingest(self, documents: list[Document]) -> IngestionResult:
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=len(documents), latency_ms=0.0)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or int(self.config.retrieval.get("top_k", 5))
        chunks = [
            RetrievedChunk(chunk_id=f"doc_{i:03d}::chunk::0", doc_id=f"doc_{i:03d}", text=f"Text about topic {i}.", score=1.0 / i, rank=i)
            for i in range(1, NUM_DOCS + 1)
        ][:k]
        return RetrievalResult(question=question, chunks=chunks)

    def _generate_answer(self, question: str, chunks: list[RetrievedChunk]):
        type(self).generated_with.append(len(chunks))
        return super()._generate_answer(question, chunks)


class ExplodingSystem(RankedFakeSystem):
    """Raises for any question containing the word 'explode'."""

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        if "explode" in question:
            raise RuntimeError("boom from fake retriever")
        return super().fetch_context(question, top_k)


class BrokenIngestSystem(RankedFakeSystem):
    def ingest(self, documents: list[Document]) -> IngestionResult:
        raise RuntimeError("ingestion exploded")


FAKE_TYPES = {"fake_ranked": RankedFakeSystem, "fake_exploding": ExplodingSystem, "fake_broken_ingest": BrokenIngestSystem}


def register_fakes(monkeypatch) -> None:
    RankedFakeSystem.generated_with = []
    for name, cls in FAKE_TYPES.items():
        monkeypatch.setitem(SYSTEM_REGISTRY, name, cls)


def write_experiment(
    root: Path,
    questions: list[dict[str, Any]],
    systems: list[dict[str, Any]],
    evaluation: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    for i in range(1, NUM_DOCS + 1):
        (docs / f"doc_{i:03d}.md").write_text(f"# Topic {i}\n\nText about topic {i}.\n", encoding="utf-8")
    questions_path = root / "questions.jsonl"
    questions_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8")
    config = {
        "run": {"name": "fake_run", "output_dir": str(root / "results")},
        "dataset": {"documents_path": str(docs), "questions_path": str(questions_path)},
        "systems": systems,
        "evaluation": {"max_workers": 1, **(evaluation or {})},
        **(extra or {}),
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def question(qid: str, text: str, relevant: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": qid,
        "question": text,
        "reference_answer": "Text about topic.",
        "expected_keywords": ["topic"],
        "relevant_doc_ids": relevant if relevant is not None else [],
        "category": "direct_fact",
    }
