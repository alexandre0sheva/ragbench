from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.documents.schema import TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EmbeddingModel
from ragbench.rag_systems.base import RetrievalResult, RetrievedChunk
from ragbench.utils.timing import timer


class VectorStore:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        backend: str = "chroma",
        collection_name: str = "ragbench",
        persist_directory: str | Path | None = None,
    ):
        self.embedding_model = embedding_model
        self.requested_backend = backend
        self.backend = "in_memory"
        self.collection_name = _safe_collection_name(collection_name)
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.chunks: list[TextChunk] = []
        self.vectors: np.ndarray | None = None
        self.ingestion_cost = CostBreakdown()
        self.collection: Any | None = None
        self.fallback_reason: str | None = None

    def build(self, chunks: list[TextChunk]) -> CostBreakdown:
        self.chunks = chunks
        result = self.embedding_model.embed_texts([chunk.text for chunk in chunks])
        self.vectors = result.vectors
        self.ingestion_cost = result.cost
        if self.requested_backend == "chroma":
            self._build_chroma(result.vectors)
        else:
            self.backend = "in_memory"
        return result.cost

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        with timer() as t:
            if self.vectors is None or not self.chunks:
                return RetrievalResult(question=query, chunks=[], latency_ms=0.0)
            query_result = self.embedding_model.embed_query(query)
            query_vec = query_result.vectors[0]
            if self.backend == "chroma" and self.collection is not None:
                chunks = self._search_chroma(query_vec, top_k)
            else:
                chunks = self._search_in_memory(query_vec, top_k)
        return RetrievalResult(
            question=query,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=query_result.cost,
            metadata={"vector_backend": self.backend, "fallback_reason": self.fallback_reason},
        )

    def _build_chroma(self, vectors: np.ndarray) -> None:
        try:
            import chromadb

            if self.persist_directory:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                client = chromadb.EphemeralClient()
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if self.chunks:
                self.collection.add(
                    ids=[chunk.chunk_id for chunk in self.chunks],
                    documents=[chunk.text for chunk in self.chunks],
                    embeddings=[vector.astype(float).tolist() for vector in vectors],
                    metadatas=[_sanitize_metadata(chunk.metadata | {"doc_id": chunk.doc_id}) for chunk in self.chunks],
                )
            self.backend = "chroma"
            self.fallback_reason = None
        except Exception as exc:
            self.collection = None
            self.backend = "in_memory"
            self.fallback_reason = f"Chroma unavailable; using in-memory NumPy search: {exc}"

    def _search_chroma(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self.collection is None:
            return self._search_in_memory(query_vec, top_k)
        result = self.collection.query(
            query_embeddings=[query_vec.astype(float).tolist()],
            n_results=min(top_k, len(self.chunks)),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances, strict=False), start=1):
            metadata = dict(metadata or {})
            doc_id = str(metadata.get("doc_id", ""))
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    doc_id=doc_id,
                    text=str(text),
                    score=1.0 - float(distance),
                    rank=rank,
                    metadata=metadata,
                )
            )
        return chunks

    def _search_in_memory(self, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self.vectors is None:
            return []
        scores = self.vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunks[int(idx)].chunk_id,
                doc_id=self.chunks[int(idx)].doc_id,
                text=self.chunks[int(idx)].text,
                score=float(scores[int(idx)]),
                rank=rank,
                metadata=dict(self.chunks[int(idx)].metadata),
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]


def _safe_collection_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if len(safe) < 3:
        safe = f"ragbench_{safe}"
    safe = safe[:63].strip("_-")
    if not safe or not safe[0].isalnum():
        safe = f"ragbench{safe}"
    if not safe[-1].isalnum():
        safe = f"{safe}0"
    return safe[:63]


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False, default=str)
    return sanitized
