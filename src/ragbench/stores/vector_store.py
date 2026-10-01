from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ragbench.documents.schema import TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EmbeddingModel
from ragbench.rag_systems.base import RetrievalResult, RetrievedChunk
from ragbench.stores.index import VectorIndex, create_index, sanitize_payload
from ragbench.utils.timing import timer


class VectorStore:
    """Embeds chunks and searches them through a pluggable `VectorIndex` backend.

    The backend is chosen by name (`numpy`, `chroma`, `faiss`, `faiss_hnsw`, `qdrant`; see `ragbench.stores.index`). A
    backend whose library is missing raises when the store is constructed; there is no fallback to another backend, so
    `RetrievalResult.metadata["vector_backend"]` is always what actually ran.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        backend: str = "numpy",
        collection_name: str = "ragbench",
        persist_directory: str | Path | None = None,
    ):
        self.embedding_model = embedding_model
        self.requested_backend = backend
        self.collection_name = _safe_collection_name(collection_name)
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.index: VectorIndex = create_index(backend, collection_name=self.collection_name, persist_directory=self.persist_directory)
        self.backend = self.index.backend  # canonical name (`in_memory` -> `numpy`)
        self.chunks: list[TextChunk] = []
        self._vectors: np.ndarray | None = None  # kept so MMR can compare candidates with each other
        self._row_of: dict[str, int] = {}
        self.ingestion_cost = CostBreakdown()

    def build(self, chunks: list[TextChunk]) -> CostBreakdown:
        self.chunks = chunks
        self._row_of = {chunk.chunk_id: row for row, chunk in enumerate(chunks)}
        result = self.embedding_model.embed_texts([chunk.text for chunk in chunks])
        self.ingestion_cost = result.cost
        self._vectors = result.vectors
        if chunks:
            self.index.build(
                [chunk.chunk_id for chunk in chunks],
                result.vectors,
                [sanitize_payload(chunk.metadata | {"doc_id": chunk.doc_id}) for chunk in chunks],
            )
        return result.cost

    def vectors_for(self, chunk_ids: list[str]) -> np.ndarray:
        """Unit-length embeddings of already indexed chunks, in the order asked (for diversity re-ranking)."""
        if self._vectors is None:
            raise ValueError("The store has no embeddings yet; call build() first.")
        return self._vectors[[self._row_of[chunk_id] for chunk_id in chunk_ids]]

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        with timer() as t:
            if not self.chunks:
                return RetrievalResult(question=query, chunks=[], latency_ms=0.0)
            query_result = self.embedding_model.embed_query(query)
            hits = self.index.search(query_result.vectors[0], top_k)
            chunks = [
                RetrievedChunk(
                    chunk_id=self.chunks[row].chunk_id,
                    doc_id=self.chunks[row].doc_id,
                    text=self.chunks[row].text,
                    score=score,
                    rank=rank,
                    metadata=dict(self.chunks[row].metadata),
                )
                for rank, (row, score) in enumerate(hits, start=1)
            ]
        return RetrievalResult(
            question=query,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=query_result.cost,
            metadata={"vector_backend": self.backend},
        )


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
