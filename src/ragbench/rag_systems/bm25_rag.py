from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.stores.bm25_store import BM25Store
from ragbench.utils.timing import timer


class BM25RAG(BaseRAGSystem):
    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = create_chunker(config.chunker)
        self.store = BM25Store()
        self.num_chunks = 0

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            self.store.build(chunks)
            self.num_chunks = len(chunks)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=len(chunks), latency_ms=t.elapsed_ms)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or int(self.config.retrieval.get("top_k", 5))
        result = self.store.search(question, top_k=k)
        result.metadata["retriever"] = "bm25"
        return result

