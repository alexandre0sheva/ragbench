from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker
from ragbench.rag_systems.options import BM25Options
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.bm25_store import BM25Store
from ragbench.utils.timing import timer


@SYSTEMS.register("bm25")
class BM25RAG(BaseRAGSystem):
    spec = SystemSpec(
        type="bm25",
        title="BM25",
        summary="Lexical BM25 over chunks",
        best_for="Cheap baseline and exact-term matching",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=BM25Options,
    )
    options: BM25Options

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker)
        self.store = BM25Store()
        self.num_chunks = 0

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            self.store.build(chunks)
            self.num_chunks = len(chunks)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=len(chunks), latency_ms=t.elapsed_ms)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        with self.trace.step("retrieve", "bm25_search", top_k=k) as step:
            step.set_input(question)
            result = self.store.search(question, top_k=k)
            step.set_chunks(result.chunks, result.cost)
        result.metadata["retriever"] = "bm25"
        return result
