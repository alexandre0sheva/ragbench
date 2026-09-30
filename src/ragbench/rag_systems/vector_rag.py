from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, build_vector_index
from ragbench.rag_systems.options import VectorSystemOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.utils.timing import timer


@SYSTEMS.register("vector")
class VectorRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="vector",
        title="Vector",
        summary="Embedding search with cosine similarity (Chroma-backed, exact in-memory fallback)",
        best_for="Semantic baseline",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=VectorSystemOptions,
    )
    options: VectorSystemOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.store = build_vector_index(self.embedding_model, self.options, self.name)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            cost = self.store.build(chunks)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"embedding_model": self.embedding_model.model_name},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        with self.trace.step("retrieve", "vector_search", top_k=k) as step:
            step.set_input(question)
            result = self.store.search(question, top_k=k)
            step.set_chunks(result.chunks, result.cost)
        result.metadata["retriever"] = "vector"
        result.metadata["embedding_model"] = self.embedding_model.model_name
        return result
