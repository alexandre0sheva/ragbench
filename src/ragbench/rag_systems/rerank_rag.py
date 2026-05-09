from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import create_embedding_model
from ragbench.models.rerankers import create_reranker
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.stores.vector_store import VectorStore
from ragbench.utils.query_planning import generate_query_variants
from ragbench.utils.timing import timer


class RerankRAG(BaseRAGSystem):
    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = create_chunker(config.chunker)
        self.embedding_model = create_embedding_model(config.models.get("embedding", "text-embedding-3-small"), force_mock=force_mock)
        self.store = VectorStore(
            self.embedding_model,
            backend=config.retrieval.get("vector_store", "chroma"),
            collection_name=self.name,
            persist_directory=config.retrieval.get("persist_directory"),
        )
        self.reranker = create_reranker(config.retrieval.get("reranker", "simple_keyword_overlap"), llm=self.llm)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            cost = self.store.build(chunks)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=len(chunks), latency_ms=t.elapsed_ms, cost=cost)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        candidate_top_k = int(self.config.retrieval.get("candidate_top_k", 30))
        final_top_k = top_k or int(self.config.retrieval.get("final_top_k", 5))
        queries = (
            generate_query_variants(question, max_queries=int(self.config.retrieval.get("max_query_variants", 4)))
            if bool(self.config.retrieval.get("multi_query", False))
            else [question]
        )
        with timer() as t:
            by_id: dict[str, RetrievedChunk] = {}
            retrieval_cost: CostBreakdown | None = None
            for query in queries:
                result = self.store.search(query, top_k=candidate_top_k)
                retrieval_cost = result.cost if retrieval_cost is None else retrieval_cost.plus(result.cost)
                for chunk in result.chunks:
                    current = by_id.get(chunk.chunk_id)
                    if current is None or chunk.score > current.score:
                        by_id[chunk.chunk_id] = chunk
            candidates = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
            for rank, chunk in enumerate(candidates, start=1):
                chunk.rank = rank
            reranked = self.reranker.rerank(question, candidates, final_top_k)
        return RetrievalResult(
            question=question,
            chunks=reranked.chunks,
            latency_ms=t.elapsed_ms,
            cost=(retrieval_cost or CostBreakdown()).plus(reranked.cost),
            metadata={"retriever": "vector_rerank", "reranker": getattr(self.reranker, "name", "unknown"), "queries": queries},
        )
