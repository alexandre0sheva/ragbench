from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import create_embedding_model
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.stores.bm25_store import BM25Store
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.stores.vector_store import VectorStore
from ragbench.utils.query_planning import generate_query_variants
from ragbench.utils.timing import timer


class HybridRAG(BaseRAGSystem):
    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = create_chunker(config.chunker)
        self.embedding_model = create_embedding_model(config.models.get("embedding", "text-embedding-3-small"), force_mock=force_mock)
        self.bm25_store = BM25Store()
        self.vector_store = VectorStore(
            self.embedding_model,
            backend=config.retrieval.get("vector_store", "chroma"),
            collection_name=self.name,
            persist_directory=config.retrieval.get("persist_directory"),
        )

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            self.bm25_store.build(chunks)
            cost = self.vector_store.build(chunks)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"embedding_model": self.embedding_model.model_name},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        cfg = self.config.retrieval
        bm25_top_k = int(cfg.get("bm25_top_k", 20))
        vector_top_k = int(cfg.get("vector_top_k", 20))
        final_top_k = top_k or int(cfg.get("final_top_k", cfg.get("top_k", 5)))
        rrf_k = int(cfg.get("rrf_k", 60))
        queries = (
            generate_query_variants(question, max_queries=int(cfg.get("max_query_variants", 4)))
            if bool(cfg.get("multi_query", False))
            else [question]
        )
        with timer() as t:
            rankings = []
            cost: CostBreakdown | None = None
            for query in queries:
                bm25_result = self.bm25_store.search(query, top_k=bm25_top_k)
                vector_result = self.vector_store.search(query, top_k=vector_top_k)
                rankings.extend([bm25_result.chunks, vector_result.chunks])
                pair_cost = bm25_result.cost.plus(vector_result.cost)
                cost = pair_cost if cost is None else cost.plus(pair_cost)
            fused = reciprocal_rank_fusion(rankings, top_k=final_top_k, rrf_k=rrf_k)
        return RetrievalResult(
            question=question,
            chunks=fused,
            latency_ms=t.elapsed_ms,
            cost=cost or CostBreakdown(),
            metadata={"retriever": "hybrid_rrf", "rrf_k": rrf_k, "queries": queries},
        )
