from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, build_reranker, build_vector_index, chunk_documents
from ragbench.rag_systems.options import HybridRerankOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.bm25_store import BM25Store
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.query_planning import generate_query_variants
from ragbench.utils.timing import timer


@SYSTEMS.register("hybrid_rerank")
class HybridRerankRAG(BaseRAGSystem):
    """Hybrid BM25 + vector retrieval fused with RRF, then reranked.

    Combines the recall of hybrid retrieval with the precision of a reranking
    pass — in practice one of the strongest non-LLM retrieval pipelines.
    """

    spec = SystemSpec(
        type="hybrid_rerank",
        title="Hybrid + rerank",
        summary="Hybrid BM25 + vector RRF retrieval followed by a reranking pass",
        best_for="Recall of hybrid plus rerank precision",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=HybridRerankOptions,
    )
    options: HybridRerankOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker, models=config.models, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.bm25_store = BM25Store()
        self.vector_store = build_vector_index(self.embedding_model, self.options, self.name)
        self.reranker = build_reranker(self.options.reranker, llm=self.llm, model=self.options.reranker_model, force_mock=self.force_mock)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks, chunk_cost = chunk_documents(self.chunker, documents)
            self.bm25_store.build(chunks)
            cost = self.vector_store.build(chunks).plus(chunk_cost)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"embedding_model": self.embedding_model.model_name},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        opts = self.options
        final_top_k = opts.resolve_top_k(top_k)
        bm25_top_k = max(opts.bm25_top_k, final_top_k)
        vector_top_k = max(opts.vector_top_k, final_top_k)
        candidate_top_k = max(opts.candidate_top_k, final_top_k)
        queries = generate_query_variants(question, max_queries=opts.max_query_variants) if opts.multi_query else [question]
        with timer() as t:
            rankings = []
            weights: list[float] = []
            cost: CostBreakdown | None = None
            for query in queries:
                with self.trace.step("retrieve", "bm25_search", top_k=bm25_top_k) as step:
                    step.set_input(query)
                    bm25_result = self.bm25_store.search(query, top_k=bm25_top_k)
                    step.set_chunks(bm25_result.chunks, bm25_result.cost)
                with self.trace.step("retrieve", "vector_search", top_k=vector_top_k) as step:
                    step.set_input(query)
                    vector_result = self.vector_store.search(query, top_k=vector_top_k)
                    step.set_chunks(vector_result.chunks, vector_result.cost)
                rankings.extend([bm25_result.chunks, vector_result.chunks])
                weights.extend([opts.bm25_weight, opts.vector_weight])
                pair_cost = bm25_result.cost.plus(vector_result.cost)
                cost = pair_cost if cost is None else cost.plus(pair_cost)
            candidates = reciprocal_rank_fusion(rankings, top_k=candidate_top_k, rrf_k=opts.rrf_k, weights=weights)
            with self.trace.step("rerank", getattr(self.reranker, "name", "rerank"), candidates=len(candidates), top_k=final_top_k) as step:
                step.set_input(question)
                reranked = self.reranker.rerank(question, candidates, final_top_k)
                step.set_chunks(reranked.chunks, reranked.cost)
        return RetrievalResult(
            question=question,
            chunks=reranked.chunks,
            latency_ms=t.elapsed_ms,
            cost=(cost or CostBreakdown()).plus(reranked.cost),
            metadata={
                "retriever": "hybrid_rrf_rerank",
                "reranker": getattr(self.reranker, "name", "unknown"),
                "rrf_k": opts.rrf_k,
                "queries": queries,
            },
        )
