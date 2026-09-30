from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.rag_systems.components import build_chunker, build_embedder, build_reranker, build_vector_index, chunk_documents
from ragbench.rag_systems.options import RerankOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.utils.query_planning import generate_query_variants
from ragbench.utils.timing import timer


@SYSTEMS.register("rerank")
class RerankRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="rerank",
        title="Vector + rerank",
        summary="Vector retrieval followed by a reranking pass",
        best_for="Higher precision context selection",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=RerankOptions,
    )
    options: RerankOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker, models=config.models, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.store = build_vector_index(self.embedding_model, self.options, self.name)
        self.reranker = build_reranker(self.options.reranker, llm=self.llm, model=self.options.reranker_model, force_mock=self.force_mock)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks, chunk_cost = chunk_documents(self.chunker, documents)
            cost = self.store.build(chunks).plus(chunk_cost)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=len(chunks), latency_ms=t.elapsed_ms, cost=cost)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        opts = self.options
        final_top_k = opts.resolve_top_k(top_k)
        candidate_top_k = max(opts.candidate_top_k, final_top_k)
        queries = generate_query_variants(question, max_queries=opts.max_query_variants) if opts.multi_query else [question]
        with timer() as t:
            by_id: dict[str, RetrievedChunk] = {}
            retrieval_cost: CostBreakdown | None = None
            for query in queries:
                with self.trace.step("retrieve", "vector_search", top_k=candidate_top_k) as step:
                    step.set_input(query)
                    result = self.store.search(query, top_k=candidate_top_k)
                    step.set_chunks(result.chunks, result.cost)
                retrieval_cost = result.cost if retrieval_cost is None else retrieval_cost.plus(result.cost)
                for chunk in result.chunks:
                    current = by_id.get(chunk.chunk_id)
                    if current is None or chunk.score > current.score:
                        by_id[chunk.chunk_id] = chunk
            candidates = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
            for rank, chunk in enumerate(candidates, start=1):
                chunk.rank = rank
            with self.trace.step("rerank", getattr(self.reranker, "name", "rerank"), candidates=len(candidates), top_k=final_top_k) as step:
                step.set_input(question)
                reranked = self.reranker.rerank(question, candidates, final_top_k)
                step.set_chunks(reranked.chunks, reranked.cost)
        return RetrievalResult(
            question=question,
            chunks=reranked.chunks,
            latency_ms=t.elapsed_ms,
            cost=(retrieval_cost or CostBreakdown()).plus(reranked.cost),
            metadata={"retriever": "vector_rerank", "reranker": getattr(self.reranker, "name", "unknown"), "queries": queries},
        )
