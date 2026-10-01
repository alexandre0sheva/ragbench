"""Shared plumbing of the systems that let an LLM plan the queries (`rag_fusion`, `decompose`) and then search each one."""

from __future__ import annotations

from collections.abc import Callable

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.options import LLMQueryOptions
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.query_planning import QueryPlan


class LLMQueryRAG(HybridRAG):
    """Indexes like `hybrid` (a vector store and BM25 over the same chunks); subclasses decide which queries to search and how to merge them."""

    options: LLMQueryOptions  # type: ignore[assignment]  # narrows HybridRAG's options: no BM25 weights or MMR here

    def _plan(self, name: str, planner: Callable[..., QueryPlan], question: str, limit: int) -> tuple[QueryPlan, CostBreakdown]:
        """Run an LLM planner as one traced `llm` step; its spend is booked as query-rewrite cost."""
        with self.trace.step("llm", name) as step:
            plan = planner(self.llm, question, limit)
            cost = CostBreakdown(query_rewrite_cost=plan.result.cost.total_cost)
            step.set_llm(plan.result, cost=cost)
            step.set_input(question)
            step.set_meta(fallback=plan.fallback, planned=len(plan.items))
        return plan, cost

    def _rank(self, query: str, k: int, retriever: str | None = None) -> tuple[list[RetrievedChunk], CostBreakdown]:
        """Rank the chunks for one query (with the configured retriever unless `retriever` says otherwise), without recording a step."""
        retriever = retriever or self.options.retriever
        vector = self.vector_store.search(query, top_k=k)
        if retriever != "hybrid":
            return vector.chunks, vector.cost
        bm25 = self.bm25_store.search(query, top_k=k)
        return reciprocal_rank_fusion([bm25.chunks, vector.chunks], top_k=k, rrf_k=self.options.rrf_k), vector.cost.plus(bm25.cost)

    def _search(self, query: str, k: int, name: str, /, *, retriever: str | None = None, **metadata: object) -> tuple[list[RetrievedChunk], CostBreakdown]:
        """`_rank` as one traced `retrieve` step."""
        retriever = retriever or self.options.retriever
        with self.trace.step("retrieve", name, retriever=retriever, top_k=k, **metadata) as step:
            step.set_input(query)
            chunks, cost = self._rank(query, k, retriever)
            step.set_chunks(chunks, cost)
        return chunks, cost
