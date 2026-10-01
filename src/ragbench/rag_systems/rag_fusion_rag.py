from __future__ import annotations

from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import RetrievalResult
from ragbench.rag_systems.llm_query_base import LLMQueryRAG
from ragbench.rag_systems.options import RagFusionOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.query_planning import plan_query_variants_llm
from ragbench.utils.timing import timer


@SYSTEMS.register("rag_fusion")
class RagFusionRAG(LLMQueryRAG):
    """RAG-Fusion: the LLM writes several alternative queries, each is searched, and the rankings are merged with RRF.

    A chunk that many differently-worded queries agree on rises to the top, which helps when the question's wording does not
    match the documents'. Unlike `hybrid`'s `multi_query` option, the variants come from the LLM, not from local rules.
    """

    spec = SystemSpec(
        type="rag_fusion",
        title="RAG-Fusion",
        summary="The LLM writes alternative queries; each is searched (hybrid or vector) and the rankings are merged with Reciprocal Rank Fusion",
        best_for="Questions worded differently from the documents",
        cost_profile="medium",
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=RagFusionOptions,
    )
    options: RagFusionOptions

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        opts = self.options
        final_top_k = opts.resolve_top_k(top_k)
        per_query_k = max(opts.per_query_top_k, final_top_k)
        with timer() as t:
            plan, cost = self._plan("query_variants", plan_query_variants_llm, question, opts.num_queries)
            queries = [question, *plan.items] if opts.include_original or not plan.items else plan.items
            rankings = []
            for index, query in enumerate(queries):
                chunks, search_cost = self._search(query, per_query_k, "variant_search", index=index)
                rankings.append(chunks)
                cost = cost.plus(search_cost)
            fused = reciprocal_rank_fusion(rankings, top_k=final_top_k, rrf_k=opts.rrf_k)
        return RetrievalResult(
            question=question,
            chunks=fused,
            latency_ms=t.elapsed_ms,
            cost=cost or CostBreakdown(),
            metadata={"retriever": "rag_fusion", "queries": queries, "fallback": plan.fallback, "search": opts.retriever, "rrf_k": opts.rrf_k},
        )
