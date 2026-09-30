from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, build_vector_index
from ragbench.rag_systems.options import HyDEOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.timing import timer

HYDE_SYSTEM_PROMPT = (
    "Write a short hypothetical passage (3-5 sentences) that would answer the user's question. "
    "Write it as if it came from a reference document: factual tone, concrete entities, no hedging. "
    "If you do not know real facts, invent plausible ones — the passage is only used as a search probe."
)


@SYSTEMS.register("hyde")
class HyDERAG(BaseRAGSystem):
    """Hypothetical Document Embeddings (HyDE), Gao et al. 2022.

    Asks the LLM for a hypothetical passage that *would* answer the question
    and searches the vector store with that passage's embedding. A document
    that reads like an answer is often closer in embedding space to other
    answers than a terse question is. The raw-question ranking is fused in via
    RRF by default as a safety net against bad hypotheses.
    """

    spec = SystemSpec(
        type="hyde",
        title="HyDE",
        summary="Hypothetical Document Embeddings: the LLM writes a hypothetical answer used as the search probe",
        best_for="Short or vaguely-worded questions",
        cost_profile="medium",
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=HyDEOptions,
    )
    options: HyDEOptions

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
        opts = self.options
        final_top_k = opts.resolve_top_k(top_k)
        probe_top_k = max(opts.probe_top_k if opts.probe_top_k is not None else max(final_top_k * 2, 10), final_top_k)
        fuse_with_question = opts.fuse_with_question
        rrf_k = opts.rrf_k
        with timer() as t:
            hypothetical, generation_cost = self._generate_hypothetical(question)
            with self.trace.step("retrieve", "vector_search:hypothesis", top_k=probe_top_k) as step:
                step.set_input(hypothetical)
                probe_result = self.store.search(hypothetical, top_k=probe_top_k)
                step.set_chunks(probe_result.chunks, probe_result.cost)
            cost = generation_cost.plus(probe_result.cost)
            if fuse_with_question:
                with self.trace.step("retrieve", "vector_search:question", top_k=probe_top_k) as step:
                    step.set_input(question)
                    question_result = self.store.search(question, top_k=probe_top_k)
                    step.set_chunks(question_result.chunks, question_result.cost)
                cost = cost.plus(question_result.cost)
                chunks = reciprocal_rank_fusion([probe_result.chunks, question_result.chunks], top_k=final_top_k, rrf_k=rrf_k)
            else:
                chunks = probe_result.chunks[:final_top_k]
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={
                "retriever": "hyde",
                "hypothetical_document": hypothetical[:500],
                "fuse_with_question": fuse_with_question,
            },
        )

    def _generate_hypothetical(self, question: str) -> tuple[str, CostBreakdown]:
        messages = [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        with self.trace.step("llm", "hypothetical_document") as step:
            result = self.llm.generate(messages, temperature=0, max_tokens=220)
            cost = CostBreakdown(query_rewrite_cost=result.cost.total_cost)
            step.set_llm(result, messages, cost=cost)
            step.set_input(question)
        text = result.text.strip() or question
        return text, cost
