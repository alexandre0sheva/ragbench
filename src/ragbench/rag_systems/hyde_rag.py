from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import create_embedding_model
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.stores.vector_store import VectorStore
from ragbench.utils.timing import timer

HYDE_SYSTEM_PROMPT = (
    "Write a short hypothetical passage (3-5 sentences) that would answer the user's question. "
    "Write it as if it came from a reference document: factual tone, concrete entities, no hedging. "
    "If you do not know real facts, invent plausible ones — the passage is only used as a search probe."
)


class HyDERAG(BaseRAGSystem):
    """Hypothetical Document Embeddings (HyDE), Gao et al. 2022.

    Asks the LLM for a hypothetical passage that *would* answer the question
    and searches the vector store with that passage's embedding. A document
    that reads like an answer is often closer in embedding space to other
    answers than a terse question is. The raw-question ranking is fused in via
    RRF by default as a safety net against bad hypotheses.
    """

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
        cfg = self.config.retrieval
        final_top_k = top_k or int(cfg.get("top_k", 5))
        probe_top_k = int(cfg.get("probe_top_k", max(final_top_k * 2, 10)))
        fuse_with_question = bool(cfg.get("fuse_with_question", True))
        rrf_k = int(cfg.get("rrf_k", 60))
        with timer() as t:
            hypothetical, generation_cost = self._generate_hypothetical(question)
            probe_result = self.store.search(hypothetical, top_k=probe_top_k)
            cost = generation_cost.plus(probe_result.cost)
            if fuse_with_question:
                question_result = self.store.search(question, top_k=probe_top_k)
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
        result = self.llm.generate(messages, temperature=0, max_tokens=220)
        text = result.text.strip() or question
        return text, CostBreakdown(query_rewrite_cost=result.cost.total_cost)
