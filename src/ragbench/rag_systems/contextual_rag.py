from __future__ import annotations

from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.prompts import SITUATE_CHUNK_MARKER
from ragbench.rag_systems.base import IngestionResult, RetrievalResult
from ragbench.rag_systems.components import chunk_documents
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.llm_ingest import map_llm_calls
from ragbench.rag_systems.options import ContextualOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.runtime.context import current_runtime
from ragbench.utils.timing import timer

CONTEXT_MAX_TOKENS = 120
CONTEXT_MAX_CHARS = 500


@SYSTEMS.register("contextual")
class ContextualRAG(HybridRAG):
    """Contextual Retrieval: an LLM situates every chunk within its document; that sentence is indexed with the chunk.

    The context is prepended to the chunk before it is embedded *and* before BM25 indexes it, so both halves of the hybrid
    search can match a chunk on what the document is about, not only on the chunk's own words. Answers are generated
    from, and judged against, the original chunk text.
    """

    spec = SystemSpec(
        type="contextual",
        title="Contextual retrieval",
        summary="An LLM situates each chunk within its document at ingestion; the context is embedded and BM25-indexed with the chunk, then hybrid RRF retrieval",
        best_for="Corpora where chunks lose their meaning out of context (many similar documents, pronouns, section-relative facts)",
        cost_profile="high",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=ContextualOptions,
    )
    options: ContextualOptions

    def ingest(self, documents: list[Document]) -> IngestionResult:
        by_id = {document.doc_id: document for document in documents}
        with timer() as t:
            chunks, chunk_cost = chunk_documents(self.chunker, documents)
            # One independent, temperature-0 (so disk-cached) call per chunk, `evaluation.ingest_workers` at a time.
            outcomes, failures = map_llm_calls(
                lambda chunk: self._situate(chunk, by_id[chunk.doc_id]),
                chunks,
                workers=current_runtime().ingest_workers,
                what="a chunk's context",
                thread_name_prefix="ragbench-context",
            )
            llm_cost = CostBreakdown()
            indexed: list[TextChunk] = []
            for chunk, outcome in zip(chunks, outcomes, strict=True):
                context, cost = outcome if outcome is not None else ("", CostBreakdown())
                llm_cost = llm_cost.plus(cost)
                text = f"{context}\n\n{chunk.text}" if context else chunk.text
                indexed.append(TextChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=text, metadata={**chunk.metadata, "context": context, "original_text": chunk.text}))
            self.bm25_store.build(indexed)
            embedding_cost = self.vector_store.build(indexed)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=embedding_cost.plus(chunk_cost).plus(llm_cost),
            metadata={
                "embedding_model": self.embedding_model.model_name,
                "expensive": True,
                "contexts_generated": len(chunks) - failures,
                "context_failures": failures,
            },
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        result = super().fetch_context(question, top_k=top_k)
        # The context only steers the search; the generator and the judge read the chunk as it appears in the document.
        result.chunks = [_with_original_text(chunk) for chunk in result.chunks]
        result.metadata["retriever"] = "hybrid_rrf"
        return result

    def _situate(self, chunk: TextChunk, document: Document) -> tuple[str, CostBreakdown]:
        excerpt = _excerpt(document.text, chunk.metadata.get("start_char"), chunk.metadata.get("end_char"), self.options.document_max_chars)
        messages = [
            {"role": "system", "content": "You write short context for document chunks to improve search retrieval."},
            {
                "role": "user",
                "content": (
                    f"Document title: {document.title}\n<document>\n{excerpt}\n</document>\n"
                    f"Here is the chunk we want to situate within the whole document:\n<chunk>\n{chunk.text}\n</chunk>\n"
                    f"Please give a short succinct context to {SITUATE_CHUNK_MARKER} for the purposes of improving search retrieval of the chunk. "
                    "Answer only with the succinct context and nothing else."
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, max_tokens=CONTEXT_MAX_TOKENS)
        return " ".join(result.text.split())[:CONTEXT_MAX_CHARS], result.cost


def _with_original_text(chunk: RetrievedChunk) -> RetrievedChunk:
    original = chunk.metadata.get("original_text")
    return chunk if original is None else chunk.model_copy(update={"text": original})


def _excerpt(text: str, start: int | None, end: int | None, limit: int) -> str:
    """The document, or when it is longer than `limit` the window of `limit` characters centred on the chunk."""
    if len(text) <= limit:
        return text
    if start is None or end is None:
        return text[:limit]
    first = max(0, min((start + end) // 2 - limit // 2, len(text) - limit))
    return text[first : first + limit]
