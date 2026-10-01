from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.documents.tokenizer import TokenCounter, count_tokens
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.options import FullContextOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.bm25_store import BM25Store
from ragbench.utils.timing import timer


@SYSTEMS.register("full_context")
class FullContextRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="full_context",
        title="Full context",
        summary=(
            "Puts whole documents (the whole corpus when it fits) in the prompt, most relevant first by BM25, up to a token budget "
            "(the ceiling baseline: do you need retrieval at all?)"
        ),
        best_for="Small corpora, and measuring how much a retrieval pipeline loses against reading everything",
        cost_profile="high",
        latency_profile="medium",
        requires_llm=False,
        agentic=False,
        options=FullContextOptions,
        chunker=None,
    )
    options: FullContextOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.documents: dict[str, Document] = {}
        self.token_counts: dict[str, int] = {}
        self.bm25 = BM25Store()

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            ordered = sorted(documents, key=lambda document: document.doc_id)  # input order must never change a ranking tie
            self.documents = {document.doc_id: document for document in ordered}
            self.token_counts = {document.doc_id: count_tokens(document.text) for document in ordered}
            self.bm25.build([TextChunk(chunk_id=_chunk_id(document.doc_id), doc_id=document.doc_id, text=document.text) for document in ordered])
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(documents),
            latency_ms=t.elapsed_ms,
            metadata={"corpus_tokens": sum(self.token_counts.values())},
        )

    def configured_context_k(self) -> int | None:
        # The generator reads every document that fit the budget; `top_k` is only a retrieval depth for the metrics.
        return max(1, len(self.documents))

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        budget = self.options.context_token_budget
        with timer() as t, self.trace.step("retrieve", "fill_context", budget_tokens=budget) as step:
            step.set_input(question)
            ranked = self.bm25.search(question, top_k=len(self.documents))
            # Documents sharing no term with the question follow the BM25 matches, in doc_id order.
            matched = [chunk.doc_id for chunk in ranked.chunks]
            order = matched + [doc_id for doc_id in self.documents if doc_id not in set(matched)]
            scores = {chunk.doc_id: chunk.score for chunk in ranked.chunks}
            chunks: list[RetrievedChunk] = []
            used = 0
            truncated = False
            for doc_id in order:
                document, tokens = self.documents[doc_id], self.token_counts[doc_id]
                text, kept = document.text, tokens
                if used + tokens > budget:
                    text, kept = _cut(document.text, budget - used)
                    truncated = True
                if kept:
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=_chunk_id(doc_id),
                            doc_id=doc_id,
                            text=text,
                            score=scores.get(doc_id, 0.0),
                            rank=len(chunks) + 1,
                            metadata={"title": document.title, "tokens": kept, "truncated": kept < tokens},
                        )
                    )
                    used += kept
                if truncated:
                    break
            step.set_chunks(chunks)
            step.set_meta(context_tokens=used, truncated=truncated)
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            metadata={
                "retriever": "full_context",
                "truncated": truncated,
                "context_tokens": used,
                "context_token_budget": budget,
                "documents_included": len(chunks),
                "documents_total": len(self.documents),
            },
        )


def _chunk_id(doc_id: str) -> str:
    return f"{doc_id}::chunk::full"


def _cut(text: str, tokens: int) -> tuple[str, int]:
    """The longest prefix of `text` that is at most `tokens` tokens, cut on a token boundary; `("", 0)` when nothing fits."""
    spans = TokenCounter(text).spans
    n = min(tokens, len(spans))
    while n > 0:
        prefix = text[: spans[n - 1][1]].rstrip()
        if prefix and count_tokens(prefix) <= tokens:  # re-tokenising a prefix can differ from the whole by a token at the seam
            return prefix, count_tokens(prefix)
        n -= 1
    return "", 0
