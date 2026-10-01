from __future__ import annotations

import numpy as np

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.prompts import SUMMARIZE_DOCUMENT_MARKER
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, chunk_documents
from ragbench.rag_systems.llm_ingest import map_llm_calls
from ragbench.rag_systems.options import HierarchicalOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.runtime.context import current_runtime
from ragbench.stores.bm25_store import BM25Store
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.timing import timer

SUMMARY_MAX_TOKENS = 200
LEAD_CHARS = 400


@SYSTEMS.register("hierarchical")
class HierarchicalRAG(BaseRAGSystem):
    """Two-level retrieval: pick the documents first from LLM-written summaries, then pick chunks only inside them.

    Each document gets a card (title plus a short LLM summary). A question is matched against the cards with BM25 and
    embedding similarity fused by RRF; the top `docs_k` documents advance, and only their chunks are scored. Search is
    exact (NumPy), so this system takes no `vector_store` option.
    """

    spec = SystemSpec(
        type="hierarchical",
        title="Hierarchical (summary-routed)",
        summary="Choose documents first from LLM-written summaries and titles, then retrieve chunks only inside the chosen documents",
        best_for='"Which document?" questions and corpora too large to search chunk by chunk',
        cost_profile="high",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=HierarchicalOptions,
    )
    options: HierarchicalOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker, models=config.models, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.card_index = BM25Store()
        self.card_doc_ids: list[str] = []
        self.card_texts: list[str] = []
        self.card_vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[TextChunk] = []
        self.chunk_vectors = np.zeros((0, 0), dtype=np.float32)
        self.chunk_rows_by_doc: dict[str, list[int]] = {}

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            self.chunks, chunk_cost = chunk_documents(self.chunker, documents)
            outcomes, failures = map_llm_calls(
                self._summarize,
                documents,
                workers=current_runtime().ingest_workers,
                what="a document summary",
                thread_name_prefix="ragbench-summary",
            )
            llm_cost = CostBreakdown()
            cards: list[TextChunk] = []
            without_summary = 0
            for document, outcome in zip(documents, outcomes, strict=True):
                summary, cost = outcome if outcome is not None else ("", CostBreakdown())
                llm_cost = llm_cost.plus(cost)
                if not summary:  # the call failed or answered nothing: the start of the document stands in for it
                    summary, without_summary = _lead(document.text), without_summary + 1
                cards.append(TextChunk(chunk_id=f"{document.doc_id}::card", doc_id=document.doc_id, text=f"{document.title}\n{summary}", metadata={"title": document.title}))
            self.card_doc_ids = [card.doc_id for card in cards]
            self.card_texts = [card.text for card in cards]
            self.card_index.build(cards)
            card_embedding = self.embedding_model.embed_texts(self.card_texts) if cards else None
            chunk_embedding = self.embedding_model.embed_texts([chunk.text for chunk in self.chunks]) if self.chunks else None
            self.card_vectors = card_embedding.vectors if card_embedding else np.zeros((0, 0), dtype=np.float32)
            self.chunk_vectors = chunk_embedding.vectors if chunk_embedding else np.zeros((0, 0), dtype=np.float32)
            self.chunk_rows_by_doc = {}
            for row, chunk in enumerate(self.chunks):
                self.chunk_rows_by_doc.setdefault(chunk.doc_id, []).append(row)
            cost = chunk_cost.plus(llm_cost)
            for embedding in (card_embedding, chunk_embedding):
                if embedding is not None:
                    cost = cost.plus(embedding.cost)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(self.chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={
                "embedding_model": self.embedding_model.model_name,
                "expensive": True,
                "summaries_generated": len(documents) - without_summary,
                "docs_without_summary": without_summary,
                "summary_failures": failures,
            },
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        if not self.card_doc_ids:
            return RetrievalResult(question=question, chunks=[], metadata={"retriever": "hierarchical", "routed_docs": []})
        with timer() as t:
            query = self.embedding_model.embed_query(question)
            vector = query.vectors[0]
            with self.trace.step("retrieve", "doc_search", docs_k=self.options.docs_k) as step:
                step.set_input(question)
                routed = self._route(question, vector)
                step.set_chunks(routed, query.cost)
            routed_docs = [card.doc_id for card in routed]
            with self.trace.step("retrieve", "chunk_search", docs=len(routed_docs), chunks_per_doc=self.options.chunks_per_doc) as step:
                step.set_input(question)
                chunks = self._chunks_within(routed_docs, vector, k)
                step.set_chunks(chunks)
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=query.cost,
            metadata={"retriever": "hierarchical", "routed_docs": routed_docs, "docs_k": self.options.docs_k, "chunks_per_doc": self.options.chunks_per_doc},
        )

    def _route(self, question: str, vector: np.ndarray) -> list[RetrievedChunk]:
        """The `docs_k` best documents: BM25 over the cards and cosine over the card embeddings, fused with RRF."""
        scores = self.card_vectors @ vector
        order = np.argsort(-scores, kind="stable")
        dense = [
            RetrievedChunk(chunk_id=f"{self.card_doc_ids[i]}::card", doc_id=self.card_doc_ids[i], text=self.card_texts[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
        lexical = self.card_index.search(question, top_k=len(self.card_doc_ids)).chunks
        return reciprocal_rank_fusion([dense, lexical], top_k=self.options.docs_k, rrf_k=self.options.rrf_k)

    def _chunks_within(self, doc_ids: list[str], vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Cosine ranking of the chunks of the chosen documents: the best `chunks_per_doc` of each, the best `k` overall."""
        picked: list[tuple[float, int, int]] = []  # (score, doc rank, chunk row)
        for doc_rank, doc_id in enumerate(doc_ids, start=1):
            rows = self.chunk_rows_by_doc.get(doc_id, [])
            if not rows:
                continue
            scores = self.chunk_vectors[rows] @ vector
            best = np.argsort(-scores, kind="stable")[: self.options.chunks_per_doc]
            picked.extend((float(scores[i]), doc_rank, rows[i]) for i in best)
        picked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            RetrievedChunk(
                chunk_id=self.chunks[row].chunk_id,
                doc_id=self.chunks[row].doc_id,
                text=self.chunks[row].text,
                score=score,
                rank=rank,
                metadata={**self.chunks[row].metadata, "routed_doc_rank": doc_rank},
            )
            for rank, (score, doc_rank, row) in enumerate(picked[:k], start=1)
        ]

    def _summarize(self, document: Document) -> tuple[str, CostBreakdown]:
        messages = [
            {"role": "system", "content": "You write short summaries that help a search system find the right document."},
            {
                "role": "user",
                "content": (
                    f"{SUMMARIZE_DOCUMENT_MARKER} In 2 to 3 sentences say what the document is about, which products, people, or topics it covers, "
                    "and what kinds of facts it contains. Answer only with the summary.\n"
                    f"Document title: {document.title}\nDocument text:\n{document.text[: self.options.summary_max_chars]}"
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, max_tokens=SUMMARY_MAX_TOKENS)
        return " ".join(result.text.split()), result.cost


def _lead(text: str) -> str:
    """The opening of a document without its title line: what stands in for a summary that could not be written."""
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return " ".join(body.split())[:LEAD_CHARS]
