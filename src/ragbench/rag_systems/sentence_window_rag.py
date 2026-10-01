from __future__ import annotations

from dataclasses import dataclass

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers._split import sentence_spans
from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_embedder, build_vector_index
from ragbench.rag_systems.options import SentenceWindowOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.utils.ids import stable_chunk_id
from ragbench.utils.timing import timer


@dataclass
class _Window:
    doc_id: str
    first: int  # sentence index range, inclusive
    last: int
    score: float
    matched: int = 1


@SYSTEMS.register("sentence_window")
class SentenceWindowRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="sentence_window",
        title="Sentence window",
        summary="Search single sentences, then hand the generator each matched sentence with the sentences around it",
        best_for="Precise matching with enough surrounding text to answer from",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=SentenceWindowOptions,
        chunker=None,
    )
    options: SentenceWindowOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.store = build_vector_index(self.embedding_model, self.options, self.name)
        self.documents: dict[str, Document] = {}
        self.spans: dict[str, list[tuple[int, int]]] = {}  # per document: character span of each non-blank sentence
        self.sentence_of: dict[str, tuple[str, int]] = {}  # chunk_id -> (doc_id, sentence index)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            sentences: list[TextChunk] = []
            for document in documents:
                spans = [(start, end) for start, end in sentence_spans(document.text) if document.text[start:end].strip()]
                self.documents[document.doc_id] = document
                self.spans[document.doc_id] = spans
                for index, (start, end) in enumerate(spans):
                    text = document.text[start:end].strip()
                    chunk_id = stable_chunk_id(document.doc_id, len(sentences), text, start, end)
                    self.sentence_of[chunk_id] = (document.doc_id, index)
                    sentences.append(TextChunk(chunk_id=chunk_id, doc_id=document.doc_id, text=text, metadata={"sentence_index": index, "title": document.title}))
            cost = self.store.build(sentences)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(sentences),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"embedding_model": self.embedding_model.model_name, "unit": "sentence", "window": self.options.window},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        pool = max(self.options.candidate_top_k, 4 * k)
        with self.trace.step("retrieve", "sentence_search", top_k=pool) as step:
            step.set_input(question)
            hits = self.store.search(question, top_k=pool)
            step.set_chunks(hits.chunks, hits.cost)
        with timer() as t:
            windows = self._windows(hits.chunks, k)
            chunks = [self._chunk(window, rank) for rank, window in enumerate(windows, start=1)]
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=hits.latency_ms + t.elapsed_ms,
            cost=hits.cost,
            metadata={"retriever": "sentence_window", "window": self.options.window, "sentences_searched": len(hits.chunks)},
        )

    def _windows(self, hits: list[RetrievedChunk], k: int) -> list[_Window]:
        """Windows around the `k` best sentences, best first; windows of one document that overlap are merged into one.

        A sentence already inside a better hit's window is not a new hit (it would only repeat that text), so the `k`
        seeds are `k` distinct places in the corpus. Lower-ranked sentences never widen a window.
        """
        seeds: list[_Window] = []
        for hit in hits:
            doc_id, index = self.sentence_of[hit.chunk_id]
            if any(w.doc_id == doc_id and w.first <= index <= w.last for w in seeds):
                continue
            last_sentence = len(self.spans[doc_id]) - 1
            seeds.append(_Window(doc_id, max(0, index - self.options.window), min(last_sentence, index + self.options.window), hit.score))
            if len(seeds) == k:
                break
        merged: list[_Window] = []
        for seed in seeds:
            touching = [w for w in merged if w.doc_id == seed.doc_id and w.first <= seed.last and seed.first <= w.last]
            if not touching:
                merged.append(seed)
                continue
            target = touching[0]
            for other in [*touching[1:], seed]:
                target.first, target.last = min(target.first, other.first), max(target.last, other.last)
                target.matched += other.matched
            for other in touching[1:]:
                merged.remove(other)
        return merged

    def _chunk(self, window: _Window, rank: int) -> RetrievedChunk:
        document = self.documents[window.doc_id]
        spans = self.spans[window.doc_id]
        start, end = spans[window.first][0], spans[window.last][1]
        text = document.text[start:end].strip()
        end = start + len(document.text[start:end].rstrip())
        return RetrievedChunk(
            chunk_id=stable_chunk_id(window.doc_id, rank - 1, text, start, end),
            doc_id=window.doc_id,
            text=text,
            score=window.score,
            rank=rank,
            metadata={
                "title": document.title,
                "start_char": start,
                "end_char": end,
                "sentence_range": [window.first, window.last],
                "matched_sentences": window.matched,
                "window": self.options.window,
            },
        )
