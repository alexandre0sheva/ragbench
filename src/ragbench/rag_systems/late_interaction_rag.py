from __future__ import annotations

import threading
from typing import ClassVar

import numpy as np

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.models.colbert import load_token_encoder, maxsim
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import IngestionResult, RetrievalResult
from ragbench.rag_systems.components import chunk_documents
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.options import LateInteractionOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.utils.timing import timer

SCORE_BLOCK_ROWS = 200_000  # token vectors scored at once in `full` mode (float32 working copy of the fp16 index)


@SYSTEMS.register("late_interaction")
class LateInteractionRAG(HybridRAG):
    """ColBERT-style late interaction: every token gets its own vector, and a chunk scores the sum over query tokens of its best-matching token.

    `rerank` reorders the first stage's candidates (hybrid or BM25) by MaxSim, encoding each candidate the first time it is seen. `full` encodes
    every chunk at ingestion into one fp16 matrix and scores all of them exactly. The model runs locally ($0 API cost), loads once per run and is
    shared by every system naming it; mock runs use deterministic hashing token vectors and load nothing.
    """

    spec = SystemSpec(
        type="late_interaction",
        title="Late interaction (ColBERT MaxSim)",
        summary="Per-token embeddings scored by MaxSim (ColBERT): reranks hybrid or BM25 candidates, or searches every chunk exactly",
        best_for="Questions whose wording must match precisely (names, terms of art) where one pooled vector per chunk blurs the detail",
        cost_profile="low",
        latency_profile="medium",
        requires_llm=False,
        agentic=False,
        options=LateInteractionOptions,
    )
    options: LateInteractionOptions
    import_modules: ClassVar[tuple[str, ...]] = ("torch", "transformers")  # imported on the main thread before a live run starts its workers

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.encoder = load_token_encoder(self.options.model, force_mock=force_mock)
        self.chunks: list[TextChunk] = []
        self.index = np.zeros((0, 0), dtype=np.float16)  # `full`: every kept token vector of every chunk, chunk after chunk
        self.starts = np.zeros(0, dtype=np.int64)  # `full`: first row of each chunk in `index`
        self.lengths = np.zeros(0, dtype=np.int64)
        self._vectors: dict[str, np.ndarray] = {}  # `rerank`: candidates encoded so far
        self._lock = threading.Lock()

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            self.chunks, cost = chunk_documents(self.chunker, documents)
            index_vectors = 0
            if self.options.mode == "full":
                index_vectors = self._build_index()
            else:
                self.bm25_store.build(self.chunks)
                if self.options.first_stage == "hybrid":
                    cost = cost.plus(self.vector_store.build(self.chunks))
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(self.chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"token_model": self.encoder.name, "mode": self.options.mode, "index_vectors": index_vectors},
        )

    def _build_index(self) -> int:
        prepared = self.encoder.prepare_documents([chunk.text for chunk in self.chunks])  # tokenised once, for the guard and the encoding
        total = sum(self.encoder.vector_count(item) for item in prepared)
        if total > self.options.max_index_vectors:
            raise ValueError(
                f"late_interaction with mode: full would hold {total:,} token vectors for {len(prepared):,} chunks, above max_index_vectors "
                f"({self.options.max_index_vectors:,}); use mode: rerank (MaxSim over first-stage candidates only), or raise max_index_vectors"
            )
        vectors = self.encoder.encode_prepared(prepared)
        self.lengths = np.array([len(v) for v in vectors], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64) if len(vectors) else np.zeros(0, dtype=np.int64)
        nonempty = [v for v in vectors if len(v)]
        self.index = np.vstack(nonempty).astype(np.float16) if nonempty else np.zeros((0, 0), dtype=np.float16)
        return int(self.lengths.sum())

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        with timer() as t:
            if self.options.mode == "full":
                with self.trace.step("retrieve", "maxsim_search", chunks=len(self.chunks)) as step:
                    step.set_input(question)
                    chunks = self._search_all(question, k)
                    step.set_chunks(chunks)
                cost = CostBreakdown()
            else:
                candidates, cost = self._first_stage(question, max(self.options.candidate_k, k))
                with self.trace.step("rerank", "maxsim", candidates=len(candidates)) as step:
                    step.set_input(question)
                    chunks = self._rerank(question, candidates, k)
                    step.set_chunks(chunks)
        return RetrievalResult(question=question, chunks=chunks, latency_ms=t.elapsed_ms, cost=cost, metadata={"retriever": f"late_interaction_{self.options.mode}", "token_model": self.encoder.name})

    def _first_stage(self, question: str, depth: int) -> tuple[list[RetrievedChunk], CostBreakdown]:
        if self.options.first_stage == "hybrid":
            result = super().fetch_context(question, top_k=depth)
            return result.chunks, result.cost
        with self.trace.step("retrieve", "bm25_search", top_k=depth) as step:
            step.set_input(question)
            result = self.bm25_store.search(question, top_k=depth)
            step.set_chunks(result.chunks, result.cost)
        return result.chunks, result.cost

    def _rerank(self, question: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        query = self.encoder.encode_query(question)
        missing = [chunk for chunk in candidates if chunk.chunk_id not in self._vectors]
        if missing:
            encoded = self.encoder.encode_documents([chunk.text for chunk in missing])
            with self._lock:
                self._vectors.update({chunk.chunk_id: vectors for chunk, vectors in zip(missing, encoded, strict=True)})
        scored = sorted(((maxsim(query, self._vectors[c.chunk_id]), c.rank, c) for c in candidates), key=lambda item: (-item[0], item[1]))
        return [
            chunk.model_copy(update={"score": score, "rank": rank, "metadata": {**chunk.metadata, "first_stage_rank": first_rank}})
            for rank, (score, first_rank, chunk) in enumerate(scored[:k], start=1)
        ]

    def _search_all(self, question: str, k: int) -> list[RetrievedChunk]:
        """Exact MaxSim of every chunk, scored in blocks of rows so the float32 working copy stays small."""
        query = self.encoder.encode_query(question).astype(np.float32)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        filled = np.flatnonzero(self.lengths)
        block: list[int] = []
        rows = 0
        for position, chunk_row in enumerate(filled):
            block.append(int(chunk_row))
            rows += int(self.lengths[chunk_row])
            if rows >= SCORE_BLOCK_ROWS or position == len(filled) - 1:
                first, last = self.starts[block[0]], self.starts[block[-1]] + self.lengths[block[-1]]
                similarity = self.index[first:last].astype(np.float32) @ query.T
                best = np.maximum.reduceat(similarity, self.starts[block] - first, axis=0)
                scores[block] = best.sum(axis=1)
                block, rows = [], 0
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            RetrievedChunk(chunk_id=self.chunks[i].chunk_id, doc_id=self.chunks[i].doc_id, text=self.chunks[i].text, score=float(scores[i]), rank=rank, metadata=dict(self.chunks[i].metadata))
            for rank, i in enumerate(order, start=1)
        ]
