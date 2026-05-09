from __future__ import annotations

import math
from collections import Counter

from ragbench.documents.schema import TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import RetrievalResult, RetrievedChunk
from ragbench.utils.text import tokenize
from ragbench.utils.timing import timer


class _FallbackBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freq: Counter[str] = Counter()
        self.term_freqs = [Counter(doc) for doc in corpus]
        for doc in corpus:
            for term in set(doc):
                self.doc_freq[term] += 1
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.n = len(corpus)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for tf, doc_len in zip(self.term_freqs, self.doc_lens, strict=False):
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


class BM25Store:
    def __init__(self):
        self.chunks: list[TextChunk] = []
        self.tokenized: list[list[str]] = []
        self.index = None

    def build(self, chunks: list[TextChunk]) -> None:
        self.chunks = chunks
        self.tokenized = [tokenize(chunk.text) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.index = BM25Okapi(self.tokenized)
        except Exception:
            self.index = _FallbackBM25(self.tokenized)

    def search(self, query: str, top_k: int = 5) -> RetrievalResult:
        with timer() as t:
            query_tokens = tokenize(query)
            scores = list(self.index.get_scores(query_tokens)) if self.index else []
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
            chunks = [
                RetrievedChunk(
                    chunk_id=self.chunks[idx].chunk_id,
                    doc_id=self.chunks[idx].doc_id,
                    text=self.chunks[idx].text,
                    score=float(score),
                    rank=rank,
                    metadata=dict(self.chunks[idx].metadata),
                )
                for rank, (idx, score) in enumerate(ranked, start=1)
            ]
        return RetrievalResult(question=query, chunks=chunks, latency_ms=t.elapsed_ms, cost=CostBreakdown())

