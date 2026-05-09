from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.text import tokenize


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    cost: CostBreakdown


class SimpleKeywordOverlapReranker:
    name = "simple_keyword_overlap"

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult:
        query_tokens = {t for t in tokenize(question) if len(t) > 2}
        rescored: list[RetrievedChunk] = []
        for chunk in chunks:
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(query_tokens.intersection(chunk_tokens))
            score = float(overlap) + (1.0 / (chunk.rank + 1000)) + (chunk.score * 0.001)
            metadata = dict(chunk.metadata)
            metadata["reranker"] = self.name
            metadata["original_rank"] = chunk.rank
            rescored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=score,
                    rank=chunk.rank,
                    metadata=metadata,
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        reranked = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=chunk.score,
                rank=rank,
                metadata=chunk.metadata,
            )
            for rank, chunk in enumerate(rescored[:top_k], start=1)
        ]
        return RerankResult(chunks=reranked, cost=CostBreakdown())


class LocalRelevanceReranker:
    name = "local_relevance"

    def __init__(self):
        self.keyword_fallback = SimpleKeywordOverlapReranker()

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult:
        if not chunks:
            return RerankResult(chunks=[], cost=CostBreakdown())
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform([question, *[chunk.text for chunk in chunks]])
            query_vec = matrix[0]
            chunk_matrix = matrix[1:]
            similarities = (chunk_matrix @ query_vec.T).toarray().ravel()
            rescored: list[RetrievedChunk] = []
            query_terms = {t for t in tokenize(question) if len(t) > 2}
            for chunk, sim in zip(chunks, similarities, strict=False):
                chunk_terms = set(tokenize(chunk.text))
                coverage = len(query_terms.intersection(chunk_terms)) / max(1, len(query_terms))
                score = float(sim) + (0.25 * coverage) + (0.05 / max(1, chunk.rank))
                metadata = dict(chunk.metadata)
                metadata["reranker"] = self.name
                metadata["original_rank"] = chunk.rank
                metadata["tfidf_similarity"] = float(sim)
                rescored.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        score=score,
                        rank=chunk.rank,
                        metadata=metadata,
                    )
                )
            order = np.argsort([chunk.score for chunk in rescored])[::-1][:top_k]
            final = []
            for rank, idx in enumerate(order, start=1):
                chunk = rescored[int(idx)]
                final.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        score=chunk.score,
                        rank=rank,
                        metadata=chunk.metadata,
                    )
                )
            return RerankResult(chunks=final, cost=CostBreakdown())
        except Exception:
            return self.keyword_fallback.rerank(question, chunks, top_k)


class LLMReranker:
    name = "llm_reranker"

    def __init__(self, llm: LLM):
        self.llm = llm
        self.fallback = LocalRelevanceReranker()

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult:
        candidate_text = "\n\n".join(f"{idx}. {chunk.chunk_id}\n{chunk.text[:700]}" for idx, chunk in enumerate(chunks, start=1))
        messages = [
            {"role": "system", "content": "Rank chunks by usefulness for answering the question. Return JSON only."},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nCandidates:\n{candidate_text}\n\n"
                    f"Return JSON like {{\"chunk_ids\": [\"id1\"]}} with up to {top_k} chunk IDs."
                ),
            },
        ]
        result = self.llm.generate(messages, json_mode=True, temperature=0)
        try:
            parsed = json.loads(result.text)
            desired = [str(x) for x in parsed.get("chunk_ids", [])]
            by_id = {chunk.chunk_id: chunk for chunk in chunks}
            ordered = [by_id[chunk_id] for chunk_id in desired if chunk_id in by_id]
            if not ordered:
                raise ValueError("No valid chunk IDs returned")
            final = []
            for rank, chunk in enumerate(ordered[:top_k], start=1):
                metadata = dict(chunk.metadata)
                metadata["reranker"] = self.name
                final.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        text=chunk.text,
                        score=float(top_k - rank + 1),
                        rank=rank,
                        metadata=metadata,
                    )
                )
            return RerankResult(chunks=final, cost=CostBreakdown(rerank_cost=result.cost.total_cost))
        except Exception:
            fallback = self.fallback.rerank(question, chunks, top_k)
            return RerankResult(chunks=fallback.chunks, cost=result.cost.plus(fallback.cost))


def create_reranker(name: str, llm: LLM | None = None):
    if name in {"llm", "llm_reranker"} and llm is not None:
        return LLMReranker(llm)
    if name in {"local", "local_relevance", "tfidf", "tfidf_relevance"}:
        return LocalRelevanceReranker()
    return SimpleKeywordOverlapReranker()
