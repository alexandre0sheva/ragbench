"""Free TF-IDF reranker (registered as `local_relevance`)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.rerankers.base import RerankResult
from ragbench.models.rerankers.keyword import SimpleKeywordOverlapReranker
from ragbench.registry import RERANKERS
from ragbench.utils.text import tokenize


@RERANKERS.register("local_relevance", aliases=("local", "tfidf", "tfidf_relevance"))
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
