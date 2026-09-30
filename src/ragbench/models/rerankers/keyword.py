"""Free keyword-overlap reranker."""

from __future__ import annotations

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.rerankers.base import RerankResult
from ragbench.registry import RERANKERS
from ragbench.utils.text import tokenize


@RERANKERS.register("simple_keyword_overlap")
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
