"""Reranking with an LLM call."""

from __future__ import annotations

import json

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM
from ragbench.models.rerankers.base import RerankResult
from ragbench.models.rerankers.tfidf import LocalRelevanceReranker
from ragbench.registry import RERANKERS


@RERANKERS.register("llm", aliases=("llm_reranker",))
class LLMReranker:
    name = "llm_reranker"
    needs_llm = True

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
                metadata["original_rank"] = chunk.rank
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
