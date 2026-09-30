"""The reranker contract shared by every implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    cost: CostBreakdown


class Reranker(Protocol):
    """Reorders `chunks` for `question` and keeps the best `top_k`.

    Returned chunks are ranked 1..n and carry `metadata["reranker"]` and `metadata["original_rank"]` (their rank before
    reranking). Optional class attributes drive `create_reranker`: `needs_llm`, `takes_model`, `offline_fallback`,
    and `import_modules` (libraries to import on the main thread before worker threads start).
    """

    name: str

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult: ...
