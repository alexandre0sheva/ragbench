"""Maximal marginal relevance: pick chunks that are relevant to the question and different from the ones already picked."""

from __future__ import annotations

import numpy as np

from ragbench.documents.schema import RetrievedChunk


def mmr_select(vectors: np.ndarray, relevance: np.ndarray, k: int, lam: float) -> list[int]:
    """Indices of up to `k` rows, chosen greedily by `lam * relevance - (1 - lam) * (max cosine similarity to a row already chosen)`.

    `vectors` are unit length, so a dot product is a cosine. At `lam == 1.0` the diversity term vanishes and the order is the
    plain ranking by `relevance` (ties keep the original order).
    """
    n = len(relevance)
    chosen: list[int] = []
    remaining = list(range(n))
    closest = np.zeros(n)  # each row's highest similarity to anything chosen so far (0 before the first pick: no penalty)
    while remaining and len(chosen) < k:
        scores = lam * relevance[remaining] - (1.0 - lam) * closest[remaining]
        best = remaining[int(np.argmax(scores))]  # argmax returns the first maximum, so ties keep the plain order
        chosen.append(best)
        remaining.remove(best)
        if remaining:
            closest[remaining] = np.maximum(closest[remaining], vectors[remaining] @ vectors[best])
    return chosen


def mmr_rerank(chunks: list[RetrievedChunk], vectors: np.ndarray, lam: float, k: int) -> list[RetrievedChunk]:
    """The `k` best chunks after MMR, re-ranked 1..k. Scores stay the original retrieval scores; metadata records where each chunk came from."""
    if not chunks:
        return []
    scores = np.array([chunk.score for chunk in chunks], dtype=np.float64)
    top = scores.max()
    relevance = scores / top if top > 0 else scores
    picked = mmr_select(vectors, relevance, k, lam)
    return [
        chunk.model_copy(update={"rank": rank, "metadata": {**chunk.metadata, "mmr_original_rank": chunk.rank}})
        for rank, chunk in enumerate((chunks[i] for i in picked), start=1)
    ]
