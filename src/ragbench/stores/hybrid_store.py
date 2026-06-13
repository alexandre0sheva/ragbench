from __future__ import annotations

from ragbench.rag_systems.base import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    top_k: int = 5,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse rankings with (optionally weighted) Reciprocal Rank Fusion.

    `weights` aligns with `rankings`; a missing or shorter list defaults the
    remaining sources to weight 1.0.
    """
    by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source_idx, ranking in enumerate(rankings):
        source = f"source_{source_idx}"
        weight = weights[source_idx] if weights and source_idx < len(weights) else 1.0
        for item in ranking:
            by_id.setdefault(item.chunk_id, item)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (rrf_k + item.rank)
            source_ranks.setdefault(item.chunk_id, {})[source] = item.rank
    ordered_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:top_k]
    fused: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered_ids, start=1):
        item = by_id[chunk_id]
        metadata = dict(item.metadata)
        metadata["rrf_source_ranks"] = source_ranks.get(chunk_id, {})
        fused.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                text=item.text,
                score=scores[chunk_id],
                rank=rank,
                metadata=metadata,
            )
        )
    return fused

