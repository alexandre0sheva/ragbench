from __future__ import annotations

import math
from typing import Any

from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.text import unique_preserve_order


def dedup_doc_ranking(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    return unique_preserve_order(chunk.doc_id for chunk in retrieved_chunks)


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def compute_retrieval_metrics(
    retrieved_chunks: list[RetrievedChunk],
    relevant_doc_ids: list[str],
    qrels: dict[str, int] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    k_values = k_values or [1, 3, 5, 10]
    doc_ranking = dedup_doc_ranking(retrieved_chunks)
    relevant = set(relevant_doc_ids)
    qrels = qrels or {doc_id: 1 for doc_id in relevant}
    metrics: dict[str, float] = {"num_relevant_docs": float(len(relevant)), "num_retrieved_docs": float(len(doc_ranking))}
    if not relevant and not qrels:
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"precision@{k}"] = 0.0
            metrics[f"hit@{k}"] = 0.0
            metrics[f"mrr@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    relevant_from_qrels = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    relevant_all = relevant or relevant_from_qrels
    for k in k_values:
        top_docs = doc_ranking[:k]
        hits = [doc_id for doc_id in top_docs if doc_id in relevant_all]
        metrics[f"recall@{k}"] = len(set(hits)) / max(1, len(relevant_all))
        metrics[f"precision@{k}"] = len(hits) / k
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        rr = 0.0
        for idx, doc_id in enumerate(top_docs, start=1):
            if doc_id in relevant_all:
                rr = 1.0 / idx
                break
        metrics[f"mrr@{k}"] = rr
        gains = [int(qrels.get(doc_id, 0)) for doc_id in top_docs]
        ideal = sorted([rel for rel in qrels.values() if rel > 0], reverse=True)[:k]
        metrics[f"ndcg@{k}"] = _dcg(gains) / _dcg(ideal) if ideal and _dcg(ideal) else 0.0
    return metrics


def metrics_to_flat_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}

