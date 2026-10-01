"""Metrics on the context the generator was actually given, as opposed to the ranking retrieval returned.

Retrieval metrics score the whole ranking (`evaluation.retrieval_depth`); the LLM only reads the first `context_k`
chunks of it. A system can rank the evidence 7th and still starve its generator, which these metrics expose.
Relevance is judged per document, like the retrieval metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.rag_systems.base import RetrievedChunk


def context_recall_doc(chunks_given_to_llm: Sequence[RetrievedChunk], relevant_doc_ids: Sequence[str]) -> float | None:
    """Share of the relevant documents that at least one given chunk comes from; None when there is nothing to find."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None
    seen = {chunk.doc_id for chunk in chunks_given_to_llm}
    return len(relevant & seen) / len(relevant)


def context_precision(chunks: Sequence[RetrievedChunk], relevant_doc_ids: Sequence[str]) -> float | None:
    """Share of the given chunks that come from a relevant document; None without relevant documents or without chunks."""
    relevant = set(relevant_doc_ids)
    if not relevant or not chunks:
        return None
    return sum(1 for chunk in chunks if chunk.doc_id in relevant) / len(chunks)
