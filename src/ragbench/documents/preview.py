"""Chunk statistics for `ragbench chunk-preview`: how a chunker would cut a corpus, before any benchmark is paid for."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ragbench.documents.schema import TextChunk
from ragbench.documents.tokenizer import count_tokens, get_tokenizer

DEFAULT_MIN_TOKENS = 50  # "small chunk" threshold when the chunker has no `min_chunk_size`


@dataclass
class ChunkStats:
    documents: int
    chunks: int
    tokenizer: str
    min_tokens: int
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    threshold_tokens: int  # the "small chunk" threshold used for `percent_below_threshold`
    percent_below_threshold: float


def chunk_stats(chunks: list[TextChunk], documents: int, threshold_tokens: int = DEFAULT_MIN_TOKENS) -> ChunkStats:
    """Token-size statistics of `chunks` (sizes are of the chunk text, prefixes included)."""
    sizes = np.array([count_tokens(chunk.text) for chunk in chunks] or [0])
    return ChunkStats(
        documents=documents,
        chunks=len(chunks),
        tokenizer=get_tokenizer().name,
        min_tokens=int(sizes.min()),
        mean_tokens=float(sizes.mean()),
        median_tokens=float(np.median(sizes)),
        p95_tokens=float(np.percentile(sizes, 95)),
        max_tokens=int(sizes.max()),
        threshold_tokens=threshold_tokens,
        percent_below_threshold=float((sizes < threshold_tokens).mean() * 100) if chunks else 0.0,
    )


def stats_as_dict(stats: ChunkStats) -> dict[str, float | int | str]:
    return asdict(stats)
