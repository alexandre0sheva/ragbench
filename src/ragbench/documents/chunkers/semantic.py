"""Semantic chunking: embed the sentences and start a new chunk where the meaning moves most.

Consecutive sentences are compared by cosine distance; a chunk boundary goes after every sentence whose distance to the next
exceeds the `breakpoint_percentile` of the document's distances. Groups are then packed up to `chunk_size`. Embedding every
sentence is paid ingestion work: it is reported in `last_cost` and charged to the system's ingestion cost.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ragbench.documents.chunkers._split import merge_small, pack, sentence_spans, split_region
from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import TokenCounter, get_tokenizer
from ragbench.models.embeddings import EmbeddingModel
from ragbench.registry import CHUNKERS

MIN_SENTENCES_TO_SPLIT = 3


@CHUNKERS.register("semantic")
class SemanticChunker(BaseChunker):
    name = "semantic"
    default_chunk_overlap = 0
    default_breakpoint_percentile = 90.0
    options: ClassVar[frozenset[str]] = frozenset({"min_chunk_size", "breakpoint_percentile"})
    needs_embedder = True
    overlap_below_size = False

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        *,
        embedder: EmbeddingModel,
        breakpoint_percentile: float | None = None,
        **options: Any,
    ):
        super().__init__(chunk_size, chunk_overlap, **options)
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile if breakpoint_percentile is not None else self.default_breakpoint_percentile

    def _common_metadata(self) -> dict[str, Any]:
        return {**super()._common_metadata(), "tokenizer": get_tokenizer().name, "breakpoint_percentile": self.breakpoint_percentile}

    def _spans(self, document: Document) -> list[Span]:
        text = document.text
        count = TokenCounter(text)
        sentences = sentence_spans(text)
        groups = self._groups(text, sentences)
        chunks: list[tuple[int, int]] = []
        for group in groups:
            units = [piece for start, end in group for piece in split_region(text, start, end, self.chunk_size, count, ())]
            chunks.extend(pack(units, self.chunk_size, self.chunk_overlap, count, unit="units"))
        if self.min_chunk_size:
            chunks = merge_small(chunks, self.min_chunk_size, self.chunk_size, count)
        return [Span(start, end) for start, end in chunks]

    def _groups(self, text: str, sentences: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        if len(sentences) < MIN_SENTENCES_TO_SPLIT:
            return [sentences] if sentences else []
        result = self.embedder.embed_texts([text[start:end].strip() or " " for start, end in sentences])
        self.last_cost = self.last_cost.plus(result.cost)
        vectors = np.asarray(result.vectors, dtype=np.float64)
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        distances = 1.0 - np.einsum("ij,ij->i", unit[:-1], unit[1:])
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        groups: list[list[tuple[int, int]]] = [[sentences[0]]]
        for sentence, distance in zip(sentences[1:], distances, strict=True):
            if distance > threshold:
                groups.append([])
            groups[-1].append(sentence)
        return groups
