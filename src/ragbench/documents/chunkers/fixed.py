"""Fixed-size character windows."""

from __future__ import annotations

from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.registry import CHUNKERS


@CHUNKERS.register("fixed_char", aliases=("fixed", "character"))
class FixedCharacterChunker(BaseChunker):
    name = "fixed_char"
    default_chunk_size = 1200
    default_chunk_overlap = 150
    unit = "characters"

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150, **options):
        super().__init__(chunk_size, chunk_overlap, **options)

    def _spans(self, document: Document) -> list[Span]:
        text, spans, start = document.text, [], 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            spans.append(Span(start, end))
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return spans
