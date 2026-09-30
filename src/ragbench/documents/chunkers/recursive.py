"""Recursive splitting: paragraphs, then lines, then sentences, then words, then a hard token split; packed up to the size."""

from __future__ import annotations

from typing import Any, ClassVar

from ragbench.documents.chunkers._split import SEPARATORS, merge_small, pack, split_region
from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import TokenCounter, get_tokenizer
from ragbench.registry import CHUNKERS


@CHUNKERS.register("recursive")
class RecursiveChunker(BaseChunker):
    name = "recursive"
    options: ClassVar[frozenset[str]] = frozenset({"min_chunk_size"})

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, **options):
        super().__init__(chunk_size, chunk_overlap, **options)

    def _common_metadata(self) -> dict[str, Any]:
        return {**super()._common_metadata(), "tokenizer": get_tokenizer().name}

    def _spans(self, document: Document) -> list[Span]:
        text = document.text
        count = TokenCounter(text)
        segments = split_region(text, 0, len(text), self.chunk_size, count, SEPARATORS)
        chunks = pack(segments, self.chunk_size, self.chunk_overlap, count)
        if self.min_chunk_size:
            chunks = merge_small(chunks, self.min_chunk_size, self.chunk_size, count)
        return [Span(start, end) for start, end in chunks]
