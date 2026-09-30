"""Whole sentences packed up to the size; overlap counts sentences."""

from __future__ import annotations

from typing import Any, ClassVar

from ragbench.documents.chunkers._split import merge_small, pack, sentence_spans, split_region
from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import TokenCounter, get_tokenizer
from ragbench.registry import CHUNKERS


@CHUNKERS.register("sentence")
class SentenceChunker(BaseChunker):
    name = "sentence"
    default_chunk_overlap = 1
    options: ClassVar[frozenset[str]] = frozenset({"min_chunk_size"})
    overlap_below_size = False

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 1, **options):
        super().__init__(chunk_size, chunk_overlap, **options)

    def _common_metadata(self) -> dict[str, Any]:
        return {**super()._common_metadata(), "tokenizer": get_tokenizer().name}

    def _spans(self, document: Document) -> list[Span]:
        text = document.text
        count = TokenCounter(text)
        # A sentence longer than the limit is split by tokens instead of overflowing it.
        units = [piece for start, end in sentence_spans(text) for piece in split_region(text, start, end, self.chunk_size, count, ())]
        chunks = pack(units, self.chunk_size, self.chunk_overlap, count, unit="units")
        if self.min_chunk_size:
            chunks = merge_small(chunks, self.min_chunk_size, self.chunk_size, count)
        return [Span(start, end) for start, end in chunks]
