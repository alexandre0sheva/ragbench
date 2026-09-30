"""Windows of whitespace-separated words (what `token` meant before 0.3.0)."""

from __future__ import annotations

import re

from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.registry import CHUNKERS

_WORD = re.compile(r"\S+")


@CHUNKERS.register("word", aliases=("tokenish",))
class WordChunker(BaseChunker):
    name = "word"
    unit = "words"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, **options):
        super().__init__(chunk_size, chunk_overlap, **options)

    def _spans(self, document: Document) -> list[Span]:
        matches = list(_WORD.finditer(document.text))
        spans: list[Span] = []
        step = self.chunk_size - self.chunk_overlap
        for first in range(0, len(matches), step):
            last = min(len(matches), first + self.chunk_size)
            spans.append(Span(matches[first].start(), matches[last - 1].end()))
            if last == len(matches):
                break
        return spans
