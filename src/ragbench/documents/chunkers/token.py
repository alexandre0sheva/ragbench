"""Windows of real model tokens (tiktoken `o200k_base`; an approximation when its vocabulary cannot be loaded)."""

from __future__ import annotations

from typing import Any

from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import get_tokenizer
from ragbench.registry import CHUNKERS


@CHUNKERS.register("token")
class TokenChunker(BaseChunker):
    name = "token"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, **options):
        super().__init__(chunk_size, chunk_overlap, **options)

    def _common_metadata(self) -> dict[str, Any]:
        return {**super()._common_metadata(), "tokenizer": get_tokenizer().name}

    def _spans(self, document: Document) -> list[Span]:
        tokens = get_tokenizer().spans(document.text)
        spans: list[Span] = []
        step = self.chunk_size - self.chunk_overlap
        for first in range(0, len(tokens), step):
            last = min(len(tokens), first + self.chunk_size)
            spans.append(Span(tokens[first][0], tokens[last - 1][1]))
            if last == len(tokens):
                break
        return spans
