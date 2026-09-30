"""The chunker contract. A chunker turns each document into character spans; this base class turns spans into `TextChunk`s.

Keeping that step in one place gives every chunker the same guarantees: no empty chunks, `start_char`/`end_char` that
reproduce the chunk text (`document.text[start:end].strip()`), consecutive `chunk_index`, deterministic ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ragbench.documents.schema import Document, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.utils.ids import stable_chunk_id


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    meta: dict[str, Any] = field(default_factory=dict)
    prefix: str = ""  # prepended to the chunk text (e.g. a heading breadcrumb); never part of the span


def _pages(document: Document, start: int, end: int) -> dict[str, int]:
    """`page` (and `page_end` when the chunk runs across pages) for a chunk at `[start, end)`, from `metadata["page_spans"]`."""
    spans = document.metadata.get("page_spans")
    if not spans:
        return {}
    first = next((page for _, stop, page in spans if stop > start), spans[-1][2])
    last = next((page for begin, _, page in reversed(spans) if begin < end), first)
    return {"page": first, **({"page_end": last} if last != first else {})}


class BaseChunker(ABC):
    name: ClassVar[str]
    default_chunk_size: ClassVar[int] = 500
    default_chunk_overlap: ClassVar[int] = 80
    default_min_chunk_size: ClassVar[int | None] = None
    # `ChunkerConfig` fields beyond chunk_size / chunk_overlap / prefix_title that this chunker honours
    # (`min_chunk_size`, `breakpoint_percentile`, `prefix_heading`); setting any other one is a config error.
    options: ClassVar[frozenset[str]] = frozenset()
    needs_embedder: ClassVar[bool] = False  # semantic chunking embeds sentences; `create_chunker` must be given an embedder
    unit: ClassVar[str] = "tokens"  # what chunk_size counts
    overlap_below_size: ClassVar[bool] = True  # overlap is in the same unit as the size (False when it counts sentences)

    def __init__(self, chunk_size: int, chunk_overlap: int, *, min_chunk_size: int | None = None, prefix_title: bool = False):
        if self.overlap_below_size and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size if min_chunk_size is not None else self.default_min_chunk_size
        self.prefix_title = prefix_title
        # Cost the chunker itself incurred while chunking (semantic chunking embeds every sentence); systems add it to ingestion.
        self.last_cost = CostBreakdown()

    @abstractmethod
    def _spans(self, document: Document) -> list[Span]:
        """Character spans to index for `document`, in order."""
        raise NotImplementedError

    def _common_metadata(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    def chunk(self, documents: list[Document]) -> list[TextChunk]:
        self.last_cost = CostBreakdown()
        chunks: list[TextChunk] = []
        for document in documents:
            index = 0
            for span in self._spans(document):
                raw = document.text[span.start : span.end]
                if not raw.strip():
                    continue
                chunks.append(self._make_chunk(document, raw, index, span.start, span.end, {**self._common_metadata(), **span.meta}, span.prefix))
                index += 1
        return chunks

    def _make_chunk(
        self,
        document: Document,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        extra_metadata: dict | None = None,
        prefix: str = "",
    ) -> TextChunk:
        metadata = {
            "doc_id": document.doc_id,
            "source_path": document.path,
            "title": document.title,
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "chunker": self.name,
        }
        pages = _pages(document, start_char, end_char)  # PDF page numbers, when the loader recorded them
        if pages:
            metadata.update(pages)
        if extra_metadata:
            metadata.update(extra_metadata)
        title = f"{document.title}\n\n" if self.prefix_title and document.title else ""
        return TextChunk(
            chunk_id=stable_chunk_id(document.doc_id, chunk_index, text, start_char, end_char),
            doc_id=document.doc_id,
            text=f"{title}{prefix}{text.strip()}",
            metadata=metadata,
        )
