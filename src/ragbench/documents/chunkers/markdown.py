"""Markdown-aware chunking: one chunk per section, with the heading path in the metadata.

Sections are cut at headings (`#` to `######`, ignoring `#` lines inside code fences) and carry their breadcrumb
(`["Guide", "Returns"]`) as `metadata["heading_path"]`. Sections smaller than `min_chunk_size` tokens are merged with the
section after them while the result fits `chunk_size`; a section larger than `chunk_size` is split recursively and keeps
its path. With `prefix_heading` the breadcrumb is also prepended to the chunk text, so it is embedded and searched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ragbench.documents.chunkers._split import SEPARATORS, pack, split_region
from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import TokenCounter, get_tokenizer
from ragbench.registry import CHUNKERS

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = ("```", "~~~")


@dataclass
class _Section:
    start: int
    end: int
    path: list[str]
    headings: list[str] = field(default_factory=list)


def markdown_sections(text: str) -> list[_Section]:
    """Sections that tile `text`: any text above the first heading, then one section per heading."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(_FENCE):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line.rstrip("\r\n"))
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if sections:
                sections[-1].end = offset
            sections.append(_Section(offset, len(text), [name for _, name in stack], [title]))
        elif not sections:
            sections.append(_Section(0, len(text), []))  # text above the first heading
        offset += len(line)
    return sections


@CHUNKERS.register("markdown", aliases=("md",))
class MarkdownAwareChunker(BaseChunker):
    name = "markdown"
    default_min_chunk_size = 50
    options: ClassVar[frozenset[str]] = frozenset({"min_chunk_size", "prefix_heading"})

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, *, prefix_heading: bool | None = None, **options: Any):
        super().__init__(chunk_size, chunk_overlap, **options)
        self.prefix_heading = bool(prefix_heading)

    def _common_metadata(self) -> dict[str, Any]:
        return {**super()._common_metadata(), "tokenizer": get_tokenizer().name}

    def _spans(self, document: Document) -> list[Span]:
        text = document.text
        count = TokenCounter(text)
        spans: list[Span] = []
        for index, section in enumerate(self._merge(markdown_sections(text), count)):
            meta: dict[str, Any] = {"heading_path": list(section.path), "section_index": index}
            if len(section.headings) > 1:
                meta["merged_headings"] = list(section.headings)
            prefix = " > ".join(section.path) + "\n\n" if self.prefix_heading and section.path else ""
            if count(section.start, section.end) <= self.chunk_size:
                spans.append(Span(section.start, section.end, meta, prefix))
                continue
            segments = split_region(text, section.start, section.end, self.chunk_size, count, SEPARATORS)
            spans.extend(Span(start, end, meta, prefix) for start, end in pack(segments, self.chunk_size, self.chunk_overlap, count))
        return spans

    def _merge(self, sections: list[_Section], count: TokenCounter) -> list[_Section]:
        """Fold a section smaller than `min_chunk_size` into the one after it (or, for the last one, the one before) while it fits."""
        minimum = self.min_chunk_size or 0
        merged: list[_Section] = []
        for section in sections:
            previous = merged[-1] if merged else None
            if previous and count(previous.start, previous.end) < minimum and count(previous.start, section.end) <= self.chunk_size:
                previous.end = section.end
                previous.headings += section.headings
            else:
                merged.append(section)
        if len(merged) > 1 and count(merged[-1].start, merged[-1].end) < minimum and count(merged[-2].start, merged[-1].end) <= self.chunk_size:
            merged[-2].end = merged[-1].end
            merged[-2].headings += merged[-1].headings
            merged.pop()
        return merged
