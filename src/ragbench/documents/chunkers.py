from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ragbench.documents.schema import Document, TextChunk
from ragbench.utils.ids import stable_chunk_id


class BaseChunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[TextChunk]:
        raise NotImplementedError

    def _make_chunk(
        self,
        document: Document,
        text: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        extra_metadata: dict | None = None,
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
        if extra_metadata:
            metadata.update(extra_metadata)
        return TextChunk(
            chunk_id=stable_chunk_id(document.doc_id, chunk_index, text, start_char, end_char),
            doc_id=document.doc_id,
            text=text.strip(),
            metadata=metadata,
        )


class FixedCharacterChunker(BaseChunker):
    name = "fixed_char"

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for document in documents:
            text = document.text
            start = 0
            index = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append(
                        self._make_chunk(
                            document,
                            chunk_text,
                            index,
                            start,
                            end,
                            {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
                        )
                    )
                    index += 1
                if end == len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks


class TokenChunker(BaseChunker):
    name = "token"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, documents: list[Document]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        token_re = re.compile(r"\S+")
        for document in documents:
            matches = list(token_re.finditer(document.text))
            if not matches:
                continue
            step = self.chunk_size - self.chunk_overlap
            index = 0
            for token_start in range(0, len(matches), step):
                token_end = min(len(matches), token_start + self.chunk_size)
                start_char = matches[token_start].start()
                end_char = matches[token_end - 1].end()
                chunk_text = document.text[start_char:end_char]
                chunks.append(
                    self._make_chunk(
                        document,
                        chunk_text,
                        index,
                        start_char,
                        end_char,
                        {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
                    )
                )
                index += 1
                if token_end == len(matches):
                    break
        return chunks


class MarkdownAwareChunker(BaseChunker):
    name = "markdown"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._token_chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, documents: list[Document]) -> list[TextChunk]:
        sectioned_docs: list[Document] = []
        for document in documents:
            sections = self._split_sections(document.text)
            if not sections:
                sectioned_docs.append(document)
                continue
            for idx, (heading, start, end, text) in enumerate(sections):
                sectioned_docs.append(
                    Document(
                        doc_id=document.doc_id,
                        path=document.path,
                        title=heading or document.title,
                        text=text,
                        metadata={**document.metadata, "section_index": idx, "section_start_char": start, "section_end_char": end},
                    )
                )
        chunks = self._token_chunker.chunk(sectioned_docs)
        fixed: list[TextChunk] = []
        for i, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["chunker"] = self.name
            fixed.append(
                TextChunk(
                    chunk_id=stable_chunk_id(chunk.doc_id, i, chunk.text, meta.get("start_char", 0), meta.get("end_char", 0)),
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    metadata=meta,
                )
            )
        return fixed

    @staticmethod
    def _split_sections(text: str) -> list[tuple[str, int, int, str]]:
        headings = list(re.finditer(r"(?m)^#{1,4}\s+(.+)$", text))
        if not headings:
            return []
        sections: list[tuple[str, int, int, str]] = []
        for idx, match in enumerate(headings):
            start = match.start()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
            sections.append((match.group(1).strip(), start, end, text[start:end]))
        return sections


def create_chunker(config: dict | None = None) -> BaseChunker:
    cfg = config or {}
    chunker_type = cfg.get("type", "token")
    if chunker_type in {"fixed", "fixed_char", "character"}:
        return FixedCharacterChunker(chunk_size=int(cfg.get("chunk_size", 1200)), chunk_overlap=int(cfg.get("chunk_overlap", 150)))
    if chunker_type in {"markdown", "md"}:
        return MarkdownAwareChunker(chunk_size=int(cfg.get("chunk_size", 500)), chunk_overlap=int(cfg.get("chunk_overlap", 80)))
    if chunker_type in {"token", "tokenish", "word"}:
        return TokenChunker(chunk_size=int(cfg.get("chunk_size", 500)), chunk_overlap=int(cfg.get("chunk_overlap", 80)))
    raise ValueError(f"Unknown chunker type: {chunker_type}")

