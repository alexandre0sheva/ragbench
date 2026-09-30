from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    path: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)



class RetrievedChunk(BaseModel):
    """A single chunk returned by a RAG system's retrieval step.

    Lives here (not in `rag_systems`) so rerankers and stores can use it without importing the systems package.
    `ragbench.rag_systems.base` re-exports it.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)
