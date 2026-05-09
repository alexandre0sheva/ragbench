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

