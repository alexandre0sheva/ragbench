from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    question: str
    reference_answer: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    category: str = "unknown"
    difficulty: str = "unknown"
    answer_type: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_answerable(self) -> bool:
        return bool(self.relevant_doc_ids)


class Qrel(BaseModel):
    query_id: str
    doc_id: str
    relevance: int = 1


class Dataset(BaseModel):
    questions: list[Question]
    qrels: dict[str, dict[str, int]] = Field(default_factory=dict)

