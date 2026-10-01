from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Question(BaseModel):
    id: str
    question: str
    reference_answer: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    category: str = "unknown"
    difficulty: str = "unknown"
    answer_type: str = "unknown"
    # Registered tools (`calculator`, `date_calc`, ...; checked against the tool registry) an agent needs to answer this question. Metadata only: it never changes retrieval
    # or scoring, it lets reports show how much a tool-using system gains on the questions that need one.
    requires_tools: list[str] = Field(default_factory=list)
    # The route an `adaptive` system should send this question to (`default`, `lexical`, `computation`, `multi_hop`, or a route name of your own).
    # Metadata only: it never changes retrieval or scoring, it lets the run report the router's accuracy.
    routing_hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("requires_tools")
    @classmethod
    def _known_tools(cls, names: list[str]) -> list[str]:
        from ragbench.registry import TOOLS
        from ragbench.tools import resolve_tools  # noqa: F401  (registers the built-in tools)

        for name in names:
            TOOLS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
        return names

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

