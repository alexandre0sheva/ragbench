from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, create_llm
from ragbench.utils.timing import timer


class RetrievedChunk(BaseModel):
    """A single chunk returned by a RAG system's retrieval step."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Output of a single retrieval call: the chunks plus timing and cost."""

    question: str
    chunks: list[RetrievedChunk]
    latency_ms: float = 0.0
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    """End-to-end answer for a question, bundling retrieval, generation, and cost."""

    question: str
    answer: str
    retrieval_result: RetrievalResult
    model_name: str
    latency_ms: float
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    token_usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Summary of a one-time ingestion pass over a document corpus."""

    system: str
    num_documents: int
    num_chunks: int
    latency_ms: float
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    metadata: dict[str, Any] = Field(default_factory=dict)


ANSWER_SYSTEM_PROMPT = """You are answering questions using only the provided context.

Rules:
- Use only the provided context.
- If the answer is not available in the context, say: "I could not find the answer in the provided documents."
- Be concise but complete.
- Cite source document IDs in square brackets when possible, for example [doc_014].
"""


class BaseRAGSystem(ABC):
    """Abstract interface every RAG strategy implements.

    Subclasses must provide `ingest` (build whatever indexes they need from
    a corpus) and `fetch_context` (retrieve relevant chunks for a question).
    `answer_question` is supplied here and threads the retrieved context into
    the configured LLM, but subclasses can override it when they need custom
    generation (e.g. query rewriting before retrieval, multi-hop reasoning).
    """

    name: str

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        self.config = config
        self.name = config.resolved_name
        self.force_mock = force_mock
        self.llm: LLM = create_llm(config.models.get("generator", "gpt-5.4-nano"), force_mock=force_mock)

    @abstractmethod
    def ingest(self, documents: list[Document]) -> IngestionResult:
        """Build the system's indexes from the document corpus.

        Called once per benchmark run. Implementations should track ingestion
        cost on the returned `IngestionResult.cost` so the cost tracker can
        report it separately from per-question query cost.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve the top-k most relevant chunks for `question`.

        `top_k` overrides the system's configured default when provided. The
        returned `RetrievalResult` must include latency and any retrieval-side
        cost (embeddings, reranker calls, etc.).
        """
        raise NotImplementedError

    def answer_question(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Retrieve context and generate an answer in one call.

        Default implementation: `fetch_context` -> format chunks -> LLM. Override
        to add query rewriting, multi-hop, or custom prompt construction.
        """
        with timer() as total_timer:
            retrieval = self.fetch_context(question, top_k=top_k)
            llm_result = self._generate_answer(question, retrieval.chunks)
        cost = retrieval.cost.plus(llm_result.cost)
        return AnswerResult(
            question=question,
            answer=llm_result.text,
            retrieval_result=retrieval,
            model_name=llm_result.model,
            latency_ms=total_timer.elapsed_ms,
            cost=cost,
            token_usage={
                "prompt_tokens": llm_result.prompt_tokens,
                "completion_tokens": llm_result.completion_tokens,
            },
            metadata={"system_type": self.config.type},
        )

    def _generate_answer(self, question: str, chunks: list[RetrievedChunk]):
        context = self._format_context(chunks)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return self.llm.generate(messages, temperature=0)

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        formatted: list[str] = []
        for chunk in chunks:
            short_id = chunk.chunk_id.split("::chunk::")[-1]
            formatted.append(f"[{chunk.doc_id} | {short_id}]\n{chunk.text}")
        return "\n\n".join(formatted)
