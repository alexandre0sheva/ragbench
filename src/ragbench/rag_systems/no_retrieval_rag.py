from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ragbench.documents.schema import Document, RetrievedChunk
from ragbench.harness.prompts import NO_RETRIEVAL_PROMPTS
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.options import NoRetrievalOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS


@SYSTEMS.register("no_retrieval")
class NoRetrievalRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="no_retrieval",
        title="No retrieval",
        summary="The model answers from its own knowledge with no documents at all (the floor baseline)",
        best_for="Showing how much retrieval helps and how often the model makes things up without it",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=NoRetrievalOptions,
        chunker=None,
        retrieves=False,
    )
    options: NoRetrievalOptions

    @classmethod
    def default_harness(cls, options: Any) -> Mapping[str, str]:
        return NO_RETRIEVAL_PROMPTS

    def ingest(self, documents: list[Document]) -> IngestionResult:
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=0, latency_ms=0.0)

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        return RetrievalResult(question=question, chunks=[], metadata={"retriever": "none"})

    def _generate_answer(self, question: str, chunks: list[RetrievedChunk]):
        messages = [{"role": "system", "content": self.harness.answer_system()}, {"role": "user", "content": f"Question: {question}"}]
        with self.trace.step("generate", "answer", context_chunks=0) as step:
            result = self.llm.generate(messages, temperature=0)
            step.set_llm(result)
            step.set_input(question)
        return result
