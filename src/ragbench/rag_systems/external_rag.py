"""What `external_http` and `python_callable` share: a system that runs outside RAGBench and hands back an answer, its passages and its cost.

RAGBench cannot see inside it, so the whole answer is one `external` step, unless the system records its own steps (a callable that
traces on RAGBench's active tracer, or returns steps; this is how framework adapters split retrieval from generation). Its passages are mapped back to corpus documents so that
retrieval can be scored; a question whose passages do not all map gets blank retrieval metrics (never zeros). Its cost is what it
reports, else its token usage at the declared model's price, else unknown ($0, and the run says so).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

from ragbench.documents.schema import Document, RetrievedChunk
from ragbench.models import cost as pricing
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.options import ExternalOptions
from ragbench.rag_systems.trace import PREVIEW_CHARS, Step, Tracer, activate, reconcile_steps, sum_costs
from ragbench.utils.text import truncate
from ragbench.utils.timing import timer

NO_ID = "(no id)"


@dataclass
class ExternalReply:
    """One answer from an external system. `contexts` is None when the system exposes no passages at all."""

    answer: str
    contexts: list[dict[str, Any]] | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    cost: CostBreakdown | None = None  # a full breakdown (an `AnswerResult`'s), which wins over `cost_usd` and the tokens
    steps: list[Step] = field(default_factory=list)  # the system's own trace of this answer, when it reports one
    agent: dict[str, Any] | None = None  # an agentic external system's loop summary (steps, llm_calls, budget_exhausted, ...)


def unscored_retrieval(metadata: dict[str, Any]) -> bool:
    """Whether an answer's retrieval cannot be scored: an external system that showed no passages, or passages that name no corpus document."""
    external = metadata.get("external")
    return bool(external and (external.get("no_contexts") or external.get("unmapped_doc_ids")))


class ExternalRAG(BaseRAGSystem):
    options: ExternalOptions

    @classmethod
    def default_harness(cls, options: Any) -> Mapping[str, str]:
        """Its prompts live outside RAGBench."""
        return {}

    def __init__(self, config, force_mock: bool = False):
        super().__init__(config, force_mock=True)  # RAGBench calls no model for an external system, so it gets the inert mock, never a real client
        self.force_mock = force_mock
        self._aliases: dict[str, str] = {}

    @abstractmethod
    def _call(self, question: str, top_k: int) -> ExternalReply:
        """Ask the external system one question."""

    def _ingest_external(self, documents: list[Document]) -> None:
        """Hand the corpus to the external system, when it takes one."""

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as elapsed:
            self._aliases = _document_aliases(documents)
            self._ingest_external(documents)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=0, latency_ms=elapsed.elapsed_ms, metadata={"external": True})

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        with timer() as elapsed:
            reply = self._call(question, self.options.resolve_top_k(top_k))
        chunks, _ = self._chunks(reply)
        cost, _ = self._cost(reply)
        return RetrievalResult(question=question, chunks=chunks, latency_ms=elapsed.elapsed_ms, cost=cost, metadata={"retriever": self.config.type})

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None) -> AnswerResult:
        """One call to the external system; `context_k` does not apply (the system decides what its generator read, and reports it)."""
        tracer = Tracer()
        with activate(tracer), timer() as total:
            reply = self._call(question, self.options.resolve_top_k(top_k))
        # Steps the system recorded on the active tracer win over steps it returned; with neither, the answer is one `external` step.
        own = list(tracer.steps) or list(reply.steps)
        cost, cost_known = self._cost(reply, own)
        prompt = reply.prompt_tokens if reply.prompt_tokens is not None else sum(step.prompt_tokens for step in own)
        completion = reply.completion_tokens if reply.completion_tokens is not None else sum(step.completion_tokens for step in own)
        steps = own or [
            Step(
                kind="external",
                name=self.config.type,
                latency_ms=total.elapsed_ms,
                cost=cost,
                prompt_tokens=prompt,
                completion_tokens=completion,
                input_preview=truncate(question, PREVIEW_CHARS),
                output_preview=truncate(reply.answer, PREVIEW_CHARS),
                metadata={"passages": len(reply.contexts or [])},
            )
        ]
        chunks, unmapped = self._chunks(reply)
        external: dict[str, Any] = {"unmapped_doc_ids": unmapped, "cost_known": cost_known}
        if reply.contexts is None:
            external["no_contexts"] = True
        metadata: dict[str, Any] = {"system_type": self.config.type, "context_chunk_ids": [chunk.chunk_id for chunk in chunks], "external": external}
        if reply.agent is not None:
            metadata["agent"] = reply.agent
        return AnswerResult(
            question=question,
            answer=reply.answer,
            retrieval_result=RetrievalResult(
                question=question,
                chunks=chunks,
                latency_ms=total.elapsed_ms,
                metadata={"retriever": self.config.type},
                steps=[step for step in own if step.kind != "generate"],
            ),
            model_name=self.options.model or self.config.type,
            latency_ms=total.elapsed_ms,
            cost=cost,
            token_usage={"prompt_tokens": prompt, "completion_tokens": completion},
            metadata=metadata,
            steps=reconcile_steps(steps, cost),
        )

    def _chunks(self, reply: ExternalReply) -> tuple[list[RetrievedChunk], list[str]]:
        """The reply's passages as chunks of corpus documents, and the ids that named no document (`(no id)` for a passage without one)."""
        chunks: list[RetrievedChunk] = []
        unmapped: list[str] = []
        for rank, context in enumerate(reply.contexts or [], start=1):
            raw = context.get("doc_id")
            doc_id = self._resolve(str(raw)) if raw is not None else None
            if doc_id is None:
                unmapped.append(NO_ID if raw is None else str(raw))
            score = context.get("score")
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{doc_id or 'external'}::external::{rank}",
                    doc_id=doc_id or "",
                    text=str(context.get("text") or ""),
                    score=float(score) if isinstance(score, int | float) else 1.0 / rank,
                    rank=rank,
                    metadata={} if raw is None or str(raw) == doc_id else {"external_doc_id": str(raw)},
                )
            )
        return chunks, unmapped

    def _resolve(self, raw: str) -> str | None:
        """The corpus document an external id names: the id itself, else its path, file name or stem (`docs/doc_001.md` -> `doc_001`)."""
        path = PurePath(raw)
        return next((self._aliases[name] for name in (raw, path.as_posix(), path.name, path.stem) if name in self._aliases), None)

    def _cost(self, reply: ExternalReply, steps: list[Step] | None = None) -> tuple[CostBreakdown, bool]:
        """What the answer cost, and whether that is known: a reported breakdown, else the reported `cost_usd`, else what its own steps
        cost, else its token usage at the declared model's price."""
        steps = steps or []
        if reply.cost is not None:
            return reply.cost, True
        if reply.cost_usd is None and any(step.cost.total_cost > 0 for step in steps):
            return sum_costs(step.cost for step in steps), True
        prompt = reply.prompt_tokens if reply.prompt_tokens is not None else sum(step.prompt_tokens for step in steps)
        completion = reply.completion_tokens if reply.completion_tokens is not None else sum(step.completion_tokens for step in steps)
        model = self.options.model
        if reply.cost_usd is not None:
            usd, known = float(reply.cost_usd), True
        elif (prompt or completion) and model and pricing.price_known(model):
            usd, known = pricing.estimate_model_cost(model, prompt, completion), True
        else:
            usd, known = 0.0, False
        return CostBreakdown(llm_prompt_tokens=prompt, llm_completion_tokens=completion, llm_cost=usd), known


def _document_aliases(documents: list[Document]) -> dict[str, str]:
    """Every name an external system may use for a document (its id, path, file name, file stem) -> the document id; ambiguous names are left out."""
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for document in documents:
        path = PurePath(document.path)
        for alias in {path.as_posix(), path.name, path.stem}:
            if aliases.get(alias, document.doc_id) != document.doc_id:
                ambiguous.add(alias)
            aliases[alias] = document.doc_id
    for alias in ambiguous:
        del aliases[alias]
    aliases.update({document.doc_id: document.doc_id for document in documents})  # an id always names its own document
    return aliases


def as_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
