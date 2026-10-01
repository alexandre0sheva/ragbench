"""Shared plumbing of the tool-using agents: build the tools, run a `ToolAgent` for a question, and turn its run into retrieval and answer results."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ragbench.agents import AgentBudget, llm_calls
from ragbench.agents.tool_agent import ToolAgent, ToolAgentRun
from ragbench.config.schema import SystemConfig, ToolRef
from ragbench.documents.schema import RetrievedChunk
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem, RetrievalResult
from ragbench.rag_systems.options import ToolAgentOptions
from ragbench.rag_systems.trace import Tracer, activate, reconcile_steps, sum_costs
from ragbench.runtime.context import current_runtime
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.tools import CorpusView, ToolBox, ToolContext
from ragbench.tools.base import Retriever
from ragbench.utils.timing import timer


class ToolAgentRAG(BaseRAGSystem):
    """A system whose answer is written by an agent that calls tools.

    The agent's passages (search hits, grep matches, documents read) are what retrieval is scored on: each evidence-producing tool
    call is one ranking, and the rankings are merged with RRF. Every model call and tool call is a trace step; the model call that
    gave the final answer (or the explicit best-effort request when a budget ended the loop) is the `generate` step.
    """

    options: ToolAgentOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.box = ToolBox.from_refs(self.tool_refs(config), current_runtime().tools)
        self.corpus = CorpusView([])

    @classmethod
    @abstractmethod
    def tool_refs(cls, config: SystemConfig) -> list[ToolRef]:
        """The tools this system gives its agent under `config`."""

    @classmethod
    def offered_tools(cls, config: SystemConfig) -> list[str]:
        return [ref if isinstance(ref, str) else str(ref["name"]) for ref in cls.tool_refs(config)]

    def _retrievers(self) -> dict[str, Retriever]:
        return {}

    def _instructions(self) -> str:
        return ""

    def _run_agent(self, question: str, tracer: Tracer) -> ToolAgentRun:
        opts = self.options
        budget = AgentBudget(max_steps=opts.max_steps, max_cost_usd=opts.max_cost_usd, max_tokens=opts.max_tokens, max_tool_calls=opts.max_tool_calls)
        ctx = ToolContext.for_question(self.corpus, self._retrievers())  # a fresh context per question: concurrent questions share nothing
        return ToolAgent(self.llm, self.box, budget, mode=opts.agent_mode, instructions=self._instructions()).run(question, ctx, tracer)

    def _retrieval(self, question: str, run: ToolAgentRun, tracer: Tracer, first_step: int, depth: int) -> RetrievalResult:
        chunks: list[RetrievedChunk] = reciprocal_rank_fusion(run.rankings, top_k=depth, rrf_k=60) if run.rankings else []
        steps = [step for step in tracer.steps[first_step:] if step.kind != "generate"]
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=sum(step.latency_ms for step in steps),
            cost=sum_costs(step.cost for step in steps),
            metadata={"retriever": self.config.type, "agent": self._agent_metadata(run, tracer.steps[first_step:])},
            steps=steps,
        )

    @staticmethod
    def _agent_metadata(run: ToolAgentRun, steps: list) -> dict[str, Any]:
        return {**run.state.metadata(), "llm_calls": llm_calls(steps)}

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        with self._recording() as tracer:
            first_step = len(tracer.steps)
            run = self._run_agent(question, tracer)
            return self._retrieval(question, run, tracer, first_step, self.options.resolve_top_k(top_k))

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None) -> AnswerResult:
        tracer = Tracer()
        with activate(tracer), timer() as total:
            run = self._run_agent(question, tracer)
        retrieval = self._retrieval(question, run, tracer, 0, self.options.resolve_top_k(top_k))
        limit = context_k if context_k is not None else self.configured_context_k()
        context_chunks = retrieval.chunks[:limit] if limit is not None else retrieval.chunks
        cost = sum_costs(step.cost for step in tracer.steps)
        return AnswerResult(
            question=question,
            answer=run.answer,
            retrieval_result=retrieval,
            model_name=run.model or self.llm.model_name,
            latency_ms=total.elapsed_ms + tracer.latency_credit_ms,
            cost=cost,
            token_usage={"prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens},
            metadata={
                "system_type": self.config.type,
                "context_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
                "agent": retrieval.metadata["agent"],
            },
            steps=reconcile_steps(tracer.steps, cost),
        )
