"""Shared plumbing of the agentic systems (`corrective`, `iterative`): an `AgentLoop` inside `fetch_context`, traced control calls, a budget."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from ragbench.agents import AgentBudget, AgentState, llm_calls
from ragbench.agents.budget import steps_cost_usd, steps_tokens, steps_tool_calls
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import AnswerResult, RetrievalResult
from ragbench.rag_systems.llm_query_base import LLMQueryRAG
from ragbench.rag_systems.options import AgenticOptions
from ragbench.rag_systems.trace import StepKind, Tracer, sum_costs
from ragbench.utils.timing import timer

T = TypeVar("T")


class AgenticRAG(LLMQueryRAG):
    """Indexes like `hybrid`. Subclasses implement `_agent`: run an `AgentLoop` and return the evidence it settled on.

    `fetch_context` wraps that in one traced retrieval whose cost is everything the agent spent; `answer_question` adds the loop's
    `agent` block (steps, hops, termination, LLM calls, ...) to the answer's metadata.
    """

    options: AgenticOptions

    @abstractmethod
    def _agent(self, question: str, final_top_k: int, tracer: Tracer) -> tuple[AgentState, list[RetrievedChunk], dict]:
        """Run the agent; return its final state, the ranked chunks (at most `final_top_k`), and extra retrieval metadata."""

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        final_top_k = self.options.resolve_top_k(top_k)
        with self._recording() as tracer, timer() as t:
            first_step = len(tracer.steps)
            state, chunks, metadata = self._agent(question, final_top_k, tracer)
            cost = sum_costs(step.cost for step in tracer.steps[first_step:])
        return RetrievalResult(
            question=question, chunks=chunks, latency_ms=t.elapsed_ms, cost=cost, metadata={**metadata, "agent": state.metadata()}
        )

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None) -> AnswerResult:
        result = super().answer_question(question, top_k=top_k, context_k=context_k)
        agent = dict(result.retrieval_result.metadata.get("agent", {}))
        agent["llm_calls"] = llm_calls(result.steps)
        result.metadata["agent"] = agent
        return result

    def _budget(self, max_steps: int) -> AgentBudget:
        return AgentBudget(max_steps=max_steps, max_cost_usd=self.options.max_cost_usd, max_tokens=self.options.max_tokens)

    def _over_budget(self) -> bool:
        """Whether this question's spending so far has reached a cap (for code that runs outside the loop, e.g. the answer self-check)."""
        steps = self.trace.steps
        return self._budget(1).exhausted(steps_cost_usd(steps), steps_tokens(steps), steps_tool_calls(steps))

    def _ask(
        self, kind: StepKind, name: str, messages: list[dict], parse: Callable[[str], T | None], *, max_tokens: int, **metadata: object
    ) -> tuple[T | None, CostBreakdown]:
        """One traced JSON-mode control call. Returns the parsed reply (`None` if unusable; the step is then flagged `fallback`) and its booked cost."""
        with self.trace.step(kind, name, **metadata) as step:
            result = self.llm.generate(messages, temperature=0, json_mode=True, max_tokens=max_tokens)
            cost = CostBreakdown(query_rewrite_cost=result.cost.total_cost)
            step.set_llm(result, messages, cost=cost)
            parsed = parse(result.text)
            if parsed is None:
                step.set_meta(fallback=True)
        return parsed, cost
