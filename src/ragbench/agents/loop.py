"""A bounded controller for systems that decide, step by step, whether to keep searching.

The loop owns the stopping rules (a hard step cap, a cost cap, a token cap); the *policy* owns the work of one iteration (retrieve,
grade, rewrite, ...) and says what to do next. Whatever the policy does, the loop ends: it never raises because a budget ran out,
it just stops, and the caller answers from the evidence gathered so far (see `AgentState.termination`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ragbench.agents.budget import AgentBudget, steps_cost_usd, steps_tokens, steps_tool_calls

if TYPE_CHECKING:
    from ragbench.rag_systems.trace import Tracer

Termination = Literal["finished", "budget", "max_steps", "abort"]
# `Abort(BUDGET)` is how a policy reports "I ran out of budget before an LLM call"; the loop records it as a budget stop.
BUDGET = "budget"


@dataclass(frozen=True)
class Continue:
    """Run another iteration, with `query` as the new `AgentState.query`."""

    query: str


@dataclass(frozen=True)
class Finish:
    """Stop: there is enough evidence. `answer` is optional; callers that generate the answer themselves leave it out."""

    answer: str | None = None


@dataclass(frozen=True)
class Abort:
    """Stop without a result worth trusting (`reason` is recorded). `Abort(BUDGET)` records a budget stop."""

    reason: str


AgentAction = Continue | Finish | Abort


@dataclass
class AgentState:
    """Handed to the policy on every iteration; the policy keeps its own working data in `data`."""

    question: str
    query: str
    budget: AgentBudget
    tracer: Tracer
    iteration: int = 0  # 1 during the first policy call
    hops: int = 0  # follow-up rounds requested so far (`Continue` actions)
    answer: str | None = None
    termination: Termination | None = None
    abort_reason: str | None = None
    data: dict[str, Any] = field(default_factory=dict)  # policy scratch space
    info: dict[str, Any] = field(default_factory=dict)  # extra facts the policy wants reported next to the loop's own (e.g. `sufficient`)

    @property
    def cost_usd(self) -> float:
        return steps_cost_usd(self.tracer.steps)

    @property
    def tokens(self) -> int:
        return steps_tokens(self.tracer.steps)

    @property
    def tool_calls(self) -> int:
        return steps_tool_calls(self.tracer.steps)

    def over_budget(self) -> bool:
        """Whether the cost, token or tool-call cap is already spent. Check before every extra LLM or tool call a policy makes."""
        return self.budget.exhausted(self.cost_usd, self.tokens, self.tool_calls)

    def metadata(self) -> dict[str, Any]:
        """The `agent` block of `AnswerResult.metadata`."""
        return {
            "steps": self.iteration,
            "hops": self.hops,
            "budget_exhausted": self.termination == "budget",
            "termination": self.termination,
            **self.info,
        }


Policy = Callable[[AgentState], AgentAction]


class AgentLoop:
    """Runs a policy until it finishes, aborts, or the budget says stop. Every decision is a zero-cost `route` step in the trace."""

    def __init__(self, budget: AgentBudget, tracer: Tracer):
        self.budget = budget
        self.tracer = tracer

    def run(self, policy: Policy, *, question: str = "", query: str | None = None) -> AgentState:
        state = AgentState(question=question, query=question if query is None else query, budget=self.budget, tracer=self.tracer)
        for _ in range(self.budget.step_limit):  # the hard cap holds whatever the policy returns
            if state.over_budget():
                return self._stop(state, "budget", "a cost, token or tool-call cap was reached before the next iteration")
            state.iteration += 1
            action = policy(state)
            if isinstance(action, Continue):
                state.hops += 1
                state.query = action.query
                self._decision(state, "continue", query=action.query)
            elif isinstance(action, Finish):
                state.answer = action.answer
                if action.answer is None:  # a `Finish` that carries the answer needs no marker: the answer step the policy recorded says it
                    self._decision(state, "finish")
                state.termination = "finished"
                return state
            elif isinstance(action, Abort):
                state.abort_reason = action.reason
                self._decision(state, "abort", reason=action.reason)
                state.termination = "budget" if action.reason == BUDGET else "abort"
                return state
            else:  # a policy that returns something else must not turn into an endless loop
                state.abort_reason = f"policy returned {type(action).__name__}, not an AgentAction"
                self._decision(state, "abort", reason=state.abort_reason)
                state.termination = "abort"
                return state
        return self._stop(state, "max_steps", f"reached the step limit of {self.budget.step_limit}")

    def _decision(self, state: AgentState, action: str, **details: Any) -> None:
        with self.tracer.step("route", "agent_decision", iteration=state.iteration, action=action, **details) as step:
            if "query" in details:
                step.set_input(details["query"])

    def _stop(self, state: AgentState, termination: Termination, reason: str) -> AgentState:
        state.termination = termination
        with self.tracer.step("route", "agent_stop", iteration=state.iteration, termination=termination, reason=reason):
            pass
        return state
