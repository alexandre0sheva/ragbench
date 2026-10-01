"""What an agent may spend on one question, and how to measure what it has spent from the question's trace."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `rag_systems/__init__` imports the agentic systems, which import this package
    from ragbench.rag_systems.trace import Step

# However large a budget asks for, a loop never runs more iterations than this.
ABSOLUTE_MAX_STEPS = 50


@dataclass(frozen=True)
class AgentBudget:
    """Limits for one question. `None` means no limit on that axis; `max_steps` is always enforced.

    `max_steps` counts loop iterations. `max_cost_usd`, `max_tokens` and `max_tool_calls` cover everything the question has used so far (every
    traced step; a tool call is a `tool` step), and are checked before each iteration and before each extra LLM call a policy makes (`AgentState.over_budget()`).
    The final answer call is always made, so a question can finish slightly over the cap by that one call.
    """

    max_steps: int = 6
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_tool_calls: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be at least 1, got {self.max_steps}")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError(f"max_cost_usd must be positive, got {self.max_cost_usd}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.max_tool_calls is not None and self.max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must not be negative, got {self.max_tool_calls}")

    @property
    def step_limit(self) -> int:
        return min(self.max_steps, ABSOLUTE_MAX_STEPS)

    def exhausted(self, cost_usd: float, tokens: int, tool_calls: int = 0) -> bool:
        """Whether usage has reached a cap (reaching it counts: a cap of $0.01 stops at $0.01, a cap of 3 tool calls after the third)."""
        return (
            (self.max_cost_usd is not None and cost_usd >= self.max_cost_usd)
            or (self.max_tokens is not None and tokens >= self.max_tokens)
            or (self.max_tool_calls is not None and tool_calls >= self.max_tool_calls)
        )


def steps_cost_usd(steps: Iterable[Step]) -> float:
    return sum(step.cost.total_cost for step in steps)


def steps_tokens(steps: Iterable[Step]) -> int:
    return sum(step.prompt_tokens + step.completion_tokens for step in steps)


def steps_tool_calls(steps: Iterable[Step]) -> int:
    return sum(1 for step in steps if step.kind == "tool")


def llm_calls(steps: Iterable[Step]) -> int:
    """How many steps called a chat model (the ones that recorded tokens)."""
    return sum(1 for step in steps if step.prompt_tokens or step.completion_tokens)
