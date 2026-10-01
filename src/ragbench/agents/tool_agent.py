"""A tool-using agent: ask the model, run the tools it calls, feed the results back, until it answers or the budget says stop.

Two protocols: `native` uses the provider's function calling (`LLM.generate(tools=...)`); `react_json` describes the tools in the prompt and
parses one JSON action per reply, for models without function calling. Both run inside `AgentLoop`, so the step cap and the cost, token and
tool-call caps apply, and when a cap ends the loop the agent still answers (from what it found), flagged `best_effort`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ragbench.agents import prompts
from ragbench.agents.budget import AgentBudget
from ragbench.agents.loop import BUDGET, Abort, AgentAction, AgentLoop, AgentState, Continue, Finish
from ragbench.agents.react_json import describe_tools, parse_action
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.llms import LLM, LLMResult
from ragbench.tools.base import ToolContext, ToolResult
from ragbench.tools.registry import ToolBox

if TYPE_CHECKING:
    from ragbench.rag_systems.trace import Tracer

NO_ANSWER = "I could not complete the research within the allowed steps."
TOOL_LIMIT_MESSAGE = "The tool-call limit was reached; this call was not run. Answer with what you have."
ANSWER_MAX_TOKENS = 800


@dataclass
class ToolAgentRun:
    answer: str
    state: AgentState
    # One ranking per tool call that produced evidence (search hits, grep matches, document windows), in call order.
    rankings: list[list[RetrievedChunk]]
    prompt_tokens: int
    completion_tokens: int
    model: str
    best_effort: bool  # the loop ended (budget, step limit, abort) before the model answered, so the answer was requested explicitly


@dataclass
class _Tally:
    """What the policy collects while the loop runs."""

    rankings: list[list[RetrievedChunk]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def count(self, result: LLMResult) -> None:
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.model = result.model


def evidence_from(tool: str, result: ToolResult) -> list[RetrievedChunk]:
    """The document passages a tool call put in front of the model, as ranked chunks (for retrieval metrics and the judge)."""
    if result.error or not result.data:
        return []
    if tool == "search" and isinstance(result.data, list):
        return [
            RetrievedChunk(chunk_id=hit["chunk_id"], doc_id=hit["doc_id"], text=hit["text"], score=float(hit.get("score", 0.0)), rank=rank)
            for rank, hit in enumerate(result.data, start=1)
        ]
    if tool == "corpus_grep" and isinstance(result.data, list):
        return [
            RetrievedChunk(chunk_id=f"{hit['doc_id']}::chunk::line{hit['line']}", doc_id=hit["doc_id"], text=hit["text"], score=1.0, rank=rank)
            for rank, hit in enumerate(result.data, start=1)
        ]
    if tool == "read_document" and isinstance(result.data, dict):
        data = result.data
        return [RetrievedChunk(chunk_id=f"{data['doc_id']}::chunk::read{data['start']}", doc_id=data["doc_id"], text=result.text, score=1.0, rank=1)]
    return []


class ToolAgent:
    def __init__(self, llm: LLM, box: ToolBox, budget: AgentBudget, *, mode: str = "native", instructions: str = ""):
        if mode not in ("native", "react_json"):
            raise ValueError(f"agent_mode must be 'native' or 'react_json', not {mode!r}")
        self.llm, self.box, self.budget, self.mode = llm, box, budget, mode
        react_tools = describe_tools(box) if mode == "react_json" else None
        self.system_message = {"role": "system", "content": prompts.tool_agent_system(react_tools, instructions)}

    def run(self, question: str, ctx: ToolContext, tracer: Tracer) -> ToolAgentRun:
        messages: list[dict[str, Any]] = [self.system_message, {"role": "user", "content": prompts.question_message(question)}]
        tally = _Tally()
        schemas = self.box.schemas() if self.mode == "native" else None
        turn = self._native_turn if self.mode == "native" else self._react_turn

        def policy(state: AgentState) -> AgentAction:
            if state.over_budget():
                return Abort(BUDGET)
            text, finished = turn(state, messages, schemas, ctx, tally)
            return Finish(text or None) if finished else Continue(question)

        state = AgentLoop(self.budget, tracer).run(policy, question=question)
        answer = state.answer or ""
        best_effort = not answer
        if best_effort:
            answer = self._force_answer(messages, schemas, tracer, tally)
        state.info.update(tool_calls=state.tool_calls, best_effort=best_effort)
        return ToolAgentRun(answer, state, tally.rankings, tally.prompt_tokens, tally.completion_tokens, tally.model, best_effort)

    # -- one turn of each protocol ---------------------------------------------------------------------------------------

    def _ask(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]] | None,
        tally: _Tally,
        is_answer: Callable[[LLMResult], bool],
        *,
        json_mode: bool = False,
    ) -> LLMResult:
        """One model call, traced as an `agent_turn`; a turn that turns out to be the final answer is recorded as the `generate` step instead."""
        with state.tracer.step("llm", "agent_turn", turn=state.iteration) as step:
            result = self.llm.generate(messages, temperature=0, tools=schemas, json_mode=json_mode)
            step.set_llm(result, messages)
            if is_answer(result):
                step.relabel("generate", "answer")
        tally.count(result)
        return result

    def _native_turn(self, state: AgentState, messages: list, schemas: list | None, ctx: ToolContext, tally: _Tally) -> tuple[str, bool]:
        result = self._ask(state, messages, schemas, tally, lambda r: not r.tool_calls)
        messages.append(result.assistant_message())
        if not result.tool_calls:
            return result.text.strip(), True
        for call in result.tool_calls:
            observation = self._call(state, call.name, call.arguments, ctx, tally)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
        return "", False

    def _react_turn(self, state: AgentState, messages: list, schemas: list | None, ctx: ToolContext, tally: _Tally) -> tuple[str, bool]:
        # A reply that is not an action at all is taken to be the answer in prose; so is an explicit `answer` action.
        result = self._ask(state, messages, None, tally, lambda r: (action := parse_action(r.text)) is None or action.kind == "answer", json_mode=True)
        messages.append({"role": "assistant", "content": result.text})
        action = parse_action(result.text)
        if action is None:
            state.info["unparsed_action"] = state.info.get("unparsed_action", 0) + 1
            return result.text.strip(), True
        if action.kind == "answer":
            return (action.answer or "").strip(), True
        observation = self._call(state, action.tool or "", action.args, ctx, tally)
        messages.append({"role": "user", "content": prompts.observation(action.tool or "", observation)})
        return "", False

    def _call(self, state: AgentState, name: str, arguments: Any, ctx: ToolContext, tally: _Tally) -> str:
        """Run one tool call (or refuse it once the tool-call cap is spent) and return the observation text for the model."""
        cap = self.budget.max_tool_calls
        if cap is not None and state.tool_calls >= cap:
            state.info["tool_limit_hit"] = True
            return TOOL_LIMIT_MESSAGE
        result = self.box.call(name, arguments, ctx, state.tracer)
        if name not in self.box.tools:
            state.info["unknown_tool_calls"] = state.info.get("unknown_tool_calls", 0) + 1
        if evidence := evidence_from(name, result):
            tally.rankings.append(evidence)
        return result.text

    def _force_answer(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]] | None, tracer: Tracer, tally: _Tally) -> str:
        """The loop ended without an answer: ask for one now. Native mode still sends the tool schemas (some providers reject tool results without them)."""
        messages.append({"role": "user", "content": prompts.force_answer()})
        with tracer.step("generate", "answer", best_effort=True) as step:
            result = self.llm.generate(messages, temperature=0, tools=schemas, max_tokens=ANSWER_MAX_TOKENS)
            step.set_llm(result, messages)
        tally.count(result)
        return result.text.strip() or NO_ANSWER
