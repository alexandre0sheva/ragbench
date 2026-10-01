"""Turning a system's `tools:` list into tools, and calling them safely: timeout, output cap, never an exception, always a trace step."""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ragbench.config.schema import ToolRef, ToolsConfig
from ragbench.models.cost import CostBreakdown
from ragbench.registry import TOOLS
from ragbench.tools.base import Tool, ToolContext, ToolResult
from ragbench.tools.custom import load_custom_tool

if TYPE_CHECKING:
    from ragbench.rag_systems.trace import Tracer

TRUNCATION_NOTE = "\n[output truncated]"


def tool_ref_name(ref: ToolRef) -> str:
    return ref if isinstance(ref, str) else str(ref["name"])


def tool_ref_options(ref: ToolRef) -> dict[str, Any]:
    return {} if isinstance(ref, str) else {key: value for key, value in ref.items() if key != "name"}


def resolve_tools(refs: Sequence[ToolRef], settings: ToolsConfig | None = None) -> list[Tool]:
    """Instantiate the tools a `tools:` list names. Raises `ValueError` for an unknown name, bad options, a duplicate, or a tool with
    side effects that `tools.allow` (and, for network tools, `tools.allow_network`) does not permit."""
    settings = settings or ToolsConfig()
    tools: list[Tool] = []
    seen: set[str] = set()
    for ref in refs:
        name = tool_ref_name(ref)
        options = tool_ref_options(ref)
        if name in seen:
            raise ValueError(f"tool '{name}' is listed twice")
        seen.add(name)
        if "path" in options:
            if name in TOOLS:
                raise ValueError(f"the custom tool '{name}' has the name of a built-in tool; pick another name")
            tool: Tool = load_custom_tool(name, options)
        else:
            try:
                tool = TOOLS.get(name)(**options)
            except ValueError as exc:  # unknown name (with a did-you-mean hint) or invalid options
                raise ValueError(f"tool '{name}': {exc}") from None
        if tool.side_effects != "none":
            if name not in settings.allow:
                raise ValueError(f"tool '{name}' has {tool.side_effects} side effects; list it under `tools.allow` to use it")
            if tool.side_effects == "network" and not settings.allow_network:
                raise ValueError(f"tool '{name}' uses the network; set `tools.allow_network: true` to use it")
        tools.append(tool)
    return tools


class ToolBox:
    """A system's tools plus the rules for calling them. `call` never raises and never hangs past the timeout."""

    def __init__(self, tools: Sequence[Tool], settings: ToolsConfig | None = None):
        self.settings = settings or ToolsConfig()
        self.tools: dict[str, Tool] = {tool.spec.name: tool for tool in tools}

    @classmethod
    def from_refs(cls, refs: Sequence[ToolRef], settings: ToolsConfig | None = None) -> ToolBox:
        return cls(resolve_tools(refs, settings), settings)

    @property
    def names(self) -> list[str]:
        return list(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def schemas(self) -> list[dict[str, Any]]:
        """The tools in the OpenAI function-calling form `LLM.generate(tools=...)` takes."""
        return [tool.spec.openai_schema() for tool in self.tools.values()]

    def call(self, name: str, args: Any, ctx: ToolContext, tracer: Tracer | None = None) -> ToolResult:
        """Run tool `name`. Unknown tools, bad arguments, exceptions, timeouts and oversized output all come back as a `ToolResult`
        (with `error` set where something went wrong); with a `tracer` the call is recorded as a `tool` step."""
        arguments = _arguments(args)
        started = perf_counter()
        if tracer is None:
            result = self._execute(name, arguments, ctx)
            result.latency_ms = (perf_counter() - started) * 1000
            return result
        with tracer.step("tool", name) as step:
            step.set_input(json.dumps(arguments if isinstance(arguments, dict) else {"arguments": arguments}, default=str, ensure_ascii=False))
            result = self._execute(name, arguments, ctx)
            result.latency_ms = (perf_counter() - started) * 1000
            step.set_output(result.text)
            step.set_cost(CostBreakdown(tool_cost=result.cost_usd))
            step.set_meta(truncated=result.truncated, **({"error": result.error} if result.error else {}))
        return result

    def _execute(self, name: str, arguments: Any, ctx: ToolContext) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.fail(f"unknown tool {name!r}; available tools: {', '.join(self.tools) or 'none'}")
        if not isinstance(arguments, dict):
            return ToolResult.fail("the arguments must be a JSON object")
        timeout = getattr(tool, "timeout_s", None) or self.settings.timeout_s
        result = _run_with_timeout(name, tool, arguments, ctx, timeout)
        if result.error and not result.text:
            result.text = f"Error: {result.error}"
        return self._cap(result)

    def _cap(self, result: ToolResult) -> ToolResult:
        limit = self.settings.max_output_chars
        if len(result.text) > limit:
            result.text = result.text[: limit - len(TRUNCATION_NOTE)] + TRUNCATION_NOTE
            result.truncated = True
        return result


def _arguments(args: Any) -> Any:
    """Tool-call arguments as the model sent them: a dict, or a JSON string of one (some providers); anything else is passed on to be rejected."""
    if isinstance(args, str):
        try:
            return json.loads(args) if args.strip() else {}
        except ValueError:
            return args
    return {} if args is None else args


def _run_with_timeout(name: str, tool: Tool, arguments: dict[str, Any], ctx: ToolContext, timeout: float) -> ToolResult:
    outcome: list[Any] = []

    def target() -> None:
        try:
            outcome.append(tool.run(arguments, ctx))
        except Exception as exc:
            outcome.append(exc)

    thread = threading.Thread(target=target, name=f"ragbench-tool-{name}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():  # the thread cannot be killed; as a daemon it will not keep the process alive
        return ToolResult.fail(f"{name} timed out after {timeout:g} s")
    result = outcome[0]
    if isinstance(result, Exception):
        return ToolResult.fail(f"{name} failed: {type(result).__name__}: {result}")
    if not isinstance(result, ToolResult):
        return ToolResult.fail(f"{name} returned {type(result).__name__} instead of a ToolResult")
    return result
