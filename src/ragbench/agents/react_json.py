"""ReAct-style JSON actions for models without native tool calling: describe the tools in the prompt, parse one JSON action per reply."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ragbench.tools.registry import ToolBox
from ragbench.utils.query_planning import loads_lenient

ANSWER_ACTIONS = {"answer", "final", "final_answer", "finish", "respond", "reply"}
TOOL_KEYS = ("tool", "tool_name", "name", "function")
ARGUMENT_KEYS = ("args", "arguments", "input", "parameters", "action_input", "tool_input")
ANSWER_KEYS = ("answer", "final_answer", "response", "output", "text")


@dataclass(frozen=True)
class ReactAction:
    kind: Literal["tool", "answer"]
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None


def describe_tools(box: ToolBox) -> str:
    """The tools as prompt text: one line each with the description and the JSON schema of the arguments."""
    lines = []
    for tool in box.tools.values():
        parameters = json.dumps(tool.spec.parameters.get("properties", {}), ensure_ascii=False, separators=(",", ":"))
        required = tool.spec.parameters.get("required", [])
        lines.append(f"- {tool.spec.name}: {tool.spec.description} Arguments: {parameters}. Required: {', '.join(required) or 'none'}.")
    return "\n".join(lines)


def parse_action(text: str) -> ReactAction | None:
    """The action in a model reply, or None when there is none. Tolerates code fences, prose around the JSON, a second JSON object after
    the first, and the usual variants of the key names (`tool` / `name`, `args` / `arguments` / `input`, `answer` / `final_answer`)."""
    data = loads_lenient(text)
    if not isinstance(data, dict):
        data = _first_object(text)
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    action_name = action.strip().lower() if isinstance(action, str) else ""
    answer = next((data[key] for key in ANSWER_KEYS if isinstance(data.get(key), str)), None)
    if action_name in ANSWER_ACTIONS or (not action_name and answer is not None and not _tool_name(data)):
        return ReactAction("answer", answer=answer if answer is not None else "") if answer is not None else None
    name = _tool_name(data) or (action_name if action_name not in {"tool", "call", "use_tool", "tool_call"} else "")
    if not name:
        return None
    return ReactAction("tool", tool=name, args=_arguments(data))


def _tool_name(data: dict[str, Any]) -> str:
    return next((str(data[key]).strip() for key in TOOL_KEYS if isinstance(data.get(key), str) and data[key].strip()), "")


def _arguments(data: dict[str, Any]) -> dict[str, Any]:
    raw: Any = next((data[key] for key in ARGUMENT_KEYS if key in data), {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {"input": raw}
    return raw if isinstance(raw, dict) else {}


def _first_object(text: str) -> Any:
    """The first complete JSON object in `text`, even when more text (or a second object) follows it."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text, index)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None
