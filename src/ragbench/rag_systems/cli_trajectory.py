"""Reading what an agent harness printed: its event stream (Claude Code stream-JSON, Codex JSONL, OpenCode JSON) as a trajectory of model
turns and tool calls, and which corpus files each tool call searched or read.

Parsers are fed one stdout line at a time, so the caller can stop a harness the moment it goes past its step cap. A line that is not a JSON
object (a banner, a warning) is skipped.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

ActionKind = Literal["read", "search", "list", "other"]

# Tool names (lowercased) and shell programs by what they do to the corpus. A listing names files without showing their text, so it is no evidence.
READ_TOOLS = {"read", "view", "read_file", "cat", "open"}
SEARCH_TOOLS = {"grep", "search", "rg", "find_in_files", "codebase_search"}
LIST_TOOLS = {"glob", "ls", "list", "list_dir", "find", "tree"}
SHELL_TOOLS = {"shell", "bash", "exec", "command"}
READ_PROGRAMS = {"cat", "head", "tail", "sed", "nl", "less", "more", "bat", "awk"}
SEARCH_PROGRAMS = {"rg", "grep", "egrep", "fgrep", "ag", "ack", "git"}
LIST_PROGRAMS = {"ls", "find", "fd", "tree", "pwd", "wc", "cd"}
_SHELL_WRAPPER = re.compile(r"^(?:\S*/)?(?:ba|z|da)?sh\s+-l?c\s+(.+)$", re.DOTALL)
_LISTING_FLAGS = re.compile(r"(?<!\S)(?:--files|--type-list)(?!\S)")  # `rg --files` lists the corpus, it does not search it
_CONTENT = re.compile(r"^[:-]\d*[:-]?\s*\S|^:\s*\S")  # after a path: `:12:text`, `-12-text` (context lines) or `: text`
_SEGMENT = re.compile(r"&&|\|\||[;|\n]")
# A workspace document path: `docs/...` not preceded by a word character (so `/tmp/ws/docs/a.md` and `./docs/a.md` match, `mydocs/a.md` does not).
_DOC_PATH = re.compile(r"(?<![\w.-])docs/([^\s'\"`:,;)\]}>|*?]+)")


@dataclass
class Turn:
    """One model call of the harness (Codex reports one per `exec` turn)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    text: str = ""
    cost_usd: float | None = None


@dataclass
class ToolCall:
    name: str
    input: Any = None
    output: str = ""
    error: str | None = None


@dataclass
class Trajectory:
    items: list[Turn | ToolCall] = field(default_factory=list)
    answer: str | None = None
    cost_usd: float | None = None  # the total the harness reported
    model: str | None = None
    error: str | None = None  # the harness said the run failed
    events: bool = True  # False for `text`: no trajectory, stdout is the answer

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [item for item in self.items if isinstance(item, ToolCall)]

    @property
    def turns(self) -> list[Turn]:
        return [item for item in self.items if isinstance(item, Turn)]

    def final_text(self) -> str:
        """The answer the harness gave: its reported result, else the text of its last turn that said something."""
        if self.answer is not None:
            return self.answer
        return next((turn.text for turn in reversed(self.turns) if turn.text.strip()), "")


def _event(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _int(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _text(content: Any) -> str:
    """A tool result's content as text: a string, or the text blocks of a content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text")
    return "" if content is None else json.dumps(content, ensure_ascii=False)


class ClaudeCodeParser:
    """`claude -p --output-format stream-json --verbose`: `assistant` messages (text and `tool_use` blocks, usage per API call; one message can
    arrive as several events with the same id), `user` messages carrying `tool_result`s, and a final `result` with the answer and `total_cost_usd`."""

    def __init__(self) -> None:
        self.trajectory = Trajectory()
        self._turns: dict[str, Turn] = {}
        self._calls: dict[str, ToolCall] = {}

    def feed(self, line: str) -> None:
        event = _event(line)
        if event is None:
            return
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            self.trajectory.model = event.get("model")
        elif kind == "assistant":
            self._assistant(event.get("message") or {})
        elif kind == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result" and (call := self._calls.get(str(block.get("tool_use_id")))):
                    call.output = _text(block.get("content"))
                    if block.get("is_error"):
                        call.error = call.output[:200] or "error"
        elif kind == "result":
            result = event.get("result")
            self.trajectory.answer = result if isinstance(result, str) else None
            # A model this Claude Code release does not know is priced on a guess (`costBasis: unknown`): RAGBench prices its tokens instead.
            guessed = any(isinstance(usage, dict) and usage.get("costBasis") == "unknown" for usage in (event.get("modelUsage") or {}).values())
            self.trajectory.cost_usd = None if guessed else _float(event.get("total_cost_usd"))
            self._settle(event.get("usage") or {})
            if event.get("is_error") or event.get("subtype") not in (None, "success"):
                detail = result if isinstance(result, str) and result else "; ".join(map(str, event.get("errors") or [])) or "no detail"
                self.trajectory.error = f"{event.get('subtype') or 'error'}: {detail}"

    def _settle(self, usage: dict[str, Any]) -> None:
        """Book what the final usage counts beyond the turns' own numbers on the last turn: the stream reports each message's prompt, but
        not all of its output (thinking and the text after the first block)."""
        turns = self.trajectory.turns
        if not turns or not usage:
            return
        prompt = sum(_int(usage.get(name)) for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        turns[-1].prompt_tokens += max(0, prompt - sum(turn.prompt_tokens for turn in turns))
        turns[-1].completion_tokens += max(0, _int(usage.get("output_tokens")) - sum(turn.completion_tokens for turn in turns))

    def _assistant(self, message: dict[str, Any]) -> None:
        key = str(message.get("id") or f"turn-{len(self._turns)}")
        turn = self._turns.get(key)
        if turn is None:
            turn = self._turns[key] = Turn()
            self.trajectory.items.append(turn)
        usage = message.get("usage") or {}
        prompt = sum(_int(usage.get(name)) for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        turn.prompt_tokens = max(turn.prompt_tokens, prompt)  # repeated on every event of the message, not additive
        turn.completion_tokens = max(turn.completion_tokens, _int(usage.get("output_tokens")))
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                turn.text += str(block.get("text") or "")
            elif block.get("type") == "tool_use":
                call = ToolCall(name=str(block.get("name") or "tool"), input=block.get("input"))
                self._calls[str(block.get("id"))] = call
                self.trajectory.items.append(call)


class CodexParser:
    """`codex exec --json`: `item.started` / `item.completed` events (`command_execution`, `mcp_tool_call`, `web_search`, `file_change`,
    `agent_message`), `turn.completed` with the turn's usage, `turn.failed` and `error`."""

    def __init__(self) -> None:
        self.trajectory = Trajectory()
        self._calls: dict[str, ToolCall] = {}
        self._text: list[str] = []

    def feed(self, line: str) -> None:
        event = _event(line)
        if event is None:
            return
        kind = event.get("type")
        if kind in ("item.started", "item.updated", "item.completed"):
            self._item(event.get("item") or {})
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
            # `input_tokens` already counts the cached ones (`cached_input_tokens`).
            self.trajectory.items.append(Turn(prompt_tokens=_int(usage.get("input_tokens")), completion_tokens=_int(usage.get("output_tokens")), text="\n".join(self._text)))
            self._text = []
        elif kind == "turn.failed":
            self.trajectory.error = str((event.get("error") or {}).get("message") or "turn failed")
        elif kind == "error":
            self.trajectory.error = str(event.get("message") or "error")

    def _item(self, item: dict[str, Any]) -> None:
        kind, key = item.get("type"), str(item.get("id"))
        if kind == "agent_message":
            if item.get("text"):
                self._text.append(str(item["text"]))
                self.trajectory.answer = str(item["text"])
            return
        call = self._calls.get(key)
        if call is None:
            if kind == "command_execution":
                call = ToolCall(name="shell", input=_command(item.get("command")))
            elif kind == "mcp_tool_call":
                call = ToolCall(name=f"{item.get('server')}.{item.get('tool')}", input=item.get("arguments"))
            elif kind == "web_search":
                call = ToolCall(name="web_search", input=item.get("query"))
            elif kind == "file_change":
                call = ToolCall(name="file_change", input=item.get("changes"))
            else:
                return  # reasoning, todo lists: not actions
            self._calls[key] = call
            self.trajectory.items.append(call)
        if kind == "command_execution":
            call.output = str(item.get("aggregated_output") or "")
            exit_code = item.get("exit_code")
            # grep and rg exit 1 when nothing matched, which is an answer, not an error.
            allowed = 1 if action_kind(call) == "search" else 0
            if exit_code > allowed if isinstance(exit_code, int) else item.get("status") == "failed":
                call.error = f"exit code {exit_code}"
        elif kind == "mcp_tool_call":
            call.output = _text((item.get("result") or {}).get("content") if isinstance(item.get("result"), dict) else item.get("result"))
            if item.get("error") or item.get("status") == "failed":
                call.error = str(item.get("error") or "failed")


def _command(command: Any) -> str:
    return shlex.join(command) if isinstance(command, list) else str(command or "")


class OpenCodeParser:
    """`opencode run --format json`: `step_start` / `step_finish` around each model call (tokens and cost on the finish), `text` parts,
    `tool_use` parts with the call's final state, and `error`."""

    def __init__(self) -> None:
        self.trajectory = Trajectory()
        self._turn: Turn | None = None

    def feed(self, line: str) -> None:
        event = _event(line)
        if event is None:
            return
        kind, part = event.get("type"), event.get("part") or {}
        if kind == "step_start" or (kind in ("text", "step_finish") and self._turn is None):
            self._turn = Turn()
            self.trajectory.items.append(self._turn)
        if kind == "text" and self._turn is not None:
            self._turn.text += str(part.get("text") or "")
        elif kind == "tool_use":
            state = part.get("state") or {}
            call = ToolCall(name=str(part.get("tool") or "tool"), input=state.get("input"), output=_text(state.get("output")))
            if state.get("status") == "error":
                call.error = str(state.get("error") or "error")
            self.trajectory.items.append(call)
        elif kind == "step_finish" and self._turn is not None:
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            self._turn.prompt_tokens = _int(tokens.get("input")) + _int(cache.get("read")) + _int(cache.get("write"))
            self._turn.completion_tokens = _int(tokens.get("output")) + _int(tokens.get("reasoning"))
            self._turn.cost_usd = _float(part.get("cost"))
            self._turn = None
        elif kind == "error":
            error = event.get("error") or {}
            self.trajectory.error = str((error.get("data") or {}).get("message") or error.get("name") or "error") if isinstance(error, dict) else str(error)
        turns = self.trajectory.turns
        if turns and all(turn.cost_usd is not None for turn in turns):
            self.trajectory.cost_usd = sum(turn.cost_usd or 0.0 for turn in turns)


class TextParser:
    """Any harness: stdout is the answer, and there is no trajectory."""

    def __init__(self) -> None:
        self.trajectory = Trajectory(events=False)
        self._lines: list[str] = []

    def feed(self, line: str) -> None:
        self._lines.append(line)
        self.trajectory.answer = "".join(self._lines).strip()


Parser = ClaudeCodeParser | CodexParser | OpenCodeParser | TextParser
PARSERS: dict[str, type[Parser]] = {"claude_code": ClaudeCodeParser, "codex": CodexParser, "opencode": OpenCodeParser, "text": TextParser}


def parse(output_format: str, text: str) -> Trajectory:
    """A whole captured stdout as a trajectory (for fixtures and tests)."""
    parser = PARSERS[output_format]()
    for line in text.splitlines(keepends=True):
        parser.feed(line)
    return parser.trajectory


def input_text(call: ToolCall) -> str:
    if isinstance(call.input, str):
        return call.input
    return "" if call.input is None else json.dumps(call.input, ensure_ascii=False, sort_keys=True)


def _programs(command: str) -> list[str]:
    match = _SHELL_WRAPPER.match(command.strip())
    if match:
        inner = match.group(1).strip()
        try:
            inner = shlex.split(inner)[0] if inner[:1] in "'\"" else inner
        except ValueError:
            pass
        command = inner
    programs = []
    for segment in _SEGMENT.split(command):
        words = segment.split()
        while words and "=" in words[0] and not words[0].startswith("="):  # `LC_ALL=C grep ...`
            words = words[1:]
        if words:
            programs.append(words[0].rsplit("/", 1)[-1])
    return programs


def action_kind(call: ToolCall) -> ActionKind:
    """What a tool call did to the corpus: `read` a file, `search` its text, `list` its files, or something `other`."""
    name = call.name.lower().rsplit(".", 1)[-1]
    if name in SHELL_TOOLS:
        programs = _programs(input_text(call))
        if any(program in SEARCH_PROGRAMS for program in programs):
            return "list" if _LISTING_FLAGS.search(input_text(call)) else "search"
        if any(program in READ_PROGRAMS for program in programs):
            return "read"
        if programs and all(program in LIST_PROGRAMS for program in programs):
            return "list"
        return "other"
    if name in READ_TOOLS:
        return "read"
    if name in SEARCH_TOOLS:
        return "search"
    if name in LIST_TOOLS:
        return "list"
    return "other"


def doc_paths(text: str) -> list[str]:
    """The workspace document paths (relative to `docs/`) a text mentions, in order, each once."""
    return list(dict.fromkeys(match.rstrip(".") for match in _DOC_PATH.findall(text)))


def matched_paths(output: str) -> list[str]:
    """The files a search's output shows matching text of: `path:12:text` (grep, rg) or a `path:` line followed by indented matches
    (OpenCode). A bare list of file names (`rg -l`, Claude Code's default `files_with_matches`) shows no text, so it is no evidence."""
    lines = output.splitlines()
    found: list[str] = []
    for index, line in enumerate(lines):
        for match in _DOC_PATH.finditer(line):
            rest = line[match.end() :]
            follows = index + 1 < len(lines) and lines[index + 1][:1].isspace() and bool(lines[index + 1].strip())
            if _CONTENT.search(rest) or (rest.strip() == ":" and follows):
                found.append(match.group(1).rstrip("."))
    return list(dict.fromkeys(found))


def evidence_paths(call: ToolCall) -> list[str]:
    """The documents a call put in front of the agent: the files a read names, the files a search's output shows, none for a listing."""
    kind = action_kind(call)
    if kind == "read":
        return doc_paths(input_text(call))
    if kind == "search":
        return matched_paths(call.output)
    if kind == "list":
        return []
    return list(dict.fromkeys([*doc_paths(input_text(call)), *doc_paths(call.output)]))
