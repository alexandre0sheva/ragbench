"""The offline harness `cli_agent` runs under `--mock`: no model, no network, no cost, deterministic.

Run in a `cli_agent` workspace (`python -m ragbench.rag_systems.cli_agent_mock --format claude_code`), it reads the question from `TASK.md`,
greps `docs/` for the question's words, reads the two best files and answers with their best-matching sentence, printing what it did in the
event format of the harness it stands in for, so the parser, the step cap and the trajectory metrics run exactly as with the real one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ragbench.utils.text import tokenize

QUESTION = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
MOCK_MODEL = "mock-harness"


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _plan(workspace: Path) -> tuple[str, list[tuple[str, Any, str]], str]:
    """The task text, the tool calls the mock makes (name, input, output) and its answer."""
    task = (workspace / "TASK.md").read_text(encoding="utf-8")
    match = QUESTION.search(task)
    question = match.group(1).strip() if match else task.strip()
    words = sorted({word for word in tokenize(question) if len(word) > 3})
    files = sorted(path for path in (workspace / "docs").rglob("*") if path.is_file())
    texts = {path.relative_to(workspace).as_posix(): path.read_text(encoding="utf-8", errors="replace") for path in files}
    scores = {name: sum(1 for word in words if word in set(tokenize(text))) for name, text in texts.items()}
    hits = [f"{name}:{number}:{line}" for name, text in texts.items() for number, line in enumerate(text.splitlines(), start=1) if words and set(words) & set(tokenize(line))]
    calls: list[tuple[str, Any, str]] = [("search", {"pattern": "|".join(words), "path": "docs"}, "\n".join(hits[:10]) or "No matches found")]
    best = [name for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if score > 0][:2]
    calls.extend(("read", {"file_path": name}, texts[name][:4000]) for name in best)  # relative, so two runs print the same
    sentences = [sentence for name in best for sentence in SENTENCE.split(texts[name]) if sentence.strip()]
    answer = max(sentences, key=lambda s: (len(set(words) & set(tokenize(s))), -len(s)), default="I cannot find the answer in the documents.").strip()
    return task, calls, answer


def _claude_code(task: str, calls: list[tuple[str, Any, str]], answer: str) -> list[dict[str, Any]]:
    names = {"search": "Grep", "read": "Read"}
    events: list[dict[str, Any]] = [{"type": "system", "subtype": "init", "model": MOCK_MODEL, "tools": ["Read", "Grep", "Glob"]}]
    context, cost = _tokens(task), 0.0
    for turn, batch in enumerate(([calls[0]], calls[1:]), start=1):
        if not batch:
            continue
        uses = [{"type": "tool_use", "id": f"toolu_{turn}_{index}", "name": names[name], "input": args} for index, (name, args, _) in enumerate(batch)]
        events.append({"type": "assistant", "message": {"id": f"msg_{turn}", "model": MOCK_MODEL, "content": uses, "usage": {"input_tokens": context, "output_tokens": 20}}})
        cost += (context * 0.1 + 20 * 0.5) / 1e6
        results = [{"type": "tool_result", "tool_use_id": use["id"], "content": output, "is_error": False} for use, (_, _, output) in zip(uses, batch, strict=True)]
        events.append({"type": "user", "message": {"role": "user", "content": results}})
        context += sum(_tokens(output) for _, _, output in batch) + 20
    out = _tokens(answer)
    events.append({"type": "assistant", "message": {"id": "msg_final", "model": MOCK_MODEL, "content": [{"type": "text", "text": answer}], "usage": {"input_tokens": context, "output_tokens": out}}})
    cost += (context * 0.1 + out * 0.5) / 1e6
    events.append({"type": "result", "subtype": "success", "is_error": False, "result": answer, "num_turns": 3, "total_cost_usd": round(cost, 10)})
    return events


def _codex(task: str, calls: list[tuple[str, Any, str]], answer: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{"type": "thread.started", "thread_id": "mock"}, {"type": "turn.started"}]
    for index, (name, args, output) in enumerate(calls):
        command = f"rg -n '{args['pattern']}' docs" if name == "search" else f"sed -n '1,200p' {args['file_path']}"
        item = {"id": f"item_{index}", "type": "command_execution", "command": f"bash -lc {json.dumps(command)}", "aggregated_output": output, "exit_code": 0, "status": "completed"}
        events.append({"type": "item.completed", "item": item})
    events.append({"type": "item.completed", "item": {"id": "item_answer", "type": "agent_message", "text": answer}})
    prompt = _tokens(task) * 3 + sum(_tokens(output) for _, _, output in calls)
    events.append({"type": "turn.completed", "usage": {"input_tokens": prompt, "cached_input_tokens": 0, "output_tokens": _tokens(answer) + 40}})
    return events


def _opencode(task: str, calls: list[tuple[str, Any, str]], answer: str) -> list[dict[str, Any]]:
    names = {"search": "grep", "read": "read"}
    events: list[dict[str, Any]] = []
    context = _tokens(task)
    for name, args, output in calls:
        events.append({"type": "step_start", "part": {"type": "step-start"}})
        state_input = {"pattern": args["pattern"], "path": "docs"} if name == "search" else {"filePath": args["file_path"]}
        events.append({"type": "tool_use", "part": {"type": "tool", "tool": names[name], "state": {"status": "completed", "input": state_input, "output": output}}})
        events.append({"type": "step_finish", "part": {"type": "step-finish", "tokens": {"input": context, "output": 20, "reasoning": 0, "cache": {"read": 0, "write": 0}}, "cost": (context * 0.1 + 20 * 0.5) / 1e6}})
        context += _tokens(output) + 20
    out = _tokens(answer)
    events.append({"type": "step_start", "part": {"type": "step-start"}})
    events.append({"type": "text", "part": {"type": "text", "text": answer}})
    events.append({"type": "step_finish", "part": {"type": "step-finish", "tokens": {"input": context, "output": out, "reasoning": 0, "cache": {"read": 0, "write": 0}}, "cost": (context * 0.1 + out * 0.5) / 1e6}})
    return events


FORMATS = {"claude_code": _claude_code, "codex": _codex, "opencode": _opencode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=[*FORMATS, "text"], default="text")
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("prompt", nargs="?", default="")
    args = parser.parse_args(argv)
    workspace = Path.cwd()
    task, calls, answer = _plan(workspace)
    if args.answer_file:
        (workspace / args.answer_file).write_text(answer + "\n", encoding="utf-8")
    if args.format == "text":
        print(answer)
    else:
        for event in FORMATS[args.format](task, calls, answer):
            print(json.dumps(event, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
