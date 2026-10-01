"""The mock model's tool-using behaviour: a deterministic rule-based policy, so `--mock` runs exercise the whole agent loop (tool calls,
observations, budgets, traces) without a network. It exists to test machinery, not to answer well.

Policy, per question: an arithmetic-looking question gets one `calculator` call and a question with dates one `date_calc` call (when those
tools exist); anything else gets one `search` (or `corpus_grep` when there is no `search`). A grep that found a document is followed by one
`read_document` of its first hit, and a grep that found nothing is retried once on its longest word. Then it answers from the observations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ragbench.models.prompts import FORCE_ANSWER_MARKER, REACT_JSON_MARKER
from ragbench.utils.text import tokenize

STOPWORDS = {"what", "which", "when", "where", "does", "that", "this", "with", "from", "have", "were", "been", "their", "about", "there", "would", "much", "many"}
ARITHMETIC_WORDS = re.compile(r"\b(total|sum|difference|percent|percentage|times|multiply|multiplied|divide|divided|ratio|average|increase|decrease|grow|growth|cost|per)\b|%", re.I)
DATE_WORDS = re.compile(r"\b(days?|weeks?|months?|years?|between|since|until|after|before)\b", re.I)
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w-])")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}\b")
GREP_HIT = re.compile(r"^([^\s:]+):\d+[:-]", re.M)
MAX_TOOL_CALLS = 3


@dataclass
class MockTurn:
    tool: str | None = None  # None: answer
    args: dict[str, Any] = field(default_factory=dict)


def question_of(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            match = re.match(r"Question:\s*(.*)", str(message.get("content") or ""), flags=re.S)
            if match:
                return match.group(1).strip()
    return ""


def observations_of(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The `(tool, result text)` pairs already in the conversation, from native tool messages or ReAct `Observation from ...` messages."""
    names: dict[str, str] = {}
    found: list[tuple[str, str]] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
        if message.get("role") == "tool":
            found.append((names.get(message.get("tool_call_id", ""), "tool"), str(message.get("content") or "")))
        elif message.get("role") == "user":
            match = re.match(r"Observation from ([\w-]+):\n(.*)", str(message.get("content") or ""), flags=re.S)
            if match:
                found.append((match.group(1), match.group(2)))
    return found


def forced_to_answer(messages: list[dict[str, Any]]) -> bool:
    return FORCE_ANSWER_MARKER in str(messages[-1].get("content") or "")


def available_react_tools(prompt: str) -> list[str]:
    return re.findall(r"^- ([A-Za-z0-9_-]+): ", prompt.split("Available tools:", 1)[-1], flags=re.M) if REACT_JSON_MARKER in prompt else []


def keywords(question: str, limit: int = 8) -> list[str]:
    words = [w for w in dict.fromkeys(tokenize(question)) if len(w) > 3 and w not in STOPWORDS]
    return words[:limit]


def decide(question: str, observations: list[tuple[str, str]], available: list[str], force_answer: bool = False) -> MockTurn:
    if force_answer or len(observations) >= MAX_TOOL_CALLS:
        return MockTurn()
    if not observations:
        return _first_call(question, available)
    used = [name for name, _ in observations]
    last_tool, last_text = observations[-1]
    if last_tool == "corpus_grep":
        hit = GREP_HIT.search(last_text)
        if hit and "read_document" in available and "read_document" not in used:
            return MockTurn("read_document", {"doc_id": hit.group(1), "length": 1500})
        if last_text.startswith("No matches") and used.count("corpus_grep") < 2 and (words := sorted(keywords(question), key=len, reverse=True)):
            return MockTurn("corpus_grep", {"pattern": words[0]})
    return MockTurn()


def _first_call(question: str, available: list[str]) -> MockTurn:
    numbers = NUMBER.findall(question)
    dates = DATE.findall(question)
    if "calculator" in available and len(numbers) >= 2 and ARITHMETIC_WORDS.search(question):
        operator = "-" if re.search(r"difference|decrease", question, re.I) else "*" if re.search(r"times|multipl|per", question, re.I) else "/" if re.search(r"ratio|divid", question, re.I) else "+"
        return MockTurn("calculator", {"expression": f" {operator} ".join(n.replace(",", "") for n in numbers[:2])})
    if "date_calc" in available and dates and DATE_WORDS.search(question):
        return MockTurn("date_calc", {"operation": "diff", "date": dates[0], "other_date": dates[1] if len(dates) > 1 else "today"})
    words = keywords(question)
    if "search" in available:
        return MockTurn("search", {"query": " ".join(words) or question})
    if "corpus_grep" in available:
        longest = sorted(words, key=len, reverse=True)[:2]
        return MockTurn("corpus_grep", {"pattern": " ".join(longest) or question})
    if "list_documents" in available:
        return MockTurn("list_documents", {})
    return MockTurn()


def native_call_id(observations: list[tuple[str, str]]) -> str:
    return f"mock_call_{len(observations) + 1}"


def react_reply(turn: MockTurn, answer: str) -> str:
    if turn.tool:
        return json.dumps({"action": "tool", "tool": turn.tool, "args": turn.args})
    return json.dumps({"action": "answer", "answer": answer})
