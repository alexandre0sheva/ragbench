"""Mock-safe answers to the prompts of `graph_rag`: entity/relation extraction, community reports and the map step of global search.

Capitalised phrases stand in for named entities and two names in one sentence for a relation, so a mock run builds a real graph
(with communities) from the corpus and exercises every path offline. The replies are about as long as a real model's, which keeps
`ragbench estimate` honest about extraction output tokens.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragbench.models.prompts import EXTRACT_GRAPH_MARKER, MAP_COMMUNITY_MARKER, SUMMARIZE_COMMUNITY_MARKER
from ragbench.utils.text import tokenize

_NAME = re.compile(r"[A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)*")
# Capitalised only because they open a sentence: dropped from the front of a name.
_LEADING = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "in", "on", "at", "for", "and", "but", "or", "if", "when", "each",
    "every", "all", "our", "we", "you", "they", "he", "she", "there", "what", "which", "who", "how", "where", "why", "after", "before",
}  # fmt: skip
DESCRIPTION_CHARS = 200
REPORT_CHARS = 700


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip() and not s.lstrip().startswith("#")]


def _names(sentence: str) -> list[str]:
    names: list[str] = []
    for match in _NAME.finditer(sentence):
        words = match.group(0).split()
        while words and words[0].lower() in _LEADING:
            words = words[1:]
        if words and " ".join(words) not in names:
            names.append(" ".join(words))
    return names


def _extract(prompt: str) -> str:
    text = prompt.split("<text>", 1)[-1].split("</text>", 1)[0]
    match = re.search(r"at most (\d+) entities", prompt)
    limit = int(match[1]) if match else 15
    entities: dict[str, dict[str, str]] = {}
    relations: list[dict[str, str]] = []
    for sentence in _sentences(text):
        names = []
        for name in _names(sentence):
            if name in entities or len(entities) < limit:
                entities.setdefault(name, {"name": name, "type": "concept", "description": sentence[:DESCRIPTION_CHARS]})
                names.append(name)
        relations.extend({"source": a, "target": b, "description": sentence[:DESCRIPTION_CHARS]} for a, b in zip(names, names[1:], strict=False))
    return json.dumps({"entities": list(entities.values()), "relations": relations})


def _report(prompt: str) -> str:
    names = re.findall(r"^- (.+?) \(", prompt.split("Entities:", 1)[-1].split("Relations:", 1)[0], flags=re.M)
    entities, _, relations = prompt.split("Entities:", 1)[-1].partition("Relations:")
    facts = re.findall(r"^- .+? — .+?: (.+)$", relations, flags=re.M) + re.findall(r"^- .+? \(.+?\): (.+)$", entities, flags=re.M)
    body = " ".join(dict.fromkeys(fact.strip() for fact in facts if fact.strip()))
    return f"Community of {', '.join(names[:3])}\n{body}"[:REPORT_CHARS]


def _map(prompt: str) -> str:
    match = re.search(r"Question:\s*(.*?)\n\nReport:", prompt, flags=re.S)
    question = match[1] if match else ""
    wanted = {word for word in tokenize(question) if len(word) > 3}
    points = []
    for sentence in _sentences(prompt.split("Report:", 1)[-1]):
        shared = len(wanted & set(tokenize(sentence)))
        if shared:
            points.append({"text": sentence, "score": min(100, 25 * shared)})
    return json.dumps({"points": points})


def register(mock_llm: Any) -> None:
    """Add the responders to `MockLLM` (called when `ragbench.models.llms` is imported); registered after JSON mode so they win over it."""
    for name, marker, respond in (
        ("extract_graph", EXTRACT_GRAPH_MARKER, _extract),
        ("summarize_community", SUMMARIZE_COMMUNITY_MARKER, _report),
        ("map_community", MAP_COMMUNITY_MARKER, _map),
    ):
        mock_llm.register_responder(lambda prompt, json_mode, marker=marker: marker in prompt, respond, name=name)
