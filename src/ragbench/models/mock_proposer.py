"""Mock-safe harness proposals: a scripted edit per kind and proposal number, so `ragbench optimize --mock` runs the whole loop offline.

The edit is read off the request's `Kind:`, `Proposal number:`, `Fields you may change:` and `Current harness (JSON):` lines
(`optimize/proposer.py`). INV edits are general (answer form, grounding, depth); DOM edits name the most common question kind from
`task.md`. Proposal number N takes the Nth edit the system has fields for, skipping one that would change nothing. COMPOSE merges the
`Candidate changes (JSON):` (later names win a field); proposal number N > 1 leaves out the (N-1)th candidate that changes something.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ragbench.models.prompts import PROPOSE_HARNESS_MARKER

Edit = Callable[[dict[str, Any], str], tuple[str, Any]]


def _append(field: str, sentence: str) -> Edit:
    return lambda current, topic: (field, f"{current.get(field) or ''} {sentence.format(topic=topic)}".strip())


def _depth(step: int) -> Edit:
    return lambda current, topic: ("top_k", min(50, int(current.get("top_k") or 5) + step))


INV_EDITS: list[Edit] = [
    lambda current, topic: ("answer_instructions", "Lead with the direct answer in one sentence, then add only the details the question asks for."),
    _append("grounding", "Check every fact against the context before you write it."),
    _depth(2),
    _append("refusal_policy", "Say so only after reading every passage."),
]
DOM_EDITS: list[Edit] = [
    _append("system_prompt", "Most questions here are `{topic}` questions: answer them with the exact names, dates and figures the documents give."),
    _depth(4),
    lambda current, topic: ("answer_instructions", "Quote figures, dates and names exactly as the documents write them."),
    _append("citation_style", "Cite the document of every figure you give."),
]


def _line(prompt: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}\s*(.*)$", prompt, flags=re.M)
    return match[1].strip() if match else ""


def _proposal(prompt: str) -> str:
    request = prompt.rsplit("=== request ===", 1)[-1]  # the evidence above it has `Kind:` lines of its own (a rollout's question kind)
    kind = next((name for name in ("DOM", "COMPOSE") if _line(request, "Kind:").startswith(name)), "INV")
    numbered = re.match(r"\d+", _line(request, "Proposal number:"))
    number = int(numbered[0]) if numbered else 1
    fields = {name.strip() for name in _line(request, "Fields you may change:").split(",") if name.strip()}
    current = json.loads(_line(request, "Current harness (JSON):") or "{}")
    if kind == "COMPOSE":
        return _composition(json.loads(_line(request, "Candidate changes (JSON):") or "{}"), number)
    kinds = re.search(r"^- Kinds: `([^`]+)`", prompt, flags=re.M)
    topic = kinds[1] if kinds else "factual"
    usable = [(name, value) for name, value in (edit(current, topic) for edit in (DOM_EDITS if kind == "DOM" else INV_EDITS)) if name in fields]
    changing = [(name, value) for name, value in usable if current.get(name) != value]
    if not changing:
        return json.dumps({"rationale": f"Scripted {kind} proposal {number}: nothing left to change.", "harness": {}})
    name, value = changing[(number - 1) % len(changing)]
    return json.dumps({"rationale": f"Scripted {kind} proposal {number}: change `{name}`.", "harness": {name: value}})


def _composition(candidates: dict[str, dict[str, Any]], number: int) -> str:
    changing = [name for name in sorted(candidates) if candidates[name]]
    left_out = changing[(number - 2) % len(changing)] if number > 1 and len(changing) > 1 else None
    merged: dict[str, Any] = {}
    for name in changing:
        if name != left_out:
            merged.update(candidates[name])
    note = f", without `{left_out}`" if left_out else ""
    return json.dumps({"rationale": f"Scripted composition {number}: every candidate's change{note}.", "harness": merged})


def register(mock_llm: Any) -> None:
    """Add the responder to `MockLLM` (called when `ragbench.models.llms` is imported)."""
    mock_llm.register_responder(lambda prompt, json_mode: PROPOSE_HARNESS_MARKER in prompt, _proposal, name="propose_harness")
