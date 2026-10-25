"""Agent trajectory metrics, from the steps recorded in `per_question_results.jsonl`: how often an agent repeats a search, and how many
actions it takes to reach a relevant document. Tool errors are `tool_error_rate` (`tool_metrics`)."""

from __future__ import annotations

from typing import Any

from ragbench.rag_systems.trace import UNTRACKED

ACTION_KINDS = ("retrieve", "tool")


def actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The searches, reads and other tool calls of one answer, in order."""
    return [step for step in row.get("steps") or [] if step["kind"] in ACTION_KINDS and step["name"] != UNTRACKED]


def redundant(steps: list[dict[str, Any]]) -> int:
    """Actions that repeat an earlier one of the same answer: same tool, same input."""
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for step in steps:
        if step.get("input_preview") is None:
            continue
        key = (step["name"], " ".join(str(step["input_preview"]).lower().split()))
        repeats += key in seen
        seen.add(key)
    return repeats


def steps_to_evidence(steps: list[dict[str, Any]], relevant: set[str]) -> int | None:
    """How many actions the agent took up to and including the first one that surfaced a relevant document; None if none did."""
    for index, step in enumerate(steps, start=1):
        if relevant & set(step.get("metadata", {}).get("doc_ids") or []):
            return index
    return None


def summary_fields(rows: list[dict[str, Any]], agentic: bool) -> dict[str, float | None]:
    """`redundant_search_rate` and `avg_steps_to_evidence` for an agentic system; nothing for one that is not.

    `avg_steps_to_evidence` is over the questions with relevant documents where the agent reached one (the others are not counted:
    `hit@k` says how often it never did)."""
    if not agentic or not rows:
        return {}
    per_row = [actions(row) for row in rows]
    with_input = sum(1 for steps in per_row for step in steps if step.get("input_preview") is not None)
    found = [
        reached
        for row, steps in zip(rows, per_row, strict=True)
        if row.get("relevant_doc_ids") and (reached := steps_to_evidence(steps, set(row["relevant_doc_ids"]))) is not None
    ]
    return {
        "redundant_search_rate": sum(redundant(steps) for steps in per_row) / with_input if with_input else None,
        "avg_steps_to_evidence": sum(found) / len(found) if found else None,
    }
