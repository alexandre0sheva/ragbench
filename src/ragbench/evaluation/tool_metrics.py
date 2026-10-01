"""Tool-use metrics derived from the `tool` steps recorded in `per_question_results.jsonl`."""

from __future__ import annotations

from typing import Any


def tool_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for step in row.get("steps", []) if step["kind"] == "tool"]


def _failed(step: dict[str, Any]) -> bool:
    return bool(step.get("metadata", {}).get("error"))


def usage_rows(system: str, configured: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per tool the system was offered or used: calls, errors, truncated outputs, questions that used it, latency and cost.

    A tool that was offered but never called still gets a row (with 0 calls), which is itself a result."""
    used: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        for step in tool_steps(row):
            used.setdefault(step["name"], []).append((index, step))
    names = [*configured, *(name for name in used if name not in configured)]
    result = []
    for name in names:
        calls = used.get(name, [])
        errors = sum(1 for _, step in calls if _failed(step))
        result.append(
            {
                "system": system,
                "tool": name,
                "calls": len(calls),
                "errors": errors,
                "error_rate": errors / len(calls) if calls else None,
                "truncated": sum(1 for _, step in calls if step.get("metadata", {}).get("truncated")),
                "questions_using": len({index for index, _ in calls}),
                "avg_latency_ms": sum(step["latency_ms"] for _, step in calls) / len(calls) if calls else None,
                "total_cost_usd": sum(step["cost"]["total_cost"] for _, step in calls),
            }
        )
    return result


def summary_fields(rows: list[dict[str, Any]], tooled: bool) -> dict[str, float | None]:
    """`avg_tool_calls`, `tool_error_rate` and `required_tool_used_rate` for a system that has tools; nothing for one that does not.

    `required_tool_used_rate` is over the questions that declare `requires_tools`: the share where the system called *every*
    tool the question needs at least once (a question that needs none is not counted)."""
    if not tooled or not rows:
        return {}
    steps = [tool_steps(row) for row in rows]
    total = sum(len(s) for s in steps)
    errors = sum(1 for s in steps for step in s if _failed(step))
    needing = [(row, s) for row, s in zip(rows, steps, strict=True) if row.get("requires_tools")]
    used_all = sum(1 for row, s in needing if set(row["requires_tools"]) <= {step["name"] for step in s})
    return {
        "avg_tool_calls": total / len(rows),
        "tool_error_rate": errors / total if total else None,
        "required_tool_used_rate": used_all / len(needing) if needing else None,
    }
