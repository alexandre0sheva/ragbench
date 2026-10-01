"""Routing metrics of the `adaptive` system, derived from the `route` recorded on each question row."""

from __future__ import annotations

from typing import Any


def route_accuracy(rows: list[dict[str, Any]]) -> float | None:
    """Share of the questions with a `routing_hint` that went to that route; None when no question carries a hint."""
    hinted = [row for row in rows if row.get("route") and row.get("routing_hint")]
    return sum(1 for row in hinted if row["route"] == row["routing_hint"]) / len(hinted) if hinted else None


def route_rows(system: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per route a system used: how many questions it took, their mean answer score and cost, and (with hints) its precision and recall."""
    routed = [row for row in rows if row.get("route")]
    if not routed:
        return []
    result = []
    for route in sorted({row["route"] for row in routed}, key=lambda name: (-sum(1 for r in routed if r["route"] == name), name)):
        mine = [row for row in routed if row["route"] == route]
        wanted = [row for row in routed if row.get("routing_hint") == route]
        hinted_here = [row for row in mine if row.get("routing_hint")]
        result.append(
            {
                "system": system,
                "route": route,
                "questions": len(mine),
                "share": len(mine) / len(routed),
                "avg_answer_score": sum(row["answer_judge"]["answer_score"] for row in mine) / len(mine),
                "avg_cost_per_question": sum(row["cost"]["total_cost"] for row in mine) / len(mine),
                "precision": sum(1 for row in hinted_here if row["routing_hint"] == route) / len(hinted_here) if hinted_here else None,
                "recall": sum(1 for row in wanted if row["route"] == route) / len(wanted) if wanted else None,
            }
        )
    return result
