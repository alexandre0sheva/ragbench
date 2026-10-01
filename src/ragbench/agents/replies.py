"""Parsing of the agentic systems' JSON replies. Every parser returns `None` (or a failure flag) for an unusable reply, never raises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragbench.utils.query_planning import loads_lenient

RELEVANT, AMBIGUOUS, IRRELEVANT = "relevant", "ambiguous", "irrelevant"
DONE = "DONE"


def normalize_grade(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text.startswith(("irrelevant", "not ", "no")):
        return IRRELEVANT
    if "ambig" in text or "partial" in text:
        return AMBIGUOUS
    if "relevant" in text or text.startswith("yes"):
        return RELEVANT
    return None


def parse_grades(text: str, count: int) -> list[str] | None:
    """One grade per passage (`count` of them) from `{"grades": [{"id": 1, "grade": ...}]}`, a list of grades, or an `{id: grade}` map.

    A passage the reply does not cover is `ambiguous`. `None` when no grade could be read at all.
    """
    data = loads_lenient(text)
    if isinstance(data, dict) and isinstance(data.get("grades"), (list, dict)):
        data = data["grades"]
    grades: dict[int, str] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).strip().isdigit() and (grade := normalize_grade(value)):
                grades[int(key)] = grade
    elif isinstance(data, list):
        for position, entry in enumerate(data, start=1):
            number, value = (entry.get("id", position), entry.get("grade")) if isinstance(entry, dict) else (position, entry)
            grade = normalize_grade(value) if value is not None else None
            if grade and str(number).strip().isdigit():
                grades[int(number)] = grade
    if not any(1 <= number <= count for number in grades):
        return None
    return [grades.get(number, AMBIGUOUS) for number in range(1, count + 1)]


def parse_query(text: str) -> str | None:
    data = loads_lenient(text)
    value = data.get("query") if isinstance(data, dict) else None
    return " ".join(value.split()) if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class NextHop:
    known: str
    missing: str
    next_query: str | None  # None: the evidence is enough (the model said DONE or gave no query)


def parse_next_hop(text: str) -> NextHop | None:
    data = loads_lenient(text)
    if not isinstance(data, dict) or not any(key in data for key in ("next_query", "known", "missing")):
        return None
    query = data.get("next_query")
    query = " ".join(query.split()) if isinstance(query, str) else ""
    done = not query or query.strip(" .!").upper() == DONE
    return NextHop(known=str(data.get("known") or "").strip(), missing=str(data.get("missing") or "").strip(), next_query=None if done else query)


def parse_groundedness(text: str) -> tuple[bool, list[str]] | None:
    """`(supported, unsupported claims)`; a reply that lists claims counts as unsupported even if it says `supported: true`."""
    data = loads_lenient(text)
    if not isinstance(data, dict) or "supported" not in data:
        return None
    raw = data.get("unsupported_claims")
    claims = [" ".join(str(item).split()) for item in raw if str(item).strip()] if isinstance(raw, list) else []
    supported = data["supported"] is True or str(data["supported"]).strip().lower() in {"true", "yes"}
    return supported and not claims, claims


def parse_route(text: str) -> tuple[str, str] | None:
    """`(route, reason)` from `{"route": ..., "reason": ...}`, or None when there is no route."""
    data = loads_lenient(text)
    route = data.get("route") if isinstance(data, dict) else None
    if not isinstance(route, str) or not route.strip():
        return None
    return route.strip(), " ".join(str(data.get("reason") or "").split())
