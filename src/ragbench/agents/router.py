"""Routing a question to the pipeline suited to it: a cheap rule-based router (no LLM, no cost) and the parsing of an LLM router's choice.

The rule-based router knows four roles by route name: `lexical` (exact identifiers), `computation` (arithmetic, percentages, date arithmetic),
`multi_hop` (comparisons and questions with several linked parts) and `default`. It never invents a route: a role the config does not define
falls through to the next matching role, and finally to `default`.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from ragbench.agents.replies import parse_route
from ragbench.utils.query_planning import split_subquestions_locally

DEFAULT_ROUTE = "default"
HEURISTIC_ROLES = ("lexical", "computation", "multi_hop")
ROLE_DESCRIPTIONS = {
    "default": "General questions: look up facts stated in the documents.",
    "lexical": "Questions about an exact identifier, code, product or error name, or quoted term that must match the text literally.",
    "computation": "Questions that need arithmetic, percentages, totals, or date calculations on figures found in the documents.",
    "multi_hop": "Questions that compare things or combine facts from several documents, or need several lookups one after another.",
}

IDENTIFIERS = (
    re.compile(r"\b[A-Za-z]{1,8}[-_]\d{2,}[A-Za-z0-9_-]*\b"),  # HS-4127, ID-1234, INV_2024
    re.compile(r"\b[A-Z]{2,}\d{2,}\b"),  # HS4127
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),  # UUID
    re.compile(r"\bv?\d+\.\d+\.\d+\b"),  # 2.1.3
    re.compile(r'"[^"]{3,}"|`[^`]{2,}`'),  # a quoted phrase or code span
)
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?(?![\w-])")
YEAR = re.compile(r"^(19|20)\d{2}$")
ARITHMETIC = re.compile(
    r"%|\b(percent(age)?|ratio|average|mean of|sum of|difference between|by how much|how much (more|less|higher|lower|faster|slower)|"
    r"growth rate|times as|multiplied|divided|increase[sd]? by|decrease[sd]? by|grow(th|n)? from)\b",
    re.I,
)
TOTALS = re.compile(r"\b(total|combined|altogether|in total|overall)\b", re.I)
QUANTITIES = re.compile(r"\b(number|amount|cost|price|revenue|volume|count|spend|budget)\b", re.I)
DATE_ARITHMETIC = re.compile(
    r"\bhow many (days|weeks|months|years)\b|\b(days|weeks|months|years) (between|after|before|since|until|from)\b|\bhow long (did|was|is|has|have|until|since|between)\b",
    re.I,
)
COMPARISON = re.compile(r"\b(compare|contrast|versus|vs\.?|differ(s|ence|ences)? between|both|and then|after that|relates? to|which two)\b", re.I)
HYPOTHETICAL_COST = re.compile(r"\b(would|will)\b.*\b(cost|total|come to|amount to)\b", re.I)
LINKED_PART = re.compile(r"\band (what|which|who|when|where|how|why)\b", re.I)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    fallback: bool = False  # the router could not decide (unreadable or unknown choice) and the default route was used


def _identifier(question: str) -> str | None:
    for pattern in IDENTIFIERS:
        if match := pattern.search(question):
            return f"exact identifier or phrase {match.group(0)!r}"
    return None


def _computation(question: str) -> str | None:
    figures = [n for n in NUMBER.findall(question) if not YEAR.match(n.strip("%,"))]
    if DATE_ARITHMETIC.search(question):
        return "date arithmetic wording"
    if len(figures) >= 2 and (ARITHMETIC.search(question) or TOTALS.search(question)):
        return "several figures to combine"
    if ARITHMETIC.search(question):
        return "percentage, ratio or average wording"
    if figures and HYPOTHETICAL_COST.search(question):
        return "a figure to scale into a cost"
    if TOTALS.search(question) and QUANTITIES.search(question):
        return "a total over quantities"
    return None


def _multi_hop(question: str) -> str | None:
    if COMPARISON.search(question):
        return "comparison wording"
    if len(split_subquestions_locally(question)) >= 2 or LINKED_PART.search(question) or question.count("?") >= 2:
        return "several linked parts"
    return None


_DETECTORS = (("lexical", _identifier), ("computation", _computation), ("multi_hop", _multi_hop))


def heuristic_route(question: str, available: Collection[str]) -> RouteDecision:
    """The route for `question` among the `available` route names, by the first matching role that is configured, else `default`."""
    skipped: list[str] = []
    for role, detect in _DETECTORS:
        if (why := detect(question)) is None:
            continue
        if role in available:
            return RouteDecision(role, why)
        skipped.append(role)
    reason = "no special pattern" if not skipped else f"looks like {', '.join(skipped)}, which is not configured"
    return RouteDecision(DEFAULT_ROUTE, reason)


def describe_routes(names: Collection[str], descriptions: dict[str, str]) -> dict[str, str]:
    """One description per route for the LLM router: the config's own, else the built-in one for a known role, else a placeholder."""
    return {name: descriptions.get(name) or ROLE_DESCRIPTIONS.get(name, "(no description)") for name in names}


def decide_from_reply(text: str, available: Collection[str]) -> RouteDecision:
    """An LLM router's reply as a decision; an unreadable reply or an unknown route name falls back to `default`."""
    parsed = parse_route(text)
    if parsed is None:
        return RouteDecision(DEFAULT_ROUTE, "the router's reply had no route", fallback=True)
    route, reason = parsed
    if route not in available:
        close = {name.lower(): name for name in available}.get(route.lower())
        if close is None:
            return RouteDecision(DEFAULT_ROUTE, f"the router chose unknown route {route!r}", fallback=True)
        route = close
    return RouteDecision(route, reason or "chosen by the LLM router")
