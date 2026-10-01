from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragbench.models.prompts import GENERATE_QUERY_VARIANTS_MARKER, PLAN_SUBQUESTIONS_MARKER
from ragbench.utils.text import normalize_text, unique_preserve_order

if TYPE_CHECKING:  # `models.llms` imports this module for its mock responders
    from ragbench.models.llms import LLM, LLMResult

QUESTION_PREFIX_RE = re.compile(r"^(compare|contrast|summarize|explain|what|which|who|when|where|how|why|did|does|do|is|are)\b", re.I)
CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-\s]+(?:AI|API|Suite|Portal|Pilot|Risk|[A-Z][A-Za-z0-9]*))*\b")
CLAUSE_SPLIT_RE = re.compile(r"\s+\band\s+(?=(?:what|which|who|when|where|how|why|did|does|do|is|are)\b)", re.I)


def generate_query_variants(question: str, max_queries: int = 4) -> list[str]:
    """Generate lightweight local query variants for multi-hop retrieval.

    This is intentionally domain-agnostic. It handles common complex-question
    patterns without using an LLM: comparison sides, explicit subclauses, and
    salient capitalized phrases.
    """
    clean = normalize_text(question).strip(" ?")
    if not clean:
        return []
    variants = [clean]
    variants.extend(_comparison_variants(clean))
    variants.extend(_clause_variants(clean))
    variants.extend(_entity_variants(clean))
    return unique_preserve_order(v for v in variants if len(v.split()) >= 2)[:max_queries]


def _comparison_variants(question: str) -> list[str]:
    lower = question.lower()
    variants: list[str] = []
    if " vs " in lower or " versus " in lower:
        parts = re.split(r"\s+(?:vs\.?|versus)\s+", question, maxsplit=1, flags=re.I)
        variants.extend(part.strip(" ?") for part in parts)
    match = re.search(r"\bcompare\s+(.+?)\s+\band\s+(.+?)(?:\s+\bby\b|\s+\bfor\b|$)", question, flags=re.I)
    if match:
        suffix_match = re.search(r"\b(?:by|for)\s+(.+)$", question, flags=re.I)
        suffix = suffix_match.group(1).strip(" ?") if suffix_match else ""
        for side in match.groups():
            side = side.strip(" ?")
            variants.append(f"{side} {suffix}".strip())
    return variants


def _clause_variants(question: str) -> list[str]:
    parts = [part.strip(" ?") for part in CLAUSE_SPLIT_RE.split(question) if part.strip(" ?")]
    if len(parts) <= 1:
        return []
    return parts


def _entity_variants(question: str) -> list[str]:
    phrases = []
    for match in CAPITALIZED_PHRASE_RE.finditer(question):
        phrase = match.group(0).strip()
        if QUESTION_PREFIX_RE.match(phrase):
            continue
        if len(phrase) > 2:
            phrases.append(phrase)
    return phrases


@dataclass(frozen=True)
class QueryPlan:
    """What an LLM planner produced: the usable items (queries or sub-questions), the raw call, and whether the answer was unusable."""

    items: list[str]
    result: LLMResult
    fallback: bool = False  # the reply held no usable items; the caller falls back to the original question


def parse_string_list(text: str, keys: tuple[str, ...]) -> list[str]:
    """Strings out of an LLM's JSON reply: `{"<key>": [...]}` for any of `keys`, or a bare list; tolerates code fences and surrounding prose.

    Items may be strings or objects with a `question` / `query` / `text` field. Blank and duplicate items (ignoring case) are dropped;
    anything unparseable yields `[]`.
    """
    data = loads_lenient(text)
    if isinstance(data, dict):
        data = next((data[key] for key in keys if isinstance(data.get(key), list)), None)
    if not isinstance(data, list):
        return []
    items: list[str] = []
    for entry in data:
        if isinstance(entry, dict):
            entry = next((entry[field] for field in ("question", "query", "text") if isinstance(entry.get(field), str)), None)
        if isinstance(entry, str) and entry.strip():
            items.append(normalize_text(entry))
    return _dedupe(items)


def plan_query_variants_llm(llm: LLM, question: str, num_queries: int) -> QueryPlan:
    """Ask `llm` for up to `num_queries` alternative search queries (RAG-Fusion); the question itself is never among them."""
    messages = [
        {"role": "system", "content": "You write search queries for a document search system. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{GENERATE_QUERY_VARIANTS_MARKER}: write {num_queries} different queries that look for the same answer with different wording, "
                "synonyms, or angles (one may name a key entity or fact directly). Do not repeat the question. "
                'Return JSON like {"queries": ["...", "..."]}.\n'
                f"Question: {question}"
            ),
        },
    ]
    result = llm.generate(messages, temperature=0, json_mode=True, max_tokens=60 * num_queries + 60)
    items = parse_string_list(result.text, keys=("queries",))
    own = normalize_text(question).lower()
    return QueryPlan(items=[item for item in items if item.lower() != own][:num_queries], result=result, fallback=not items)


def plan_subquestions_llm(llm: LLM, question: str, max_subquestions: int) -> QueryPlan:
    """Ask `llm` to split `question` into ordered sub-questions (at most `max_subquestions`); an unusable reply yields the question itself."""
    messages = [
        {"role": "system", "content": "You plan how to answer questions from a document collection. Answer with JSON only."},
        {
            "role": "user",
            "content": (
                f"{PLAN_SUBQUESTIONS_MARKER}, at most {max_subquestions}. Each sub-question must be self-contained and answerable from a single "
                "passage; later ones may build on earlier ones. A simple question needs only one. "
                'Return JSON like {"sub_questions": ["...", "..."]}.\n'
                f"Question: {question}"
            ),
        },
    ]
    result = llm.generate(messages, temperature=0, json_mode=True, max_tokens=80 * max_subquestions + 60)
    items = parse_string_list(result.text, keys=("sub_questions", "subquestions", "questions"))[:max_subquestions]
    return QueryPlan(items=items or [question], result=result, fallback=not items)


def split_subquestions_locally(question: str) -> list[str]:
    """Rule-based split of a compound question (comparison sides, `... and what ...` clauses); `[]` when it does not obviously split."""
    clean = normalize_text(question).strip(" ?")
    return _dedupe([part for part in (_clause_variants(clean) or _comparison_variants(clean)) if len(part.split()) >= 2])


def loads_lenient(text: str):
    """`json.loads` that tolerates code fences and prose around the JSON; `None` when nothing parses."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text).strip()
    for candidate in (text, *_bracketed(text)):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _bracketed(text: str) -> list[str]:
    spans = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            spans.append(text[start : end + 1])
    return spans


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique
