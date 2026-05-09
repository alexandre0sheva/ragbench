from __future__ import annotations

import re

from ragbench.utils.text import normalize_text, unique_preserve_order

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
