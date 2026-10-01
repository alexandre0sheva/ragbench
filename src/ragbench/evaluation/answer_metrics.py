"""Deterministic answer metrics: no LLM, no network, identical on every run.

They complement the LLM judge: they cannot read meaning, but they never drift, never cost anything and cannot prefer
one model family over another. Token F1 and exact match compare against `reference_answer` (SQuAD-style
normalization), keyword recall against `expected_keywords`, and the refusal metrics against whether the question is
answerable at all.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_CITATION = re.compile(r"\[[^\[\]\n]{1,80}\]")  # [doc_005], [doc_005 | chunk 2]: a citation, not part of the answer
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_PUNCTUATION = re.compile(r"[^\w\s.]")
_STRAY_PERIOD = re.compile(r"\.(?!\d)|(?<!\d)\.")  # a period that is not a decimal point
_ARTICLES = frozenset({"a", "an", "the"})

# What the generation prompts tell a model to say when the context has no answer, plus the usual paraphrases of it.
_REFUSAL = re.compile(
    r"could(?: not|n't) find (?:the|an?|any) (?:answer|information)"
    r"|cannot find (?:the|an?|any) (?:answer|information)"
    r"|can't find (?:the|an?|any) (?:answer|information)"
    r"|(?:do|does)(?: not|n't) (?:contain|provide|mention|include|have) (?:the |any |enough |sufficient |relevant )?(?:information|answer|details)"
    r"|(?:not enough|insufficient|no relevant) information"
    r"|no information (?:is |was )?(?:available|provided|found)"
    r"|unable to (?:find|answer|determine)"
    r"|not available in the (?:provided |given )?(?:documents?|context|sources?)"
)


def normalize_answer(text: str) -> str:
    """Lowercase, drop citations, punctuation (decimal points stay), digit grouping commas and the articles a/an/the."""
    text = _THOUSANDS.sub("", _CITATION.sub(" ", text.lower()))
    text = _STRAY_PERIOD.sub(" ", _PUNCTUATION.sub(" ", text))
    return " ".join(token for token in text.split() if token not in _ARTICLES)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0


def token_f1(pred: str, ref: str) -> float:
    """Harmonic mean of token precision and recall over the normalized texts (two empty texts match)."""
    pred_tokens, ref_tokens = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, keywords: list[str]) -> float:
    """Share of `keywords` that appear in `pred` as whole tokens (vacuously 1.0 for none; the evaluator reports those as missing)."""
    if not keywords:
        return 1.0
    haystack = f" {normalize_answer(pred)} "
    hits = sum(1 for keyword in keywords if (needle := normalize_answer(keyword)) and f" {needle} " in haystack)
    return hits / len(keywords)


def is_refusal(answer: str) -> bool:
    """Whether the answer declines to answer ("I could not find the answer in the provided documents.")."""
    return _REFUSAL.search(answer.lower().replace("’", "'")) is not None


def refusal_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """Abstention quality from rows carrying an `answerable` and a `refused` flag. A rate with an empty denominator is None.

    `abstain_precision`: of the refusals, the share that were right (the question had no answer).
    `abstain_recall`: of the unanswerable questions, the share that were refused.
    `false_refusal_rate`: of the answerable questions, the share that were refused anyway.
    """
    counts = Counter((bool(row["answerable"]), bool(row["refused"])) for row in rows)
    refused = counts[(True, True)] + counts[(False, True)]
    unanswerable = counts[(False, True)] + counts[(False, False)]
    answerable = counts[(True, True)] + counts[(True, False)]
    return {
        "abstain_precision": counts[(False, True)] / refused if refused else None,
        "abstain_recall": counts[(False, True)] / unanswerable if unanswerable else None,
        "false_refusal_rate": counts[(True, True)] / answerable if answerable else None,
    }


def answer_metrics(answer: str, reference_answer: str | None, expected_keywords: list[str], answerable: bool = True) -> dict[str, float | None]:
    """Per-question `exact_match`, `token_f1` and `keyword_recall`.

    None where the question has no reference or no keywords, and for every unanswerable question: its reference only says
    "not in the documents", so overlap with it would score wording, not correctness (the refusal metrics judge those).
    """
    return {
        "exact_match": exact_match(answer, reference_answer) if reference_answer and answerable else None,
        "token_f1": token_f1(answer, reference_answer) if reference_answer and answerable else None,
        "keyword_recall": keyword_recall(answer, expected_keywords) if expected_keywords and answerable else None,
    }
