"""The prompts of `generate-questions` and `label`. The markers (`ragbench.models.prompts`) are what `MockLLM` recognizes, so offline runs and the
fake-server tests answer them deterministically. Every prompt asks for one JSON object."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ragbench.models.prompts import (
    GRADE_DOCUMENT_MARKER,
    PARAPHRASE_QUESTION_MARKER,
    WRITE_MULTIHOP_MARKER,
    WRITE_QUESTION_MARKER,
    WRITE_UNANSWERABLE_MARKER,
)
from ragbench.utils.text import tokenize

MAX_DOC_CHARS = 6000  # the part of a document a prompt shows; longer ones are cut (graders look at the window that best matches the question)

KIND_GUIDANCE = {
    "single_hop": "a self-contained question about one specific fact stated in it",
    "numeric": "a self-contained question whose answer is a number, amount, percentage, date or duration stated in it (or computed from numbers in it)",
}

_RULES = (
    "Rules: the question must make sense to someone who has not seen the document (never say 'the document' or 'the text'), "
    "the reference answer must be short and stated by the document, and expected_keywords are the 1-4 words or numbers an answer must contain."
)


def clip(text: str, limit: int = MAX_DOC_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def best_window(text: str, question: str, limit: int = MAX_DOC_CHARS) -> str:
    """`text` if it fits, else the `limit`-character window sharing the most words with `question` (the part a grader should read)."""
    if len(text) <= limit:
        return text
    wanted = set(tokenize(question))
    step = max(1, limit // 2)
    starts = [*range(0, len(text) - limit, step), len(text) - limit]
    start = max(starts, key=lambda s: (len(wanted & set(tokenize(text[s : s + limit]))), -s))
    return ("[...] " if start else "") + text[start : start + limit].strip() + (" [...]" if start + limit < len(text) else "")


def doc_block(label: str, title: str, text: str) -> str:
    return f"{label} (title: {title}):\n<<<\n{clip(text)}\n>>>"


def _avoid(questions: Sequence[str]) -> str:
    return "Already asked (write a different one):\n" + "\n".join(f"- {question}" for question in questions) + "\n\n" if questions else ""


def write_question_prompt(kind: str, title: str, text: str, avoid: Sequence[str]) -> str:
    return (
        f"{WRITE_QUESTION_MARKER}: {KIND_GUIDANCE[kind]}.\nKind: {kind}\n{_RULES}\n\n{doc_block('Document', title, text)}\n\n{_avoid(avoid)}"
        'Return JSON: {"question": "...", "reference_answer": "...", "expected_keywords": ["..."]}'
    )


def write_multihop_prompt(title_a: str, text_a: str, title_b: str, text_b: str, avoid: Sequence[str]) -> str:
    return (
        f"{WRITE_MULTIHOP_MARKER}: it must combine a fact from each, so neither document alone is enough.\n{_RULES}\n\n"
        f"{doc_block('Document A', title_a, text_a)}\n\n{doc_block('Document B', title_b, text_b)}\n\n{_avoid(avoid)}"
        'Return JSON: {"question": "...", "reference_answer": "...", "expected_keywords": ["..."]}'
    )


def paraphrase_prompt(title: str, text: str, question: str) -> str:
    return (
        f"{PARAPHRASE_QUESTION_MARKER} that the document uses (a user who does not know its vocabulary would ask it differently), keeping the meaning "
        f"and the answer exactly the same.\n\n{doc_block('Document', title, text)}\n\nQuestion: {question}\n\n"
        'Return JSON: {"question": "..."}'
    )


def unanswerable_prompt(digest: Sequence[tuple[str, str]], avoid: Sequence[str]) -> str:
    listing = "\n".join(f"- {title}: {snippet}" for title, snippet in digest)
    return (
        f"{WRITE_UNANSWERABLE_MARKER}, but that a user of this collection could plausibly ask (same domain, same style). Name the specific thing it "
        "asks about (a product, person, place, number or event) as `entity`: it must not appear anywhere in the documents.\n\n"
        f"Some of the documents:\n{listing}\n\n{_avoid(avoid)}"
        'Return JSON: {"question": "...", "entity": "..."}'
    )


GRADE_RUBRIC = (
    "Grades: 0 = not relevant, 1 = partially relevant (related, but does not answer it), 2 = relevant (contains most of what is needed), "
    "3 = contains the exact answer. Judge only from the document text."
)


def grade_prompt(question: str, doc_id: str, title: str, text: str) -> str:
    return (
        f"{GRADE_DOCUMENT_MARKER}.\n{GRADE_RUBRIC}\n\nQuestion: {question}\n\n{doc_block(f'Document {doc_id}', title, best_window(text, question))}\n\n"
        'Return JSON: {"grade": 0-3, "reason": "one short sentence"}'
    )


def block_text(prompt: str, label: str = "Document") -> tuple[str, str]:
    """(title, text) of the first `<label ...>` block of a prompt; what mock responders read back."""
    match = re.search(rf"{label}[^\n(]*\(title: (.*?)\):\n<<<\n(.*?)\n>>>", prompt, flags=re.S)
    return (match.group(1), match.group(2)) if match else ("", "")
