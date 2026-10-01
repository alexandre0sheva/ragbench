"""Deterministic, model-free question writing: what `generate-questions --mock` produces, and what `MockLLM` answers synthesis prompts with.

The questions are templates over the document's own sentences ("What does the document say about <topic>?"). They exist to prove the
pipeline (generation, validation, `inspect-dataset`, a run) works offline; they say nothing about how real questions would score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragbench.utils.text import normalize_text

NOT_IN_DOCUMENTS = "The documents do not contain this information."
_STOPWORDS = frozenset(
    "that this with from have been were will would could should their there which about what when where while these those into than then them they "
    "also each other some such only over more most very your yours ours".split()
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"[$€£]?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class Candidate:
    """A question a writer proposes, before synthesis validates it."""

    question: str
    reference_answer: str
    keywords: list[str] = field(default_factory=list)
    entity: str | None = None  # unanswerable questions: the specific thing asked about, which must not occur in any document


def sentences(text: str) -> list[str]:
    """The prose sentences of a document: headings and markup lines dropped, very short fragments ignored."""
    prose = " ".join(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "|", "```", "---")))
    return [part.strip() for part in _SENTENCE_SPLIT.split(normalize_text(prose)) if len(part.split()) >= 5]


def content_words(sentence: str, limit: int = 4) -> list[str]:
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", sentence):
        if len(raw) > 3 and raw.lower() not in _STOPWORDS and raw not in words:
            words.append(raw)
        if len(words) == limit:
            break
    return words


def _topic(sentence: str) -> str:
    return " ".join(content_words(sentence)) or sentence[:40]


def _pick(items: list[str], index: int) -> str | None:
    return items[index % len(items)] if items else None


def single_candidate(title: str, text: str, index: int) -> Candidate:
    sentence = _pick(sentences(text), index) or normalize_text(text)[:200] or title
    return Candidate(f"What does the document say about {_topic(sentence)}?", sentence, content_words(sentence, 3))


def numeric_candidate(title: str, text: str, index: int) -> Candidate | None:
    sentence = _pick([s for s in sentences(text) if _NUMBER.search(s)], index)
    if sentence is None:
        return None
    numbers = _NUMBER.findall(sentence)
    return Candidate(f"What number does the document give for {_topic(sentence)}?", f"{numbers[0]}. {sentence}", numbers[:3])


def multihop_candidate(title_a: str, text_a: str, title_b: str, text_b: str, index: int) -> Candidate:
    sentence_a = _pick(sentences(text_a), index) or title_a
    sentence_b = _pick(sentences(text_b), index) or title_b
    name_a, name_b = title_a.strip() or _topic(sentence_a), title_b.strip() or _topic(sentence_b)
    return Candidate(f"How do {name_a} and {name_b} relate according to the documents?", f"{sentence_a} {sentence_b}", content_words(f"{name_a} {name_b}", 4))


def paraphrase_question(question: str) -> str:
    match = re.match(r"(?i)what does the document say about (.*?)\??$", question.strip())
    return f"Tell me what is stated regarding {match.group(1)}." if match else f"In other words: {question.strip()}"


def unanswerable_candidate(index: int) -> Candidate:
    entity = f"Zorblax{index + 1:02d}"
    return Candidate(f"What does the document say about {entity}?", NOT_IN_DOCUMENTS, [], entity)
