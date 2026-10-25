"""How `decision_gate` words its questions: predicate names, default instructions, and the text each request carries.

The offline `mock` backend reads the same formats back, so the system and the mock cannot drift apart. Instructions are
templates: `{passage}` stands for "the passage" (one passage per request) or "passage [3]" (several in one request).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

SINGLE = "passage"  # predicate name when a request judges one passage
SUFFICIENT = "sufficient"  # predicate name of the set-level sufficiency question
BATCH_PREFIX = "passage_"  # `passage_3` judges passage [3] of a batched request

# The wording is a precision/recall dial, not a detail (docs/decision-gate-evaluation.md measures it). The defaults are the plain questions the decision-gate
# pattern is usually described with. The strict pair below demands the answer itself: on the demo dataset it dropped chunks that only carry part of an answer
# and refused answerable multi-passage questions, in exchange for refusing every unanswerable one.
ANSWERABILITY_INSTRUCTIONS = "Does {passage} help answer the query?"
SUFFICIENCY_INSTRUCTIONS = "Is this evidence enough to answer the query from the documents alone?"
STRICT_ANSWERABILITY_INSTRUCTIONS = (
    "Does {passage} state a fact that answers the question, or a fact the answer depends on? "
    "Passages that are merely about the same domain, product line or industry, without such a fact, do not count."
)
STRICT_SUFFICIENCY_INSTRUCTIONS = (
    "Is the specific fact the question asks for stated in the evidence? Mentioning the same entity or topic is not enough: "
    "the evidence must give the actual answer to every part of the question."
)

# `evaluation.refusal_detector: model` puts the answer in the passage slot and asks this about it.
DECLINES_INSTRUCTIONS = (
    "The passage is a model's answer to the question. Does it decline to answer, or say that the answer is not available, cannot be found "
    "or cannot be determined from the documents? Answer no when it gives an answer, even a hedged one."
)


# `evaluation.claims` puts one claim in the question slot and the context the generator saw in the passage slot.
SUPPORTED_INSTRUCTIONS = (
    "Is the claim fully supported by the passage? Judge only from the passage, not from what you know. Answer no when any part of the claim "
    "goes beyond the passage or contradicts it."
)


def claim_input(claim: str, passage: str) -> str:
    return f"Claim: {claim}\n\nPassage:\n{passage}"


def single_input(question: str, passage: str) -> str:
    return f"Question: {question}\n\nPassage:\n{passage}"


def batch_input(question: str, passages: Sequence[str]) -> str:
    return f"Question: {question}\n\nPassages:\n" + "\n".join(f"[{i}] {_one_line(text)}" for i, text in enumerate(passages, start=1))


def evidence_input(question: str, passages: Sequence[str]) -> str:
    return f"Question: {question}\n\nEvidence:\n" + "\n".join(f"[{i}] {_one_line(text)}" for i, text in enumerate(passages, start=1))


def target(index: int | None) -> str:
    """What `{passage}` becomes: `the passage` for a single-passage request, `passage [i]` inside a batch."""
    return "the passage" if index is None else f"passage [{index}]"


def _one_line(text: str) -> str:
    return " ".join(text.split())


# -- reading the formats back (the mock backend) ----------------------------------------------------------------------

_QUESTION = re.compile(r"(?:Question|Claim):\s*(.*?)\n\n", re.S)


def parse_question(text: str) -> str:
    match = _QUESTION.search(text)
    return match.group(1) if match else ""


def parse_passages(text: str) -> dict[str, str]:
    """Predicate name -> the text it is about, for any request this module builds."""
    if "\n\nPassage:\n" in text:
        return {SINGLE: text.split("\n\nPassage:\n", 1)[1]}
    numbered = {f"[{i}]": body for i, body in re.findall(r"^\[(\d+)\] (.*)$", text.split("\n\n", 1)[-1], flags=re.M)}
    if "\n\nEvidence:\n" in text:
        return {SUFFICIENT: " ".join(numbered.values())}
    return {f"{BATCH_PREFIX}{key[1:-1]}": body for key, body in numbered.items()}
