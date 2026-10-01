"""Mock-safe answers to the synthesis and grading prompts (`ragbench generate-questions`, `ragbench label`).

They read the document out of the prompt and answer with the deterministic templates, so a run against `MockLLM` (or the fake OpenAI
server, which is backed by it) produces valid questions and grades with no network.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragbench.models.prompts import (
    GRADE_DOCUMENT_MARKER,
    PARAPHRASE_QUESTION_MARKER,
    WRITE_MULTIHOP_MARKER,
    WRITE_QUESTION_MARKER,
    WRITE_UNANSWERABLE_MARKER,
)
from ragbench.utils.text import tokenize


def _avoided(prompt: str) -> int:
    """How many questions the prompt says were already asked: the index of the next template."""
    return len(re.findall(r"^- ", prompt.split("Already asked", 1)[1].split("\n\n", 1)[0], flags=re.M)) if "Already asked" in prompt else 0


def _candidate_json(candidate: Any) -> str:
    payload = {"question": candidate.question, "reference_answer": candidate.reference_answer, "expected_keywords": candidate.keywords}
    return json.dumps(payload)


def _write_question(prompt: str) -> str:
    from ragbench.datasets import templates
    from ragbench.datasets.prompts import block_text

    title, text = block_text(prompt)
    kind = (re.search(r"^Kind: (\w+)", prompt, flags=re.M) or [None, "single_hop"])[1]
    candidate = templates.numeric_candidate(title, text, _avoided(prompt)) if kind == "numeric" else templates.single_candidate(title, text, _avoided(prompt))
    return _candidate_json(candidate) if candidate else json.dumps({"question": "", "reference_answer": "", "expected_keywords": []})


def _write_multihop(prompt: str) -> str:
    from ragbench.datasets import templates
    from ragbench.datasets.prompts import block_text

    (title_a, text_a), (title_b, text_b) = block_text(prompt, "Document A"), block_text(prompt, "Document B")
    return _candidate_json(templates.multihop_candidate(title_a, text_a, title_b, text_b, _avoided(prompt)))


def _paraphrase(prompt: str) -> str:
    from ragbench.datasets import templates

    match = re.search(r"^Question: (.*)$", prompt, flags=re.M)
    return json.dumps({"question": templates.paraphrase_question(match[1] if match else "")})


def _unanswerable(prompt: str) -> str:
    from ragbench.datasets import templates

    candidate = templates.unanswerable_candidate(_avoided(prompt))
    return json.dumps({"question": candidate.question, "entity": candidate.entity})


def _grade(prompt: str) -> str:
    from ragbench.datasets.prompts import block_text

    match = re.search(r"^Question: (.*)$", prompt, flags=re.M)
    wanted = {word for word in tokenize(match[1] if match else "") if len(word) > 3}
    _, text = block_text(prompt)
    overlap = len(wanted & set(tokenize(text))) / len(wanted) if wanted else 0.0
    grade = 3 if overlap >= 0.8 else 2 if overlap >= 0.5 else 1 if overlap >= 0.25 else 0
    return json.dumps({"grade": grade, "reason": f"Mock grader: {overlap:.0%} of the question's words occur in the document."})


def register(mock_llm: Any) -> None:
    """Add the responders to `MockLLM` (called when `ragbench.models.llms` is imported). JSON mode would otherwise win, so they are registered last."""
    for name, marker, respond in (
        ("write_question", WRITE_QUESTION_MARKER, _write_question),
        ("write_multihop", WRITE_MULTIHOP_MARKER, _write_multihop),
        ("paraphrase_question", PARAPHRASE_QUESTION_MARKER, _paraphrase),
        ("write_unanswerable", WRITE_UNANSWERABLE_MARKER, _unanswerable),
        ("grade_document", GRADE_DOCUMENT_MARKER, _grade),
    ):
        mock_llm.register_responder(lambda prompt, json_mode, marker=marker: marker in prompt, respond, name=name)
