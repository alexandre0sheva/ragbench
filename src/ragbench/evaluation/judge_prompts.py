"""The judge prompt. Bump `JUDGE_PROMPT_VERSION` whenever the wording changes: it is stored with every judgment, and
scores from different prompt versions should not be compared.

Keep the phrase "evaluation judge" in the system prompt: `tests/fake_openai_server.py` recognizes judge calls by it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ragbench.datasets.schema import Question
from ragbench.rag_systems.base import RetrievedChunk

JUDGE_PROMPT_VERSION = "v3"
MAX_CONTEXT_CHARS = 12000

SYSTEM_PROMPT = """You are a strict RAG evaluation judge. You grade one model answer to one question.

Ground every judgment only in the material you are given: the reference answer (when there is one) and the retrieved context. Do not use your own knowledge to decide what is true. The field "answerable" says whether the documents can answer the question. If it is false, the only correct response is to say so. If it is true and the reference answer is null, no reference exists: grade correctness and completeness against the retrieved context alone (does the answer say what the context says about the question, and does it use what the context offers), and rely on the other axes as usual.

Score each axis from 0 to 5 (decimals are allowed) with these anchors:
- correctness: does the answer state what the reference answer states? 0 = contradicts it or says nothing relevant, 3 = right in substance but with a wrong or missing detail, 5 = fully matches. For an unanswerable question, 5 = a clear refusal and 0 = a confident answer.
- faithfulness: is every claim in the answer supported by the retrieved context? 0 = mostly invented, 3 = some claims go beyond the context, 5 = every claim is in the context.
- completeness: does the answer cover everything the reference answer asks for? 0 = nothing, 3 = about half, 5 = all of it.
- relevance: does the answer address the question that was asked? 0 = off topic, 3 = partly on topic or padded, 5 = directly on point.
- citation_quality: are the sources cited (as [doc_id]) the ones that actually contain the supporting evidence? 0 = none cited or all wrong, 3 = some right, 5 = all right and nothing irrelevant cited.

Also decide two booleans: is_supported_by_context (the context contains what the answer claims) and is_hallucinated (the answer asserts something the context does not support).

Return JSON only, with exactly these keys:
{"correctness": 0-5, "faithfulness": 0-5, "completeness": 0-5, "relevance": 0-5, "citation_quality": 0-5, "is_supported_by_context": true|false, "is_hallucinated": true|false, "reasoning": "one or two sentences"}"""


def build_messages(question: Question, answer: str, contexts: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    """The chat messages that ask the judge to grade `answer` against the reference and the context the generator saw."""
    context_text = "\n\n".join(f"[{chunk.doc_id} | {chunk.chunk_id}]\n{chunk.text}" for chunk in contexts)
    payload = {
        "question": question.question,
        "answerable": question.is_answerable,
        "reference_answer": question.reference_answer,
        "model_answer": answer,
        "retrieved_context": context_text[:MAX_CONTEXT_CHARS],
        "expected_keywords": question.expected_keywords,
        "category": question.category,
        "answer_type": question.answer_type,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


RETRY_PROMPT = (
    "That reply was not a valid judgment. Reply again with one JSON object and nothing else, containing numeric scores 0-5 for "
    "correctness, faithfulness, completeness, relevance and citation_quality, the booleans is_supported_by_context and is_hallucinated, and a reasoning string."
)


def retry_messages(messages: list[dict[str, str]], bad_reply: str) -> list[dict[str, str]]:
    """`messages` followed by the unusable reply and a reminder of the format.

    A retry must differ from the first request: an identical temperature-0 request would just be answered (or replayed from the cache) the same way.
    """
    return [*messages, {"role": "assistant", "content": bad_reply}, {"role": "user", "content": RETRY_PROMPT}]
