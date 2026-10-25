"""A RAG pipeline of your own, benchmarked in-process with `type: python_callable`.

    systems:
      - type: python_callable
        name: my_rag
        retrieval:
          target: examples/simple_rag.py:answer
          ingest_target: examples/simple_rag.py:ingest

`ingest(documents)` gets RAGBench's documents (objects with `doc_id`, `path`, `title`, `text`) once before the questions.
`answer(question, top_k)` returns a dict: `answer`, and optionally `contexts` (`text`, `doc_id`, `score`), `usage`
(`prompt_tokens`, `completion_tokens`) and `cost_usd`. Questions run concurrently, so `answer` must be thread-safe (this one
only reads what `ingest` built). Replace the word-overlap retrieval below with your own retriever and model call.
"""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9$][a-z0-9$\-]*")
_passages: list[tuple[str, str]] = []


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def ingest(documents: list[Any]) -> None:
    _passages[:] = [(doc.doc_id, part.strip()) for doc in documents for part in doc.text.split("\n\n") if part.strip() and not part.startswith("#")]


def answer(question: str, top_k: int) -> dict[str, Any]:
    wanted = _terms(question)
    ranked = sorted(_passages, key=lambda passage: len(wanted & _terms(passage[1])), reverse=True)[:top_k]
    contexts = [{"doc_id": doc_id, "text": text, "score": float(len(wanted & _terms(text)))} for doc_id, text in ranked if wanted & _terms(text)]
    text = contexts[0]["text"].split(". ")[0].rstrip(".") + "." if contexts else "I could not find the answer in the provided documents."
    return {"answer": text, "contexts": contexts, "usage": {"prompt_tokens": len(question.split()) + sum(len(c["text"].split()) for c in contexts), "completion_tokens": len(text.split())}}
