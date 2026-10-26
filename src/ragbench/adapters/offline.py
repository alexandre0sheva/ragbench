"""The no-key path of the reference pipelines (`examples/{langchain,llamaindex,haystack}_default.py`).

Those pipelines call OpenAI only when `RAGBENCH_REFERENCE_LIVE=1` and a key is set; an external system is never stubbed by `--mock`, so spending
money must be a decision of its own. Otherwise they embed and answer with RAGBench's deterministic mock models, wrapped in each framework's classes by the example.
"""

from __future__ import annotations

import os
import threading
from functools import cache

from ragbench.models.embeddings import HashingEmbeddingModel
from ragbench.models.llms import MockLLM

_lock = threading.Lock()
_used = 0  # offline embeddings and completions so far; `python_callable` refuses them in a live run (see `calls`)


def live() -> bool:
    return os.environ.get("RAGBENCH_REFERENCE_LIVE") == "1" and bool(os.environ.get("OPENAI_API_KEY"))


def calls() -> int:
    """How many times the offline models have been used in this process. A live run compares it before and after a `python_callable` call:
    mock answers must never be scored, and priced, as a real pipeline's."""
    return _used


def _note() -> None:
    global _used
    with _lock:
        _used += 1


@cache
def _embedder() -> HashingEmbeddingModel:
    return HashingEmbeddingModel(dim=256)


def embed(texts: list[str]) -> list[list[float]]:
    _note()
    return [[float(x) for x in row] for row in _embedder().embed_texts(texts).vectors]


def complete(prompt: str) -> tuple[str, int, int]:
    """`(text, prompt_tokens, completion_tokens)` of the mock model's answer to `prompt` (it reads the prompt's `Context:` and `Question:` blocks)."""
    _note()
    result = MockLLM().generate([{"role": "user", "content": prompt}])
    return result.text, result.prompt_tokens, result.completion_tokens


def answer(context: str, question: str) -> tuple[str, int, int]:
    """`complete` for a framework whose prompt template is not shaped `Context: ... Question: ...`: pass the two parts."""
    return complete(f"Context:\n{context}\n\nQuestion: {question}")
