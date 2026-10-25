"""The no-key path of the reference pipelines (`examples/{langchain,llamaindex,haystack}_default.py`).

Those pipelines call OpenAI only when `RAGBENCH_REFERENCE_LIVE=1` and a key is set; an external system is never stubbed by `--mock`, so spending
money must be a decision of its own. Otherwise they embed and answer with RAGBench's deterministic mock models, wrapped in each framework's classes by the example.
"""

from __future__ import annotations

import os
from functools import cache

from ragbench.models.embeddings import HashingEmbeddingModel
from ragbench.models.llms import MockLLM


def live() -> bool:
    return os.environ.get("RAGBENCH_REFERENCE_LIVE") == "1" and bool(os.environ.get("OPENAI_API_KEY"))


@cache
def _embedder() -> HashingEmbeddingModel:
    return HashingEmbeddingModel(dim=256)


def embed(texts: list[str]) -> list[list[float]]:
    return [[float(x) for x in row] for row in _embedder().embed_texts(texts).vectors]


def complete(prompt: str) -> tuple[str, int, int]:
    """`(text, prompt_tokens, completion_tokens)` of the mock model's answer to `prompt` (it reads the prompt's `Context:` and `Question:` blocks)."""
    result = MockLLM().generate([{"role": "user", "content": prompt}])
    return result.text, result.prompt_tokens, result.completion_tokens


def answer(context: str, question: str) -> tuple[str, int, int]:
    """`complete` for a framework whose prompt template is not shaped `Context: ... Question: ...`: pass the two parts."""
    return complete(f"Context:\n{context}\n\nQuestion: {question}")
