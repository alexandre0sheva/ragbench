"""Rerankers. Importing this package registers every built-in one in `RERANKERS`.

`ragbench.models.rerankers` depends only on `ragbench.documents` and `ragbench.models`, so it can be imported first in a
fresh interpreter (it used to loop through `ragbench.rag_systems`).
"""

from __future__ import annotations

from ragbench.models.llms import LLM
from ragbench.models.rerankers.base import Reranker, RerankResult
from ragbench.models.rerankers.cross_encoder import DEFAULT_CROSS_ENCODER, CrossEncoderReranker
from ragbench.models.rerankers.keyword import SimpleKeywordOverlapReranker
from ragbench.models.rerankers.llm import LLMReranker
from ragbench.models.rerankers.tfidf import LocalRelevanceReranker
from ragbench.registry import RERANKERS


def create_reranker(name: str, llm: LLM | None = None, model: str | None = None, force_mock: bool = False) -> Reranker:
    """Build the reranker registered as `name`.

    `llm` is given to rerankers that need one (without it they degrade to the free keyword reranker); `model` to rerankers
    that load a model (`takes_model`). In mock mode, rerankers that would download a model use their `offline_fallback`.
    """
    cls = RERANKERS.get(name)
    fallback = getattr(cls, "offline_fallback", None)
    if force_mock and fallback:
        return create_reranker(fallback, llm=llm)
    if getattr(cls, "needs_llm", False):
        return cls(llm) if llm is not None else SimpleKeywordOverlapReranker()  # type: ignore[call-arg]
    if getattr(cls, "takes_model", False):
        return cls(model)  # type: ignore[call-arg]
    return cls()


__all__ = [
    "DEFAULT_CROSS_ENCODER",
    "CrossEncoderReranker",
    "LLMReranker",
    "LocalRelevanceReranker",
    "RerankResult",
    "Reranker",
    "SimpleKeywordOverlapReranker",
    "create_reranker",
]
