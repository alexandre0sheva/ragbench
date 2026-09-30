"""Cross-encoder reranker (optional extra: `pip install 'ragbench[rerank]'`).

Scores every (question, chunk) pair with a Hugging Face cross-encoder through sentence-transformers. The model is loaded on
first use and shared by every reranker instance in the process (two systems reranking with the same model pay for one
copy); predictions are serialized per model because torch already uses every core. Cost is $0; latency is what the tracer
measures around `rerank`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.errors import MissingExtraError
from ragbench.models.rerankers.base import RerankResult
from ragbench.registry import RERANKERS
from ragbench.utils.extras import require_extra

DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-base"
_NEEDS_EXTRA = "The cross_encoder reranker needs the optional `sentence-transformers` package. Install it with: pip install 'ragbench[rerank]'"


@dataclass
class _LoadedModel:
    model: Any
    predict_lock: threading.Lock


_LOAD_LOCK = threading.Lock()
_MODELS: dict[tuple[str, int], _LoadedModel] = {}


def clear_model_cache() -> None:
    """Drop loaded models (tests; or to free memory between runs in one process)."""
    with _LOAD_LOCK:
        _MODELS.clear()


def _load(model_name: str, max_length: int) -> _LoadedModel:
    with _LOAD_LOCK:  # concurrent first calls must not each load (and download) the weights
        key = (model_name, max_length)
        if key not in _MODELS:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise MissingExtraError(_NEEDS_EXTRA) from exc
            # device=None lets the library pick CUDA, then Apple MPS, then CPU.
            _MODELS[key] = _LoadedModel(CrossEncoder(model_name, max_length=max_length, device=None), threading.Lock())
        return _MODELS[key]


@RERANKERS.register("cross_encoder")
class CrossEncoderReranker:
    name = "cross_encoder"
    takes_model = True  # `retrieval.reranker_model` selects the Hugging Face model
    offline_fallback = "local_relevance"  # mock runs are offline, so they use the free TF-IDF reranker instead
    import_modules = ("sentence_transformers",)

    def __init__(self, model: str | None = None, *, max_length: int = 512, batch_size: int = 32):
        require_extra("sentence_transformers", "The cross_encoder reranker", "rerank", "sentence-transformers")  # fail when the system is built, not on the first question
        self.model_name = model or DEFAULT_CROSS_ENCODER
        self.max_length = max_length
        self.batch_size = batch_size

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> RerankResult:
        if not chunks:
            return RerankResult(chunks=[], cost=CostBreakdown())
        loaded = _load(self.model_name, self.max_length)
        pairs = [(question, chunk.text) for chunk in chunks]
        with loaded.predict_lock:
            raw_scores = loaded.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = [float(score) for score in raw_scores]
        order = sorted(range(len(chunks)), key=lambda index: -scores[index])[:top_k]  # stable: ties keep retrieval order
        final = [
            RetrievedChunk(
                chunk_id=chunks[index].chunk_id,
                doc_id=chunks[index].doc_id,
                text=chunks[index].text,
                score=scores[index],
                rank=rank,
                metadata={
                    **chunks[index].metadata,
                    "reranker": self.name,
                    "original_rank": chunks[index].rank,
                    "cross_encoder_model": self.model_name,
                },
            )
            for rank, index in enumerate(order, start=1)
        ]
        return RerankResult(chunks=final, cost=CostBreakdown())
