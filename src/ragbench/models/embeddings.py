from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.utils.env import has_openai_key
from ragbench.utils.hashing import stable_hash
from ragbench.utils.text import estimate_tokens, tokenize


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model: str
    input_tokens: int
    cost: CostBreakdown


class EmbeddingModel(ABC):
    model_name: str

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        raise NotImplementedError

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.embed_texts([text])


class HashingEmbeddingModel(EmbeddingModel):
    def __init__(self, dim: int = 384):
        self.model_name = "hashing-embedding"
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        input_tokens = 0
        for row, text in enumerate(texts):
            toks = tokenize(text)
            input_tokens += max(1, int(len(toks) * 1.3))
            for tok in toks:
                digest = int(stable_hash(tok, 16), 16)
                idx = digest % self.dim
                sign = 1.0 if (digest // self.dim) % 2 == 0 else -1.0
                vectors[row, idx] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return EmbeddingResult(vectors=vectors, model=self.model_name, input_tokens=input_tokens, cost=CostBreakdown())


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 96):
        self.model_name = model_name
        self.batch_size = batch_size
        from openai import OpenAI

        self.client = OpenAI()

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), model=self.model_name, input_tokens=0, cost=CostBreakdown())
        vectors: list[list[float]] = []
        input_tokens = 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend([item.embedding for item in response.data])
            if getattr(response, "usage", None):
                input_tokens += int(response.usage.prompt_tokens)
            else:
                input_tokens += sum(estimate_tokens(text, self.model_name) for text in batch)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = np.divide(array, np.maximum(norms, 1e-12))
        cost = estimate_model_cost(self.model_name, input_tokens=input_tokens)
        return EmbeddingResult(
            vectors=array,
            model=self.model_name,
            input_tokens=input_tokens,
            cost=CostBreakdown(embedding_input_tokens=input_tokens, embedding_cost=cost),
        )


class _EmbeddingCache:
    """Process-wide cache of corpus embeddings shared by all systems in a run.

    Systems in one benchmark run embed the same corpus, so without the cache a
    six-system comparison pays the embedding bill six times. Cached hits reuse
    the vector but are still *charged* to the requesting system at standalone
    prices, keeping per-system cost comparable; the real API savings are
    tracked separately and reported in the run summary.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
        self.hits = 0
        self.misses = 0
        self.saved_cost_usd = 0.0
        self.enabled = True

    def lookup(self, model: str, key: str) -> tuple[np.ndarray, int] | None:
        with self._lock:
            entry = self._vectors.get((model, key))
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self.saved_cost_usd += estimate_model_cost(model, input_tokens=entry[1])
            return entry

    def store(self, model: str, key: str, vector: np.ndarray, tokens: int) -> None:
        with self._lock:
            self._vectors[(model, key)] = (vector, tokens)

    def clear(self) -> None:
        with self._lock:
            self._vectors.clear()
            self.hits = 0
            self.misses = 0
            self.saved_cost_usd = 0.0

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "saved_cost_usd": self.saved_cost_usd}


EMBEDDING_CACHE = _EmbeddingCache()


class CachedEmbeddingModel(EmbeddingModel):
    """Wraps an embedding model with the shared corpus cache.

    Only `embed_texts` (corpus ingestion) consults the cache. `embed_query`
    always calls the underlying model so per-question latency measurements
    stay honest for every system regardless of run order.
    """

    def __init__(self, inner: EmbeddingModel):
        self.inner = inner
        self.model_name = inner.model_name

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts or not EMBEDDING_CACHE.enabled:
            return self.inner.embed_texts(texts)
        keys = [stable_hash(text, 32) for text in texts]
        cached: dict[int, tuple[np.ndarray, int]] = {}
        for idx, key in enumerate(keys):
            entry = EMBEDDING_CACHE.lookup(self.model_name, key)
            if entry is not None:
                cached[idx] = entry
        miss_indices = [idx for idx in range(len(texts)) if idx not in cached]
        # Dedupe within the batch so repeated texts are embedded once.
        unique_texts: dict[str, int] = {}
        for idx in miss_indices:
            unique_texts.setdefault(keys[idx], idx)
        miss_result: EmbeddingResult | None = None
        vector_by_key: dict[str, np.ndarray] = {}
        if unique_texts:
            ordered_keys = list(unique_texts)
            miss_result = self.inner.embed_texts([texts[unique_texts[key]] for key in ordered_keys])
            for row, key in enumerate(ordered_keys):
                vector = miss_result.vectors[row]
                tokens = estimate_tokens(texts[unique_texts[key]], self.model_name)
                vector_by_key[key] = vector
                EMBEDDING_CACHE.store(self.model_name, key, vector, tokens)
        dim = (
            miss_result.vectors.shape[1]
            if miss_result is not None and miss_result.vectors.size
            else next(iter(cached.values()))[0].shape[0]
        )
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        charged_tokens = 0
        for idx, key in enumerate(keys):
            if idx in cached:
                vectors[idx] = cached[idx][0]
                charged_tokens += cached[idx][1]
            else:
                vectors[idx] = vector_by_key[key]
        if miss_result is not None:
            charged_tokens += miss_result.input_tokens
        # Charge cache hits at standalone prices so per-system cost stays comparable.
        cost = estimate_model_cost(self.model_name, input_tokens=charged_tokens)
        return EmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            input_tokens=charged_tokens,
            cost=CostBreakdown(embedding_input_tokens=charged_tokens, embedding_cost=cost),
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.inner.embed_query(text)


def create_embedding_model(model_name: str | None = None, force_mock: bool = False) -> EmbeddingModel:
    if force_mock or not has_openai_key():
        return CachedEmbeddingModel(HashingEmbeddingModel())
    try:
        return CachedEmbeddingModel(OpenAIEmbeddingModel(model_name or "text-embedding-3-small"))
    except Exception:
        return CachedEmbeddingModel(HashingEmbeddingModel())
