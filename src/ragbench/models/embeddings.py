from __future__ import annotations

import logging
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragbench.cache import CacheRuntime, active_cache, cache_key
from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL
from ragbench.models.errors import ModelInitError
from ragbench.models.refs import parse_model_ref, provider_reachable
from ragbench.models.usage import note_embedding, recording
from ragbench.utils.hashing import stable_hash
from ragbench.utils.text import estimate_tokens, tokenize

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    model: str
    input_tokens: int
    cost: CostBreakdown


class EmbeddingModel(ABC):
    model_name: str
    # False for models that are free to recompute: they are never written to the persistent disk cache.
    cacheable: bool = True

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        raise NotImplementedError

    def embed_query(self, text: str) -> EmbeddingResult:
        return self.embed_texts([text])


class HashingEmbeddingModel(EmbeddingModel):
    cacheable = False

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
        if recording():  # `ragbench estimate`: count tokens the way a hosted API would (tiktoken), not with this model's rough heuristic
            note_embedding(sum(estimate_tokens(text) for text in texts))
        return EmbeddingResult(vectors=vectors, model=self.model_name, input_tokens=input_tokens, cost=CostBreakdown())


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


def _pack(vector: np.ndarray, tokens: int) -> bytes:
    return struct.pack("<I", tokens) + np.asarray(vector, dtype="<f4").tobytes()


def _unpack(blob: bytes) -> tuple[np.ndarray, int]:
    (tokens,) = struct.unpack("<I", blob[:4])
    return np.frombuffer(blob[4:], dtype="<f4").astype(np.float32), tokens


class CachedEmbeddingModel(EmbeddingModel):
    """Wraps an embedding model with two cache tiers for corpus embeddings.

    L1 is the in-process `EMBEDDING_CACHE` shared by all systems of a run; L2 is the run's persistent disk cache,
    which lets later runs skip embedding an unchanged corpus altogether. Hits from either tier are still *charged*
    at standalone prices (see `_EmbeddingCache`); the avoided spend is tracked separately.

    `embed_query` goes straight to the underlying model so per-question latency stays honest, unless the user opts
    in with `cache.cache_query_embeddings`. Models that are free to recompute (`cacheable = False`, e.g. the mock
    hashing embeddings) never touch the disk.
    """

    def __init__(self, inner: EmbeddingModel):
        self.inner = inner
        self.model_name = inner.model_name

    def _runtime(self) -> CacheRuntime | None:
        runtime = active_cache()
        if runtime is None or not runtime.config.embeddings or not getattr(self.inner, "cacheable", True):
            return None
        return runtime

    def _disk_key(self, kind: str, text_hash: str) -> str:
        return cache_key("embeddings", model=self.model_name, kind=kind, text=text_hash)

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        runtime = self._runtime()
        if not texts or (not EMBEDDING_CACHE.enabled and runtime is None):
            return self.inner.embed_texts(texts)
        keys = [stable_hash(text, 32) for text in texts]
        cached: dict[int, tuple[np.ndarray, int]] = {}
        if EMBEDDING_CACHE.enabled:
            for idx, key in enumerate(keys):
                entry = EMBEDDING_CACHE.lookup(self.model_name, key)
                if entry is not None:
                    cached[idx] = entry
        if runtime is not None:
            wanted = {self._disk_key("doc", keys[idx]): keys[idx] for idx in range(len(texts)) if idx not in cached}
            stored = runtime.disk.get_many("embeddings", wanted)
            entries = {wanted[disk_key]: _unpack(blob) for disk_key, blob in stored.items()}
            for key, (vector, tokens) in entries.items():
                runtime.disk.record_saved("embeddings", estimate_model_cost(self.model_name, input_tokens=tokens))
                if EMBEDDING_CACHE.enabled:
                    EMBEDDING_CACHE.store(self.model_name, key, vector, tokens)
            for idx, key in enumerate(keys):
                if idx not in cached and key in entries:
                    cached[idx] = entries[key]
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
            fresh: list[tuple[str, bytes]] = []
            for row, key in enumerate(ordered_keys):
                vector = miss_result.vectors[row]
                tokens = estimate_tokens(texts[unique_texts[key]], self.model_name)
                vector_by_key[key] = vector
                if EMBEDDING_CACHE.enabled:
                    EMBEDDING_CACHE.store(self.model_name, key, vector, tokens)
                fresh.append((self._disk_key("doc", key), _pack(vector, tokens)))
            if runtime is not None:
                runtime.disk.put_many("embeddings", fresh, meta={"model": self.model_name})
        dim = (
            miss_result.vectors.shape[1]
            if miss_result is not None and miss_result.vectors.size
            else next(iter(cached.values()))[0].shape[0]
        )
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        # Every text is charged its own token estimate, whether it was a cache hit or had to be embedded, so a
        # system's bill does not depend on which other system (or thread, or earlier run) embedded it first.
        miss_tokens: dict[str, int] = {}
        charged_tokens = 0
        for idx, key in enumerate(keys):
            if idx in cached:
                vectors[idx] = cached[idx][0]
                charged_tokens += cached[idx][1]
            else:
                vectors[idx] = vector_by_key[key]
                if key not in miss_tokens:
                    miss_tokens[key] = estimate_tokens(texts[idx], self.model_name)
                charged_tokens += miss_tokens[key]
        # Charge cache hits at standalone prices so per-system cost stays comparable.
        cost = estimate_model_cost(self.model_name, input_tokens=charged_tokens)
        return EmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            input_tokens=charged_tokens,
            cost=CostBreakdown(embedding_input_tokens=charged_tokens, embedding_cost=cost),
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        runtime = self._runtime()
        if runtime is None or not runtime.config.cache_query_embeddings:
            return self.inner.embed_query(text)
        disk_key = self._disk_key("query", stable_hash(text, 32))
        blob = runtime.disk.get("embeddings", disk_key)
        if blob is not None:
            vector, tokens = _unpack(blob)
            cost = estimate_model_cost(self.model_name, input_tokens=tokens)
            runtime.disk.record_saved("embeddings", cost)
            return EmbeddingResult(vector[np.newaxis, :], self.model_name, tokens, CostBreakdown(embedding_input_tokens=tokens, embedding_cost=cost))
        result = self.inner.embed_query(text)
        runtime.disk.put("embeddings", disk_key, _pack(result.vectors[0], result.input_tokens), meta={"model": self.model_name})
        return result


def create_embedding_model(
    model_name: str | None = None, force_mock: bool = False, strict: bool = True, providers: dict[str, Any] | None = None
) -> EmbeddingModel:
    """Build the (cache-wrapped) embedding model for a ref such as `text-embedding-3-small` or `local:BAAI/bge-small-en-v1.5`.

    See `create_llm` for `strict`, `providers` and the mock / missing-extra rules. Local models load their weights on first use.
    """
    import ragbench.models.providers  # noqa: F401  (registers the built-in providers)
    from ragbench.registry import EMBEDDERS
    from ragbench.runtime.context import current_runtime

    ref = model_name or DEFAULT_EMBEDDING_MODEL
    provider, model = parse_model_ref(ref)
    if force_mock or not provider_reachable(provider):
        return CachedEmbeddingModel(HashingEmbeddingModel())
    factory = EMBEDDERS.get(provider)
    endpoints = providers if providers is not None else current_runtime().providers
    try:
        return CachedEmbeddingModel(factory(model, providers=endpoints))
    except ImportError:
        raise
    except Exception as exc:
        if strict:
            raise ModelInitError(f"Could not create the {provider} embedding client for {ref!r}: {exc}") from exc
        logger.warning("%s embedding client init failed (%s); falling back to hashing embeddings.", provider, exc)
        return CachedEmbeddingModel(HashingEmbeddingModel())
