from __future__ import annotations

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


def create_embedding_model(model_name: str | None = None, force_mock: bool = False) -> EmbeddingModel:
    if force_mock or not has_openai_key():
        return HashingEmbeddingModel()
    try:
        return OpenAIEmbeddingModel(model_name or "text-embedding-3-small")
    except Exception:
        return HashingEmbeddingModel()
