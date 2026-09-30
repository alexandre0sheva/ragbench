"""Local embedding models through sentence-transformers (optional extra: `pip install 'ragbench[local]'`).

Chosen over fastembed because it runs any Hugging Face embedding model, has wheels for Python 3.11-3.14 on macOS and Linux
(fastembed's onnxruntime dependency has none for 3.14 on Apple Silicon at the time of writing), and the `rerank` extra
(cross-encoders, Task 9) needs it anyway. Refs look like `local:BAAI/bge-small-en-v1.5`. Local models cost $0.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EmbeddingModel, EmbeddingResult
from ragbench.models.errors import MissingExtraError
from ragbench.registry import EMBEDDERS
from ragbench.utils.text import estimate_tokens


class LocalEmbeddingModel(EmbeddingModel):
    def __init__(self, hf_model: str, batch_size: int = 32, model: Any | None = None):
        self.hf_model = hf_model
        self.model_name = f"local:{hf_model}"
        self.batch_size = batch_size
        self.provider = "local"
        self._model = model
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:  # the first concurrent callers must not each load (and download) the weights
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise MissingExtraError(
                        "Local embeddings need the optional `sentence-transformers` package. Install it with: pip install 'ragbench[local]'"
                    ) from exc
                self._model = SentenceTransformer(self.hf_model)
            return self._model

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, 0), dtype=np.float32), model=self.model_name, input_tokens=0, cost=CostBreakdown())
        model = self._load()
        with self._lock:  # one encode at a time: torch already uses every core, and concurrent calls only add contention
            encoded = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(encoded, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, np.maximum(norms, 1e-12))
        tokens = sum(estimate_tokens(text, self.model_name) for text in texts)
        return EmbeddingResult(vectors=vectors, model=self.model_name, input_tokens=tokens, cost=CostBreakdown(embedding_input_tokens=tokens))


@EMBEDDERS.register("local")
def _local_embedder(model: str, *, providers: dict[str, Any]) -> EmbeddingModel:
    return LocalEmbeddingModel(model)
