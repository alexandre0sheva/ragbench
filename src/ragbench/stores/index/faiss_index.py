"""FAISS inner-product indexes (vectors are unit length, so inner product is cosine). Optional extra: `pip install 'ragbench[faiss]'`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ragbench.registry import VECTOR_BACKENDS
from ragbench.utils.extras import require_extra


@VECTOR_BACKENDS.register("faiss")
class FaissIndex:
    """Exact flat index."""

    backend: ClassVar[str] = "faiss"
    approximate: ClassVar[bool] = False
    persistent: ClassVar[bool] = False
    import_modules: ClassVar[tuple[str, ...]] = ("faiss",)

    def __init__(self, *, collection_name: str, persist_directory: Path | None = None) -> None:
        require_extra("faiss", "The faiss vector backend", "faiss", "faiss-cpu")
        self._index: Any | None = None

    def _make_index(self, faiss: Any, dim: int) -> Any:
        return faiss.IndexFlatIP(dim)

    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        import faiss

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        index = self._make_index(faiss, matrix.shape[1])
        index.add(matrix)
        self._index = index

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        self._prepare_search(k)
        scores, rows = self._index.search(np.ascontiguousarray(query, dtype=np.float32)[None, :], k)
        return [(int(row), float(score)) for row, score in zip(rows[0], scores[0], strict=True) if row >= 0]

    def _prepare_search(self, k: int) -> None:
        pass


@VECTOR_BACKENDS.register("faiss_hnsw")
class FaissHNSWIndex(FaissIndex):
    """HNSW graph index (approximate): faster than flat on large corpora, at some recall cost."""

    backend: ClassVar[str] = "faiss_hnsw"
    approximate: ClassVar[bool] = True
    neighbors = 32
    ef_construction = 200
    ef_search = 128

    def _make_index(self, faiss: Any, dim: int) -> Any:
        index = faiss.IndexHNSWFlat(dim, self.neighbors, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        return index

    def _prepare_search(self, k: int) -> None:
        assert self._index is not None
        self._index.hnsw.efSearch = max(self.ef_search, k)  # the search breadth must be at least k
