"""Exact brute-force cosine search with NumPy: the default backend, no extra dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ragbench.registry import VECTOR_BACKENDS


@VECTOR_BACKENDS.register("numpy", aliases=("in_memory",))
class NumpyIndex:
    backend: ClassVar[str] = "numpy"
    approximate: ClassVar[bool] = False
    persistent: ClassVar[bool] = False
    import_modules: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, collection_name: str, persist_directory: Path | None = None) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        self._vectors = vectors

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._vectors is None or len(self._vectors) == 0:
            return []
        scores = self._vectors @ query
        k = min(top_k, len(scores))
        candidates = np.argpartition(scores, -k)[-k:]
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        return [(int(row), float(scores[int(row)])) for row in ranked]
