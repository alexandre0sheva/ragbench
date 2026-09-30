"""The contract for a vector backend, and helpers shared by the implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np


class VectorIndex(Protocol):
    """Nearest-neighbour index over chunk embeddings.

    Vectors are unit length (the embedding models normalize them), so every backend ranks by cosine similarity and
    reports it as the score. Rows are identified by their position in the `build` call; `VectorStore` maps them back to
    chunks. Constructing an index never falls back to another backend: a missing library raises `MissingExtraError`.

    Class attributes: `backend` (canonical registry name), `approximate` (may miss true neighbours), `persistent`
    (honours `persist_directory`), `import_modules` (libraries imported on the main thread before worker threads start).
    """

    backend: ClassVar[str]
    approximate: ClassVar[bool]
    persistent: ClassVar[bool]
    import_modules: ClassVar[tuple[str, ...]]

    def __init__(self, *, collection_name: str, persist_directory: Path | None = None) -> None: ...

    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Up to `top_k` `(row index, cosine score)` pairs, best first."""
        ...


def sanitize_payload(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Metadata as scalars only (what Chroma and Qdrant payloads accept); other values are stored as JSON text."""
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False, default=str)
    return sanitized
