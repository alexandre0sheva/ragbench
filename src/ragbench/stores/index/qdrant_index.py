"""Qdrant in local mode (exact; in-memory, or on disk with `persist_directory`). Optional extra: `pip install 'ragbench[qdrant]'`.

A server-backed Qdrant is out of scope: this backend exists to compare retrieval behaviour, not deployment topologies.
Local mode locks its directory, so give each system its own `persist_directory`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ragbench.registry import VECTOR_BACKENDS
from ragbench.stores.index.base import sanitize_payload
from ragbench.utils.extras import require_extra

_UPSERT_BATCH = 256


@VECTOR_BACKENDS.register("qdrant")
class QdrantIndex:
    backend: ClassVar[str] = "qdrant"
    approximate: ClassVar[bool] = False
    persistent: ClassVar[bool] = True
    import_modules: ClassVar[tuple[str, ...]] = ("qdrant_client",)

    def __init__(self, *, collection_name: str, persist_directory: Path | None = None) -> None:
        require_extra("qdrant_client", "The qdrant vector backend", "qdrant", "qdrant-client")
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: Any | None = None
        self._size = 0

    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        from qdrant_client import QdrantClient, models

        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(self.persist_directory))
        else:
            client = QdrantClient(":memory:")
        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)
        client.create_collection(self.collection_name, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE))
        for start in range(0, len(ids), _UPSERT_BATCH):
            client.upsert(
                self.collection_name,
                points=[
                    models.PointStruct(id=row, vector=vectors[row].astype(float).tolist(), payload={"chunk_id": ids[row], **sanitize_payload(payloads[row])})
                    for row in range(start, min(start + _UPSERT_BATCH, len(ids)))
                ],
            )
        self._client, self._size = client, len(ids)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._client is None or self._size == 0:
            return []
        result = self._client.query_points(self.collection_name, query=query.astype(float).tolist(), limit=min(top_k, self._size))
        return [(int(point.id), float(point.score)) for point in result.points]

    def close(self) -> None:
        """Release the on-disk lock (local mode allows one open client per directory)."""
        if self._client is not None:
            self._client.close()
            self._client = None
