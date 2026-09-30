"""Chroma (HNSW, approximate). Optional extra: `pip install 'ragbench[chroma]'`."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ragbench.registry import VECTOR_BACKENDS
from ragbench.utils.extras import require_extra

# Systems may be built on several threads; Chroma's in-process client is shared, so index builds take turns.
_BUILD_LOCK = threading.Lock()


@VECTOR_BACKENDS.register("chroma")
class ChromaIndex:
    backend: ClassVar[str] = "chroma"
    approximate: ClassVar[bool] = True
    persistent: ClassVar[bool] = True
    import_modules: ClassVar[tuple[str, ...]] = ("chromadb",)

    def __init__(self, *, collection_name: str, persist_directory: Path | None = None) -> None:
        require_extra("chromadb", "The chroma vector backend", "chroma")
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._collection: Any | None = None
        self._row_of: dict[str, int] = {}

    def build(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        with _BUILD_LOCK:
            import chromadb  # first import inside the lock: two threads must not import it at once

            if self.persist_directory:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(self.persist_directory))
            else:
                client = chromadb.EphemeralClient()
            try:
                client.delete_collection(self.collection_name)
            except Exception:  # noqa: BLE001  (no such collection yet)
                pass
            collection = client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
            # A single add() above the client's limit (5461 on Chroma 1.5) is rejected, so send bounded batches.
            batch = max(1, int(client.get_max_batch_size()))
            metadatas: Any = payloads if all(payloads) else None  # Chroma refuses empty metadata dicts
            for start in range(0, len(ids), batch):
                stop = start + batch
                collection.add(
                    ids=ids[start:stop],
                    embeddings=[vector.astype(float).tolist() for vector in vectors[start:stop]],
                    metadatas=metadatas[start:stop] if metadatas is not None else None,
                )
            self._collection = collection
        self._row_of = {chunk_id: row for row, chunk_id in enumerate(ids)}

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self._collection is None or not self._row_of:
            return []
        result = self._collection.query(
            query_embeddings=[query.astype(float).tolist()], n_results=min(top_k, len(self._row_of)), include=["distances"]
        )
        ids, distances = result["ids"][0], result["distances"][0]
        return [(self._row_of[str(chunk_id)], 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances, strict=False)]
