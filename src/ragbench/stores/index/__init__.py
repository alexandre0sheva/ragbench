"""Vector backends. Importing this package registers every built-in one in `VECTOR_BACKENDS`.

Each backend's library is imported lazily, so importing the package needs none of the optional extras.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ragbench.registry import VECTOR_BACKENDS
from ragbench.stores.index.base import VectorIndex, sanitize_payload
from ragbench.stores.index.chroma_index import ChromaIndex
from ragbench.stores.index.faiss_index import FaissHNSWIndex, FaissIndex
from ragbench.stores.index.numpy_index import NumpyIndex
from ragbench.stores.index.qdrant_index import QdrantIndex

logger = logging.getLogger(__name__)

# Old names that still load but should be replaced: name -> replacement.
DEPRECATED_ALIASES = {"in_memory": "numpy"}
_WARNED_ALIASES: set[str] = set()

VECTOR_BACKENDS.load_entry_points("ragbench.vector_backends")


def create_index(name: str, *, collection_name: str, persist_directory: Path | None = None) -> VectorIndex:
    """Instantiate the backend registered as `name`. Raises `UnknownComponentError` / `MissingExtraError`; never falls back."""
    if name in DEPRECATED_ALIASES and name not in _WARNED_ALIASES:
        _WARNED_ALIASES.add(name)
        logger.warning("vector_store %r is deprecated; use %r (it is the same exact in-memory search).", name, DEPRECATED_ALIASES[name])
    return VECTOR_BACKENDS.get(name)(collection_name=collection_name, persist_directory=persist_directory)


__all__ = [
    "DEPRECATED_ALIASES",
    "ChromaIndex",
    "FaissHNSWIndex",
    "FaissIndex",
    "NumpyIndex",
    "QdrantIndex",
    "VectorIndex",
    "create_index",
    "sanitize_payload",
]
