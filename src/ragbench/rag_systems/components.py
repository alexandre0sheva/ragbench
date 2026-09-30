"""Shared builders so systems do not each re-implement chunker / embedder / vector-store construction."""

from __future__ import annotations

from typing import Any

from ragbench.config.schema import ChunkerConfig
from ragbench.documents.chunkers import BaseChunker, create_chunker
from ragbench.models.embeddings import EmbeddingModel, create_embedding_model
from ragbench.models.llms import LLM
from ragbench.models.rerankers import Reranker, create_reranker
from ragbench.rag_systems.options import VectorOptions
from ragbench.stores.vector_store import VectorStore

DEFAULT_GENERATOR = "gpt-5.4-nano"
DEFAULT_EMBEDDING = "text-embedding-3-small"


def build_chunker(cfg: ChunkerConfig | dict[str, Any] | None) -> BaseChunker:
    return create_chunker(cfg)


def build_embedder(models: dict[str, Any], force_mock: bool = False) -> EmbeddingModel:
    """Embedding model named by the system's `models.embedding` (cache-wrapped; hashing embeddings in mock mode)."""
    return create_embedding_model(models.get("embedding", DEFAULT_EMBEDDING), force_mock=force_mock)


def build_vector_index(embedder: EmbeddingModel, options: VectorOptions, name: str) -> VectorStore:
    """Vector store for one system; `name` keeps Chroma collections of different systems apart."""
    return VectorStore(embedder, backend=options.vector_store, collection_name=name, persist_directory=options.persist_directory)


def build_reranker(name: str, llm: LLM | None = None) -> Reranker:
    return create_reranker(name, llm=llm)
