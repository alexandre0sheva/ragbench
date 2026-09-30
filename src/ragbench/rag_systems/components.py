"""Shared builders so systems do not each re-implement chunker / embedder / vector-store construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragbench.config.schema import ChunkerConfig
from ragbench.documents.chunkers import BaseChunker, create_chunker
from ragbench.documents.schema import Document, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATOR_MODEL
from ragbench.models.embeddings import EmbeddingModel, create_embedding_model
from ragbench.models.llms import LLM
from ragbench.models.rerankers import Reranker, create_reranker
from ragbench.rag_systems.options import VectorOptions

if TYPE_CHECKING:
    from ragbench.stores.vector_store import VectorStore

DEFAULT_GENERATOR = DEFAULT_GENERATOR_MODEL
DEFAULT_EMBEDDING = DEFAULT_EMBEDDING_MODEL


def build_chunker(cfg: ChunkerConfig | dict[str, Any] | None, models: dict[str, Any] | None = None, force_mock: bool = False) -> BaseChunker:
    """Chunker for a system's `chunker:` section. Chunkers that embed text (`semantic`) get the system's embedding model."""
    from ragbench.registry import CHUNKERS

    config = cfg if isinstance(cfg, ChunkerConfig) else ChunkerConfig.model_validate(cfg or {})
    embedder = build_embedder(models or {}, force_mock) if CHUNKERS.get(config.type).needs_embedder else None
    return create_chunker(config, embedder=embedder)


def chunk_documents(chunker: BaseChunker, documents: list[Document]) -> tuple[list[TextChunk], CostBreakdown]:
    """Chunk `documents` and return the chunks with the cost the chunker itself incurred (embedding sentences for `semantic`)."""
    chunks = chunker.chunk(documents)
    return chunks, chunker.last_cost


def build_embedder(models: dict[str, Any], force_mock: bool = False) -> EmbeddingModel:
    """Embedding model named by the system's `models.embedding` (cache-wrapped; hashing embeddings in mock mode)."""
    return create_embedding_model(models.get("embedding", DEFAULT_EMBEDDING), force_mock=force_mock)


def build_vector_index(embedder: EmbeddingModel, options: VectorOptions, name: str) -> VectorStore:
    """Vector store for one system; `name` keeps collections of different systems apart."""
    from ragbench.stores.vector_store import VectorStore  # lazy: the store imports `rag_systems.base`, which loads this module

    return VectorStore(embedder, backend=options.vector_store, collection_name=name, persist_directory=options.persist_directory)


def build_reranker(name: str, llm: LLM | None = None, model: str | None = None, force_mock: bool = False) -> Reranker:
    """Reranker for a system's `retrieval.reranker` / `reranker_model` (mock runs never download a model)."""
    return create_reranker(name, llm=llm, model=model, force_mock=force_mock)
