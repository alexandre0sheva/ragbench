"""Chunkers. Importing this package registers every built-in one in `CHUNKERS`.

| name | splits on | `chunk_size` / `chunk_overlap` unit |
| --- | --- | --- |
| `fixed_char` | character windows | characters |
| `word` | whitespace-separated words (what `token` meant before 0.3.0) | words |
| `token` | real model tokens (tiktoken `o200k_base`) | tokens |
| `recursive` | paragraphs, lines, sentences, words | tokens |
| `sentence` | sentences | size in tokens, overlap in sentences |
| `semantic` | topic shifts between sentences (needs an embedding model) | size in tokens, overlap in sentences |
| `markdown` | headings, with the heading path in metadata | tokens |
"""

from __future__ import annotations

from ragbench.config.schema import ChunkerConfig
from ragbench.documents.chunkers.base import BaseChunker, Span
from ragbench.documents.chunkers.fixed import FixedCharacterChunker
from ragbench.documents.chunkers.markdown import MarkdownAwareChunker
from ragbench.documents.chunkers.recursive import RecursiveChunker
from ragbench.documents.chunkers.semantic import SemanticChunker
from ragbench.documents.chunkers.sentence import SentenceChunker
from ragbench.documents.chunkers.token import TokenChunker
from ragbench.documents.chunkers.word import WordChunker
from ragbench.models.embeddings import EmbeddingModel
from ragbench.registry import CHUNKERS


def create_chunker(config: ChunkerConfig | dict | None = None, embedder: EmbeddingModel | None = None) -> BaseChunker:
    """Build a chunker from a `chunker:` config section; unset sizes fall back to the chunker's own defaults.

    `embedder` is required by chunkers that embed text (`semantic`); `build_chunker` in `rag_systems/components.py` supplies it.
    """
    cfg = config if isinstance(config, ChunkerConfig) else ChunkerConfig.model_validate(config or {})
    cls = CHUNKERS.get(cfg.type)
    kwargs: dict = {
        "chunk_size": cfg.chunk_size if cfg.chunk_size is not None else cls.default_chunk_size,
        "chunk_overlap": cfg.chunk_overlap if cfg.chunk_overlap is not None else cls.default_chunk_overlap,
        "prefix_title": cfg.prefix_title,
    }
    for option in cls.options:
        value = getattr(cfg, option)
        if value is not None:
            kwargs[option] = value
    if cls.needs_embedder:
        if embedder is None:
            raise ValueError(f"The '{cfg.type}' chunker embeds text and needs an embedding model; build it with build_chunker(cfg, models=..., force_mock=...)")
        kwargs["embedder"] = embedder
    return cls(**kwargs)


__all__ = [
    "BaseChunker",
    "FixedCharacterChunker",
    "MarkdownAwareChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "Span",
    "TokenChunker",
    "WordChunker",
    "create_chunker",
]
