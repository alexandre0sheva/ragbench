"""Named-component registries: the single place that knows which systems, chunkers, rerankers and vector backends exist.

Built-ins register themselves with `@SYSTEMS.register("name")` when their module is imported; third-party
packages can contribute components through entry points (see `Registry.load_entry_points`).
"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from ragbench.documents.chunkers import BaseChunker
    from ragbench.documents.loaders.registry import LoadContext
    from ragbench.documents.schema import Document
    from ragbench.models.embeddings import EmbeddingModel
    from ragbench.models.llms import LLM
    from ragbench.models.rerankers import Reranker
    from ragbench.rag_systems.base import BaseRAGSystem
    from ragbench.stores.index import VectorIndex
    from ragbench.tools.base import BaseTool

logger = logging.getLogger(__name__)

T = TypeVar("T")


class UnknownComponentError(ValueError):
    """A config referenced a component name that is not registered."""


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}
        self._loaded_groups: set[str] = set()

    # -- registration ---------------------------------------------------------------------------

    def add(self, name: str, obj: T, aliases: tuple[str, ...] = ()) -> T:
        for taken in (name, *aliases):
            if taken in self._aliases or (taken in self._items and (self._items[taken] is not obj or taken != name)):
                raise ValueError(f"{self.kind} {taken!r} is already registered")
        self._items[name] = obj
        for alias in aliases:
            self._aliases[alias] = name
        return obj

    def register(self, name: str, *, aliases: tuple[str, ...] = ()) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            return self.add(name, obj, aliases)

        return decorator

    # -- lookup ---------------------------------------------------------------------------------

    def get(self, name: str) -> T:
        canonical = self._aliases.get(name, name)
        if canonical in self._items:
            return self._items[canonical]
        known = self.names()
        close = difflib.get_close_matches(name, [*known, *self._aliases], n=1, cutoff=0.6)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise UnknownComponentError(f"Unknown {self.kind} {name!r}.{hint} Available: {', '.join(sorted(known))}")

    def names(self) -> list[str]:
        """Canonical names in registration order (aliases excluded)."""
        return list(self._items)

    def all_names(self) -> list[str]:
        """Canonical names followed by every alias."""
        return [*self._items, *self._aliases]

    def items(self) -> Iterator[tuple[str, T]]:
        return iter(self._items.items())

    def __contains__(self, name: object) -> bool:
        return name in self._items or name in self._aliases

    @property
    def mapping(self) -> dict[str, T]:
        """The live name -> component dict. Mutating it registers/unregisters (kept for `SYSTEM_REGISTRY` compatibility)."""
        return self._items

    # -- plugins --------------------------------------------------------------------------------

    def load_entry_points(self, group: str) -> None:
        """Register every component advertised under the entry-point `group` (once per group).

        A plugin that fails to import is logged and skipped rather than breaking RAGBench startup.
        """
        if group in self._loaded_groups:
            return
        self._loaded_groups.add(group)
        for entry_point in metadata.entry_points(group=group):
            try:
                component = entry_point.load()
            except Exception as exc:
                logger.warning("Could not load %s plugin %r from entry-point group %s: %s", self.kind, entry_point.name, group, exc)
                continue
            if entry_point.name in self:
                logger.warning("%s plugin %r ignored: the name is already registered", self.kind, entry_point.name)
                continue
            self.add(entry_point.name, component)


SYSTEMS: Registry[type[BaseRAGSystem]] = Registry("RAG system")
CHUNKERS: Registry[type[BaseChunker]] = Registry("chunker")
RERANKERS: Registry[type[Reranker]] = Registry("reranker")
# Provider factories: `factory(model, *, providers) -> LLM | EmbeddingModel`, where `model` is the ref without its provider prefix
# and `providers` maps endpoint names to their `providers:` config. Registered by `ragbench.models.providers`.
LLM_PROVIDERS: Registry[Callable[..., LLM]] = Registry("chat model provider")
EMBEDDERS: Registry[Callable[..., EmbeddingModel]] = Registry("embedding provider")
# Document loaders, keyed by file extension (`.pdf`): `loader(path, context) -> list[Document]`. Registered by `ragbench.documents.loaders`.
LOADERS: Registry[Callable[[Path, LoadContext], list[Document]]] = Registry("document loader")
# Vector index backends (`vector_store:`); classes taking `(collection_name=, persist_directory=)`. Registered by `ragbench.stores.index`.
VECTOR_BACKENDS: Registry[type[VectorIndex]] = Registry("vector backend")
# Built-in tools, keyed by the name an agent calls them by; classes taking their `Options` as keyword arguments. Registered by `ragbench.tools`.
TOOLS: Registry[type[BaseTool]] = Registry("tool")
