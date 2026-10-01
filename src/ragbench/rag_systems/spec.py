from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from ragbench.config.schema import ChunkerConfig
from ragbench.rag_systems.options import BaseOptions

CostProfile = Literal["free", "low", "medium", "high"]
LatencyProfile = Literal["fast", "medium", "slow"]
# Aliases: inside `SystemSpec` the field named `type` shadows the builtin in annotations.
OptionsModel = type[BaseOptions]
SectionModel = type[BaseModel]


@dataclass(frozen=True)
class SystemSpec:
    """Self-description of a RAG system; the single source for `list-systems`, leaderboards and `docs/systems.md`.

    `cost_profile` is relative spend per question: `low` = one generation call plus embeddings, `medium` = one
    extra short LLM call, `high` = several LLM calls per question, a prompt that carries the whole corpus, or LLM work at ingestion. `latency_profile`
    follows the same idea (`fast` = no LLM call besides generation). `requires_llm` means the retrieval side
    itself calls an LLM (beyond generating the answer); `agentic` means the system runs a multi-step loop.
    `retrieves` is False for systems that fetch nothing (the `no_retrieval` floor baseline): their retrieval metrics are
    reported as missing, never as 0. `supports_tools` systems accept a `tools:` list (see `docs/tools.md`). `chunker=None` means the system has no `chunker:` section.
    """

    type: str
    title: str
    summary: str
    best_for: str
    cost_profile: CostProfile
    latency_profile: LatencyProfile
    requires_llm: bool
    agentic: bool
    options: OptionsModel
    chunker: SectionModel | None = ChunkerConfig
    llm_features: SectionModel | None = None
    retrieves: bool = True
    supports_tools: bool = False  # the system takes a `tools:` list (agentic systems that call tools)
