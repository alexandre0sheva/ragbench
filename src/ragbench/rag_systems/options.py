"""Typed, documented `retrieval:` options shared by the built-in systems.

Every option model forbids unknown keys, so a typo such as `rerank_top_k` fails at config-load time with a
suggestion instead of being silently ignored. Field descriptions and defaults feed `docs/systems.md`.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator


def _known_reranker(name: str) -> str:
    import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
    from ragbench.registry import RERANKERS

    RERANKERS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
    return name


RerankerName = Annotated[str, AfterValidator(_known_reranker)]


class BaseOptions(BaseModel):
    """Base for all per-system `retrieval:` option models."""

    model_config = ConfigDict(extra="forbid")

    @property
    def configured_top_k(self) -> int | None:
        """How many chunks this system hands the generator when the caller does not say (None = not configured)."""
        return None


class PermissiveOptions(BaseOptions):
    """Fallback for custom systems that declare no `spec`: accepts any keys, understands `top_k` / `final_top_k`."""

    model_config = ConfigDict(extra="allow")

    top_k: int | None = None
    final_top_k: int | None = None

    @property
    def configured_top_k(self) -> int | None:
        return self.final_top_k or self.top_k


class TopKOptions(BaseOptions):
    default_top_k: ClassVar[int] = 5
    top_k: int | None = Field(
        default=None, ge=1, description="Chunks given to the generator. Default 5; when unset the evaluator uses `evaluation.context_k`."
    )

    @property
    def configured_top_k(self) -> int | None:
        return self.top_k

    def resolve_top_k(self, override: int | None) -> int:
        """`override` is the retrieval depth passed by the evaluator; otherwise the configured or default top-k."""
        return override or self.configured_top_k or self.default_top_k


class FinalTopKOptions(BaseOptions):
    default_top_k: ClassVar[int] = 5
    final_top_k: int | None = Field(
        default=None, ge=1, description="Chunks given to the generator. Default 5; when unset the evaluator uses `evaluation.context_k`."
    )
    top_k: int | None = Field(default=None, ge=1, description="Alias for `final_top_k`; `final_top_k` wins when both are set.")

    @property
    def configured_top_k(self) -> int | None:
        return self.final_top_k or self.top_k

    def resolve_top_k(self, override: int | None) -> int:
        return override or self.configured_top_k or self.default_top_k


class VectorOptions(BaseOptions):
    vector_store: Literal["chroma", "in_memory"] = Field(
        default="chroma", description="Vector backend. `chroma` falls back to exact in-memory search if Chroma cannot be used."
    )
    persist_directory: str | None = Field(default=None, description="Directory for a persistent Chroma store (default: ephemeral).")


class FusionOptions(BaseOptions):
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion smoothing constant.")


class MultiQueryOptions(BaseOptions):
    multi_query: bool = Field(default=False, description="Retrieve with local (non-LLM) query variants to help multi-hop questions.")
    max_query_variants: int = Field(default=4, ge=1, description="Maximum number of query variants when `multi_query` is on.")


class BM25Options(TopKOptions):
    pass


class VectorSystemOptions(VectorOptions, TopKOptions):
    pass


class HybridOptions(VectorOptions, FinalTopKOptions, FusionOptions, MultiQueryOptions):
    bm25_top_k: int = Field(default=20, ge=1, description="BM25 candidates per query (raised to the retrieval depth if smaller).")
    vector_top_k: int = Field(default=20, ge=1, description="Vector candidates per query (raised to the retrieval depth if smaller).")
    bm25_weight: float = Field(default=1.0, ge=0, description="Weight of the BM25 ranking in weighted RRF; raise it to favor lexical evidence.")
    vector_weight: float = Field(default=1.0, ge=0, description="Weight of the vector ranking in weighted RRF; raise it to favor semantic evidence.")


class HybridRerankOptions(HybridOptions):
    candidate_top_k: int = Field(default=30, ge=1, description="Fused candidates handed to the reranker (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="local_relevance", description="`simple_keyword_overlap`, `local_relevance` (TF-IDF), or `llm`.")


class RerankOptions(VectorOptions, FinalTopKOptions, MultiQueryOptions):
    candidate_top_k: int = Field(default=30, ge=1, description="Vector candidates handed to the reranker (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="simple_keyword_overlap", description="`simple_keyword_overlap`, `local_relevance` (TF-IDF), or `llm`.")


class ParentDocOptions(VectorOptions):
    top_k_children: int = Field(default=8, ge=1, description="Child chunks searched (raised to twice the parent depth if smaller).")
    top_k_parents: int = Field(default=4, ge=1, description="Parent chunks returned; also how many the generator reads.")
    parent_score_aggregation: Literal["max", "sum", "mean"] = Field(
        default="max",
        description="How child scores roll up to their parent: `max` keeps the best match, `sum` rewards parents hit by several children, `mean` averages.",
    )

    @field_validator("parent_score_aggregation", mode="before")
    @classmethod
    def _case_insensitive(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @property
    def configured_top_k(self) -> int | None:
        return self.top_k_parents


class HyDEOptions(VectorOptions, TopKOptions, FusionOptions):
    probe_top_k: int | None = Field(default=None, ge=1, description="Results fetched per ranking before fusion. Default: twice the retrieval depth, at least 10.")
    fuse_with_question: bool = Field(default=True, description="Fuse the hypothetical-document ranking with the raw-question ranking via RRF.")


class LLMHeavyOptions(VectorOptions, TopKOptions):
    per_query_top_k: int = Field(default=10, ge=1, description="Candidates per rewritten query (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="simple_keyword_overlap", description="Reranker used when `llm_features.enable_llm_rerank` is off.")


class LLMHeavyFeatures(BaseModel):
    """`llm_features:` section of the `llm_heavy` system."""

    model_config = ConfigDict(extra="forbid")

    enable_llm_ingestion: bool = Field(default=False, description="Enrich every chunk at ingestion with an LLM summary, entities, and hypothetical questions.")
    enable_query_rewrite: bool = Field(default=True, description="Rewrite each question into several search queries with the LLM.")
    enable_llm_rerank: bool = Field(default=False, description="Rerank candidates with an LLM call.")
