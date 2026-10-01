"""Typed, documented `retrieval:` options shared by the built-in systems.

Every option model forbids unknown keys, so a typo such as `rerank_top_k` fails at config-load time with a
suggestion instead of being silently ignored. Field descriptions and defaults feed `docs/systems.md`.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.config.schema import SystemConfig


def _known_reranker(name: str) -> str:
    import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
    from ragbench.registry import RERANKERS

    RERANKERS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
    return name


RerankerName = Annotated[str, AfterValidator(_known_reranker)]


def _known_backend(name: str) -> str:
    import ragbench.stores.index  # noqa: F401  (registers the built-in vector backends)
    from ragbench.registry import VECTOR_BACKENDS

    VECTOR_BACKENDS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
    return name


VectorBackendName = Annotated[str, AfterValidator(_known_backend)]


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
    vector_store: VectorBackendName = Field(
        default="numpy",
        description=(
            "Vector backend: `numpy` (exact, no extra), `faiss` (exact) or `faiss_hnsw` (approximate) with `pip install 'ragbench[faiss]'`, "
            "`chroma` (approximate) with `ragbench[chroma]`, `qdrant` (exact, local mode) with `ragbench[qdrant]`. `in_memory` is a deprecated alias of `numpy`. "
            "A missing library is an error, never a silent fallback."
        ),
    )
    persist_directory: str | None = Field(default=None, description="Directory for a persistent store (`chroma`, `qdrant`). Default: in memory.")

    @model_validator(mode="after")
    def _persist_directory_needs_a_persistent_backend(self) -> VectorOptions:
        import ragbench.stores.index  # noqa: F401
        from ragbench.registry import VECTOR_BACKENDS

        if self.persist_directory is not None and not VECTOR_BACKENDS.get(self.vector_store).persistent:
            persistent = ", ".join(name for name, cls in VECTOR_BACKENDS.items() if cls.persistent)
            raise ValueError(f"persist_directory only applies to the persistent backends ({persistent}), not '{self.vector_store}'")
        return self


class FusionOptions(BaseOptions):
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion smoothing constant.")


class MultiQueryOptions(BaseOptions):
    multi_query: bool = Field(default=False, description="Retrieve with local (non-LLM) query variants to help multi-hop questions.")
    max_query_variants: int = Field(default=4, ge=1, description="Maximum number of query variants when `multi_query` is on.")


class MMROptions(BaseOptions):
    """Maximal marginal relevance: re-pick the candidates so each new chunk is relevant but not a near-duplicate of one already chosen."""

    diversity: Literal["none", "mmr"] = Field(
        default="none",
        description="`mmr` re-ranks a candidate pool by maximal marginal relevance so near-duplicate chunks stop crowding out other evidence. `none` keeps the plain ranking.",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Relevance versus diversity when `diversity: mmr`: 1.0 reproduces the plain ranking, 0.0 only avoids repeats. Relevance is the retrieval score scaled so the best candidate is 1.",
    )
    mmr_candidates: int = Field(default=20, ge=1, description="Candidates MMR chooses from when `diversity: mmr` (raised to the retrieval depth if smaller).")


class BM25Options(TopKOptions):
    pass


class VectorSystemOptions(VectorOptions, TopKOptions, MMROptions):
    pass


class HybridCoreOptions(VectorOptions, FinalTopKOptions, FusionOptions, MultiQueryOptions):
    bm25_top_k: int = Field(default=20, ge=1, description="BM25 candidates per query (raised to the retrieval depth if smaller).")
    vector_top_k: int = Field(default=20, ge=1, description="Vector candidates per query (raised to the retrieval depth if smaller).")
    bm25_weight: float = Field(default=1.0, ge=0, description="Weight of the BM25 ranking in weighted RRF; raise it to favor lexical evidence.")
    vector_weight: float = Field(default=1.0, ge=0, description="Weight of the vector ranking in weighted RRF; raise it to favor semantic evidence.")


class HybridOptions(HybridCoreOptions, MMROptions):
    pass


class RerankerModelOptions(BaseOptions):
    """`reranker_model` for systems with a `reranker` option."""

    reranker_model: str | None = Field(
        default=None,
        description="Hugging Face model for `reranker: cross_encoder` (default `BAAI/bge-reranker-base`). Needs `pip install 'ragbench[rerank]'`.",
    )

    @model_validator(mode="after")
    def _model_needs_a_reranker_that_loads_one(self) -> RerankerModelOptions:
        import ragbench.models.rerankers  # noqa: F401  (registers the built-in rerankers)
        from ragbench.registry import RERANKERS

        name = getattr(self, "reranker", None)
        if self.reranker_model is not None and name is not None and not getattr(RERANKERS.get(name), "takes_model", False):
            raise ValueError(f"reranker_model only applies to rerankers that load a model (cross_encoder), not '{name}'")
        return self


_RERANKER_NAMES = "`simple_keyword_overlap`, `local_relevance` (TF-IDF), `cross_encoder` (needs the `rerank` extra), or `llm`."


class HybridRerankOptions(RerankerModelOptions, HybridCoreOptions):
    candidate_top_k: int = Field(default=30, ge=1, description="Fused candidates handed to the reranker (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="local_relevance", description=_RERANKER_NAMES)


class RerankOptions(RerankerModelOptions, VectorOptions, FinalTopKOptions, MultiQueryOptions):
    candidate_top_k: int = Field(default=30, ge=1, description="Vector candidates handed to the reranker (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="simple_keyword_overlap", description=_RERANKER_NAMES)


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


class LLMHeavyOptions(RerankerModelOptions, VectorOptions, TopKOptions):
    per_query_top_k: int = Field(default=10, ge=1, description="Candidates per rewritten query (raised to the retrieval depth if smaller).")
    reranker: RerankerName = Field(default="simple_keyword_overlap", description="Reranker used when `llm_features.enable_llm_rerank` is off: " + _RERANKER_NAMES)


class LLMHeavyFeatures(BaseModel):
    """`llm_features:` section of the `llm_heavy` system."""

    model_config = ConfigDict(extra="forbid")

    enable_llm_ingestion: bool = Field(default=False, description="Enrich every chunk at ingestion with an LLM summary, entities, and hypothetical questions.")
    enable_query_rewrite: bool = Field(default=True, description="Rewrite each question into several search queries with the LLM.")
    enable_llm_rerank: bool = Field(default=False, description="Rerank candidates with an LLM call.")


class NoRetrievalOptions(BaseOptions):
    """`no_retrieval` has nothing to configure: the model answers from its own knowledge."""


class FullContextOptions(BaseOptions):
    context_token_budget: int = Field(
        default=100_000,
        ge=1,
        description=(
            "Most tokens of document text put in the prompt (counted with the chunking tokenizer; the small `[doc_id]` headers are not counted). "
            "Documents are ordered by BM25 relevance to the question; when they do not all fit, the first one that does not is cut at a token "
            "boundary and the rest are dropped (`truncated: true` in the retrieval metadata). You pay for every token you put in."
        ),
    )


class SentenceWindowOptions(VectorOptions, TopKOptions):
    window: int = Field(
        default=2, ge=0, description="Sentences added on each side of a matched sentence (0 = the sentence alone). Overlapping windows of one document merge."
    )
    candidate_top_k: int = Field(default=30, ge=1, description="Sentences searched before windows are built (raised to four times the retrieval depth if smaller).")


class ContextualOptions(HybridOptions):
    document_max_chars: int = Field(
        default=12000,
        ge=500,
        description=(
            "Characters of the source document shown to the LLM when it writes each chunk's context. A longer document is cut to a window "
            "around the chunk. Raise it for long documents whose chunks need context from far away; every call pays for the whole excerpt."
        ),
    )


class HierarchicalOptions(TopKOptions, FusionOptions):
    docs_k: int = Field(default=5, ge=1, description="Documents the first stage selects (by BM25 and embedding similarity over each document's title and summary).")
    chunks_per_doc: int = Field(default=2, ge=1, description="Chunks taken from each selected document in the second stage; the best of them overall are returned.")
    summary_max_chars: int = Field(default=6000, ge=200, description="Characters of each document shown to the LLM when it writes the summary.")


class LLMQueryOptions(VectorOptions, FinalTopKOptions, FusionOptions):
    """Shared by the systems that search several LLM-planned queries and merge the rankings with RRF."""

    retriever: Literal["hybrid", "vector"] = Field(
        default="hybrid", description="Search used for every query: `hybrid` (BM25 + vector, fused with RRF) or `vector` alone."
    )
    per_query_top_k: int = Field(default=20, ge=1, description="Candidates per query before the rankings are merged (raised to the retrieval depth if smaller).")


class RagFusionOptions(LLMQueryOptions):
    num_queries: int = Field(default=4, ge=1, le=10, description="Alternative queries the LLM writes for each question.")
    include_original: bool = Field(default=True, description="Also search the original question and merge it in (standard RAG-Fusion).")


class DecomposeOptions(LLMQueryOptions):
    max_subquestions: int = Field(default=4, ge=1, le=10, description="Most sub-questions the LLM may split a question into; extras are dropped.")
    sequential: bool = Field(
        default=False,
        description=(
            "Answer the sub-questions in order, each from its own evidence, and add the earlier findings to the next search "
            "(for questions whose later parts depend on earlier answers). Costs one extra LLM call per sub-question; off, they are searched independently."
        ),
    )
    chunks_per_subquestion: int = Field(default=3, ge=1, description="Chunks of a sub-question's ranking the LLM reads to answer it when `sequential` is on.")
    include_original: bool = Field(default=False, description="Also search the original question, as a safety net against a bad split.")


class SpendingCaps(BaseOptions):
    """Spending caps of the agentic systems. They are checked before each extra agent call; the final answer call is always made."""

    max_cost_usd: float | None = Field(
        default=None, gt=0, description="Stop the agent's extra work once this question has cost this much (USD, at standalone prices); the answer is then written from the evidence so far. No cap by default."
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Stop the agent's extra work once this question has used this many LLM tokens (prompt + completion). No cap by default."
    )


class AgenticOptions(SpendingCaps, LLMQueryOptions):
    """Spending caps plus the shared multi-query search options, for `corrective` and `iterative`."""


class CorrectiveOptions(AgenticOptions):
    max_rounds: int = Field(default=2, ge=0, le=10, description="Retries after the first retrieval when too little relevant evidence was found (0 = grade only, never retry).")
    grade_top_k: int = Field(default=5, ge=1, description="Chunks of the ranking the LLM grades per round (graded in one call).")
    min_relevant: int = Field(default=1, ge=1, description="Chunks graded `relevant` needed to stop retrying.")
    expand_to: Literal["hybrid", "full_doc", "none"] = Field(
        default="hybrid",
        description=(
            "How a retry widens the search besides rewriting the query: `hybrid` searches with BM25 + vector at twice the depth and grades twice as many chunks, "
            "`full_doc` also adds every chunk of the two best documents (right document, wrong chunk), `none` only rewrites."
        ),
    )
    self_check: bool = Field(
        default=False,
        description="After answering, ask the LLM whether the answer is supported by the context and, if not, regenerate once from the supported claims. Costs one or two extra calls.",
    )


class IterativeOptions(AgenticOptions):
    max_hops: int = Field(default=3, ge=1, le=10, description="Most search rounds; each asks the LLM what is known, what is missing, and what to search next.")
    chunks_per_hop: int = Field(default=3, ge=1, description="Top chunks of each round the LLM sees as evidence (the final answer reads the merged ranking).")


class ToolAgentOptions(SpendingCaps, FinalTopKOptions):
    """Options of the tool-using agents (`agent_search`, `grep_agent`). Retrieval is scored on the passages the agent's tool calls returned."""

    agent_mode: Literal["native", "react_json"] = Field(
        default="native",
        description="`native` uses the provider's function calling. `react_json` describes the tools in the prompt and parses one JSON action per reply, for models without function calling.",
    )
    max_steps: int = Field(default=6, ge=1, le=50, description="Most model turns (each may call several tools). When it is reached the agent is asked to answer with what it has.")
    max_tool_calls: int | None = Field(
        default=None, ge=0, description="Most tool calls per question; further calls are refused and the agent is asked to answer. No cap by default."
    )


class AgentSearchOptions(ToolAgentOptions, LLMQueryOptions):
    """`retriever` and `per_query_top_k` configure the `search` tool the agent always has."""


class GrepAgentOptions(ToolAgentOptions):
    """`grep_agent` has no index, so it has no search options: only the agent's own limits."""


class AdaptiveOptions(BaseOptions):
    router: Literal["heuristic", "llm"] = Field(
        default="heuristic",
        description=(
            "`heuristic` routes with fixed rules and costs nothing: exact identifiers go to `lexical`, arithmetic, percentages and date calculations to `computation`, "
            "comparisons and questions with several linked parts to `multi_hop`, everything else to `default`. `llm` asks the system's generator model to choose among "
            "the routes (one short call per question), which also allows route names of your own."
        ),
    )
    routes: dict[str, SystemConfig] = Field(
        description=(
            "The pipelines a question can be sent to, by route name; each is a full system config (`type`, `chunker`, `retrieval`, `tools`, ...). "
            "`default` is required. With `router: heuristic` the other names must be `lexical`, `computation` or `multi_hop`; a role you leave out "
            "falls back to `default`. Routes inherit the adaptive system's `models` unless they set their own."
        )
    )
    route_descriptions: dict[str, str] = Field(
        default_factory=dict, description="`router: llm` only: what each route is for, shown to the model. Known role names have a built-in description."
    )

    @model_validator(mode="after")
    def _check_routes(self) -> AdaptiveOptions:
        import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
        from ragbench.agents.router import DEFAULT_ROUTE, HEURISTIC_ROLES
        from ragbench.registry import SYSTEMS

        if DEFAULT_ROUTE not in self.routes:
            raise ValueError(f"routes must include a '{DEFAULT_ROUTE}' route, the one used when no other applies (got: {', '.join(self.routes) or 'none'})")
        for name, config in self.routes.items():
            try:
                SYSTEMS.get(config.type)
            except ValueError as exc:
                raise ValueError(f"routes.{name}: {exc}") from None
            if config.type == "adaptive":
                raise ValueError(f"routes.{name}: a route cannot itself be an adaptive system")
        if self.router == "heuristic":
            unknown = [name for name in self.routes if name != DEFAULT_ROUTE and name not in HEURISTIC_ROLES]
            if unknown:
                raise ValueError(
                    f"the heuristic router only knows the routes {', '.join(HEURISTIC_ROLES)} (and '{DEFAULT_ROUTE}'), so {', '.join(repr(n) for n in unknown)} "
                    "would never be chosen; rename it, or set `router: llm` to use route names of your own"
                )
        stray = [name for name in self.route_descriptions if name not in self.routes]
        if stray:
            raise ValueError(f"route_descriptions names routes that do not exist: {', '.join(stray)}")
        return self
