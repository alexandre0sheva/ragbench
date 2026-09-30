from __future__ import annotations

import difflib
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator


class RunConfig(BaseModel):
    name: str = "ragbench_run"
    output_dir: Path = Path("results")


class DatasetConfig(BaseModel):
    documents_path: Path
    questions_path: Path
    qrels_path: Path | None = None


def _known_chunker(name: str) -> str:
    import ragbench.documents.chunkers  # noqa: F401  (registers the built-in chunkers)
    from ragbench.registry import CHUNKERS

    CHUNKERS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
    return name


ChunkerName = Annotated[str, AfterValidator(_known_chunker)]


class ChunkerConfig(BaseModel):
    """`chunker:` section of systems that index a flat list of chunks. Unset sizes use the chunker's own defaults."""

    model_config = ConfigDict(extra="forbid")

    type: ChunkerName = Field(default="token", description="Chunker name: `token`/`word`, `fixed_char`, or `markdown`.")
    chunk_size: int | None = Field(default=None, ge=1, description="Chunk size: words for `token`/`markdown`, characters for `fixed_char`. Default: 500 (`token`, `markdown`) or 1200 (`fixed_char`).")
    chunk_overlap: int | None = Field(default=None, ge=0, description="Overlap between consecutive chunks, same unit as `chunk_size`. Default: 80 (`token`, `markdown`) or 150 (`fixed_char`).")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkerConfig:
        if self.chunk_size is not None and self.chunk_overlap is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ParentChildChunkerConfig(BaseModel):
    """`chunker:` section of `parent_doc`: large parent chunks for context, small child chunks for search."""

    model_config = ConfigDict(extra="forbid")

    parent_chunk_size: int = Field(default=1000, ge=1, description="Parent chunk size in words.")
    parent_chunk_overlap: int = Field(default=150, ge=0, description="Overlap between parent chunks in words.")
    child_chunk_size: int = Field(default=250, ge=1, description="Child chunk size in words.")
    child_chunk_overlap: int = Field(default=50, ge=0, description="Overlap between child chunks in words.")

    @model_validator(mode="after")
    def _overlaps_smaller_than_sizes(self) -> ParentChildChunkerConfig:
        if self.parent_chunk_overlap >= self.parent_chunk_size or self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError("chunk overlaps must be smaller than their chunk sizes")
        return self


def _format_section_errors(label: str, section: str, model: type[BaseModel], exc: ValidationError) -> str:
    valid = sorted(model.model_fields)
    problems: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if error["type"] == "extra_forbidden":
            close = difflib.get_close_matches(location, valid, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            problems.append(f"unknown option '{section}.{location}'.{hint}")
        else:
            prefix = f"{section}.{location}: " if location else f"{section}: "
            problems.append(prefix + str(error["msg"]).removeprefix("Value error, "))
    return f"{label}: " + "; ".join(problems) + f" (valid `{section}` options: {', '.join(valid)})"


def _validate_section(label: str, section: str, model: type[BaseModel] | None, value: dict[str, Any]) -> None:
    if model is None:
        if value:
            raise ValueError(f"{label}: this system does not use `{section}`, but it was given ({', '.join(sorted(value))})")
        return
    try:
        model.model_validate(value)
    except ValidationError as exc:
        raise ValueError(_format_section_errors(label, section, model, exc)) from None


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    name: str | None = None
    chunker: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    llm_features: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_name(self) -> str:
        return self.name or self.type

    @model_validator(mode="after")
    def _validate_sections_against_spec(self) -> SystemConfig:
        """Check `chunker` / `retrieval` / `llm_features` against the system's declared option models.

        Systems that are not registered (or declare no spec) are left alone here; an unknown `type` is
        reported by `ExperimentConfig` and `create_rag_system`.
        """
        import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
        from ragbench.registry import SYSTEMS

        cls = SYSTEMS.mapping.get(self.type)
        spec = getattr(cls, "spec", None)
        if spec is None:
            return self
        label = f"System '{self.resolved_name}' (type {self.type})"
        _validate_section(label, "chunker", spec.chunker, self.chunker)
        _validate_section(label, "retrieval", spec.options, self.retrieval)
        _validate_section(label, "llm_features", spec.llm_features, self.llm_features)
        return self


def default_primary_k(k_values: list[int]) -> int:
    """Headline cut-off for recall: 5 when available, otherwise the median of `k_values`."""
    ks = sorted(set(k_values))
    return 5 if 5 in ks else ks[len(ks) // 2]


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    # How many chunks the generator sees when a system does not set its own `top_k`/`final_top_k`.
    context_k: int = Field(default=5, ge=1)
    # How many ranked chunks retrieval metrics are scored on. None = max(k_values).
    retrieval_depth: int | None = None
    # Recall cut-off used for headline columns, failure classification and the qrels audit. None = 5 or the median k.
    primary_k: int | None = None
    # A system whose share of failed questions exceeds this fails the run (results are still written).
    max_error_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    judge_enabled: bool = True
    judge_model: str = "gpt-5.4-nano"
    max_questions: int | None = None
    max_workers: int = Field(default=4, ge=1)
    # Systems evaluated at the same time (each still ingests once). Total concurrency is system_workers x max_workers.
    system_workers: int = Field(default=1, ge=1)
    # Parallel LLM enrichment calls / embedding batches during ingestion, per system.
    ingest_workers: int = Field(default=4, ge=1)
    # Questions re-run one at a time with the disk cache off, after the parallel pass, for clean latency (live runs only).
    latency_probe_questions: int = Field(default=5, ge=0)
    # Share corpus embeddings across systems in one run. Hits are still charged
    # to each system at standalone prices; real savings appear in run_summary.
    embedding_cache: bool = True

    @model_validator(mode="after")
    def _check_cutoffs(self) -> EvaluationConfig:
        if not self.k_values or any(k < 1 for k in self.k_values):
            raise ValueError("evaluation.k_values must be a non-empty list of positive integers")
        if self.retrieval_depth is not None and self.retrieval_depth < max(self.k_values):
            raise ValueError(f"evaluation.retrieval_depth ({self.retrieval_depth}) must be >= max(k_values) ({max(self.k_values)})")
        if self.primary_k is not None and self.primary_k not in self.k_values:
            raise ValueError(f"evaluation.primary_k ({self.primary_k}) must be one of k_values {self.k_values}")
        return self

    @property
    def resolved_retrieval_depth(self) -> int:
        return self.retrieval_depth if self.retrieval_depth is not None else max(self.k_values)

    @property
    def resolved_primary_k(self) -> int:
        return self.primary_k if self.primary_k is not None else default_primary_k(self.k_values)


class CacheConfig(BaseModel):
    """`cache:` section: persistent caches for paid calls. Mock runs never use the disk cache (nothing to save)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Master switch for the disk cache (`--no-cache` turns it off for one run).")
    dir: Path = Field(default=Path(".ragbench_cache"), description="Cache directory; `RAGBENCH_CACHE_DIR` overrides it.")
    llm: bool = Field(default=True, description="Cache LLM responses (generation, rewriting, judging) for temperature-0 requests.")
    embeddings: bool = Field(default=True, description="Cache corpus embeddings across runs.")
    cache_query_embeddings: bool = Field(
        default=False, description="Also cache query embeddings. Off by default because a hit makes measured query latency look too low."
    )
    llm_nonzero_temperature: bool = Field(default=False, description="Also cache LLM requests made with temperature > 0 (freezes one sample).")
    ttl_days: int | None = Field(default=None, ge=1, description="Ignore entries older than this many days (default: keep forever).")


class LimitsConfig(BaseModel):
    """`limits:` section: throttles shared by every request to the provider (today: OpenAI)."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_requests: int | None = Field(default=None, ge=1, description="Requests in flight at once, across all systems and questions.")
    requests_per_minute: int | None = Field(default=None, ge=1, description="Requests started per rolling minute.")
    tokens_per_minute: int | None = Field(default=None, ge=1, description="Estimated prompt tokens per rolling minute.")


class ModelPrice(BaseModel):
    """USD per 1M tokens; `output` stays 0 for embedding models."""

    input: float = 0.0
    output: float = 0.0


class ExperimentConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    systems: list[SystemConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # Per-model price overrides (take precedence over the built-in table in models/cost.py).
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @model_validator(mode="after")
    def _check_systems(self) -> ExperimentConfig:
        import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
        from ragbench.registry import SYSTEMS, UnknownComponentError

        seen: set[str] = set()
        for index, system in enumerate(self.systems):
            try:
                SYSTEMS.get(system.type)
            except UnknownComponentError as exc:
                raise ValueError(f"systems[{index}]: {exc}") from None
            name = system.resolved_name
            if name in seen:
                raise ValueError(f"systems[{index}]: duplicate system name '{name}'. Give each system a unique `name`.")
            seen.add(name)
        return self
