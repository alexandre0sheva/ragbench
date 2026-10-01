from __future__ import annotations

import difflib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from ragbench.config.sweep import expand_sweeps
from ragbench.models.defaults import DEFAULT_JUDGE_MODEL
from ragbench.selection.constraints import Constraints
from ragbench.selection.scoring import PROFILES, Weights


class RunConfig(BaseModel):
    name: str = "ragbench_run"
    output_dir: Path = Path("results")


class TabularConfig(BaseModel):
    """`dataset.tabular:` options for `.csv`, `.tsv`, `.json` and `.jsonl` files (one document per row / record)."""

    model_config = ConfigDict(extra="forbid")

    text_columns: list[str] | None = Field(default=None, description="Columns that make up the document text (default: all of them).")
    id_column: str | None = Field(default=None, description="Column whose value names the document (`<file>#<value>`; default: the row number).")


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents_path: Path
    questions_path: Path
    qrels_path: Path | None = None
    include: list[str] | None = Field(default=None, description="Only load files whose path (relative to `documents_path`) matches one of these globs.")
    exclude: list[str] | None = Field(default=None, description="Skip files matching any of these globs. `.ragbenchignore` in the documents folder adds gitignore-style rules.")
    on_error: Literal["raise", "skip"] = Field(default="raise", description="What to do with a file that cannot be loaded: fail, or skip it and report a warning.")
    max_file_mb: float = Field(default=25, gt=0, description="Files larger than this many MB are treated as load errors.")
    tabular: TabularConfig = Field(default_factory=TabularConfig)


def _known_chunker(name: str) -> str:
    import ragbench.documents.chunkers  # noqa: F401  (registers the built-in chunkers)
    from ragbench.registry import CHUNKERS

    CHUNKERS.get(name)  # raises UnknownComponentError (a ValueError) with a did-you-mean hint
    return name


ChunkerName = Annotated[str, AfterValidator(_known_chunker)]


class ChunkerConfig(BaseModel):
    """`chunker:` section of systems that index a flat list of chunks. Unset sizes use the chunker's own defaults."""

    model_config = ConfigDict(extra="forbid")

    type: ChunkerName = Field(
        default="token",
        description="Chunker: `token` (real model tokens), `word` (whitespace words), `fixed_char`, `recursive`, `sentence`, `semantic`, or `markdown`.",
    )
    chunk_size: int | None = Field(
        default=None,
        ge=1,
        description="Maximum chunk size. Unit: tokens (`token`, `recursive`, `sentence`, `semantic`, `markdown`), words (`word`) or characters (`fixed_char`). Default: 500 (1200 for `fixed_char`).",
    )
    chunk_overlap: int | None = Field(
        default=None,
        ge=0,
        description="Overlap between consecutive chunks, in the unit of `chunk_size`; `sentence` and `semantic` count sentences instead. Default: 80 (150 for `fixed_char`, 1 for `sentence`, 0 for `semantic`).",
    )
    min_chunk_size: int | None = Field(
        default=None,
        ge=1,
        description="`recursive`, `sentence`, `semantic`, `markdown`: fold chunks (for `markdown`, sections) smaller than this many tokens into a neighbour while the result fits `chunk_size`. Default: 50 for `markdown`, off otherwise.",
    )
    prefix_title: bool = Field(default=False, description="Prepend the document title to every chunk's text, so it is embedded and searched with the chunk.")
    prefix_heading: bool | None = Field(default=None, description="`markdown` only: prepend the heading breadcrumb (`Guide > Returns`) to every chunk's text.")
    breakpoint_percentile: float | None = Field(
        default=None,
        gt=0,
        lt=100,
        description="`semantic` only: start a new chunk where the distance between consecutive sentences exceeds this percentile of the document's distances. Default: 90.",
    )

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkerConfig:
        if self.chunk_size is not None and self.chunk_overlap is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @model_validator(mode="after")
    def _options_supported_by_the_chunker(self) -> ChunkerConfig:
        import ragbench.documents.chunkers  # noqa: F401  (registers the built-in chunkers)
        from ragbench.registry import CHUNKERS

        cls = CHUNKERS.get(self.type)
        optional = {"min_chunk_size", "prefix_heading", "breakpoint_percentile"}
        unsupported = sorted((optional & self.model_fields_set) - set(cls.options))
        if unsupported:
            supported = ", ".join(sorted(cls.options)) or "none of them"
            raise ValueError(f"chunker '{self.type}' does not use {', '.join(unsupported)} (it supports: {supported})")
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


# A tool in a system's `tools:` list: a built-in's name, or `{name: ..., <tool options>}`; a mapping with `path: "pkg.module:function"` is a custom tool.
ToolRef = str | dict[str, Any]


class ToolsConfig(BaseModel):
    """Top-level `tools:` section: the rules every tool call runs under."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(default_factory=list, description="Tools with side effects (network, filesystem) that may be used; a tool with side effects that is not listed here is refused when the config loads.")
    allow_network: bool = Field(default=False, description="Permit tools that declare `side_effects: network` (they must also be listed in `allow`). No network tool is built in.")
    timeout_s: float = Field(default=10.0, gt=0, description="A tool call that takes longer returns an error result instead of hanging the question.")
    max_output_chars: int = Field(default=4000, ge=100, description="Tool output longer than this is cut and flagged `truncated`, so one call cannot flood the model's context.")


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    name: str | None = None
    chunker: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    llm_features: dict[str, Any] = Field(default_factory=dict)
    tools: list[ToolRef] = Field(default_factory=list)

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
        if self.tools and not spec.supports_tools:
            raise ValueError(f"{label} does not use tools; remove its `tools:` list (see docs/tools.md for the systems that do)")
        return self

    @model_validator(mode="after")
    def _validate_tool_refs_are_well_formed(self) -> SystemConfig:
        for index, ref in enumerate(self.tools):
            if isinstance(ref, dict) and not isinstance(ref.get("name"), str):
                raise ValueError(f"System '{self.resolved_name}': tools[{index}] must be a tool name or a mapping with a `name`")
        return self


def default_primary_k(k_values: list[int]) -> int:
    """Headline cut-off for recall: 5 when available, otherwise the median of `k_values`."""
    ks = sorted(set(k_values))
    return 5 if 5 in ks else ks[len(ks) // 2]


class JudgeConfig(BaseModel):
    """`evaluation.judge:` section: how answers are graded by an LLM (a heuristic judge scores mock runs and `judge_enabled: false`)."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, description="Model ref of the judge. Default: `evaluation.judge_model`. Prefer a different model family than the generators.")
    samples: int = Field(default=1, ge=1, le=10, description="Independent judgments per question, averaged; the spread is recorded. Needs `temperature` above 0.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Judge sampling temperature. Leave at 0 for one deterministic (and cacheable) judgment.")
    independent: bool = Field(
        default=True,
        description="Expect the judge to be a different model than the generators: the report warns when it is the same one (self-preference). Set false to accept that.",
    )


class StatsConfig(BaseModel):
    """`evaluation.stats:` section: how uncertainty and significance are computed (see docs/methodology.md#statistics)."""

    model_config = ConfigDict(extra="forbid")

    n_boot: int = Field(default=2000, ge=100, le=100_000, description="Bootstrap resamples for every confidence interval and paired comparison.")
    seed: int = Field(default=0, description="Random seed: the same results and seed give the same intervals and p-values.")
    baseline: str | None = Field(
        default=None, description="System name every other system is compared with in `significance.csv`. Default: the cheapest system ($/Q)."
    )


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
    # Model ref (see docs/configuration.md#providers--model-refs): `gpt-6-luna`, `anthropic:claude-haiku-4-5`, ...
    judge_model: str = DEFAULT_JUDGE_MODEL
    # Judge details. `judge.model`, when set, wins over `judge_model` (setting both to different models is an error); afterwards both agree.
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    max_questions: int | None = None
    # Hard spending cap in dollars: once the run has been charged this much it starts no more questions, finishes the ones in flight and
    # reports the systems that completed. Counts charged cost (standalone prices), an upper bound of real spend when the cache is warm.
    max_cost_usd: float | None = Field(default=None, gt=0)
    # A live run whose estimate is above this asks for confirmation first (`--yes` skips it; with no terminal, or when CI is set, it refuses).
    cost_confirm_threshold_usd: float = Field(default=1.0, ge=0)
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
    # What "today" is for the `date_calc` tool, an ISO date or datetime: frozen for the whole run so tool answers never depend on the clock.
    tools_now: str = "2026-01-01"
    _judge_model_in_section: bool = PrivateAttr(default=False)

    @property
    def judge_model_key(self) -> str:
        """The config key the judge's model was set under, for error messages."""
        return "evaluation.judge.model" if self._judge_model_in_section else "evaluation.judge_model"

    @model_validator(mode="after")
    def _sync_judge_model(self) -> EvaluationConfig:
        explicit = "judge_model" in self.model_fields_set
        self._judge_model_in_section = self.judge.model is not None
        if self.judge.model is None:
            self.judge.model = self.judge_model
        elif explicit and self.judge_model != self.judge.model:
            raise ValueError(f"evaluation.judge_model ({self.judge_model}) and evaluation.judge.model ({self.judge.model}) disagree; set only one of them")
        else:
            self.judge_model = self.judge.model
        if self.judge.samples > 1 and self.judge.temperature == 0:
            raise ValueError("evaluation.judge.samples > 1 needs evaluation.judge.temperature above 0, or every sample would be identical")
        return self

    @model_validator(mode="after")
    def _check_tools_now(self) -> EvaluationConfig:
        try:
            datetime.fromisoformat(self.tools_now)
        except ValueError:
            raise ValueError(f"evaluation.tools_now must be an ISO date such as 2026-01-01, not {self.tools_now!r}") from None
        return self

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
    def resolved_tools_now(self) -> datetime:
        return datetime.fromisoformat(self.tools_now)

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
    """Throttles shared by every request to one provider."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_requests: int | None = Field(default=None, ge=1, description="Requests in flight at once, across all systems and questions.")
    requests_per_minute: int | None = Field(default=None, ge=1, description="Requests started per rolling minute.")
    tokens_per_minute: int | None = Field(default=None, ge=1, description="Estimated prompt tokens per rolling minute.")

    @property
    def is_set(self) -> bool:
        return any(value is not None for value in (self.max_concurrent_requests, self.requests_per_minute, self.tokens_per_minute))


class ProviderConfig(BaseModel):
    """One `providers:` entry: a named OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, OpenRouter, Together, ...).

    Use it in a model ref as `openai_compatible:<name>/<model>`.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai_compatible"] = "openai_compatible"
    base_url: str = Field(description="Endpoint root including the version path, e.g. `http://localhost:11434/v1`.")
    api_key_env: str | None = Field(default=None, description="Environment variable holding the API key. Omit for servers that need none.")
    limits: LimitsConfig = Field(default_factory=LimitsConfig, description="Throttles for this endpoint only.")


class ModelPrice(BaseModel):
    """USD per 1M tokens; `output` stays 0 for embedding models."""

    input: float = 0.0
    output: float = 0.0


def _route_configs(system: SystemConfig) -> list[tuple[str, SystemConfig]]:
    """The pipelines inside an `adaptive` system's `routes:`, as `(route name, config)`, each inheriting the system's `models`."""
    routes = system.retrieval.get("routes")
    if system.type != "adaptive" or not isinstance(routes, dict):
        return []
    configs = [(name, raw if isinstance(raw, SystemConfig) else SystemConfig.model_validate(raw)) for name, raw in routes.items()]
    return [(name, config.model_copy(update={"models": {**system.models, **config.models}})) for name, config in configs]


def _with_routes(systems: list[SystemConfig]) -> list[tuple[str, SystemConfig]]:
    """Every system config to validate, with the label to report it under: each system, then each of its routes."""
    found: list[tuple[str, SystemConfig]] = []
    for index, system in enumerate(systems):
        found.append((f"systems[{index}]", system))
        found.extend((f"systems[{index}].routes.{name}", config) for name, config in _route_configs(system))
    return found


class SelectionConfig(BaseModel):
    """`selection:` section: how the run's recommendation (`recommendation.md`, `winner.yaml`) is chosen. See docs/methodology.md#selection."""

    model_config = ConfigDict(extra="forbid")

    profile: str = Field(default="balanced", description="`balanced`, `max_quality`, `cheapest_acceptable` or `lowest_latency`.")
    constraints: Constraints = Field(default_factory=Constraints, description="Hard requirements; a system that breaks one is never recommended.")
    weights: Weights | None = Field(default=None, description="Override the profile's weights for quality, cost and latency.")

    @model_validator(mode="after")
    def _known_profile(self) -> SelectionConfig:
        if self.profile not in PROFILES:
            from ragbench.selection.scoring import resolve_profile

            resolve_profile(self.profile)  # raises with a did-you-mean hint
        return self


class ReportConfig(BaseModel):
    """`report:` section: how `report.html` is built. See docs/methodology.md#reading-the-report."""

    model_config = ConfigDict(extra="forbid")

    max_embedded_mb: float = Field(
        default=4.0,
        gt=0,
        description="Most question data (answers, contexts, traces) embedded in the page. Above it the page keeps a reduced copy and the complete data is written to `report_questions.json` next to it.",
    )


class ExperimentConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    systems: list[SystemConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # Per-model price overrides (take precedence over the built-in table in models/cost.py).
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    # Throttles applied to each hosted API in use (OpenAI, Anthropic) separately; endpoints in `providers:` carry their own.
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    # Named OpenAI-compatible endpoints, referenced as `openai_compatible:<name>/<model>`.
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    # Rules for tool calls (permissions, timeout, output cap); each system lists the tools it may use in its own `tools:`.
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    # Which system the run recommends deploying, and under what constraints and priorities.
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    # How report.html is built (the size of the question explorer's data).
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="before")
    @classmethod
    def _expand_sweeps(cls, data: Any) -> Any:
        """A system with a `sweep:` becomes one system per combination before anything else is checked (see config/sweep.py)."""
        return expand_sweeps(data) if isinstance(data, dict) else data

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

    @model_validator(mode="after")
    def _check_tools(self) -> ExperimentConfig:
        from ragbench.tools import resolve_tools

        for label, system in _with_routes(self.systems):
            if system.tools:
                try:
                    resolve_tools(system.tools, self.tools)
                except ValueError as exc:
                    raise ValueError(f"{label}.tools: {exc}") from None
        return self

    @model_validator(mode="after")
    def _check_stats_baseline(self) -> ExperimentConfig:
        baseline = self.evaluation.stats.baseline
        names = [system.resolved_name for system in self.systems]
        if baseline is not None and baseline not in names:
            close = difflib.get_close_matches(baseline, names, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise ValueError(f"evaluation.stats.baseline: no system named '{baseline}'.{hint} Systems: {', '.join(names)}")
        return self

    @model_validator(mode="after")
    def _check_model_refs(self) -> ExperimentConfig:
        from ragbench.models.refs import validate_model_ref

        for name in self.providers:
            if not name or any(char in name for char in "/:"):
                raise ValueError(f"providers: endpoint name {name!r} must be non-empty and contain neither '/' nor ':'")
        for label, system in _with_routes(self.systems):
            if "generator" in system.models:
                try:
                    validate_model_ref(str(system.models["generator"]), "llm", self.providers)
                except ValueError as exc:
                    raise ValueError(f"{label}.models.generator: {exc}") from None
            if "embedding" in system.models:
                try:
                    validate_model_ref(str(system.models["embedding"]), "embedding", self.providers)
                except ValueError as exc:
                    raise ValueError(f"{label}.models.embedding: {exc}") from None
        try:
            validate_model_ref(self.evaluation.judge_model, "llm", self.providers)
        except ValueError as exc:
            raise ValueError(f"{self.evaluation.judge_model_key}: {exc}") from None
        return self
