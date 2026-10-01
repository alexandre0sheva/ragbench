"""`ragbench estimate`: what a run will cost and roughly how long it will take, before spending anything.

How: every system is built with the offline mock models and run on the real corpus (to index it) and on a sample of the real
questions. The mock models report how many calls were made and how many tokens went in and out (`ragbench.models.usage`), so
prompt sizes, retrieved context, call counts and embedding volume are *measured*, not guessed. Those tokens are then priced with
the configured models' rates (the built-in table plus `pricing:`), and judging is added from the real judge prompt.

What it cannot know: how long a real model's answers are (the mock's are short, so output cost is likely low), how many steps a
real agent takes (the mock's tool policy takes few), and cache hits (assumed none: an upper bound for a re-run). Against a mock run
priced as if live the estimate lands within a few percent; see TOLERANCE in tests/test_estimate.py for the bound we hold it to.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

from ragbench.cache import activate_cache
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.loader import load_dataset
from ragbench.documents.loaders import load_dataset_documents
from ragbench.evaluation.judge_prompts import build_messages
from ragbench.models import cost as pricing
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.models.refs import system_model_refs
from ragbench.models.usage import UsageRecorder, record_usage
from ragbench.rag_systems import create_rag_system
from ragbench.registry import SYSTEMS
from ragbench.runtime import RuntimeContext, activate_runtime
from ragbench.runtime.parallel import evenly_sample
from ragbench.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = 6  # questions each system is actually run on; per-question figures are their mean
JUDGE_COMPLETION_TOKENS = 120  # a judge verdict (five scores, two flags, a sentence of reasoning)
STALE_PRICES_AFTER_DAYS = 90
# Rough API latencies for the time projection: a model call, its generation speed, and an embedding request.
LLM_CALL_BASE_S = 0.7
LLM_TOKENS_PER_S = 70.0
EMBED_CALL_S = 0.25


@dataclass
class SystemEstimate:
    system: str
    system_type: str
    agentic: bool
    ingestion_usd: float  # one-off cost of indexing the whole corpus
    answer_usd_per_question: float  # embeddings, retrieval-side LLM calls, generation, tools
    judge_usd_per_question: float
    ingestion_seconds: float
    question_seconds: float  # per question, including judging
    n_questions: int
    error: str | None = None  # set when the system could not be measured; its numbers are then zero

    @property
    def query_usd(self) -> float:
        return self.answer_usd_per_question * self.n_questions

    @property
    def judge_usd(self) -> float:
        return self.judge_usd_per_question * self.n_questions

    @property
    def total_usd(self) -> float:
        return self.ingestion_usd + self.query_usd + self.judge_usd


@dataclass
class Estimate:
    systems: list[SystemEstimate]
    n_questions: int
    n_documents: int
    corpus_tokens: int
    total_usd: float
    wall_seconds: float
    pricing_as_of: str
    warnings: list[str] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "systems": [{**asdict(s), "total_usd": s.total_usd} for s in self.systems]}


def confirmation_decision(total_usd: float, threshold_usd: float, *, yes: bool, interactive: bool, ci: bool) -> Literal["proceed", "ask", "refuse"]:
    """What a live run does about its estimated cost: carry on, ask a person, or refuse (nobody to ask, so it must be told with `--yes`)."""
    if total_usd <= threshold_usd or yes:
        return "proceed"
    return "ask" if interactive and not ci else "refuse"


def _usd(usage: UsageRecorder, generator: str, embedding: str | None) -> float:
    total = 0.0
    if usage.llm_prompt_tokens or usage.llm_completion_tokens:
        total += pricing.estimate_model_cost(generator, usage.llm_prompt_tokens, usage.llm_completion_tokens)
    if usage.embed_tokens and embedding is not None:
        total += pricing.estimate_model_cost(embedding, usage.embed_tokens, 0)
    return total


def _seconds(usage: UsageRecorder) -> float:
    return usage.llm_calls * LLM_CALL_BASE_S + usage.llm_completion_tokens / LLM_TOKENS_PER_S + usage.embed_calls * EMBED_CALL_S


def _for_measuring(system_config: SystemConfig) -> SystemConfig:
    """The system as it is measured: the exact NumPy index, because the vector backend changes neither the tokens nor the calls (and may not be installed)."""
    spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
    if spec is not None and "vector_store" in spec.options.model_fields and system_config.retrieval.get("vector_store", "numpy") != "numpy":
        return system_config.model_copy(update={"retrieval": {**system_config.retrieval, "vector_store": "numpy"}})
    return system_config


def _estimate_system(system_config: SystemConfig, config: ExperimentConfig, documents: list, sample: list, n_questions: int) -> SystemEstimate:
    name = system_config.resolved_name
    spec = getattr(SYSTEMS.mapping.get(system_config.type), "spec", None)
    agentic = bool(spec is not None and spec.agentic)
    empty = SystemEstimate(name, system_config.type, agentic, 0.0, 0.0, 0.0, 0.0, 0.0, n_questions)
    evaluation = config.evaluation
    try:
        refs = system_model_refs(system_config)
        generator = refs.llm[0]
        embedding = refs.embedding[0] if refs.embedding else None
        system = create_rag_system(_for_measuring(system_config), force_mock=True)
        with record_usage() as ingestion:
            system.ingest(documents)
        answer_usd = judge_usd = answer_seconds = judge_seconds = 0.0
        for question in sample:
            context_k = system.configured_context_k() or evaluation.context_k
            with record_usage() as usage:
                answer = system.answer_question(question.question, top_k=max(evaluation.resolved_retrieval_depth, context_k), context_k=context_k)
            answer_usd += _usd(usage, generator, embedding)
            answer_seconds += _seconds(usage)
            if evaluation.judge_enabled:
                given = answer.metadata.get("context_chunk_ids")
                chunks = [c for c in answer.retrieval_result.chunks if given is None or c.chunk_id in given]
                prompt = "\n".join(message["content"] for message in build_messages(question, answer.answer, chunks))
                prompt_tokens = estimate_tokens(prompt, evaluation.judge_model)
                calls = evaluation.judge.samples
                judge_usd += calls * pricing.estimate_model_cost(evaluation.judge_model, prompt_tokens, JUDGE_COMPLETION_TOKENS)
                judge_seconds += calls * (LLM_CALL_BASE_S + JUDGE_COMPLETION_TOKENS / LLM_TOKENS_PER_S)
        count = max(1, len(sample))
        return SystemEstimate(
            name,
            system_config.type,
            agentic,
            ingestion_usd=_usd(ingestion, generator, embedding),
            answer_usd_per_question=answer_usd / count,
            judge_usd_per_question=judge_usd / count,
            ingestion_seconds=_seconds(ingestion) / max(1, evaluation.ingest_workers),
            question_seconds=(answer_seconds + judge_seconds) / count,
            n_questions=n_questions,
        )
    except Exception as exc:  # noqa: BLE001 (one system that cannot be measured must not hide the others)
        logger.warning("Could not estimate %s: %s: %s", name, type(exc).__name__, exc)
        empty.error = f"{type(exc).__name__}: {exc}"[:300]
        return empty


def estimate_run(
    config: ExperimentConfig,
    *,
    sample_questions: int = SAMPLE_QUESTIONS,
    today: date | None = None,
    progress: Callable[[str], None] | None = None,
) -> Estimate:
    """Project a run's cost and time without calling any paid model. Safe to call before a live run: it is forced to the mock models."""
    today = today or date.today()
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()
    pricing.register_pricing({model: price.model_dump() for model, price in config.pricing.items()})
    was_enabled = EMBEDDING_CACHE.enabled
    EMBEDDING_CACHE.clear()
    EMBEDDING_CACHE.enabled = False  # the second system must embed the corpus too, or its tokens would not be measured
    try:
        documents = load_dataset_documents(config.dataset, warnings=[])
        dataset = load_dataset(config.dataset.questions_path, config.dataset.qrels_path)
        limit = config.evaluation.max_questions
        questions = dataset.questions[:limit] if limit else dataset.questions
        sample = evenly_sample(questions, sample_questions)
        runtime = RuntimeContext(
            ingest_workers=config.evaluation.ingest_workers, providers=config.providers, tools=config.tools, tools_now=config.evaluation.resolved_tools_now
        )
        systems: list[SystemEstimate] = []
        with activate_cache(None), activate_runtime(runtime):
            for system_config in config.systems:
                if progress is not None:
                    progress(system_config.resolved_name)
                systems.append(_estimate_system(system_config, config, documents, sample, len(questions)))
        unpriced = sorted(pricing.unknown_priced_models())
    finally:
        EMBEDDING_CACHE.clear()
        EMBEDDING_CACHE.enabled = was_enabled
        pricing.clear_pricing_overrides()
    measured = [s for s in systems if s.error is None]
    workers = max(1, config.evaluation.max_workers)
    work = [s.ingestion_seconds + s.question_seconds * s.n_questions / workers for s in measured]
    wall = max(sum(work) / max(1, config.evaluation.system_workers), max(work, default=0.0))
    return Estimate(
        systems=systems,
        n_questions=len(questions),
        n_documents=len(documents),
        corpus_tokens=sum(estimate_tokens(document.text) for document in documents),
        total_usd=sum(s.total_usd for s in measured),
        wall_seconds=wall,
        pricing_as_of=pricing.PRICING_AS_OF,
        warnings=_warnings(systems, unpriced, today),
        unpriced_models=unpriced,
    )


def _warnings(systems: list[SystemEstimate], unpriced: list[str], today: date) -> list[str]:
    warnings: list[str] = []
    age = (today - date.fromisoformat(pricing.PRICING_AS_OF)).days
    if age > STALE_PRICES_AFTER_DAYS:
        warnings.append(
            f"The price table is dated {pricing.PRICING_AS_OF}, which is older than {STALE_PRICES_AFTER_DAYS} days ({age} days ago): "
            "providers change prices, so check current rates and override them under `pricing:`."
        )
    if unpriced:
        warnings.append(f"Cost is under-estimated: {', '.join(unpriced)} {'has' if len(unpriced) == 1 else 'have'} no price registered (counted as $0). Add a price under `pricing:`.")
    agents = [s.system for s in systems if s.agentic and s.error is None]
    if agents:
        warnings.append(
            f"The agentic systems ({', '.join(agents)}) loop until they have an answer. The estimate uses the steps a scripted model takes, so real runs usually cost more; "
            "their `max_cost_usd`, `max_steps` and `evaluation.max_cost_usd` cap the real spend."
        )
    for system in systems:
        if system.error is not None:
            warnings.append(f"{system.system} could not be estimated and is left out of the total: {system.error}")
    return warnings
