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
from ragbench.config.schema import EvaluationConfig, ExperimentConfig, SystemConfig
from ragbench.datasets.loader import load_configured_dataset
from ragbench.documents.loaders import load_dataset_documents
from ragbench.evaluation.claims import SPLIT_SYSTEM_PROMPT
from ragbench.evaluation.judge_panel import scoring_judges
from ragbench.evaluation.judge_prompts import build_messages
from ragbench.evaluation.pairwise import pairwise_messages
from ragbench.models import cost as pricing
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.models.refs import system_model_refs
from ragbench.models.usage import UsageRecorder, record_usage
from ragbench.rag_systems import create_rag_system
from ragbench.registry import SYSTEMS
from ragbench.runtime import RuntimeContext, activate_runtime
from ragbench.runtime.executor import AUTO_PROBE_MAX
from ragbench.runtime.parallel import evenly_sample
from ragbench.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = 6  # questions each system is actually run on; per-question figures are their mean
JUDGE_COMPLETION_TOKENS = 120  # a judge verdict (five scores, two flags, a sentence of reasoning)
CLAIMS_PER_ANSWER = 4  # what `evaluation.claims` is assumed to split an answer into (capped by `max_claims`)
CLAIM_COMPLETION_TOKENS = 25  # one claim in the splitter's reply
DECISION_COMPLETION_TOKENS = 16  # a probability reply
DECISION_PROMPT_TOKENS = 80  # a claim check's instructions and framing around the context
PAIRWISE_COMPLETION_TOKENS = 40  # a pairwise verdict (the winner and one sentence)
STALE_PRICES_AFTER_DAYS = 90
# Rough API latencies for the time projection: a model call, its generation speed, and an embedding request.
LLM_CALL_BASE_S = 0.7
LLM_TOKENS_PER_S = 70.0
EMBED_CALL_S = 0.25
DECISION_CALL_S = 1.5  # a typed-decision request: about 2 s measured against the Decisions API on 2026-10-07 for a short input


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
    trials: int = 1  # `evaluation.trials`: every question is answered and judged this many times
    probe_calls: int = 0  # latency-probe answers a live run adds (`evaluation.latency_probe_questions`; `auto` counts its cap)
    answer_tokens: float = 0.0  # mean length of the sampled answers, for pricing pairwise verdicts

    @property
    def query_usd(self) -> float:
        return self.answer_usd_per_question * self.n_questions * self.trials

    @property
    def judge_usd(self) -> float:
        return self.judge_usd_per_question * self.n_questions * self.trials

    @property
    def probe_usd(self) -> float:
        return self.answer_usd_per_question * self.probe_calls

    @property
    def total_usd(self) -> float:
        return self.ingestion_usd + self.query_usd + self.judge_usd + self.probe_usd


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
    trials: int = 1
    pairwise_calls: int = 0  # head-to-head judge requests (`evaluation.pairwise`), both orders counted
    pairwise_usd: float = 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "systems": [{**asdict(s), "probe_usd": s.probe_usd, "total_usd": s.total_usd} for s in self.systems]}


def confirmation_decision(total_usd: float, threshold_usd: float, *, yes: bool, interactive: bool, ci: bool) -> Literal["proceed", "ask", "refuse"]:
    """What a live run does about its estimated cost: carry on, ask a person, or refuse (nobody to ask, so it must be told with `--yes`)."""
    if total_usd <= threshold_usd or yes:
        return "proceed"
    return "ask" if interactive and not ci else "refuse"


def _usd(usage: UsageRecorder, generator: str, embedding: str | None, decision: str | None = None) -> float:
    total = 0.0
    if usage.decision_tokens:  # typed decisions are billed on input tokens only, at the decision model's rate (not the generator's)
        total += pricing.estimate_decision_cost(decision or generator, usage.decision_tokens)
    if usage.llm_prompt_tokens or usage.llm_completion_tokens:
        # Summed over the question's calls: a tiered price is chosen by the mean prompt of one call.
        total += pricing.estimate_model_cost(generator, usage.llm_prompt_tokens, usage.llm_completion_tokens, calls=usage.llm_calls)
    if usage.embed_tokens and embedding is not None:
        total += pricing.estimate_model_cost(embedding, usage.embed_tokens, 0)
    return total


def _answer_checks(evaluation: EvaluationConfig, question: str, answer: str, context: str) -> tuple[float, float]:
    """Dollars and seconds of the optional checks on one answer: claim splitting and checking (`evaluation.claims`), model-based refusal detection."""
    usd = seconds = 0.0
    call = LLM_CALL_BASE_S + DECISION_COMPLETION_TOKENS / LLM_TOKENS_PER_S
    if evaluation.claims.enabled:
        model, claims = evaluation.claims_model, min(evaluation.claims.max_claims, CLAIMS_PER_ANSWER)
        split_tokens = estimate_tokens(SPLIT_SYSTEM_PROMPT + question + answer, model)
        usd += pricing.estimate_model_cost(model, split_tokens, claims * CLAIM_COMPLETION_TOKENS)
        usd += claims * pricing.estimate_model_cost(
            model, estimate_tokens(context, model) + DECISION_PROMPT_TOKENS, DECISION_COMPLETION_TOKENS
        )
        seconds += LLM_CALL_BASE_S + claims * CLAIM_COMPLETION_TOKENS / LLM_TOKENS_PER_S + claims * call
    if evaluation.refusal_detector == "model":
        model = evaluation.judge_model
        usd += pricing.estimate_model_cost(
            model, estimate_tokens(question + answer, model) + DECISION_PROMPT_TOKENS, DECISION_COMPLETION_TOKENS
        )
        seconds += call
    return usd, seconds


def _seconds(usage: UsageRecorder) -> float:
    return usage.llm_calls * LLM_CALL_BASE_S + usage.llm_completion_tokens / LLM_TOKENS_PER_S + usage.embed_calls * EMBED_CALL_S + usage.decision_calls * DECISION_CALL_S


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
    empty = SystemEstimate(name, system_config.type, agentic, 0.0, 0.0, 0.0, 0.0, 0.0, n_questions, trials=config.evaluation.trials)
    evaluation = config.evaluation
    try:
        refs = system_model_refs(system_config)
        generator = refs.llm[0]
        embedding = refs.embedding[0] if refs.embedding else None
        system = create_rag_system(_for_measuring(system_config), force_mock=True)
        decision = getattr(system, "decision_model_ref", None)
        judges = scoring_judges(evaluation.judge_models, generator, independent=evaluation.judge.independent)
        with record_usage() as ingestion:
            system.ingest(documents)
        answer_usd = judge_usd = answer_seconds = judge_seconds = answer_tokens = 0.0
        for question in sample:
            context_k = system.configured_context_k() or evaluation.context_k
            with record_usage() as usage:
                answer = system.answer_question(question.question, top_k=max(evaluation.resolved_retrieval_depth, context_k), context_k=context_k)
            answer_usd += _usd(usage, generator, embedding, decision)
            answer_tokens += estimate_tokens(answer.answer)
            answer_seconds += _seconds(usage)
            given = answer.metadata.get("context_chunk_ids")
            chunks = [c for c in answer.retrieval_result.chunks if given is None or c.chunk_id in given]
            check_usd, check_seconds = _answer_checks(evaluation, question.question, answer.answer, "\n".join(c.text for c in chunks))
            judge_usd += check_usd  # checks on the answer are charged as judge spend
            judge_seconds += check_seconds
            if evaluation.judge_enabled:
                prompt = "\n".join(message["content"] for message in build_messages(question, answer.answer, chunks))
                calls = evaluation.judge.samples
                for judge in judges:
                    judge_usd += calls * pricing.estimate_model_cost(judge, estimate_tokens(prompt, judge), JUDGE_COMPLETION_TOKENS)
                judge_seconds += len(judges) * calls * (LLM_CALL_BASE_S + JUDGE_COMPLETION_TOKENS / LLM_TOKENS_PER_S)  # panel judges run one after another
        count = max(1, len(sample))
        return SystemEstimate(
            name,
            system_config.type,
            agentic,
            ingestion_usd=_usd(ingestion, generator, embedding, decision),
            answer_usd_per_question=answer_usd / count,
            judge_usd_per_question=judge_usd / count,
            ingestion_seconds=_seconds(ingestion) / max(1, evaluation.ingest_workers),
            question_seconds=(answer_seconds + judge_seconds) / count,
            n_questions=n_questions,
            trials=evaluation.trials,
            probe_calls=_probe_calls(evaluation.latency_probe_questions, n_questions),
            answer_tokens=answer_tokens / count,
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
        dataset, _ = load_configured_dataset(config.dataset)
        limit = config.evaluation.max_questions
        questions = dataset.questions[:limit] if limit else dataset.questions
        sample = evenly_sample(questions, sample_questions)
        runtime = RuntimeContext(
            ingest_workers=config.evaluation.ingest_workers,
            providers=config.providers,
            openrouter=config.openrouter,
            tools=config.tools,
            tools_now=config.evaluation.resolved_tools_now,
        )
        systems: list[SystemEstimate] = []
        with activate_cache(None), activate_runtime(runtime):
            for system_config in config.systems:
                if progress is not None:
                    progress(system_config.resolved_name)
                systems.append(_estimate_system(system_config, config, documents, sample, len(questions)))
        pairwise_calls, pairwise_usd = _pairwise(config, [s for s in systems if s.error is None], sample, len(questions))
        unpriced = sorted(pricing.unknown_priced_models())
    finally:
        EMBEDDING_CACHE.clear()
        EMBEDDING_CACHE.enabled = was_enabled
        pricing.clear_pricing_overrides()
    measured = [s for s in systems if s.error is None]
    workers = max(1, config.evaluation.max_workers)
    work = [s.ingestion_seconds + s.question_seconds * s.n_questions * s.trials / workers for s in measured]
    wall = max(sum(work) / max(1, config.evaluation.system_workers), max(work, default=0.0))
    wall += sum(s.question_seconds * s.probe_calls for s in measured)  # the probe runs one call at a time, after everything else
    wall += pairwise_calls * (LLM_CALL_BASE_S + PAIRWISE_COMPLETION_TOKENS / LLM_TOKENS_PER_S) / workers
    return Estimate(
        systems=systems,
        n_questions=len(questions),
        n_documents=len(documents),
        corpus_tokens=sum(estimate_tokens(document.text) for document in documents),
        total_usd=sum(s.total_usd for s in measured) + pairwise_usd,
        wall_seconds=wall,
        pricing_as_of=pricing.PRICING_AS_OF,
        warnings=_warnings(systems, unpriced, today),
        unpriced_models=unpriced,
        trials=config.evaluation.trials,
        pairwise_calls=pairwise_calls,
        pairwise_usd=pairwise_usd,
    )


def _pairs_per_question(mode: str, systems: int, max_pairs: int | None) -> int:
    """How many pairs each question's answers meet in (see `pairwise_run.schedule`)."""
    every = systems * (systems - 1) // 2
    if mode == "adaptive":
        return min(max_pairs or systems, every)
    pairs = systems - 1 if mode == "vs_baseline" else every
    return min(pairs, max_pairs) if max_pairs is not None else pairs


def _pairwise(config: ExperimentConfig, measured: list[SystemEstimate], sample: list, n_questions: int) -> tuple[int, float]:
    """Calls and dollars of the pairwise stage: two verdicts per pair, priced at the lead judge's rate (a pair that no judge of the panel
    is neutral on is judged by all of them, which this leaves out)."""
    settings = config.evaluation.pairwise
    if not settings.enabled or not config.evaluation.judge_enabled or len(measured) < 2:
        return 0, 0.0
    calls = 2 * n_questions * _pairs_per_question(settings.pairs, len(measured), settings.max_pairs_per_question)
    judge = config.evaluation.judge_models[0]
    answer = sum(s.answer_tokens for s in measured) / len(measured)
    asked = sum(estimate_tokens(" ".join(m["content"] for m in pairwise_messages(q, "", ""))) for q in sample) / max(1, len(sample))
    return calls, calls * pricing.estimate_model_cost(judge, int(asked + 2 * answer), PAIRWISE_COMPLETION_TOKENS)


def _probe_calls(size: int | Literal["auto"], n_questions: int) -> int:
    """How many probe answers a live run adds per system: the configured number (at most one per question), or `auto`'s cap."""
    return AUTO_PROBE_MAX if size == "auto" else min(size, n_questions)


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
