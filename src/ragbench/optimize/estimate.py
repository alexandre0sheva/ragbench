"""What `ragbench optimize` would cost, before it runs: the system's per-question and indexing cost (measured on the mock models by
`estimate_run`) times the loop's runs, plus the proposer's requests.

Every probe is an ordinary run, so each one indexes the corpus again (the disk cache makes repeats of unchanged calls free in a live run;
like `ragbench estimate`, this assumes no cache hits). The proposer is priced at a full evidence budget per request and one retry for none
of them; the dev selection with arena judging adds pairwise verdicts not counted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ragbench.config.schema import ExperimentConfig
from ragbench.datasets.loader import configured_splits_path
from ragbench.datasets.splits import load_splits
from ragbench.evaluation.estimate import Estimate, estimate_run
from ragbench.models import cost as pricing
from ragbench.optimize.loop import OptimizeSettings
from ragbench.optimize.proposer import MAX_EVIDENCE_CHARS

CHARS_PER_TOKEN = 4
PROPOSER_FRAME_TOKENS = 1_500  # the system message and the request block
PROPOSER_COMPLETION_TOKENS = 800


@dataclass
class OptimizeEstimate:
    system: str
    k: int
    mode: str
    questions: dict[str, int]  # train / dev / test
    per_question_usd: float  # answering and judging one question
    ingestion_usd: float  # indexing the corpus once
    train_usd: float  # the 4k probes
    dev_usd: float  # base and finalists on dev
    test_usd: float  # base and pick on test
    proposer_usd: float  # 3k proposer requests
    proposer_model: str
    measured: Estimate

    @property
    def total_usd(self) -> float:
        return self.train_usd + self.dev_usd + self.test_usd + self.proposer_usd

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("measured")
        return {**data, "total_usd": self.total_usd}


def estimate_optimize(config: ExperimentConfig, settings: OptimizeSettings, proposer_model: str, *, max_evidence_chars: int = MAX_EVIDENCE_CHARS) -> OptimizeEstimate:
    splits = load_splits(configured_splits_path(config.dataset))
    target = [system for system in config.systems if system.resolved_name == settings.system]
    if not target:
        raise ValueError(f"The config has no system named '{settings.system}'.")
    narrowed = config.model_copy(update={"systems": target, "dataset": config.dataset.model_copy(update={"split": "train"})})
    measured = estimate_run(narrowed)
    system = measured.systems[0]
    if system.error is not None:
        raise ValueError(f"`{settings.system}` could not be measured: {system.error}")
    per_question, ingestion = system.answer_usd_per_question + system.judge_usd_per_question, system.ingestion_usd
    n_train, n_dev, n_test = len(splits.train), len(splits.dev), len(splits.test)
    k = settings.k
    runs = 3 * k + 1 if settings.mode == "parallel" else 4 * k  # the parallel base is one run of k trials
    train = runs * ingestion + 4 * k * n_train * per_question
    dev = (1 + settings.finalists) * (ingestion + n_dev * per_question) if n_dev else 0.0
    test = 2 * (ingestion + n_test * settings.test_trials * per_question)
    prompt_tokens = max_evidence_chars // CHARS_PER_TOKEN + PROPOSER_FRAME_TOKENS
    pricing.register_pricing({model: price.model_dump() for model, price in config.pricing.items()})  # `estimate_run` cleared them
    try:
        proposer = pricing.estimate_model_cost(proposer_model, 3 * k * prompt_tokens, 3 * k * PROPOSER_COMPLETION_TOKENS, calls=3 * k)
    finally:
        pricing.clear_pricing_overrides()
    return OptimizeEstimate(
        system=settings.system,
        k=k,
        mode=settings.mode,
        questions={"train": n_train, "dev": n_dev, "test": n_test},
        per_question_usd=per_question,
        ingestion_usd=ingestion,
        train_usd=train,
        dev_usd=dev,
        test_usd=test,
        proposer_usd=proposer,
        proposer_model=proposer_model,
        measured=measured,
    )
