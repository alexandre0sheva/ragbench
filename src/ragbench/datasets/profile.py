"""The profile `ragbench inspect-dataset` prints: what the documents and questions look like, and what that means for a benchmark.

Everything here is computed offline and cheaply (no model is called). The projected cost is separate (`projected_standard_cost`) because it
runs the systems of the `standard` preset on the mock models, which takes a few seconds on a large corpus.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ragbench.datasets.schema import Dataset, has_relevance_labels
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import count_tokens
from ragbench.utils.text import normalize_text

if TYPE_CHECKING:
    from ragbench.config.schema import DatasetConfig
    from ragbench.evaluation.estimate import Estimate

# A corpus this small is cheap to put in every prompt, so the `full_context` baseline (default budget: 100k tokens) is very hard to beat.
FULL_CONTEXT_WARN_TOKENS = 20_000
NEAR_DUPLICATE_JACCARD = 0.8
SHINGLE_WORDS = 5  # words per shingle
PREFIX_WORDS = 400  # only the start of each document is compared: cheap, and near-duplicates (versions, templates) share their start
MIN_WORDS_FOR_DUPLICATES = 8
MAX_REPORTED_PAIRS = 20
MAX_PAIR_UPDATES = 5_000_000  # a guard against corpora whose documents all share most of their text
MIN_LEAK_WORDS = 4  # a shorter question appearing in a document proves nothing
NICE_CHUNK_SIZES = (100, 150, 200, 300, 400, 500, 800, 1000)
MIN_CHUNK, MAX_CHUNK = 200, 1000

_LETTERS = re.compile(r"[^\W\d_]+")
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and of to in is that it for was with as on be at by this are from or an have not but which they his her you had were".split()),
    "es": frozenset("el la de que y en los del las un por con una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy".split()),
    "fr": frozenset("le la les de des du et en un une que qui dans pour pas sur est au avec ce il elle ne se par plus son sont mais comme".split()),
    "de": frozenset("der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach".split()),
    "it": frozenset("il lo la di che e in un una per con non sono del della le i gli si da più come ma anche è al nel suo alla ha".split()),
    "pt": frozenset("o a os as de do da em um uma que e para com não por mais se como mas dos das ao ele ela foi são tem seu sua".split()),
    "nl": frozenset("de het een en van in is dat op te zijn voor met die niet aan er ook als bij maar om uit door nog dan wordt naar".split()),
}


@dataclass
class Distribution:
    min: float
    p50: float
    p95: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Distribution:
        if not len(values):
            return cls(0, 0, 0, 0)
        array = np.asarray(values, dtype=float)
        return cls(float(array.min()), float(np.percentile(array, 50)), float(np.percentile(array, 95)), float(array.max()))


@dataclass
class DocumentStats:
    count: int
    total_tokens: int
    tokens: Distribution
    language: str
    language_confidence: float


@dataclass
class QuestionStats:
    count: int
    words: Distribution
    categories: dict[str, int]
    difficulties: dict[str, int]
    answer_types: dict[str, int]
    answerable: int
    unanswerable: int
    answerable_ratio: float
    synthetic: int = 0  # written by `generate-questions` (`metadata.synthetic`)
    needs_review: int = 0  # still flagged `metadata.needs_review`: nobody has checked them
    mock_generated: int = 0  # template questions from `generate-questions --mock`: pipeline checks, not real questions


@dataclass
class QrelStats:
    labeled_questions: int  # questions with relevant documents (ids or positive qrels): retrieval can be scored on these
    labeled_share: float
    unlabeled_answerable: int  # answerable questions with no labels: judged on their answers only
    qrel_rows: int
    documents_referenced: int  # corpus documents that some question names as relevant
    corpus_coverage: float
    label_free: bool


@dataclass
class NearDuplicate:
    a: str
    b: str
    jaccard: float


@dataclass
class DatasetProfile:
    documents: DocumentStats
    questions: QuestionStats
    qrels: QrelStats
    near_duplicates: list[NearDuplicate]
    near_duplicate_total: int
    chunk_sizes: list[int]  # suggested `chunker.chunk_size` values to compare, in tokens
    chunk_size_note: str
    leaked_questions: list[str]
    warnings: list[str] = field(default_factory=list)
    projected_cost: Estimate | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["projected_cost"] = self.projected_cost.to_dict() if self.projected_cost is not None else None
        return data


def guess_language(texts: Iterable[str], sample_words: int = 20_000) -> tuple[str, float]:
    """The most likely language of `texts` by the share of common function words, and how clear the call is (0-1); `unknown` when no list fits."""
    words: list[str] = []
    for text in texts:
        words.extend(_LETTERS.findall(text.lower()))
        if len(words) >= sample_words:
            break
    words = words[:sample_words]
    if not words:
        return "unknown", 0.0
    scores = {language: sum(word in stop for word in words) / len(words) for language, stop in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    (best, top), (_, runner_up) = ranked[0], ranked[1]
    if top < 0.08 or top < runner_up * 1.3:
        return "unknown", 0.0
    return best, round(top / sum(scores.values()), 2)


def token_weighted_median(counts: Sequence[int]) -> float:
    """The document length below which half of the corpus's *tokens* lie: what chunking mostly acts on (a pile of short FAQ pages must not hide the handbooks)."""
    ordered = sorted(counts)
    half = sum(ordered) / 2
    running = 0
    for count in ordered:
        running += count
        if running >= half:
            return float(count)
    return 0.0


def suggest_chunk_sizes(typical: float, p95: float) -> tuple[list[int], str]:
    """Chunk sizes (tokens) worth comparing for documents of this length: a chunk bigger than a typical document is just the document."""
    if typical <= 0:
        return [], ""
    if p95 < MIN_CHUNK:
        size = next((size for size in NICE_CHUNK_SIZES if size >= p95), NICE_CHUNK_SIZES[0])
        return [size], f"Documents are short (95% under {p95:.0f} tokens): a chunk is about one document, so chunk size will hardly matter."
    ceiling = min(MAX_CHUNK, p95, typical if typical < 500 else MAX_CHUNK)
    candidates = [size for size in NICE_CHUNK_SIZES if MIN_CHUNK <= size <= max(ceiling, MIN_CHUNK)]
    picked = sorted({candidates[0], candidates[len(candidates) // 2], candidates[-1]})
    return picked, f"Half of the corpus's tokens sit in documents of {typical:,.0f} tokens or more (95% of documents are under {p95:,.0f})."


def _shingles(text: str) -> set[int]:
    words = text.lower().split()[:PREFIX_WORDS]
    if len(words) < MIN_WORDS_FOR_DUPLICATES:
        return set()
    return {hash(tuple(words[i : i + SHINGLE_WORDS])) for i in range(len(words) - SHINGLE_WORDS + 1)}


def find_near_duplicates(documents: Sequence[Document]) -> tuple[list[NearDuplicate], bool]:
    """Pairs of documents whose opening words overlap by at least `NEAR_DUPLICATE_JACCARD` (Jaccard on 5-word shingles), most similar first.

    Only documents sharing some uncommon shingle are compared, so corpora of thousands of documents stay fast. The flag is True when the
    search hit `MAX_PAIR_UPDATES` and may have missed pairs.
    """
    sets = {document.doc_id: _shingles(document.text) for document in documents}
    ids = [doc_id for doc_id, shingles in sets.items() if shingles]
    index: dict[int, list[int]] = defaultdict(list)
    for position, doc_id in enumerate(ids):
        for shingle in sets[doc_id]:
            index[shingle].append(position)
    common = max(10, len(ids) // 5)  # shingles in more documents than this are boilerplate: they do not make documents candidates
    shared: Counter[tuple[int, int]] = Counter()
    updates = 0
    truncated = False
    for positions in index.values():
        if len(positions) < 2 or len(positions) > common:
            continue
        for i, first in enumerate(positions):
            for second in positions[i + 1 :]:
                shared[(first, second)] += 1
        updates += len(positions) * (len(positions) - 1) // 2
        if updates > MAX_PAIR_UPDATES:
            truncated = True
            break
    pairs: list[NearDuplicate] = []
    for (first, second), count in shared.items():
        a, b = sets[ids[first]], sets[ids[second]]
        if count < max(3, 0.2 * min(len(a), len(b))):
            continue
        jaccard = len(a & b) / len(a | b)
        if jaccard >= NEAR_DUPLICATE_JACCARD:
            pairs.append(NearDuplicate(ids[first], ids[second], round(jaccard, 3)))
    pairs.sort(key=lambda pair: (-pair.jaccard, pair.a, pair.b))
    return pairs, truncated


def _squash(text: str) -> str:
    return normalize_text(text).lower().strip(" ?!.:;,\"'")


def find_leaked_questions(documents: Sequence[Document], dataset: Dataset) -> list[str]:
    """Ids of questions whose text appears word for word in a document (ignoring case, spacing and end punctuation): retrieval then only has to match strings."""
    corpus = "\x00".join(_squash(document.text) for document in documents)
    leaked = []
    for question in dataset.questions:
        squashed = _squash(question.question)
        if len(squashed.split()) >= MIN_LEAK_WORDS and squashed in corpus:
            leaked.append(question.id)
    return leaked


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def profile_dataset(documents: Sequence[Document], dataset: Dataset) -> DatasetProfile:
    """Describe a dataset and list what is likely to distort a benchmark run on it. Warnings are advice: nothing here stops a run."""
    token_counts = [count_tokens(document.text) for document in documents]
    token_stats = Distribution.of(token_counts)
    language, confidence = guess_language(document.text for document in documents)
    doc_ids = {document.doc_id for document in documents}
    questions = dataset.questions
    answerable = sum(question.is_answerable for question in questions)
    labeled = [question for question in questions if has_relevance_labels(question, dataset.qrels.get(question.id, {}))]
    labeled_ids = {question.id for question in labeled}
    referenced = {
        doc_id
        for question in labeled
        for doc_id in [*question.relevant_doc_ids, *(d for d, grade in dataset.qrels.get(question.id, {}).items() if grade > 0)]
        if doc_id in doc_ids
    }
    near_duplicates, truncated = find_near_duplicates(documents)
    leaked = find_leaked_questions(documents, dataset)
    sizes, size_note = suggest_chunk_sizes(token_weighted_median(token_counts), token_stats.p95)
    total_tokens = sum(token_counts)

    warnings: list[str] = []
    if near_duplicates:
        shown = "; ".join(f"{pair.a} ~ {pair.b} ({pair.jaccard:.0%})" for pair in near_duplicates[:3])
        more = f" and {len(near_duplicates) - 3} more" if len(near_duplicates) > 3 else ""
        warnings.append(
            f"{len(near_duplicates)} near-duplicate document pair(s) ({shown}{more}): a system that returns the other copy scores a miss unless both are labeled relevant."
            + (" The search was cut short and may have missed more." if truncated else "")
        )
    if leaked:
        shown_ids = ", ".join(leaked[:5]) + (f" and {len(leaked) - 5} more" if len(leaked) > 5 else "")
        warnings.append(
            f"{len(leaked)} question(s) appear verbatim in a document ({shown_ids}): keyword search finds them by string match, which flatters lexical systems "
            "and says little about real questions. Reword them, or expect lexical systems to look better than they are."
        )
    if documents and total_tokens <= FULL_CONTEXT_WARN_TOKENS:
        warnings.append(
            f"The whole corpus is only ~{total_tokens:,} tokens: it fits in one prompt, so the `full_context` baseline will likely beat every retrieval system. "
            "Retrieval choices start to matter on larger corpora."
        )
    synthetic = sum(bool(question.metadata.get("synthetic")) for question in questions)
    needs_review = sum(bool(question.metadata.get("needs_review")) for question in questions)
    mocked = sum(bool(question.metadata.get("mock")) for question in questions)
    if needs_review:
        warnings.append(
            f"{needs_review} of {len(questions)} question(s) are flagged `needs_review` (synthetic questions nobody has checked): read them, fix or drop the bad ones, "
            "and remove the flag (`metadata.needs_review`) when done. Scores on unreviewed synthetic questions measure the generator's taste as much as the systems."
        )
    if mocked:
        warnings.append(f"{mocked} question(s) are mock templates (`generate-questions --mock`): they validate the pipeline and say nothing about real quality.")
    return DatasetProfile(
        documents=DocumentStats(len(documents), total_tokens, token_stats, language, confidence),
        questions=QuestionStats(
            count=len(questions),
            words=Distribution.of([len(question.question.split()) for question in questions]),
            categories=_count(question.category for question in questions),
            difficulties=_count(question.difficulty for question in questions),
            answer_types=_count(question.answer_type for question in questions),
            answerable=answerable,
            unanswerable=len(questions) - answerable,
            answerable_ratio=answerable / len(questions) if questions else 0.0,
            synthetic=synthetic,
            needs_review=needs_review,
            mock_generated=mocked,
        ),
        qrels=QrelStats(
            labeled_questions=len(labeled),
            labeled_share=len(labeled) / len(questions) if questions else 0.0,
            unlabeled_answerable=sum(question.is_answerable for question in questions if question.id not in labeled_ids),
            qrel_rows=sum(len(rows) for rows in dataset.qrels.values()),
            documents_referenced=len(referenced),
            corpus_coverage=len(referenced) / len(doc_ids) if doc_ids else 0.0,
            label_free=dataset.label_free,
        ),
        near_duplicates=near_duplicates[:MAX_REPORTED_PAIRS],
        near_duplicate_total=len(near_duplicates),
        chunk_sizes=sizes,
        chunk_size_note=size_note,
        leaked_questions=leaked,
        warnings=warnings,
    )


def projected_standard_cost(dataset: DatasetConfig) -> Estimate:
    """What the `standard` preset would cost on this dataset with the default models (`ragbench estimate`'s measurement: mock runs, priced)."""
    from ragbench.config.presets import apply_preset
    from ragbench.config.schema import ExperimentConfig
    from ragbench.evaluation.estimate import estimate_run

    config = apply_preset({"dataset": dataset.model_dump(mode="json", exclude_none=True)}, "standard")
    return estimate_run(ExperimentConfig.model_validate(config))
