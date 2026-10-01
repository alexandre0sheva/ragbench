"""`ragbench generate-questions`: write a question set for a corpus that has none.

A *writer* proposes candidates (an LLM, or the deterministic templates in mock mode); this module decides which become questions:
it samples documents stratified by length, builds multi-hop pairs from BM25 neighbours, rejects near-duplicates and unanswerable
questions whose key entity does occur in the corpus, keeps the requested category mix, and flags every row for human review.
The same seed gives the same questions (for an LLM: the same model and, with the disk cache, the same replies).
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ragbench.datasets import prompts, templates
from ragbench.datasets.templates import NOT_IN_DOCUMENTS, Candidate
from ragbench.documents.schema import Document, TextChunk
from ragbench.evaluation.budget import BudgetGuard
from ragbench.models import cost as pricing
from ragbench.models.llms import LLM
from ragbench.utils.query_planning import loads_lenient
from ragbench.utils.text import estimate_tokens, normalize_text, tokenize

logger = logging.getLogger(__name__)

CATEGORIES = ("single_hop", "multi_hop", "paraphrase", "numeric", "unanswerable")
DEFAULT_MIX = {"single_hop": 0.4, "multi_hop": 0.2, "paraphrase": 0.15, "numeric": 0.1, "unanswerable": 0.15}
DEDUPE_JACCARD = 0.8  # token-set overlap at which two questions count as the same
ATTEMPTS_PER_QUESTION = 3  # candidates tried per question before a category gives up (rejections and unusable replies count)
MIN_QUESTION_WORDS = 3
DIGEST_DOCS = 12  # documents shown to the model when it writes an unanswerable question
NEIGHBOURS = 4
MOCK_GENERATOR = "mock-template"
_DIFFICULTY = {"single_hop": "easy", "numeric": "medium", "paraphrase": "medium", "unanswerable": "medium", "multi_hop": "hard"}
_ANSWER_TYPE = {"numeric": "numeric", "unanswerable": "unanswerable"}
# Rough sizes behind the cost estimate: prompt wording around the documents, a JSON reply, and the share of candidates that are rejected and retried.
PROMPT_OVERHEAD_TOKENS = 260
REPLY_TOKENS = 110
RETRY_SLACK = 1.25


def parse_mix(text: str | None) -> dict[str, float]:
    """`single_hop=0.4,multi_hop=0.2,...` as shares that sum to 1 (they are normalized); categories left out get none. None means the default mix."""
    if not text or not text.strip():
        return dict(DEFAULT_MIX)
    mix: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in CATEGORIES:
            raise ValueError(f"Unknown category {name!r} in --mix. Available: {', '.join(CATEGORIES)}")
        try:
            share = float(value)
        except ValueError:
            raise ValueError(f"--mix needs `category=share` pairs such as single_hop=0.4; got {part.strip()!r}") from None
        if share < 0:
            raise ValueError(f"--mix share for {name} must not be negative")
        mix[name] = share
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("--mix shares must add up to more than 0")
    return {name: share / total for name, share in mix.items() if share > 0}


def allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    """How many questions of each category: `n` split by the shares, largest remainder first, so every count is within 1 of its exact share."""
    total = sum(mix.values())
    exact = {name: n * share / total for name, share in mix.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(exact, key=lambda key: (-(exact[key] - counts[key]), list(mix).index(key)))[: n - sum(counts.values())]:
        counts[name] += 1
    return {name: counts[name] for name in mix}


def stratified_order(documents: Sequence[Document], rng: random.Random) -> list[Document]:
    """Documents interleaved across three length bands (short, medium, long), each shuffled: sampling from the front covers all lengths evenly."""
    ranked = sorted(documents, key=lambda document: (len(document.text), document.doc_id))
    size = max(1, -(-len(ranked) // 3))
    bands = [ranked[start : start + size] for start in range(0, len(ranked), size)]
    for band in bands:
        rng.shuffle(band)
    order: list[Document] = []
    for position in range(size):
        order.extend(band[position] for band in bands if position < len(band))
    return order


def estimate_generation_cost(documents: Sequence[Document], n: int, mix: dict[str, float], model: str) -> float:
    """Dollars a live `generate-questions` run is expected to cost: calls per category over documents of the corpus's typical size, at `model`'s price."""
    counts = allocate(n, mix)
    doc_tokens = [estimate_tokens(document.text[: prompts.MAX_DOC_CHARS]) for document in documents] or [0]
    typical = sum(doc_tokens) / len(doc_tokens)
    digest = DIGEST_DOCS * 45
    # (calls per question, document tokens per call)
    shape = {"single_hop": (1, typical), "numeric": (1, typical), "multi_hop": (1, 2 * typical), "paraphrase": (2, typical), "unanswerable": (1, digest)}
    prompt_tokens = completion_tokens = 0.0
    for category, count in counts.items():
        calls, size = shape[category]
        prompt_tokens += count * calls * (PROMPT_OVERHEAD_TOKENS + size) * RETRY_SLACK
        completion_tokens += count * calls * REPLY_TOKENS * RETRY_SLACK
    return pricing.estimate_model_cost(model, int(prompt_tokens), int(completion_tokens))


# -- writers ---------------------------------------------------------------------------------------


class BudgetStop(Exception):
    """The spending cap was reached before the next model call."""


class Writer(Protocol):
    """Proposes candidates; returns None for an unusable reply."""

    def write(self, kind: str, document: Document, avoid: Sequence[str]) -> Candidate | None: ...

    def multihop(self, first: Document, second: Document, avoid: Sequence[str]) -> Candidate | None: ...

    def paraphrase(self, document: Document, question: str) -> str | None: ...

    def unanswerable(self, digest: Sequence[tuple[str, str]], avoid: Sequence[str], corpus: str) -> Candidate | None: ...


class TemplateWriter:
    """The model-free writer of mock mode: deterministic templates over each document's own sentences."""

    def write(self, kind: str, document: Document, avoid: Sequence[str]) -> Candidate | None:
        index = len(avoid)
        if kind == "numeric":
            return templates.numeric_candidate(document.title, document.text, index)
        return templates.single_candidate(document.title, document.text, index)

    def multihop(self, first: Document, second: Document, avoid: Sequence[str]) -> Candidate | None:
        return templates.multihop_candidate(first.title, first.text, second.title, second.text, len(avoid))

    def paraphrase(self, document: Document, question: str) -> str | None:
        return templates.paraphrase_question(question)

    def unanswerable(self, digest: Sequence[tuple[str, str]], avoid: Sequence[str], corpus: str) -> Candidate | None:
        index = len(avoid)
        while (candidate := templates.unanswerable_candidate(index)).entity and candidate.entity.lower() in corpus:
            index += 1  # a made-up name that happens to occur in the corpus is skipped
        return candidate


class LLMWriter:
    """Asks a chat model for JSON candidates. Counts calls and cost, and stops (`BudgetStop`) once the spending cap is reached."""

    def __init__(self, llm: LLM, budget: BudgetGuard | None = None):
        self.llm = llm
        self.budget = budget
        self.calls = 0
        self.cost_usd = 0.0
        self.bad_replies = 0

    def _ask(self, prompt: str) -> dict[str, Any] | None:
        if self.budget is not None and self.budget.exhausted:
            raise BudgetStop
        result = self.llm.generate([{"role": "user", "content": prompt}], json_mode=True)
        self.calls += 1
        self.cost_usd += result.cost.total_cost
        if self.budget is not None:
            self.budget.charge(result.cost.total_cost)
        parsed = loads_lenient(result.text)
        if not isinstance(parsed, dict):
            self.bad_replies += 1
            return None
        return parsed

    @staticmethod
    def _candidate(reply: dict[str, Any] | None, *, answer: bool = True) -> Candidate | None:
        if reply is None:
            return None
        question = str(reply.get("question") or "").strip()
        reference = str(reply.get("reference_answer") or "").strip()
        if not question or (answer and not reference):
            return None
        keywords = reply.get("expected_keywords")
        words = [str(word).strip() for word in keywords if str(word).strip()] if isinstance(keywords, list) else []
        entity = str(reply.get("entity") or "").strip() or None
        return Candidate(question, reference, words[:6], entity)

    def write(self, kind: str, document: Document, avoid: Sequence[str]) -> Candidate | None:
        return self._candidate(self._ask(prompts.write_question_prompt(kind, document.title, document.text, avoid)))

    def multihop(self, first: Document, second: Document, avoid: Sequence[str]) -> Candidate | None:
        return self._candidate(self._ask(prompts.write_multihop_prompt(first.title, first.text, second.title, second.text, avoid)))

    def paraphrase(self, document: Document, question: str) -> str | None:
        reply = self._ask(prompts.paraphrase_prompt(document.title, document.text, question))
        text = str(reply.get("question") or "").strip() if reply else ""
        return text or None

    def unanswerable(self, digest: Sequence[tuple[str, str]], avoid: Sequence[str], corpus: str) -> Candidate | None:
        candidate = self._candidate(self._ask(prompts.unanswerable_prompt(digest, avoid)), answer=False)
        if candidate is not None:
            candidate.reference_answer = NOT_IN_DOCUMENTS
        return candidate


# -- generation ------------------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    rows: list[dict[str, Any]]
    requested: dict[str, int]
    made: dict[str, int]
    generator: str
    mock: bool
    rejected: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    stopped_by_budget: bool = False


# Words every question is built from; they say nothing about *what* is asked, so two questions are compared without them.
_FRAME_WORDS = frozenset(
    "a an the of to in on for and or is are was were be been do does did what which who whom whose when where why how it its this that these those "
    "with by as at from about into relate relates according document documents say says tell me you can could would should there their".split()
)


def _tokens(text: str) -> frozenset[str]:
    words = tokenize(text)
    return frozenset(word for word in words if word not in _FRAME_WORDS) or frozenset(words)


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    return len(first & second) / len(first | second) if first | second else 1.0


class _Generation:
    """The state of one `generate_questions` call: accepted questions, the document cursor, and what was rejected and why."""

    def __init__(self, documents: Sequence[Document], writer: Writer, rng: random.Random, generator: str):
        self.documents = list(documents)
        self.writer = writer
        self.rng = rng
        self.generator = generator
        self.order = stratified_order(self.documents, rng)
        self.cursor = 0
        self.corpus = "\x00".join(normalize_text(document.text).lower() + " " + document.title.lower() for document in self.documents)
        self.accepted: list[dict[str, Any]] = []
        self.seen: list[frozenset[str]] = []
        self.asked: dict[str, list[str]] = {}  # doc id -> questions already written from it (the prompt tells the writer not to repeat them)
        self.rejected: Counter[str] = Counter()
        self._neighbours: Any = None  # a BM25Store over one entry per document, built when the first multi-hop pair is needed
        self._used_pairs: set[frozenset[str]] = set()

    def next_document(self, wanted: Any = None) -> Document | None:
        for _ in range(len(self.order)):
            document = self.order[self.cursor % len(self.order)]
            self.cursor += 1
            if wanted is None or wanted(document):
                return document
        return None

    def neighbour_of(self, document: Document) -> Document | None:
        """The most similar other document by BM25 that has not been paired with this one yet."""
        if self._neighbours is None:
            import ragbench.rag_systems  # noqa: F401  (the store's own imports only resolve when the systems package loads first)
            from ragbench.stores.bm25_store import BM25Store

            self._neighbours = BM25Store()
            self._neighbours.build([TextChunk(chunk_id=d.doc_id, doc_id=d.doc_id, text=f"{d.title} {d.text[:2000]}") for d in self.documents])
        by_id = {d.doc_id: d for d in self.documents}
        query = f"{document.title} {document.text[:300]}"
        for chunk in self._neighbours.search(query, top_k=NEIGHBOURS + 1).chunks:
            pair = frozenset((document.doc_id, chunk.doc_id))
            if chunk.doc_id != document.doc_id and pair not in self._used_pairs:
                self._used_pairs.add(pair)
                return by_id[chunk.doc_id]
        return None

    def accept(self, category: str, question: str, reference: str, keywords: list[str], doc_ids: list[str], extra: dict[str, Any]) -> bool:
        question = question.strip()
        if len(question.split()) < MIN_QUESTION_WORDS:
            self.rejected["unusable"] += 1
            return False
        tokens = _tokens(question)
        if any(_jaccard(tokens, other) >= DEDUPE_JACCARD for other in self.seen):
            self.rejected["duplicate"] += 1
            return False
        self.seen.append(tokens)
        row: dict[str, Any] = {
            "question": question,
            "reference_answer": reference,
            "expected_keywords": keywords,
            "relevant_doc_ids": doc_ids,
            "category": category,
            "difficulty": _DIFFICULTY[category],
            "answer_type": _ANSWER_TYPE.get(category, "single_fact"),
            "metadata": {
                "synthetic": True,
                "needs_review": True,
                "generator": self.generator,
                "source_doc_ids": doc_ids,
                **({"mock": True} if self.generator == MOCK_GENERATOR else {}),
                **extra,
            },
        }
        if category == "unanswerable":
            row["answerable"] = False
        self.accepted.append(row)
        return True

    def remember(self, document: Document, question: str) -> None:
        self.asked.setdefault(document.doc_id, []).append(question)

    # -- one category ----------------------------------------------------------------------------

    def make(self, category: str) -> bool:
        """Try to add one question of `category`; False when the attempt produced nothing usable."""
        if category == "unanswerable":
            return self._unanswerable()
        if category == "multi_hop":
            return self._multi_hop()
        document = self.next_document(lambda d: bool(re.search(r"\d", d.text))) if category == "numeric" else self.next_document()
        if document is None:
            self.rejected["no_suitable_document"] += 1
            return False
        avoid = self.asked.get(document.doc_id, [])
        candidate = self.writer.write("numeric" if category == "numeric" else "single_hop", document, avoid)
        if candidate is None:
            self.rejected["unusable"] += 1
            return False
        self.remember(document, candidate.question)
        if category == "paraphrase":
            rewritten = self.writer.paraphrase(document, candidate.question)
            if rewritten is None or normalize_text(rewritten).lower() == normalize_text(candidate.question).lower():
                self.rejected["unusable"] += 1
                return False
            return self.accept(category, rewritten, candidate.reference_answer, candidate.keywords, [document.doc_id], {"paraphrase_of": candidate.question})
        return self.accept(category, candidate.question, candidate.reference_answer, candidate.keywords, [document.doc_id], {})

    def _multi_hop(self) -> bool:
        first = self.next_document()
        second = self.neighbour_of(first) if first is not None else None
        if first is None or second is None:
            self.rejected["no_related_pair"] += 1
            return False
        avoid = self.asked.get(first.doc_id, [])
        candidate = self.writer.multihop(first, second, avoid)
        if candidate is None:
            self.rejected["unusable"] += 1
            return False
        self.remember(first, candidate.question)
        return self.accept("multi_hop", candidate.question, candidate.reference_answer, candidate.keywords, [first.doc_id, second.doc_id], {})

    def _unanswerable(self) -> bool:
        sample = [self.next_document() for _ in range(min(DIGEST_DOCS, len(self.order)))]
        unique = {d.doc_id: d for d in sample if d is not None}.values()
        digest = [(d.title, normalize_text(d.text)[:140]) for d in unique]
        avoid = [row["question"] for row in self.accepted if row["category"] == "unanswerable"]
        candidate = self.writer.unanswerable(digest, avoid, self.corpus)
        if candidate is None or not candidate.entity:
            self.rejected["unusable"] += 1
            return False
        if candidate.entity.lower() in self.corpus:
            self.rejected["entity_in_corpus"] += 1  # the corpus does mention it, so the question may well be answerable
            return False
        return self.accept("unanswerable", candidate.question, NOT_IN_DOCUMENTS, [], [], {"entity": candidate.entity})


def generate_questions(
    documents: Sequence[Document],
    *,
    n: int = 100,
    mix: dict[str, float] | None = None,
    llm: LLM | None = None,
    seed: int = 0,
    budget: BudgetGuard | None = None,
) -> SynthesisResult:
    """Write about `n` questions over `documents`. With `llm=None` the deterministic templates are used (mock mode: rows say `mock: true`).

    Each category tries up to `ATTEMPTS_PER_QUESTION` candidates per question it owes; shortfalls are reported in `warnings`, never padded with
    another category. Rows come out shuffled (so `evaluation.max_questions` takes a representative slice) and numbered `q_001`...
    """
    if not documents:
        raise ValueError("There are no documents to write questions about")
    if n < 1:
        raise ValueError("--n must be at least 1")
    mix = mix or dict(DEFAULT_MIX)
    requested = allocate(n, mix)
    rng = random.Random(seed)
    writer: Writer = TemplateWriter() if llm is None else LLMWriter(llm, budget)
    generator = MOCK_GENERATOR if llm is None else llm.model_name
    state = _Generation(documents, writer, rng, generator)
    made: dict[str, int] = {}
    stopped = False
    warnings: list[str] = []
    try:
        for category, target in requested.items():
            attempts = 0
            made[category] = 0
            while made[category] < target and attempts < target * ATTEMPTS_PER_QUESTION + 2:
                attempts += 1
                made[category] += state.make(category)
    except BudgetStop:
        stopped = True
    for category, target in requested.items():
        if made.get(category, 0) < target:
            warnings.append(f"Only {made.get(category, 0)} of {target} {category} questions were generated" + (" (stopped at the spending cap)." if stopped else "."))
    if state.rejected:
        warnings.append("Candidates rejected: " + ", ".join(f"{count} {reason.replace('_', ' ')}" for reason, count in sorted(state.rejected.items())) + ".")
    rows = state.accepted
    rng.shuffle(rows)
    for number, row in enumerate(rows, start=1):
        row["id"] = f"q_{number:03d}"
    ordered = [{"id": row["id"], **{key: value for key, value in row.items() if key != "id"}} for row in rows]
    return SynthesisResult(
        rows=ordered,
        requested=requested,
        made=made,
        generator=generator,
        mock=llm is None,
        rejected=state.rejected,
        warnings=warnings,
        cost_usd=writer.cost_usd if isinstance(writer, LLMWriter) else 0.0,
        calls=writer.calls if isinstance(writer, LLMWriter) else 0,
        stopped_by_budget=stopped,
    )
