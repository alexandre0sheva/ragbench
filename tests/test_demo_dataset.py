"""The bundled demo dataset is data, not code: these tests pin its shape so edits cannot quietly make it easy again."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ragbench.config.schema import SystemConfig
from ragbench.datasets.demo_generator import demo_source_dir, write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Question
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents
from ragbench.rag_systems import create_rag_system
from ragbench.registry import TOOLS
from ragbench.tools import resolve_tools  # noqa: F401  (registers the built-in tools)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"

KNOWN_TOOLS = set(TOOLS.names())  # the tool registry is the single list of valid names

MIN_PER_CATEGORY = {
    "direct_fact": 10,
    "paraphrase": 10,
    "multi_hop": 8,
    "comparison": 6,
    "numeric_reasoning": 8,
    "date_arithmetic": 8,
    "aggregation": 7,
    "temporal_conflict": 10,
    "distractor": 8,
    "unanswerable": 20,
    "exact_identifier": 10,
    "long_context": 10,
}


@pytest.fixture(scope="module")
def documents():
    return load_documents(DEMO / "docs")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DEMO / "questions.jsonl", DEMO / "qrels.jsonl")


def test_the_dataset_validates_with_zero_warnings(documents, dataset):
    assert validate_dataset(documents, dataset) == []


def test_size_and_the_twelve_categories(documents, dataset):
    counts = Counter(question.category for question in dataset.questions)

    assert len(documents) >= 55 and len(dataset.questions) >= 140
    assert set(counts) == set(MIN_PER_CATEGORY), "categories changed: update the table above and docs/dataset-format.md"
    for category, minimum in MIN_PER_CATEGORY.items():
        assert counts[category] >= minimum, f"{category}: {counts[category]} < {minimum}"


def test_at_least_fifteen_percent_of_questions_are_unanswerable(dataset):
    unanswerable = [q for q in dataset.questions if not q.is_answerable]

    assert len(unanswerable) / len(dataset.questions) >= 0.15
    assert {q.category for q in unanswerable} == {"unanswerable"}, "unanswerable questions and the category must coincide"
    assert all(q.answer_type == "unanswerable" for q in unanswerable)


def test_every_relevant_document_exists_and_qrels_are_graded(documents, dataset):
    doc_ids = {d.doc_id for d in documents}
    grades: Counter[int] = Counter()
    for question in dataset.questions:
        judged = dataset.qrels.get(question.id, {})
        assert set(question.relevant_doc_ids) <= doc_ids, question.id
        assert set(question.relevant_doc_ids) <= set(judged), question.id
        if question.is_answerable:
            assert max(judged.values()) == 3, f"{question.id} has no key-evidence (grade 3) document"
        else:
            assert not judged, f"{question.id} is unanswerable but has qrels"
        grades.update(judged.values())
    assert {1, 2, 3} <= set(grades), "qrels should use grades 1, 2 and 3"


def test_questions_never_quote_their_evidence_verbatim(documents, dataset):
    by_id = {d.doc_id: d.text.lower() for d in documents}
    for question in dataset.questions:
        for doc_id in question.relevant_doc_ids:
            assert question.question.lower() not in by_id[doc_id], f"{question.id} is a verbatim substring of {doc_id}"


def test_answerable_reference_answers_are_short_and_keywords_are_checkable(documents, dataset):
    doc_text = {d.doc_id: d.text.lower() for d in documents}
    for question in dataset.questions:
        assert question.reference_answer and len(question.reference_answer.split()) <= 45, question.id
        for keyword in question.expected_keywords:
            assert keyword.lower() in question.reference_answer.lower(), f"{question.id}: {keyword!r} missing from the reference answer"
            if question.is_answerable and not question.requires_tools:  # computed answers need not appear verbatim in a document
                cited = " ".join(doc_text[d] for d in question.relevant_doc_ids)
                assert keyword.lower() in cited, f"{question.id}: {keyword!r} is not in its cited documents"


def test_requires_tools_names_are_known_and_match_the_category(dataset):
    used: Counter[str] = Counter()
    for question in dataset.questions:
        assert set(question.requires_tools) <= KNOWN_TOOLS, question.id
        used.update(question.requires_tools)
        if question.category == "numeric_reasoning":
            assert "calculator" in question.requires_tools, question.id
        if question.category == "date_arithmetic":
            assert "date_calc" in question.requires_tools, question.id
        if question.category in {"direct_fact", "paraphrase", "unanswerable", "exact_identifier", "long_context", "distractor"}:
            assert not question.requires_tools, question.id
    assert used["calculator"] >= 8 and used["date_calc"] >= 8


def test_requires_tools_is_optional_metadata():
    assert Question(id="q", question="?").requires_tools == []
    assert Question.model_validate({"id": "q", "question": "?", "requires_tools": ["calculator"]}).requires_tools == ["calculator"]


def test_the_corpus_has_the_intended_mix(documents):
    words = {d.doc_id: len(d.text.split()) for d in documents}
    titles = [d.title for d in documents]

    assert sum(1 for n in words.values() if n >= 2000) >= 4 and max(words.values()) >= 3000, "long documents are where chunking matters"
    assert sum(1 for n in words.values() if n <= 120) >= 15, "very short FAQ-style documents are part of the mix"
    assert sum("Coverage Pack" in t for t in titles) == 4 and sum("Plan Sheet" in t for t in titles) == 3, "near-duplicate product pages"
    assert sum(t.startswith("Remote Work Policy") for t in titles) == 3, "versioned policy documents"
    assert any("|" in d.text and "---" in d.text for d in documents), "table-heavy content"
    assert any("```" in d.text for d in documents), "code / config snippets"


def test_bm25_beats_or_ties_vector_on_exact_identifiers_and_the_two_systems_really_differ(documents, dataset):
    def hit_rates(system_type: str, chunker: dict) -> dict[str, list[float]]:
        system = create_rag_system(SystemConfig(type=system_type, chunker=chunker, retrieval={"top_k": 5}), force_mock=True)
        system.ingest(documents)
        rates: dict[str, list[float]] = {}
        for question in dataset.questions:
            if not question.is_answerable:
                continue
            retrieved = {chunk.doc_id for chunk in system.fetch_context(question.question, top_k=5).chunks}
            rates.setdefault(question.category, []).append(len(retrieved & set(question.relevant_doc_ids)) / len(question.relevant_doc_ids))
        return rates

    bm25 = hit_rates("bm25", {"type": "word", "chunk_size": 450, "chunk_overlap": 70})
    vector = hit_rates("vector", {"type": "word", "chunk_size": 500, "chunk_overlap": 80})

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    assert mean(bm25["exact_identifier"]) >= mean(vector["exact_identifier"])
    assert bm25["paraphrase"] != vector["paraphrase"] or bm25["exact_identifier"] != vector["exact_identifier"]


def test_the_demo_command_writes_the_packaged_dataset_and_never_clobbers_edits(tmp_path):
    source = demo_source_dir()
    assert (source / "docs").is_dir() and (source / "questions.jsonl").is_file() and (source / "qrels.jsonl").is_file()

    stats = write_demo_dataset(tmp_path / "copy")
    assert stats["documents"] == len(list((DEMO / "docs").glob("*.md"))) and stats["modified"] == 0
    assert (tmp_path / "copy" / "questions.jsonl").read_bytes() == (DEMO / "questions.jsonl").read_bytes()

    edited = tmp_path / "copy" / "docs" / "doc_001.md"
    edited.write_text("my own edit\n", encoding="utf-8")
    kept = write_demo_dataset(tmp_path / "copy")
    assert kept["modified"] == 1 and edited.read_text(encoding="utf-8") == "my own edit\n"

    restored = write_demo_dataset(tmp_path / "copy", overwrite=True)
    assert restored["modified"] == 0 and edited.read_bytes() == (DEMO / "docs" / "doc_001.md").read_bytes()
