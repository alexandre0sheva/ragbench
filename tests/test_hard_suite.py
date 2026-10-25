"""The bundled hard suite (`data/hard`) is generated: these tests pin the generator to the committed files and pin the shape
that makes the suite hard (versions that conflict, chains whose bridge is never named, aggregation over many documents,
near-name decoys, facts buried in long reports, a quarter unanswerable), so an edit cannot quietly make it easy."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from ragbench.config.loader import load_config
from ragbench.config.schema import SystemConfig
from ragbench.datasets.demo_generator import SUITES, demo_source_dir, write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents
from ragbench.rag_systems import create_rag_system

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "data" / "hard"

MIN_PER_CATEGORY = {
    "version_conflict": 14,
    "temporal": 14,
    "multi_hop": 26,
    "aggregation": 16,
    "distractor": 12,
    "long_context": 12,
    "unanswerable": 34,
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_hard_suite", ROOT / "scripts" / "generate_hard_suite.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # dataclasses look their module up while the script is being executed
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def documents():
    return load_documents(HARD / "docs")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(HARD / "questions.jsonl", HARD / "qrels.jsonl")


@pytest.fixture(scope="module")
def doc_text(documents):
    return {d.doc_id: d.text for d in documents}


def test_the_generator_is_deterministic_and_the_committed_files_are_its_output():
    generator = _load_generator()
    files = generator.build_suite()

    assert files == generator.build_suite(), "two builds differ: the generator is not deterministic"
    committed = {str(path.relative_to(HARD)) for path in HARD.rglob("*") if path.is_file()}
    assert committed == set(files), "data/hard has files the generator does not write, or misses some: run scripts/generate_hard_suite.py"
    stale = [name for name, content in files.items() if (HARD / name).read_text(encoding="utf-8") != content]
    assert not stale, f"Run `python scripts/generate_hard_suite.py` to refresh: {stale[:5]}"


def test_the_generator_check_mode_reports_a_stale_file(tmp_path):
    generator = _load_generator()
    assert generator.main(["--output", str(tmp_path)]) == 0
    assert generator.main(["--check", "--output", str(tmp_path)]) == 0

    (tmp_path / "questions.jsonl").write_text("{}\n", encoding="utf-8")
    assert generator.main(["--check", "--output", str(tmp_path)]) == 1


def test_the_suite_validates_with_zero_warnings(documents, dataset):
    assert validate_dataset(documents, dataset) == []


def test_size_and_categories(documents, dataset, doc_text):
    counts = Counter(question.category for question in dataset.questions)

    assert len(documents) >= 100 and len(dataset.questions) >= 140
    assert sum(len(text.split()) for text in doc_text.values()) >= 40_000, "the corpus must be well past a small full-context budget"
    assert set(counts) == set(MIN_PER_CATEGORY), "categories changed: update the table above and docs/dataset-format.md"
    for category, minimum in MIN_PER_CATEGORY.items():
        assert counts[category] >= minimum, f"{category}: {counts[category]} < {minimum}"


def test_about_a_quarter_is_unanswerable_and_each_is_a_near_miss(dataset):
    unanswerable = [q for q in dataset.questions if not q.is_answerable]

    assert 0.22 <= len(unanswerable) / len(dataset.questions) <= 0.30
    assert {q.category for q in unanswerable} == {"unanswerable"}
    assert all(q.answer_type == "unanswerable" and not q.relevant_doc_ids for q in unanswerable)
    traps = Counter(q.metadata["trap"] for q in unanswerable)
    assert len(traps) >= 4 and min(traps.values()) >= 4, f"unanswerable questions should mix several kinds of near miss: {traps}"


def test_relevant_documents_exist_qrels_are_graded_and_evidence_is_verbatim(dataset, doc_text):
    grades: Counter[int] = Counter()
    for question in dataset.questions:
        judged = dataset.qrels.get(question.id, {})
        assert set(question.relevant_doc_ids) <= set(doc_text) and set(question.relevant_doc_ids) <= set(judged), question.id
        if question.is_answerable:
            assert max(judged.values()) == 3, question.id
            assert question.evidence, f"{question.id} has no evidence quotes"
            for item in question.evidence:
                assert item.doc_id in question.relevant_doc_ids
                assert " ".join(item.quote.lower().split()) in " ".join(doc_text[item.doc_id].lower().split()), question.id
        grades.update(judged.values())
    assert {2, 3} <= set(grades)


def test_questions_never_quote_documents_and_keywords_are_checkable(dataset, doc_text):
    for question in dataset.questions:
        assert question.reference_answer and len(question.reference_answer.split()) <= 45, question.id
        for doc_id in question.relevant_doc_ids:
            assert question.question.lower() not in doc_text[doc_id].lower(), question.id
        for keyword in question.expected_keywords:
            assert keyword.lower() in question.reference_answer.lower(), f"{question.id}: {keyword!r} missing from the reference answer"
            if not question.requires_tools:  # computed totals need not appear in a document
                cited = " ".join(doc_text[d] for d in question.relevant_doc_ids).lower()
                assert keyword.lower() in cited, f"{question.id}: {keyword!r} is not in its cited documents"


def test_multi_hop_bridges_are_never_named_in_the_question(dataset):
    hops = Counter()
    for question in (q for q in dataset.questions if q.category == "multi_hop"):
        hops[question.metadata["hops"]] += 1
        assert len(question.relevant_doc_ids) == question.metadata["hops"], question.id
        for bridge in question.metadata["bridges"]:
            assert bridge.lower() not in question.question.lower(), f"{question.id} names its bridge {bridge!r}"
    assert set(hops) == {2, 3} and hops[3] >= 10


def test_version_conflicts_have_superseded_documents_that_say_something_else(dataset, doc_text):
    for question in (q for q in dataset.questions if q.category == "version_conflict"):
        superseded = question.metadata["superseded"]
        assert superseded and not set(superseded) & set(question.relevant_doc_ids), question.id
        assert any(question.metadata["answer"].lower() not in doc_text[d].lower() for d in superseded), question.id


def test_distractors_have_a_near_name_decoy_with_a_different_answer(dataset, doc_text):
    for question in (q for q in dataset.questions if q.category == "distractor"):
        decoy = question.metadata["decoy"]
        assert decoy in doc_text and decoy not in question.relevant_doc_ids, question.id
        assert question.metadata["answer"].lower() not in doc_text[decoy].lower(), question.id


def test_aggregation_needs_several_documents_and_a_calculator(dataset):
    for question in (q for q in dataset.questions if q.category == "aggregation"):
        assert question.requires_tools == ["calculator"], question.id
        assert len(question.relevant_doc_ids) >= 2, question.id


def test_long_context_facts_sit_deep_in_long_documents(dataset, doc_text):
    for question in (q for q in dataset.questions if q.category == "long_context"):
        (doc_id,) = question.relevant_doc_ids
        text = doc_text[doc_id]
        assert len(text.split()) >= 3000, question.id
        assert text.lower().find(question.evidence[0].quote.lower()) > len(text) // 5, f"{question.id}: the fact is near the top"


def test_single_shot_retrieval_finds_the_first_hop_far_more_often_than_the_last(documents, dataset):
    system = create_rag_system(SystemConfig(type="bm25", chunker={"type": "word", "chunk_size": 450, "chunk_overlap": 70}, retrieval={"top_k": 5}), force_mock=True)
    system.ingest(documents)
    first = last = total = 0
    for question in (q for q in dataset.questions if q.category == "multi_hop"):
        retrieved = {chunk.doc_id for chunk in system.fetch_context(question.question, top_k=5).chunks}
        first += question.relevant_doc_ids[0] in retrieved
        last += question.relevant_doc_ids[-1] in retrieved
        total += 1
    assert first / total >= 0.7 and last / total <= 0.4, f"first hop {first}/{total}, last hop {last}/{total}"


def test_demo_suite_hard_writes_the_packaged_suite(tmp_path):
    assert set(SUITES) == {"demo", "hard"}
    source = demo_source_dir("hard")
    stats = write_demo_dataset(tmp_path / "hard", suite="hard")

    assert stats["documents"] == len(list((source / "docs").glob("*.md"))) and stats["modified"] == 0
    assert (tmp_path / "hard" / "questions.jsonl").read_bytes() == (HARD / "questions.jsonl").read_bytes()


def test_the_hard_config_runs_the_representative_systems_on_the_suite():
    config = load_config(ROOT / "configs" / "hard.yaml")

    assert config.dataset.documents_path == Path("data/hard/docs")
    types = {system.type for system in config.systems}
    assert {"no_retrieval", "full_context", "bm25", "iterative", "agent_search"} <= types and len(config.systems) == 8


def test_the_demo_command_takes_a_suite_and_defaults_its_directory_to_it(tmp_path, monkeypatch):
    from cli_support import invoke

    monkeypatch.chdir(tmp_path)
    result = invoke("demo", "--suite", "hard")
    assert result.exit_code == 0, result.output
    assert "Hard suite ready at data/hard" in result.output and (tmp_path / "data" / "hard" / "qrels.jsonl").is_file()

    unknown = invoke("demo", "--suite", "easy")
    assert unknown.exit_code != 0 and "Unknown suite 'easy'" in unknown.output


def test_possessives_of_plural_names_are_written_with_a_bare_apostrophe(dataset):
    texts = [q.question for q in dataset.questions] + [q.reference_answer or "" for q in dataset.questions]
    assert not [t for t in texts if "s's " in t], "write `Norrland Batteries'`, not `Norrland Batteries's`"
