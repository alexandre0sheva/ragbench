"""`inspect-dataset` v2: the rich dataset profile, its warnings, suggested chunk sizes and projected cost."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.schema import DatasetConfig
from ragbench.datasets.profile import (
    FULL_CONTEXT_WARN_TOKENS,
    guess_language,
    profile_dataset,
    projected_standard_cost,
    suggest_chunk_sizes,
    token_weighted_median,
)
from ragbench.datasets.schema import Dataset, Question
from ragbench.documents.schema import Document
from ragbench.utils.jsonl import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"

ENGLISH = "The quick brown fox jumps over the lazy dog and the cat is in the garden with a bird that was in the tree."
SPANISH = "El rápido zorro marrón salta sobre el perro perezoso y el gato está en el jardín con un pájaro que estaba en el árbol de la casa."


def _doc(doc_id: str, text: str, title: str = "t") -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=title, text=text)


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{number}" for number in range(count))


def _questions(*specs: dict) -> Dataset:
    questions = [Question(id=f"q{number}", **spec) for number, spec in enumerate(specs, start=1)]
    return Dataset(questions=questions, qrels={q.id: {doc_id: 1 for doc_id in q.relevant_doc_ids} for q in questions})


# -- documents -----------------------------------------------------------------------------------


def test_document_statistics_cover_count_size_distribution_and_language():
    docs = [_doc(f"doc_{n:03d}", ENGLISH * n) for n in range(1, 21)]
    profile = profile_dataset(docs, _questions({"question": "Where is the cat?", "relevant_doc_ids": ["doc_001"]}))
    stats = profile.documents
    assert stats.count == 20 and stats.total_tokens > 0
    assert 0 < stats.tokens.min <= stats.tokens.p50 <= stats.tokens.p95 <= stats.tokens.max
    assert stats.language == "en" and 0 < stats.language_confidence <= 1


def test_language_guess_uses_stopwords_and_admits_when_it_cannot_tell():
    assert guess_language([ENGLISH * 3])[0] == "en"
    assert guess_language([SPANISH * 3])[0] == "es"
    assert guess_language(["12345 67890 !!! ???"])[0] == "unknown"
    assert guess_language([])[0] == "unknown"


# -- near-duplicates, leakage, tiny corpora ------------------------------------------------------


def test_near_duplicate_documents_are_found_by_shingled_overlap():
    base = _words("alpha", 300)
    docs = [
        _doc("doc_001", base + " tail one"),
        _doc("doc_002", base + " tail two"),
        _doc("doc_003", _words("beta", 300)),
        _doc("doc_004", _words("gamma", 300)),
    ]
    profile = profile_dataset(docs, _questions({"question": "What is alpha1?", "relevant_doc_ids": ["doc_001"]}))
    assert [(d.a, d.b) for d in profile.near_duplicates] == [("doc_001", "doc_002")]
    assert profile.near_duplicates[0].jaccard > 0.9
    assert any("near-duplicate" in warning and "doc_001" in warning for warning in profile.warnings)


def test_distinct_documents_raise_no_near_duplicate_warning():
    docs = [_doc(f"doc_{n:03d}", _words(f"w{n}x", 200)) for n in range(1, 6)]
    profile = profile_dataset(docs, _questions({"question": "What is w1x1?", "relevant_doc_ids": ["doc_001"]}))
    assert profile.near_duplicates == [] and not any("near-duplicate" in warning for warning in profile.warnings)


def test_a_question_that_appears_verbatim_in_a_document_is_flagged_as_leakage():
    docs = [_doc("doc_001", "# FAQ\n\nHow do I reset my password?\nOpen Settings and choose Reset.\n"), _doc("doc_002", "Billing runs monthly.")]
    dataset = _questions(
        {"question": "how do I reset my  password", "relevant_doc_ids": ["doc_001"]},  # case, spacing and the question mark do not matter
        {"question": "When is billing run?", "relevant_doc_ids": ["doc_002"]},
        {"question": "Billing runs", "relevant_doc_ids": ["doc_002"]},  # too short to be evidence of anything
    )
    profile = profile_dataset(docs, dataset)
    assert profile.leaked_questions == ["q1"]
    assert any("verbatim" in warning and "q1" in warning for warning in profile.warnings)


def test_a_corpus_that_fits_in_one_prompt_warns_that_full_context_will_likely_win():
    small = profile_dataset([_doc("doc_001", ENGLISH)], _questions({"question": "Where is the cat?", "relevant_doc_ids": ["doc_001"]}))
    assert any("full_context" in warning for warning in small.warnings)
    big_text = " ".join(f"word{n}" for n in range(FULL_CONTEXT_WARN_TOKENS * 2))
    big = profile_dataset([_doc("doc_001", big_text)], _questions({"question": "Where is the cat?", "relevant_doc_ids": ["doc_001"]}))
    assert not any("full_context" in warning for warning in big.warnings)


# -- questions and qrels -------------------------------------------------------------------------


def test_question_statistics_report_balance_answerability_and_label_coverage():
    docs = [_doc("doc_001", ENGLISH), _doc("doc_002", SPANISH), _doc("doc_003", "Other text entirely about nothing.")]
    dataset = _questions(
        {"question": "Where is the cat sitting right now?", "relevant_doc_ids": ["doc_001"], "category": "fact", "difficulty": "easy", "answer_type": "single_fact"},
        {"question": "Where is the dog?", "relevant_doc_ids": ["doc_001", "doc_002"], "category": "fact", "difficulty": "hard", "answer_type": "single_fact"},
        {"question": "Who owns the moon?", "category": "unanswerable"},  # legacy inference: no relevant docs, so unanswerable
        {"question": "Is it labeled?", "answerable": True},  # answerable, but nothing to score retrieval against
    )
    profile = profile_dataset(docs, dataset)
    questions = profile.questions
    assert questions.count == 4 and questions.categories == {"fact": 2, "unanswerable": 1, "unknown": 1}
    assert questions.difficulties["easy"] == 1 and questions.answer_types["single_fact"] == 2
    assert (questions.answerable, questions.unanswerable, questions.answerable_ratio) == (3, 1, 0.75)
    assert questions.words.max >= questions.words.p50 > 0
    qrels = profile.qrels
    assert (qrels.labeled_questions, qrels.unlabeled_answerable) == (2, 1) and qrels.labeled_share == 0.5
    assert qrels.documents_referenced == 2 and qrels.corpus_coverage == pytest.approx(2 / 3)


def test_a_label_free_dataset_is_reported_as_such():
    dataset = Dataset(questions=[Question(id="q1", question="Where is the cat?", answerable=True)], label_free=True)
    profile = profile_dataset([_doc("doc_001", ENGLISH)], dataset)
    assert profile.qrels.labeled_questions == 0 and profile.qrels.label_free


# -- chunk sizes ---------------------------------------------------------------------------------


def test_chunk_size_suggestions_follow_the_document_length_distribution():
    short, note = suggest_chunk_sizes(typical=60, p95=120)
    assert len(short) == 1 and "short" in note
    sizes, _ = suggest_chunk_sizes(typical=2500, p95=3500)
    assert sizes == sorted(set(sizes)) and 2 <= len(sizes) <= 3 and min(sizes) >= 100 and max(sizes) <= 1000
    medium, _ = suggest_chunk_sizes(typical=300, p95=800)
    assert max(medium) <= 300  # a chunk larger than a typical document is just the document
    assert suggest_chunk_sizes(typical=0, p95=0)[0] == []


def test_the_token_weighted_median_follows_where_the_text_is_not_where_the_documents_are():
    counts = [50] * 20 + [3000] * 2  # twenty FAQ pages and two handbooks: the handbooks hold nearly all the text
    assert token_weighted_median(counts) == 3000
    assert token_weighted_median([100, 100, 100]) == 100 and token_weighted_median([]) == 0


# -- projected cost and the CLI ------------------------------------------------------------------


def _tiny(root: Path) -> tuple[Path, Path]:
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    questions = root / "questions.jsonl"
    write_jsonl(
        questions,
        [
            {"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"], "category": "price"},
            {"id": "q2", "question": "When does ClaimPilot ship?", "relevant_doc_ids": ["doc_002"], "category": "date"},
        ],
    )
    return docs, questions


def test_projected_cost_of_the_standard_preset_reuses_the_estimate(tmp_path):
    docs, questions = _tiny(tmp_path)
    estimate = projected_standard_cost(DatasetConfig(documents_path=docs, questions_path=questions))
    assert estimate.n_questions == 2 and estimate.n_documents == 2
    names = {system.system_type for system in estimate.systems}
    assert {"bm25", "vector", "hybrid_rerank", "hyde", "contextual"} <= names  # the `standard` preset
    assert estimate.total_usd >= 0


def test_inspect_dataset_prints_the_profile(tmp_path):
    docs, questions = _tiny(tmp_path)
    result = CliRunner().invoke(app, ["inspect-dataset", "--docs", str(docs), "--questions", str(questions)])
    assert result.exit_code == 0, result.output
    for expected in ("Dataset Summary", "Documents", "Questions", "Qrels coverage", "Suggested chunk sizes", "Projected cost", "full_context"):
        assert expected in result.output, expected


def test_inspect_dataset_can_skip_the_cost_projection(tmp_path):
    docs, questions = _tiny(tmp_path)
    result = CliRunner().invoke(app, ["inspect-dataset", "--docs", str(docs), "--questions", str(questions), "--no-estimate"])
    assert result.exit_code == 0 and "Projected cost" not in result.output


def test_inspect_dataset_works_on_a_label_free_questions_file(tmp_path):
    docs, _ = _tiny(tmp_path)
    questions = tmp_path / "free.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "How much does HarborShield cost?"}])
    result = CliRunner().invoke(app, ["inspect-dataset", "--docs", str(docs), "--questions", str(questions), "--no-estimate"])
    assert result.exit_code == 0 and "label-free" in result.output
