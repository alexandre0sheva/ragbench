"""Importers (`ragbench import`): every supported format round-trips to a dataset the loaders and validation accept."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.datasets.importers import FORMATS, ImportDatasetError, import_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _docs(root: Path) -> Path:
    _write(root / "doc_001.md", "# Pricing\n\nHarborShield costs $200 per month.\n")
    _write(root / "doc_002.md", "# Roadmap\n\nClaimPilot ships in Q3.\n")
    return root


# -- csv -----------------------------------------------------------------------------------------


def test_csv_round_trips_to_a_valid_dataset(tmp_path):
    docs = _docs(tmp_path / "docs")
    source = _write(
        tmp_path / "qa.csv",
        'question,answer,doc_ids,category\n'
        'How much does HarborShield cost?,"$200 per month, billed monthly",doc_001,pricing\n'
        'When does ClaimPilot ship?,Q3,doc_002; doc_001,roadmap\n',
    )
    result = import_dataset("csv", source, tmp_path / "out", docs=docs)

    dataset = load_dataset(result.questions_path)
    assert [q.id for q in dataset.questions] == ["q_001", "q_002"]
    assert dataset.questions[0].reference_answer == "$200 per month, billed monthly"
    assert dataset.questions[0].relevant_doc_ids == ["doc_001"] and dataset.questions[1].relevant_doc_ids == ["doc_002", "doc_001"]
    assert dataset.questions[1].category == "roadmap"
    assert not dataset.label_free
    assert validate_dataset(load_documents(docs), dataset) == []
    assert result.n_questions == 2 and result.qrels_path is None and result.warnings == []


def test_csv_without_doc_ids_is_label_free_and_marks_questions_answerable(tmp_path):
    source = _write(tmp_path / "qa.csv", "question,answer\nWhat is the refund window?,30 days\nWho owns the roadmap?,\n")
    result = import_dataset("csv", source, tmp_path / "out")
    rows = [json.loads(line) for line in result.questions_path.read_text().splitlines()]
    assert all(row["answerable"] is True and not row.get("relevant_doc_ids") for row in rows)
    assert rows[1].get("reference_answer") is None
    assert load_dataset(result.questions_path).label_free


def test_csv_honours_explicit_ids_answerable_column_and_keeps_unknown_columns_as_metadata(tmp_path):
    source = _write(
        tmp_path / "qa.csv",
        "id,query,reference,answerable,owner\nfaq-1,Is X supported?,Yes,yes,alice\nfaq-2,Does it fly?,,no,bob\n",
    )
    result = import_dataset("csv", source, tmp_path / "out")
    first, second = (json.loads(line) for line in result.questions_path.read_text().splitlines())
    assert first["id"] == "faq-1" and first["question"] == "Is X supported?" and first["answerable"] is True and first["metadata"] == {"owner": "alice"}
    assert second["answerable"] is False


def test_csv_needs_a_question_column_and_skips_blank_rows_with_a_warning(tmp_path):
    with pytest.raises(ImportDatasetError, match="question"):
        import_dataset("csv", _write(tmp_path / "bad.csv", "prompt,answer\nx,y\n"), tmp_path / "o1")
    result = import_dataset("csv", _write(tmp_path / "ok.csv", "question\nReal one?\n\n   \n"), tmp_path / "o2")
    assert result.n_questions == 1


def test_csv_with_doc_ids_missing_from_the_docs_folder_validates_with_warnings(tmp_path):
    docs = _docs(tmp_path / "docs")
    source = _write(tmp_path / "qa.csv", "question,doc_ids\nWhat?,doc_404\n")
    result = import_dataset("csv", source, tmp_path / "out", docs=docs)
    assert any("doc_404" in warning for warning in result.warnings)  # warns, does not fail


# -- beir ----------------------------------------------------------------------------------------


def _beir(root: Path) -> Path:
    _write(
        root / "corpus.jsonl",
        "\n".join(
            json.dumps(row)
            for row in [
                {"_id": "d1", "title": "Pricing", "text": "HarborShield costs $200 per month."},
                {"_id": "MED-10.2", "title": "", "text": "ClaimPilot ships in Q3."},
                {"_id": "unused", "title": "Other", "text": "Nobody asks about this."},
            ]
        )
        + "\n",
    )
    _write(
        root / "queries.jsonl",
        "\n".join(json.dumps(row) for row in [{"_id": "q1", "text": "How much does HarborShield cost?"}, {"_id": "q2", "text": "When does ClaimPilot ship?"}, {"_id": "q3", "text": "Train-only question"}])
        + "\n",
    )
    _write(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\tunused\t0\nq2\tMED-10.2\t1\n")
    _write(root / "qrels" / "train.tsv", "query-id\tcorpus-id\tscore\nq3\td1\t1\n")
    return root


def test_beir_round_trips_to_a_valid_labeled_dataset(tmp_path):
    result = import_dataset("beir", _beir(tmp_path / "beir"), tmp_path / "out")

    documents = load_documents(result.docs_dir)
    dataset = load_dataset(result.questions_path, result.qrels_path)
    ids = {d.doc_id for d in documents}
    assert len(documents) == 3 and "doc_d1" in ids  # ids are made loader-safe (doc_ prefix, no dots)
    assert [q.id for q in dataset.questions] == ["q1", "q2"]  # only the queries of the chosen split (default: test)
    q1, q2 = dataset.questions
    assert q1.relevant_doc_ids == ["doc_d1"] and len(q2.relevant_doc_ids) == 1 and q2.relevant_doc_ids[0] in ids
    assert dataset.qrels["q1"]["doc_d1"] == 2 and dataset.qrels["q1"]["doc_unused"] == 0  # graded qrels keep their grades, including 0
    assert validate_dataset(documents, dataset) == []
    assert (result.output / "id_map.json").exists()
    assert json.loads((result.output / "id_map.json").read_text())["MED-10.2"] == q2.relevant_doc_ids[0]
    title_doc = next(d for d in documents if d.doc_id == "doc_d1")
    assert title_doc.title == "Pricing" and "HarborShield costs $200" in title_doc.text


def test_beir_split_option_and_missing_split(tmp_path):
    root = _beir(tmp_path / "beir")
    train = import_dataset("beir", root, tmp_path / "train", split="train")
    assert [q.id for q in load_dataset(train.questions_path, train.qrels_path).questions] == ["q3"]
    with pytest.raises(ImportDatasetError, match="dev"):
        import_dataset("beir", root, tmp_path / "dev", split="dev")


def test_beir_sanitised_ids_that_would_collide_are_kept_apart(tmp_path):
    root = tmp_path / "beir"
    _write(root / "corpus.jsonl", json.dumps({"_id": "a.b", "text": "one"}) + "\n" + json.dumps({"_id": "a_b", "text": "two"}) + "\n")
    _write(root / "queries.jsonl", json.dumps({"_id": "q", "text": "which?"}) + "\n")
    _write(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq\ta.b\t1\n")
    result = import_dataset("beir", root, tmp_path / "out")
    assert len({d.doc_id for d in load_documents(result.docs_dir)}) == 2


def test_beir_warns_about_qrels_for_missing_documents_and_drops_unlabeled_queries(tmp_path):
    root = _beir(tmp_path / "beir")
    _write(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq1\tghost\t1\nq1\td1\t1\nq2\td1\t0\n")
    result = import_dataset("beir", root, tmp_path / "out")
    assert any("ghost" in warning for warning in result.warnings)
    assert any("q2" in warning for warning in result.warnings)  # no document graded above 0: nothing to score, so it is left out
    dataset = load_dataset(result.questions_path, result.qrels_path)
    assert [q.id for q in dataset.questions] == ["q1"] and dataset.questions[0].relevant_doc_ids == ["doc_d1"]
    with pytest.raises(ImportDatasetError, match="no questions"):
        _write(root / "qrels" / "test.tsv", "query-id\tcorpus-id\tscore\nq1\tghost\t1\n")
        import_dataset("beir", root, tmp_path / "out2")


# -- qa-md ---------------------------------------------------------------------------------------

QA_MD = """# Support FAQ

## Q: How much does HarborShield cost?
A: $200 per month,
billed monthly.
Docs: doc_001

**Question:** When does ClaimPilot ship?
**Answer:** Q3.

- Q: Does it come with a free tier?
"""


def test_qa_markdown_round_trips_to_a_valid_dataset(tmp_path):
    docs = _docs(tmp_path / "docs")
    result = import_dataset("qa-md", _write(tmp_path / "faq.md", QA_MD), tmp_path / "out", docs=docs)
    dataset = load_dataset(result.questions_path)
    first, second, third = dataset.questions
    assert (first.question, first.reference_answer, first.relevant_doc_ids) == ("How much does HarborShield cost?", "$200 per month, billed monthly.", ["doc_001"])
    assert (second.question, second.reference_answer, second.relevant_doc_ids) == ("When does ClaimPilot ship?", "Q3.", [])
    assert third.question == "Does it come with a free tier?" and third.reference_answer is None
    assert all(q.is_answerable for q in dataset.questions)  # unlabeled pairs are answerable, not silently "unanswerable"
    assert validate_dataset(load_documents(docs), dataset) != []  # q2/q3 have no labels: the dataset says so rather than hiding it
    assert result.n_questions == 3


def test_qa_markdown_accepts_a_folder_of_files_and_rejects_files_without_pairs(tmp_path):
    folder = tmp_path / "qa"
    _write(folder / "a.md", "Q: One?\nA: 1\n")
    _write(folder / "b.md", "Q: Two?\nA: 2\n")
    result = import_dataset("qa-md", folder, tmp_path / "out")
    assert [q.question for q in load_dataset(result.questions_path).questions] == ["One?", "Two?"]
    with pytest.raises(ImportDatasetError, match="Q:"):
        import_dataset("qa-md", _write(tmp_path / "none.md", "just prose\n"), tmp_path / "o2")


# -- shared behaviour ----------------------------------------------------------------------------


def test_import_refuses_to_overwrite_without_force(tmp_path):
    source = _write(tmp_path / "qa.csv", "question\nOne?\n")
    import_dataset("csv", source, tmp_path / "out")
    with pytest.raises(ImportDatasetError, match="--force"):
        import_dataset("csv", source, tmp_path / "out")
    import_dataset("csv", source, tmp_path / "out", force=True)


def test_unknown_format_and_missing_input_are_clear_errors(tmp_path):
    with pytest.raises(ImportDatasetError, match="beir"):
        import_dataset("parquet", tmp_path / "x", tmp_path / "out")
    with pytest.raises(ImportDatasetError, match="not found"):
        import_dataset("csv", tmp_path / "missing.csv", tmp_path / "out")
    assert set(FORMATS) == {"csv", "beir", "qa-md"}


def test_cli_import_writes_the_dataset_and_reports_next_steps(tmp_path):
    source = _write(tmp_path / "qa.csv", "question,answer\nWhat?,That.\n")
    result = CliRunner().invoke(app, ["import", "--format", "csv", "--input", str(source), "--output", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "questions.jsonl").exists() and "ragbench init" in result.output
    bad = CliRunner().invoke(app, ["import", "--format", "nope", "--input", str(source), "--output", str(tmp_path / "o2")])
    assert bad.exit_code == 2
