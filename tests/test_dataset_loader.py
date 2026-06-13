import json

from ragbench.datasets.loader import load_dataset


def test_dataset_loader_loads_questions_and_derives_qrels(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Who?",
                "reference_answer": "A.",
                "expected_keywords": ["A"],
                "relevant_doc_ids": ["doc_001"],
                "category": "direct_fact",
                "difficulty": "easy",
                "answer_type": "single_fact",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(questions)

    assert len(dataset.questions) == 1
    assert dataset.questions[0].id == "q1"
    assert dataset.qrels["q1"]["doc_001"] == 1



def test_pdf_without_pypdf_raises_helpful_error(tmp_path):
    import pytest

    from ragbench.documents.loaders import load_documents

    try:
        import pypdf  # noqa: F401

        pytest.skip("pypdf installed; error path not reachable")
    except ImportError:
        pass
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    with pytest.raises(ImportError, match="ragbench\\[pdf\\]"):
        load_documents(tmp_path)


def test_rst_documents_are_loaded(tmp_path):
    from ragbench.documents.loaders import load_documents

    (tmp_path / "guide.rst").write_text("Title\n=====\n\nSome reStructuredText content.", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert "reStructuredText" in documents[0].text
