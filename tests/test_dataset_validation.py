from ragbench.datasets.schema import Dataset, Question
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.schema import Document


def _doc(doc_id: str, text: str = "Some content.") -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text)


def test_clean_dataset_has_no_warnings():
    questions = [Question(id="q1", question="What?", relevant_doc_ids=["doc_001"])]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_001": 1}})
    assert validate_dataset([_doc("doc_001")], dataset) == []


def test_missing_document_reference_is_flagged():
    questions = [Question(id="q1", question="What?", relevant_doc_ids=["doc_404"])]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_404": 1}})
    warnings = validate_dataset([_doc("doc_001")], dataset)
    assert any("doc_404" in warning for warning in warnings)


def test_duplicate_question_ids_are_flagged():
    questions = [
        Question(id="q1", question="A?", relevant_doc_ids=["doc_001"]),
        Question(id="q1", question="B?", relevant_doc_ids=["doc_001"]),
    ]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_001": 1}})
    warnings = validate_dataset([_doc("doc_001")], dataset)
    assert any("Duplicate" in warning for warning in warnings)


def test_qrels_for_unknown_question_is_flagged():
    questions = [Question(id="q1", question="A?", relevant_doc_ids=["doc_001"])]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_001": 1}, "q_ghost": {"doc_001": 1}})
    warnings = validate_dataset([_doc("doc_001")], dataset)
    assert any("q_ghost" in warning for warning in warnings)


def test_relevant_doc_missing_from_qrels_is_flagged():
    questions = [Question(id="q1", question="A?", relevant_doc_ids=["doc_001", "doc_002"])]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_001": 1}})
    warnings = validate_dataset([_doc("doc_001"), _doc("doc_002")], dataset)
    assert any("doc_002" in warning for warning in warnings)


def test_empty_document_is_flagged():
    questions = [Question(id="q1", question="A?", relevant_doc_ids=["doc_001"])]
    dataset = Dataset(questions=questions, qrels={"q1": {"doc_001": 1}})
    warnings = validate_dataset([_doc("doc_001"), _doc("doc_empty", text="   ")], dataset)
    assert any("doc_empty" in warning for warning in warnings)
