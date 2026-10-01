from __future__ import annotations

from collections import Counter

from ragbench.datasets.schema import Dataset, has_relevance_labels
from ragbench.documents.schema import Document


def validate_dataset(documents: list[Document], dataset: Dataset) -> list[str]:
    """Return human-readable warnings about dataset inconsistencies.

    Warnings are non-fatal: the benchmark will run, but each one tends to
    silently distort retrieval metrics, so the CLI surfaces them up front.
    """
    warnings: list[str] = []
    doc_ids = {document.doc_id for document in documents}
    question_ids = [question.id for question in dataset.questions]

    duplicates = [question_id for question_id, count in Counter(question_ids).items() if count > 1]
    if duplicates:
        warnings.append(f"Duplicate question ids: {', '.join(sorted(duplicates))}")

    for question in dataset.questions:
        unknown = sorted(set(question.relevant_doc_ids) - doc_ids)
        if unknown:
            warnings.append(f"Question {question.id} references missing documents: {', '.join(unknown)}")

    known_questions = set(question_ids)
    for query_id, doc_relevances in dataset.qrels.items():
        if query_id not in known_questions:
            warnings.append(f"Qrels reference unknown question id: {query_id}")
        unknown_docs = sorted(set(doc_relevances) - doc_ids)
        if unknown_docs:
            warnings.append(f"Qrels for {query_id} reference missing documents: {', '.join(unknown_docs)}")

    for question in dataset.questions:
        if not question.is_answerable:
            continue
        qrel_docs = set(dataset.qrels.get(question.id, {}))
        labeled = set(question.relevant_doc_ids)
        if qrel_docs and labeled - qrel_docs:
            missing = ", ".join(sorted(labeled - qrel_docs))
            warnings.append(f"Question {question.id}: relevant_doc_ids not present in qrels: {missing}")

    if dataset.label_free and dataset.questions:
        warnings.append(
            "No question has relevant_doc_ids or qrels (label-free mode): retrieval metrics (recall, MRR, nDCG) are skipped and answers are judged "
            "against the reference answer, or against the retrieved context when there is none. Add labels to compare retrieval quality."
        )
    else:
        unlabeled = [q.id for q in dataset.questions if q.is_answerable and not has_relevance_labels(q, dataset.qrels.get(q.id, {}))]
        if unlabeled:
            shown = ", ".join(unlabeled[:5]) + (f" and {len(unlabeled) - 5} more" if len(unlabeled) > 5 else "")
            warnings.append(f"Answerable questions with no relevant_doc_ids or qrels get no retrieval metrics: {shown}")

    empty_docs = sorted(document.doc_id for document in documents if not document.text.strip())
    if empty_docs:
        warnings.append(f"Documents with empty text (will produce no chunks): {', '.join(empty_docs)}")

    return warnings
