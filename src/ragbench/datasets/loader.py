from __future__ import annotations

from pathlib import Path

from ragbench.datasets.schema import Dataset, Qrel, Question, has_relevance_labels
from ragbench.utils.jsonl import read_jsonl


def load_questions(path: Path) -> list[Question]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    return [Question.model_validate(row) for row in read_jsonl(path)]


def load_qrels(path: Path | None, questions: list[Question]) -> dict[str, dict[str, int]]:
    if path and path.exists():
        qrels: dict[str, dict[str, int]] = {}
        for row in read_jsonl(path):
            qrel = Qrel.model_validate(row)
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        return qrels
    return {q.id: {doc_id: 1 for doc_id in q.relevant_doc_ids} for q in questions}


def load_dataset(questions_path: Path, qrels_path: Path | None = None) -> Dataset:
    questions = load_questions(questions_path)
    qrels = load_qrels(qrels_path, questions)
    dataset = Dataset(questions=questions, qrels=qrels)
    if questions and not any(has_relevance_labels(question, qrels.get(question.id, {})) for question in questions):
        # Nothing is labeled, so "no relevant documents" cannot mean "unanswerable": questions not marked `answerable: false` are assumed to have answers.
        dataset.label_free = True
        dataset.questions = [question if question.answerable is not None else question.model_copy(update={"answerable": True}) for question in questions]
    else:
        # A question labeled only through the qrels file (no `relevant_doc_ids`, as `ragbench label --apply` writes them) has relevant documents, so it has an answer.
        dataset.questions = [
            question.model_copy(update={"answerable": True})
            if question.answerable is None and not question.relevant_doc_ids and has_relevance_labels(question, qrels.get(question.id, {}))
            else question
            for question in questions
        ]
    return dataset
