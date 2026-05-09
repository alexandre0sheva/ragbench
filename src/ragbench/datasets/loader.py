from __future__ import annotations

from pathlib import Path

from ragbench.datasets.schema import Dataset, Qrel, Question
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
    return Dataset(questions=questions, qrels=qrels)

