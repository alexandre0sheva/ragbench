from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ragbench.datasets.schema import Dataset, Evidence, Qrel, Question, has_relevance_labels
from ragbench.datasets.splits import SPLITS_NAME, apply_split, load_splits
from ragbench.utils.jsonl import read_jsonl

if TYPE_CHECKING:
    from ragbench.config.schema import DatasetConfig


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


def load_qrel_evidence(path: Path | None) -> dict[str, list[Evidence]]:
    """The `evidence` carried by rows of a qrels file, by question id (empty when there is no file or no row has any)."""
    found: dict[str, list[Evidence]] = {}
    if path and path.exists():
        for row in read_jsonl(path):
            qrel = Qrel.model_validate(row)
            found.setdefault(qrel.query_id, []).extend(qrel.evidence)
    return {query_id: evidence for query_id, evidence in found.items() if evidence}


def load_dataset(questions_path: Path, qrels_path: Path | None = None) -> Dataset:
    questions = load_questions(questions_path)
    for query_id, evidence in load_qrel_evidence(qrels_path).items():
        for question in questions:
            if question.id == query_id:
                question.evidence = [*question.evidence, *(item for item in evidence if item not in question.evidence)]
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


def configured_splits_path(config: DatasetConfig) -> Path:
    """Where the dataset's `splits.json` is: `dataset.splits_path`, or next to the questions."""
    return config.splits_path or config.questions_path.parent / SPLITS_NAME


def load_configured_dataset(config: DatasetConfig) -> tuple[Dataset, dict[str, str] | None]:
    """The questions a run uses: `dataset.split` applied to the loaded dataset, and `{"name", "hash"}` of that split (None for `all`)."""
    dataset = load_dataset(config.questions_path, config.qrels_path)
    if config.split == "all":
        return dataset, None
    splits = load_splits(configured_splits_path(config))
    return apply_split(dataset, splits, config.split), {"name": config.split, "hash": splits.digest}
