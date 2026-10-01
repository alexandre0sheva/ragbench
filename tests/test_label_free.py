"""Label-free mode: questions without `relevant_doc_ids` and without qrels run, are judged on their answers, and get no retrieval metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Dataset, Question, has_relevance_labels
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.schema import Document
from ragbench.evaluation.answer_judge import heuristic_judge
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.judge_prompts import build_messages
from ragbench.utils.jsonl import write_jsonl


def _write_dataset(root: Path, questions: list[dict]) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    write_jsonl(root / "questions.jsonl", questions)


def _config(tmp_path: Path, root: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
run: {{name: label_free, output_dir: {tmp_path / "results"}}}
dataset: {{documents_path: {root / "docs"}, questions_path: {root / "questions.jsonl"}}}
systems:
  - {{type: bm25, name: bm25, chunker: {{type: word, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3}}}}
  - {{type: vector, name: vector, chunker: {{type: word, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3}}}}
evaluation: {{k_values: [1, 3], max_workers: 1, latency_probe_questions: 0}}
""",
        encoding="utf-8",
    )
    return path


# -- answerability -------------------------------------------------------------------------------


def test_legacy_answerability_is_inferred_from_relevant_doc_ids():
    assert Question(id="q", question="?", relevant_doc_ids=["doc_001"]).is_answerable is True
    assert Question(id="q", question="?").is_answerable is False


def test_explicit_answerable_wins_over_inference():
    assert Question(id="q", question="?", answerable=True).is_answerable is True
    assert Question(id="q", question="?", relevant_doc_ids=["doc_001"], answerable=False).is_answerable is False


def test_has_relevance_labels_counts_ids_and_positive_qrels_only():
    bare = Question(id="q", question="?")
    assert not has_relevance_labels(bare, {})
    assert not has_relevance_labels(bare, {"doc_001": 0})
    assert has_relevance_labels(bare, {"doc_001": 2})
    assert has_relevance_labels(Question(id="q", question="?", relevant_doc_ids=["doc_001"]), {})


def test_a_dataset_without_any_labels_is_label_free_and_its_questions_are_answerable(tmp_path):
    path = tmp_path / "q.jsonl"
    write_jsonl(path, [{"id": "q1", "question": "A?"}, {"id": "q2", "question": "B?", "answerable": False}])
    dataset = load_dataset(path)
    assert dataset.label_free
    assert [q.is_answerable for q in dataset.questions] == [True, False]  # unmarked: answerable; explicit false is kept
    assert not any(dataset.qrels.values())


def test_a_dataset_with_some_labels_keeps_the_legacy_inference(tmp_path):
    path = tmp_path / "q.jsonl"
    write_jsonl(path, [{"id": "q1", "question": "A?", "relevant_doc_ids": ["doc_001"]}, {"id": "q2", "question": "Nothing about this?"}])
    dataset = load_dataset(path)
    assert not dataset.label_free
    assert [q.is_answerable for q in dataset.questions] == [True, False]


def test_labels_in_a_qrels_file_alone_make_a_dataset_labeled_and_its_questions_answerable(tmp_path):
    questions, qrels = tmp_path / "q.jsonl", tmp_path / "qrels.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "A?"}])
    write_jsonl(qrels, [{"query_id": "q1", "doc_id": "doc_001", "relevance": 2}])
    dataset = load_dataset(questions, qrels)
    assert not dataset.label_free and dataset.questions[0].is_answerable is True  # labeled through the qrels, so it has relevant documents
    assert dataset.questions[0].relevant_doc_ids == []


def test_validation_warns_about_label_free_without_failing():
    docs = [Document(doc_id="doc_001", path="doc_001.md", title="t", text="Some content.")]
    dataset = Dataset(questions=[Question(id="q1", question="A?", answerable=True)], label_free=True)
    warnings = validate_dataset(docs, dataset)
    assert any("retrieval metrics" in warning for warning in warnings)


def test_validation_flags_answerable_questions_without_labels_in_a_labeled_dataset():
    docs = [Document(doc_id="doc_001", path="doc_001.md", title="t", text="Some content.")]
    questions = [Question(id="q1", question="A?", relevant_doc_ids=["doc_001"]), Question(id="q2", question="B?", answerable=True)]
    warnings = validate_dataset(docs, Dataset(questions=questions, qrels={"q1": {"doc_001": 1}}))
    assert any("q2" in warning and "no relevant_doc_ids" in warning for warning in warnings)


# -- the judge -----------------------------------------------------------------------------------


def test_the_judge_prompt_says_when_there_is_no_reference_instead_of_calling_the_question_unanswerable():
    question = Question(id="q", question="What is the price?", answerable=True)
    system, user = build_messages(question, "It is $200.", [])
    payload = json.loads(user["content"])
    assert payload["answerable"] is True and payload["reference_answer"] is None
    assert "answerable" in system["content"]
    assert "reference answer is null the question has no answer" not in system["content"]


def test_heuristic_judge_without_a_reference_grades_on_the_context_not_on_zero():
    from ragbench.rag_systems.base import RetrievedChunk

    context = [RetrievedChunk(chunk_id="c", doc_id="doc_001", text="HarborShield costs $200 per month for the marine module.", score=1.0, rank=1)]
    question = Question(id="q", question="How much does HarborShield cost?", answerable=True)
    supported = heuristic_judge(question, "HarborShield costs $200 per month.", context)
    invented = heuristic_judge(question, "Quantum entanglement powers the pricing engine nobody mentioned.", context)
    assert supported.correctness == supported.faithfulness > 3
    assert invented.correctness == invented.faithfulness < supported.faithfulness


# -- a run ---------------------------------------------------------------------------------------


def test_label_free_run_completes_with_blank_retrieval_columns_and_scored_answers(tmp_path):
    root = tmp_path / "data"
    _write_dataset(
        root,
        [
            {"id": "q1", "question": "How much does HarborShield cost?", "reference_answer": "HarborShield costs $200 per month."},
            {"id": "q2", "question": "When does ClaimPilot ship?"},  # no reference at all: judged on the context alone
        ],
    )
    out = run_benchmark(_config(tmp_path, root), force_mock=True, max_workers=1)

    retrieval = pd.read_csv(out / "retrieval_metrics.csv")
    assert {"recall@1", "recall@3", "ndcg@3"} <= set(retrieval.columns)
    assert retrieval[["recall@1", "recall@3", "mrr@3", "ndcg@3", "hit@3"]].isna().all().all()
    summary = pd.read_csv(out / "metrics_summary.csv")
    assert summary["retrieval_recall@3"].isna().all()
    assert summary["answer_score"].notna().all() and summary["faithfulness"].notna().all()
    answers = pd.read_csv(out / "answer_metrics.csv")
    assert answers["answer_score"].notna().all() and answers["answerable"].all()
    assert answers.loc[answers["question_id"] == "q1", "token_f1"].notna().all()  # the reference, where there is one, still feeds the lexical metrics
    assert answers.loc[answers["question_id"] == "q2", "token_f1"].isna().all()

    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(row["error"] is None and row["answerable"] is True for row in rows)
    assert all(row["retrieval_metrics"] == {} for row in rows)
    assert all(row["failure_type"] not in {"retrieval_miss", "bad_reranking", "run_error"} for row in rows)

    run_summary = json.loads((out / "run_summary.json").read_text())
    assert {key: run_summary["dataset"][key] for key in ("questions", "labeled_questions", "label_free")} == {"questions": 2, "labeled_questions": 0, "label_free": True}
    assert any("retrieval metrics" in warning for warning in run_summary["dataset_warnings"])


def test_labeled_questions_in_a_mixed_dataset_keep_their_retrieval_metrics(tmp_path):
    root = tmp_path / "data"
    _write_dataset(
        root,
        [
            {"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"]},
            {"id": "q2", "question": "When does ClaimPilot ship?", "answerable": True},
        ],
    )
    out = run_benchmark(_config(tmp_path, root), force_mock=True, max_workers=1)
    retrieval = pd.read_csv(out / "retrieval_metrics.csv")
    assert retrieval.loc[retrieval["question_id"] == "q1", "recall@3"].notna().all()
    assert retrieval.loc[retrieval["question_id"] == "q2", "recall@3"].isna().all()
    summary = pd.read_csv(out / "metrics_summary.csv")
    assert summary["retrieval_recall@3"].notna().all()  # the mean is over the labeled question only
    run_summary = json.loads((out / "run_summary.json").read_text())
    assert {key: run_summary["dataset"][key] for key in ("questions", "labeled_questions", "label_free")} == {"questions": 2, "labeled_questions": 1, "label_free": False}
