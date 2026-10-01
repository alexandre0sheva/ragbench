from __future__ import annotations

import pandas as pd
import pytest
from fake_systems import question, register_fakes, write_experiment

from ragbench.evaluation.context_metrics import context_precision, context_recall_doc
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics
from ragbench.rag_systems.base import RetrievedChunk


def _chunks(*doc_ids: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"{doc}::chunk::{i}", doc_id=doc, text=doc, score=1.0, rank=i + 1) for i, doc in enumerate(doc_ids)]


def test_context_recall_is_the_share_of_relevant_documents_the_llm_saw():
    assert context_recall_doc(_chunks("a", "b", "c"), ["a", "c"]) == 1.0
    assert context_recall_doc(_chunks("a", "b", "c"), ["a", "z"]) == 0.5
    assert context_recall_doc(_chunks("b", "c"), ["a"]) == 0.0
    assert context_recall_doc([], ["a"]) == 0.0  # nothing was given, so no evidence was seen
    assert context_recall_doc(_chunks("a", "a", "a"), ["a"]) == 1.0  # repeated chunks of one document count once


def test_context_precision_is_the_share_of_given_chunks_from_relevant_documents():
    assert context_precision(_chunks("a", "b", "a", "c"), ["a"]) == 0.5
    assert context_precision(_chunks("a"), ["a"]) == 1.0
    assert context_precision(_chunks("b"), ["a"]) == 0.0


def test_context_metrics_are_undefined_without_relevant_documents_or_context():
    assert context_recall_doc(_chunks("a"), []) is None  # an unanswerable question has no evidence to see
    assert context_precision(_chunks("a"), []) is None
    assert context_precision([], ["a"]) is None  # no chunks, so no share


def test_context_recall_differs_from_retrieval_recall_when_context_k_is_smaller_than_the_ranking():
    chunks = _chunks(*[f"doc_{i:03d}" for i in range(1, 11)])
    retrieval = compute_retrieval_metrics(chunks, ["doc_007"], k_values=[5, 10])
    assert retrieval["recall@10"] == 1.0  # the ranking found it ...
    assert context_recall_doc(chunks[:5], ["doc_007"]) == 0.0  # ... but the generator was only given the top five


def test_run_reports_what_the_generator_actually_saw(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    # The relevant document is ranked 7th and the generator reads 5 chunks: found by retrieval, never seen by the LLM.
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 7?", ["doc_007"]), question("q2", "What is topic 2?", ["doc_002"])],
        [{"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 5}}],
        {"k_values": [1, 5, 10]},
    )
    out = run_benchmark(config, force_mock=True)

    answers = pd.read_csv(out / "answer_metrics.csv").set_index("question_id")
    assert answers.loc["q1", "context_recall"] == 0.0 and answers.loc["q2", "context_recall"] == 1.0
    assert answers.loc["q2", "context_precision"] == pytest.approx(1 / 5)
    summary = pd.read_csv(out / "metrics_summary.csv").iloc[0]
    assert summary["context_recall"] == pytest.approx(0.5) and summary["retrieval_recall@10"] == 1.0
    failures = {row["question_id"]: row["failure_type"] for row in pd.read_csv(out / "answer_metrics.csv").to_dict("records")}
    assert failures["q1"] == "bad_reranking"  # in the ranking, below the cut-off the generator reads

