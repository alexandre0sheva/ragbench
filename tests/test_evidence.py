"""Span-level evidence in a dataset (`evidence: [{doc_id, quote}]`): loading it, `evidence_recall@k`, and datasets without it staying as they were."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Evidence, Question
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics, evidence_recall
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.jsonl import write_jsonl

PRICE = Evidence(doc_id="doc_001", quote="HarborShield costs $200 per month")
LAUNCH = Evidence(doc_id="doc_002", quote="ClaimPilot ships in Q3")


def _chunk(chunk_id: str, doc_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id=doc_id, text=text, score=1.0 / rank, rank=rank)


# -- the metric -----------------------------------------------------------------------------------


def test_evidence_recall_counts_quotes_found_in_the_top_k_chunks():
    ranking = [
        _chunk("a", "doc_001", "Pricing.  HarborShield costs $200\nper month for the marine module.", 1),
        _chunk("b", "doc_009", "Something unrelated.", 2),
        _chunk("c", "doc_002", "The roadmap says ClaimPilot ships in Q3 with triage.", 3),
    ]
    assert evidence_recall(ranking, [PRICE, LAUNCH], k=1) == 0.5  # whitespace and case do not matter
    assert evidence_recall(ranking, [PRICE, LAUNCH], k=2) == 0.5
    assert evidence_recall(ranking, [PRICE, LAUNCH], k=3) == 1.0


def test_a_quote_only_counts_in_the_document_it_was_taken_from():
    ranking = [_chunk("a", "doc_777", "HarborShield costs $200 per month", 1)]
    assert evidence_recall(ranking, [PRICE], k=5) == 0.0


def test_a_document_that_is_relevant_does_not_make_its_wrong_chunk_evidence():
    # Document-level recall is satisfied by any chunk of doc_001; evidence recall needs the chunk that holds the quote.
    ranking = [_chunk("a", "doc_001", "An unrelated paragraph of the pricing page.", 1)]
    metrics = compute_retrieval_metrics(ranking, ["doc_001"], k_values=[1], evidence=[PRICE])
    assert metrics["recall@1"] == 1.0
    assert metrics["evidence_recall@1"] == 0.0


def test_metrics_without_evidence_have_no_evidence_keys():
    ranking = [_chunk("a", "doc_001", "text", 1)]
    assert not any("evidence" in key for key in compute_retrieval_metrics(ranking, ["doc_001"], k_values=[1, 3]))


# -- loading --------------------------------------------------------------------------------------


def _write(root: Path, questions: list[dict], qrels: list[dict] | None = None) -> tuple[Path, Path | None]:
    write_jsonl(root / "questions.jsonl", questions)
    if qrels is None:
        return root / "questions.jsonl", None
    write_jsonl(root / "qrels.jsonl", qrels)
    return root / "questions.jsonl", root / "qrels.jsonl"


def test_evidence_on_a_question_is_loaded(tmp_path):
    questions, _ = _write(tmp_path, [{"id": "q1", "question": "Price?", "relevant_doc_ids": ["doc_001"], "evidence": [PRICE.model_dump()]}])
    assert load_dataset(questions).questions[0].evidence == [PRICE]


def test_evidence_in_a_qrels_row_is_attached_to_its_question(tmp_path):
    questions, qrels = _write(
        tmp_path,
        [{"id": "q1", "question": "Price?", "relevant_doc_ids": ["doc_001"]}],
        [{"query_id": "q1", "doc_id": "doc_001", "relevance": 1, "evidence": [{"doc_id": "doc_001", "quote": PRICE.quote}]}],
    )
    dataset = load_dataset(questions, qrels)
    assert dataset.questions[0].evidence == [PRICE]
    assert dataset.qrels == {"q1": {"doc_001": 1}}


def test_a_question_without_evidence_has_an_empty_list():
    assert Question(id="q", question="?").evidence == []


def test_validation_warns_about_a_quote_missing_from_its_document(tmp_path):
    questions, _ = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "Price?",
                "relevant_doc_ids": ["doc_001"],
                "evidence": [PRICE.model_dump(), {"doc_id": "doc_001", "quote": "never written"}, {"doc_id": "doc_404", "quote": "x"}],
            }
        ],
    )
    documents = [
        Document(doc_id="doc_001", title="Pricing", text="HarborShield costs $200 per month for the marine module.", path="doc_001.md")
    ]
    warnings = validate_dataset(documents, load_dataset(questions))
    assert any("q1" in w and "never written" in w for w in warnings)
    assert any("q1" in w and "doc_404" in w for w in warnings)
    assert not any("HarborShield" in w for w in warnings)


# -- a run ----------------------------------------------------------------------------------------


def _run(tmp_path: Path, questions: list[dict]) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    write_jsonl(tmp_path / "questions.jsonl", questions)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
run: {{name: evidence, output_dir: {tmp_path / "results"}}}
dataset: {{documents_path: {docs}, questions_path: {tmp_path / "questions.jsonl"}}}
systems:
  - {{type: bm25, name: bm25, chunker: {{type: word, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3}}}}
  - {{type: no_retrieval, name: floor}}
evaluation: {{k_values: [1, 3], max_workers: 1, latency_probe_questions: 0}}
""",
        encoding="utf-8",
    )
    return run_benchmark(config, force_mock=True)


def test_a_run_reports_evidence_recall_only_for_questions_that_have_evidence(tmp_path):
    out = _run(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "How much does HarborShield cost per month?",
                "relevant_doc_ids": ["doc_001"],
                "evidence": [PRICE.model_dump()],
            },
            {"id": "q2", "question": "When does ClaimPilot ship?", "relevant_doc_ids": ["doc_002"]},
        ],
    )
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert summary.loc["bm25", "retrieval_evidence_recall@3"] == 1.0  # q2 has no evidence: it is left out of the mean, not counted as 0
    assert pd.isna(summary.loc["floor", "retrieval_evidence_recall@3"])  # a system that retrieves nothing has no evidence recall, not 0
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    q2 = next(r for r in rows if r["system"] == "bm25" and r["question_id"] == "q2")
    assert "evidence_recall@3" not in q2["retrieval_metrics"]


def test_a_run_without_evidence_has_no_evidence_columns(tmp_path):
    out = _run(tmp_path, [{"id": "q1", "question": "How much does HarborShield cost per month?", "relevant_doc_ids": ["doc_001"]}])
    assert not any("evidence" in column for column in pd.read_csv(out / "metrics_summary.csv").columns)
