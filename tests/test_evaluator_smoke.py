"""End-to-end smoke test of the BenchmarkEvaluator on a tiny synthetic dataset.

Runs in mock mode (no API key) to exercise ingestion -> retrieval -> judge ->
report writing and verify the expected output files appear.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragbench.evaluation.evaluator import run_benchmark
from ragbench.utils.jsonl import write_jsonl


def _build_tiny_dataset(root: Path) -> None:
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs_dir / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    write_jsonl(
        root / "questions.jsonl",
        [
            {
                "id": "q_001",
                "question": "How much does HarborShield cost?",
                "reference_answer": "HarborShield costs $200 per month for the marine module.",
                "expected_keywords": ["HarborShield", "$200"],
                "relevant_doc_ids": ["doc_001"],
                "category": "direct_fact",
                "difficulty": "easy",
                "answer_type": "single_fact",
            },
            {
                "id": "q_002",
                "question": "When does ClaimPilot ship?",
                "reference_answer": "ClaimPilot ships in Q3.",
                "expected_keywords": ["ClaimPilot", "Q3"],
                "relevant_doc_ids": ["doc_002"],
                "category": "date_or_timeline",
                "difficulty": "easy",
                "answer_type": "single_fact",
            },
        ],
    )


def test_run_benchmark_mock_mode_produces_expected_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset_root = tmp_path / "data"
    _build_tiny_dataset(dataset_root)
    output_root = tmp_path / "results"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
run:
  name: smoke
  output_dir: {output_root}
dataset:
  documents_path: {dataset_root / "docs"}
  questions_path: {dataset_root / "questions.jsonl"}
systems:
  - type: bm25
    name: bm25_smoke
    chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}
    retrieval: {{top_k: 3}}
    models: {{generator: gpt-5.4-nano}}
evaluation:
  k_values: [1, 3]
  judge_enabled: true
  max_workers: 1
""",
        encoding="utf-8",
    )

    out = run_benchmark(config_path, force_mock=True, max_workers=1)

    assert out.exists()
    expected = [
        "leaderboard.md",
        "metrics_summary.csv",
        "per_question_results.jsonl",
        "retrieval_metrics.csv",
        "answer_metrics.csv",
        "cost_breakdown.csv",
        "system_runtime.csv",
        "qrels_audit.md",
        "qrels_audit.csv",
        "failures.md",
        "run_summary.json",
        "report.html",
        "config.yaml",
    ]
    for name in expected:
        assert (out / name).exists(), f"missing output: {name}"

    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["num_systems"] == 1
    assert summary["num_question_rows"] == 2
    assert summary["max_workers"] == 1


def test_run_benchmark_parallel_is_order_preserving(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    dataset_root = tmp_path / "data"
    _build_tiny_dataset(dataset_root)
    output_root = tmp_path / "results"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
run: {{name: parallel, output_dir: {output_root}}}
dataset:
  documents_path: {dataset_root / "docs"}
  questions_path: {dataset_root / "questions.jsonl"}
systems:
  - type: bm25
    name: bm25_parallel
    chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}
    retrieval: {{top_k: 3}}
evaluation:
  k_values: [1, 3]
  max_workers: 4
""",
        encoding="utf-8",
    )
    out = run_benchmark(config_path, force_mock=True, max_workers=4)
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines() if line.strip()]
    assert [r["question_id"] for r in rows] == ["q_001", "q_002"]
