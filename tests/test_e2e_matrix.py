"""End to end, offline, for every registered system and every chunker: the run completes, writes every output, has no holes in the numbers it
promises, its step trace adds up to its cost, and two runs give the same answers. A system or chunker added later is covered automatically."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from dataset_support import DOCUMENTS
from paid_fakes import EMBEDDING, GENERATOR, JUDGE, PRICING
from paid_fakes import install as install_paid_fakes

import ragbench.documents.chunkers  # noqa: F401  (registers the built-in chunkers)
import ragbench.rag_systems  # noqa: F401
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.rag_systems import all_specs
from ragbench.rag_systems.components import build_chunker, chunk_documents
from ragbench.registry import CHUNKERS, SYSTEMS
from ragbench.utils.jsonl import read_jsonl

WORD = {"type": "word", "chunk_size": 30, "chunk_overlap": 0}
MODELS = {"generator": GENERATOR, "embedding": EMBEDDING}
OUTPUTS = (
    "metrics_summary.csv",
    "per_question_results.jsonl",
    "retrieval_metrics.csv",
    "answer_metrics.csv",
    "cost_breakdown.csv",
    "leaderboard.md",
    "failures.md",
    "qrels_audit.md",
    "recommendation.json",
    "run_summary.json",
    "run_manifest.json",
    "significance.csv",
    "report.html",
    "report_data.json",
)
SPECS = {spec.type: spec for spec in all_specs()}
ALWAYS_PRESENT = ("system", "system_type", "n_ok", "n_error", "answer_score", "faithfulness", "avg_cost_per_question", "avg_latency_ms")


def system_block(kind: str) -> dict[str, Any]:
    """A minimal valid config block for any registered system type."""
    block: dict[str, Any] = {"type": kind, "name": kind, "models": dict(MODELS)}
    spec = SPECS.get(kind)
    if spec is not None and spec.chunker is not None and kind != "parent_doc":
        block["chunker"] = dict(WORD)
    if kind == "adaptive":
        block["retrieval"] = {"routes": {"default": {"type": "vector"}, "lexical": {"type": "bm25"}, "computation": {"type": "agent_search", "tools": ["calculator", "date_calc"]}}}
    return block


def run_once(root: Path, systems: list[dict[str, Any]], dataset: dict[str, str], tag: str) -> Path:
    config = {
        "run": {"name": f"e2e_{tag}", "output_dir": str(root / "results")},
        "dataset": dataset,
        "systems": systems,
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0, "judge_model": JUDGE},
        "pricing": PRICING,
    }
    path = root / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(path, force_mock=False, max_workers=1, use_cache=False, run_dir=root / "results" / tag)


def _stable(value: Any) -> Any:
    """A per-question row without the things that legitimately differ between two runs: timings."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if "latency" not in k and not k.endswith("_ms")}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@pytest.mark.parametrize("kind", sorted(SYSTEMS.names()))
def test_every_registered_system_runs_end_to_end_and_is_reproducible(kind, tmp_path, tiny_dataset, monkeypatch):
    install_paid_fakes(monkeypatch)
    first = run_once(tmp_path, [system_block(kind)], tiny_dataset, "a")
    second = run_once(tmp_path, [system_block(kind)], tiny_dataset, "b")

    for name in OUTPUTS:
        assert (first / name).exists(), f"{kind}: {name} was not written"

    summary = pd.read_csv(first / "metrics_summary.csv")
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_error"] == 0 and row["n_ok"] == 8, f"{kind}: {row['n_error']} questions failed"
    holes = [column for column in ALWAYS_PRESENT if _is_missing(row[column])]
    assert not holes, f"{kind}: no value for {holes}"
    spec = SPECS[kind]
    retrieval_columns = [c for c in summary.columns if c.startswith(("retrieval_recall@", "retrieval_mrr@", "retrieval_ndcg@")) and "_ci_" not in c]
    assert retrieval_columns
    for column in retrieval_columns:
        assert _is_missing(row[column]) is (not spec.retrieves), f"{kind}: {column} should {'be blank' if not spec.retrieves else 'have a value'}"
    assert row["avg_cost_per_question"] > 0  # the paid fakes charge for every model call, so a free system would mean something is not counted

    rows = read_jsonl(first / "per_question_results.jsonl")
    assert len(rows) == 8 and all(r["error"] is None and r["steps"] is not None and r["answer"].strip() for r in rows), kind
    for r in rows:
        answer_cost = r["cost"]["total_cost"] - r["cost"]["judge_cost"]
        assert sum(step["cost"]["total_cost"] for step in r["steps"]) == pytest.approx(answer_cost, abs=1e-9), f"{kind}/{r['question_id']}: steps do not add up to the cost"
    again = read_jsonl(second / "per_question_results.jsonl")
    assert [_stable(r) for r in rows] == [_stable(r) for r in again], f"{kind}: two runs differ"
    columns = [c for c in summary.columns if "latency" not in c and "wall_time" not in c and not c.endswith("_ms")]
    pd.testing.assert_frame_equal(summary[columns], pd.read_csv(second / "metrics_summary.csv")[columns])

    report = json.loads((first / "report_data.json").read_text(encoding="utf-8"))
    assert [r["system"] for r in report["leaderboard"]["rows"]] == [kind]


def test_the_matrix_covers_every_registered_system():
    assert set(SPECS) == set(SYSTEMS.names()), "a system has no SystemSpec"


# -- chunkers -------------------------------------------------------------------------------------------------------------------------------


CHUNKER_OPTIONS = {"fixed_char": {"chunk_size": 120, "chunk_overlap": 10}, "token": {"chunk_size": 30, "chunk_overlap": 0}, "sentence": {"chunk_size": 30}, "semantic": {}}


def _chunker(kind: str) -> dict[str, Any]:
    return {"type": kind, **({"chunk_size": 30, "chunk_overlap": 0} if kind in {"word", "recursive", "markdown"} else CHUNKER_OPTIONS.get(kind, {}))}


@pytest.mark.parametrize("kind", sorted(CHUNKERS.names()))
def test_every_chunker_cuts_the_corpus_sanely_and_runs_end_to_end(kind, tmp_path, tiny_dataset, monkeypatch):
    from ragbench.config.schema import ChunkerConfig

    install_paid_fakes(monkeypatch)
    documents = [Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text) for doc_id, text in DOCUMENTS.items()]
    chunks, _ = chunk_documents(build_chunker(ChunkerConfig.model_validate(_chunker(kind)), models=dict(MODELS)), documents)
    assert chunks, f"{kind} produced no chunks"
    by_doc = {c.doc_id for c in chunks}
    assert by_doc == set(DOCUMENTS), f"{kind} dropped a document: {set(DOCUMENTS) - by_doc}"
    assert all(c.text.strip() for c in chunks), f"{kind} produced an empty chunk"
    assert len({c.chunk_id for c in chunks}) == len(chunks), f"{kind} produced duplicate chunk ids"
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_documents(build_chunker(ChunkerConfig.model_validate(_chunker(kind)), models=dict(MODELS)), documents)[0]]

    block = {"type": "bm25", "name": f"bm25_{kind}", "models": dict(MODELS), "chunker": _chunker(kind)}
    run = run_once(tmp_path, [block], tiny_dataset, "chunker")
    summary = pd.read_csv(run / "metrics_summary.csv").iloc[0]
    assert summary["n_error"] == 0 and not _is_missing(summary["retrieval_recall@5"]) and summary["retrieval_recall@5"] > 0, f"{kind}: bm25 over its chunks finds nothing"
