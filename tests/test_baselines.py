"""Baselines (`no_retrieval`, `full_context`), `sentence_window`, and the MMR diversity option."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import count_tokens
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.stores.mmr import mmr_select
from ragbench.utils.jsonl import write_jsonl


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text)


def _system(type_: str, force_mock: bool = True, **sections):
    return create_rag_system(SystemConfig(type=type_, **sections), force_mock=force_mock)


TINY = [
    _doc("doc_001", "# Pricing\n\nHarborShield costs $200 per month for the marine module."),
    _doc("doc_002", "# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows."),
    _doc("doc_003", "# Support\n\nSeverity 1 issues are acknowledged within 30 minutes."),
]


# --- every new system ingests and answers ----------------------------------------------------------------------------


@pytest.mark.parametrize("type_", ["no_retrieval", "full_context", "sentence_window"])
def test_each_new_system_ingests_and_answers_on_a_tiny_corpus(type_):
    system = _system(type_)

    ingestion = system.ingest(TINY)
    answer = system.answer_question("How much does HarborShield cost?")

    assert ingestion.num_documents == 3 and answer.answer and answer.steps
    assert answer.steps[-1].kind == "generate"
    assert system.spec is not None and system.spec.type == type_


def test_only_no_retrieval_declares_that_it_does_not_retrieve():
    flags = {spec.type: spec.retrieves for spec in all_specs()}

    assert flags["no_retrieval"] is False
    assert all(flag for name, flag in flags.items() if name != "no_retrieval")
    assert {"no_retrieval", "full_context", "sentence_window"} <= set(flags)


def test_list_systems_shows_the_new_systems():
    result = CliRunner().invoke(app, ["list-systems"])

    assert result.exit_code == 0
    for name in ("no_retrieval", "full_context", "sentence_window"):
        assert name in result.output


# --- no_retrieval --------------------------------------------------------------------------------------------------------


def test_no_retrieval_returns_no_chunks_costs_no_embeddings_and_never_shows_documents_to_the_model():
    system = _system("no_retrieval")

    ingestion = system.ingest(TINY)
    result = system.fetch_context("How much does HarborShield cost?", top_k=5)
    answer = system.answer_question("How much does HarborShield cost?")

    assert ingestion.num_chunks == 0 and ingestion.cost.embedding_input_tokens == 0
    assert result.chunks == []
    assert [step.kind for step in answer.steps] == ["retrieve", "generate"] and answer.steps[0].cost.total_cost == 0.0
    assert answer.retrieval_result.chunks == [] and answer.metadata["context_chunk_ids"] == []


def test_no_retrieval_prompt_asks_for_parametric_knowledge_only(monkeypatch):
    system = _system("no_retrieval")
    seen: list[str] = []
    original = system.llm.generate

    def spy(messages, **kwargs):
        seen.append(" ".join(m["content"] for m in messages))
        return original(messages, **kwargs)

    monkeypatch.setattr(system.llm, "generate", spy)
    system.ingest(TINY)
    system.answer_question("What is the capital of France?")

    assert "own knowledge" in seen[0] and "provided context" not in seen[0].lower()


@pytest.mark.parametrize("section", [{"retrieval": {"top_k": 3}}, {"chunker": {"type": "word"}}])
def test_no_retrieval_rejects_sections_it_would_ignore(section):
    with pytest.raises(ValueError, match="no_retrieval"):
        SystemConfig(type="no_retrieval", **section)


# --- full_context ---------------------------------------------------------------------------------------------------------


def _long_corpus() -> list[Document]:
    filler = " ".join(f"word{i}" for i in range(120))
    return [
        _doc("doc_a", f"# Alpha\n\nAlpha handbook about onboarding. {filler}"),
        _doc("doc_b", f"# Bravo\n\nBravo handbook about zebra migration patterns. {filler}"),
        _doc("doc_c", f"# Charlie\n\nCharlie handbook about invoices. {filler}"),
    ]


def test_full_context_gives_the_model_every_document_when_they_fit_and_orders_them_by_bm25():
    system = _system("full_context", retrieval={"context_token_budget": 100_000})
    system.ingest(_long_corpus())

    result = system.fetch_context("zebra migration", top_k=2)

    assert [c.doc_id for c in result.chunks][0] == "doc_b"
    assert {c.doc_id for c in result.chunks} == {"doc_a", "doc_b", "doc_c"}, "top_k is a retrieval depth; the whole corpus is in context"
    assert result.metadata["truncated"] is False and result.metadata["documents_included"] == 3 and result.metadata["documents_total"] == 3
    assert system.configured_context_k() >= 3


def test_full_context_truncates_deterministically_when_over_budget_and_says_so():
    docs = _long_corpus()
    one_doc_tokens = count_tokens(docs[0].text)
    budget = int(one_doc_tokens * 1.5)
    first, second = _system("full_context", retrieval={"context_token_budget": budget}), _system("full_context", retrieval={"context_token_budget": budget})
    first.ingest(docs)
    second.ingest(list(reversed(docs)))  # input order must not matter

    a = first.fetch_context("zebra migration")
    b = second.fetch_context("zebra migration")

    assert [(c.chunk_id, c.text) for c in a.chunks] == [(c.chunk_id, c.text) for c in b.chunks]
    assert a.metadata["truncated"] is True and a.metadata["documents_included"] < a.metadata["documents_total"]
    assert a.chunks[0].doc_id == "doc_b" and a.chunks[0].metadata.get("truncated") is not True, "the best document is kept whole"
    assert a.metadata["context_tokens"] <= budget
    assert sum(count_tokens(c.text) for c in a.chunks) <= budget
    assert any(c.metadata.get("truncated") for c in a.chunks), "the document that did not fit is cut, not skipped silently"


def test_full_context_cuts_the_first_document_when_even_it_does_not_fit():
    system = _system("full_context", retrieval={"context_token_budget": 40})
    system.ingest(_long_corpus())

    result = system.fetch_context("zebra migration")

    assert len(result.chunks) == 1 and result.chunks[0].metadata["truncated"] is True and result.metadata["truncated"] is True
    assert count_tokens(result.chunks[0].text) <= 40 and result.chunks[0].text.strip()


def test_full_context_pays_for_the_tokens_it_reads():
    docs = _long_corpus()
    full, narrow = _system("full_context", retrieval={"context_token_budget": 100_000}), _system("bm25", retrieval={"top_k": 1})
    full.ingest(docs)
    narrow.ingest(docs)

    full_answer = full.answer_question("zebra migration")
    narrow_answer = narrow.answer_question("zebra migration")

    assert full_answer.token_usage["prompt_tokens"] > 2 * narrow_answer.token_usage["prompt_tokens"]


def test_full_context_budget_must_be_positive():
    with pytest.raises(ValueError, match="context_token_budget"):
        SystemConfig(type="full_context", retrieval={"context_token_budget": 0})


# --- sentence_window --------------------------------------------------------------------------------------------------------

SENTENCES = [
    "Alpha opens the document with a generic remark.",
    "Beta mentions the zebra marker exactly once.",
    "Gamma follows with an unrelated remark.",
    "Delta adds more unrelated filler text.",
    "Epsilon closes the document with a summary.",
]


def _sentence_corpus() -> list[Document]:
    return [_doc("doc_x", "# Notes\n\n" + " ".join(SENTENCES)), _doc("doc_y", "Omega is a separate document about invoices and billing.")]


def _window_texts(window: int, query: str = "zebra marker") -> list[str]:
    system = _system("sentence_window", retrieval={"window": window, "top_k": 1})
    system.ingest(_sentence_corpus())
    return [chunk.text for chunk in system.fetch_context(query, top_k=1).chunks]


def test_sentence_window_returns_the_matched_sentence_plus_its_neighbours():
    zero, one = _window_texts(0)[0], _window_texts(1)[0]

    assert SENTENCES[1] in zero and SENTENCES[0] not in zero and SENTENCES[2] not in zero
    assert all(s in one for s in SENTENCES[:3]) and SENTENCES[3] not in one


def test_sentence_window_is_clipped_at_document_edges_and_never_crosses_documents():
    system = _system("sentence_window", retrieval={"window": 3, "top_k": 1})
    system.ingest(_sentence_corpus())

    chunk = system.fetch_context("Omega invoices billing", top_k=1).chunks[0]

    assert chunk.doc_id == "doc_y" and "zebra" not in chunk.text


def test_sentence_window_merges_overlapping_windows_from_the_same_document():
    text = " ".join(["Alpha intro line.", "Zebra appears first.", "Plain middle line.", "Zebra appears second.", "Plain closing line."])
    system = _system("sentence_window", retrieval={"window": 1, "top_k": 5})
    system.ingest([_doc("doc_z", text)])

    chunks = system.fetch_context("zebra", top_k=5).chunks

    assert len(chunks) == 1 and "Alpha intro line." in chunks[0].text and "Plain closing line." in chunks[0].text
    assert chunks[0].metadata["matched_sentences"] == 2


def test_sentence_window_chunks_point_back_into_the_source_text():
    system = _system("sentence_window", retrieval={"window": 1, "top_k": 1})
    docs = _sentence_corpus()
    system.ingest(docs)

    chunk = system.fetch_context("zebra marker", top_k=1).chunks[0]

    assert docs[0].text[chunk.metadata["start_char"] : chunk.metadata["end_char"]].strip() == chunk.text.strip()


def test_sentence_window_rejects_a_negative_window():
    with pytest.raises(ValueError, match="window"):
        SystemConfig(type="sentence_window", retrieval={"window": -1})


# --- MMR ---------------------------------------------------------------------------------------------------------------------


def test_mmr_select_equals_plain_ranking_at_lambda_one_and_penalises_duplicates_below_it():
    vectors = np.array([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    relevance = np.array([1.0, 0.95, 0.5])

    assert mmr_select(vectors, relevance, k=3, lam=1.0) == [0, 1, 2]
    assert mmr_select(vectors, relevance, k=3, lam=0.5) == [0, 2, 1], "the near-duplicate of the first pick drops behind the different one"
    assert mmr_select(vectors, relevance, k=2, lam=0.5) == [0, 2]
    assert mmr_select(vectors, relevance, k=10, lam=0.5) == [0, 2, 1] and mmr_select(vectors[:0], relevance[:0], k=3, lam=0.5) == []


def _duplicate_heavy_corpus() -> list[Document]:
    docs = [
        _doc("doc_dup0", "Refund policy: customers may request a refund within 14 days of purchase for unused items."),
        _doc("doc_dup1", "Refund policy details: customers may request a refund within 14 days of purchase for unused items and returns."),
        _doc("doc_dup2", "Refund policy summary for all customers: customers may request a refund within 14 days of purchase for unused items."),
        _doc("doc_dmg", "Damaged goods: if an item arrives damaged the refund also covers shipping costs. Report damage within 7 days."),
    ]
    fillers = [
        "The cafeteria opens at 8 am and serves lunch until 2 pm.",
        "Office parking permits are issued by the facilities team every January.",
        "Quarterly planning meetings are held in the main auditorium.",
        "Laptops are refreshed every three years by the IT desk.",
        "The annual picnic takes place in the park near the lake.",
        "Bicycle racks are located in the basement garage.",
    ]
    return docs + [_doc(f"doc_f{i}", text) for i, text in enumerate(fillers, start=1)]


def _docs_returned(system_type: str, **retrieval) -> list[str]:
    system = _system(system_type, chunker={"type": "word", "chunk_size": 100, "chunk_overlap": 0}, retrieval={"top_k": 2, **retrieval})
    system.ingest(_duplicate_heavy_corpus())
    return [c.doc_id for c in system.fetch_context("refund policy", top_k=2).chunks]


@pytest.mark.parametrize("system_type", ["vector", "hybrid"])
def test_mmr_returns_diverse_documents_on_a_duplicate_heavy_corpus(system_type):
    plain = _docs_returned(system_type)
    diverse = _docs_returned(system_type, diversity="mmr", mmr_lambda=0.5)

    assert all(doc.startswith("doc_dup") for doc in plain), "without MMR the near-duplicates crowd out the other answer"
    assert "doc_dmg" in diverse and sum(doc.startswith("doc_dup") for doc in diverse) == 1


@pytest.mark.parametrize("system_type", ["vector", "hybrid"])
def test_mmr_at_lambda_one_is_identical_to_plain_ranking(system_type):
    assert _docs_returned(system_type, diversity="mmr", mmr_lambda=1.0) == _docs_returned(system_type)


def test_mmr_records_a_rerank_step_and_leaves_the_default_untouched():
    system = _system("vector", retrieval={"top_k": 2, "diversity": "mmr", "mmr_lambda": 0.5})
    system.ingest(_duplicate_heavy_corpus())
    answer = system.answer_question("refund policy")

    assert [step.kind for step in answer.steps] == ["retrieve", "rerank", "generate"] and answer.steps[1].name == "mmr"
    assert _system("vector").options.diversity == "none"


def test_mmr_options_are_validated_and_not_silently_accepted_by_other_systems():
    with pytest.raises(ValueError, match="mmr_lambda"):
        SystemConfig(type="vector", retrieval={"diversity": "mmr", "mmr_lambda": 1.5})
    with pytest.raises(ValueError, match="diversity"):
        SystemConfig(type="hybrid_rerank", retrieval={"diversity": "mmr"})


# --- evaluator and reports: no retrieval means blank, not zero ----------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_run(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("baselines_run")
    docs = root / "docs"
    docs.mkdir()
    for document in TINY:
        (docs / f"{document.doc_id}.md").write_text(document.text + "\n", encoding="utf-8")
    write_jsonl(
        root / "questions.jsonl",
        [
            {"id": "q_001", "question": "How much does HarborShield cost?", "reference_answer": "HarborShield costs $200 per month.", "expected_keywords": ["$200"],
             "relevant_doc_ids": ["doc_001"], "category": "direct_fact"},
            {"id": "q_002", "question": "When does ClaimPilot ship?", "reference_answer": "ClaimPilot ships in Q3.", "expected_keywords": ["Q3"],
             "relevant_doc_ids": ["doc_002"], "category": "direct_fact"},
            {"id": "q_003", "question": "Who is the CEO?", "reference_answer": "The documents do not say.", "expected_keywords": [], "relevant_doc_ids": [],
             "category": "unanswerable", "answer_type": "unanswerable"},
        ],
    )
    config = root / "config.yaml"
    config.write_text(
        f"""
run: {{name: baselines, output_dir: {root / "results"}}}
dataset: {{documents_path: {docs}, questions_path: {root / "questions.jsonl"}}}
systems:
  - {{type: bm25, name: bm25_x, retrieval: {{top_k: 3}}}}
  - {{type: no_retrieval, name: floor}}
  - {{type: full_context, name: ceiling}}
  - {{type: sentence_window, name: window_x}}
evaluation: {{k_values: [1, 3], max_workers: 1}}
""",
        encoding="utf-8",
    )
    return run_benchmark(config, force_mock=True, max_workers=1)


def test_a_run_with_a_non_retrieving_system_completes_and_scores_every_system(mixed_run):
    summary = pd.read_csv(mixed_run / "metrics_summary.csv").set_index("system")

    assert set(summary.index) == {"bm25_x", "floor", "ceiling", "window_x"}
    assert (summary["n_ok"] == 3).all() and (summary["n_error"] == 0).all()


def test_retrieval_columns_of_a_non_retrieving_system_are_empty_not_zero(mixed_run):
    summary = pd.read_csv(mixed_run / "metrics_summary.csv").set_index("system")
    retrieval_columns = [c for c in summary.columns if c.startswith("retrieval_")]

    assert retrieval_columns and summary.loc["floor", retrieval_columns].isna().all()
    assert summary.loc["bm25_x", "retrieval_recall@3"] >= 0 and not math.isnan(summary.loc["bm25_x", "retrieval_recall@3"])
    assert not pd.isna(summary.loc["floor", "answer_score"]), "answer quality is still measured"
    per_system = pd.read_csv(mixed_run / "retrieval_metrics.csv")
    assert per_system[per_system["system"] == "floor"]["recall@3"].isna().all()


def test_leaderboard_and_report_show_an_em_dash_for_the_floor_baseline(mixed_run):
    row = next(line for line in (mixed_run / "leaderboard.md").read_text(encoding="utf-8").splitlines() if "floor" in line and line.startswith("|"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]

    assert cells.count("—") >= 3 and "0.000" not in cells[1:4]
    html = (mixed_run / "report.html").read_text(encoding="utf-8")
    assert "floor" in html and "NaN" not in re.sub(r"<script.*?</script>", "", html, flags=re.S)  # shown values, not the page's own script


def test_per_question_rows_of_a_non_retrieving_system_have_no_retrieval_failures(mixed_run):
    rows = [json.loads(line) for line in (mixed_run / "per_question_results.jsonl").read_text().splitlines() if line.strip()]
    floor = [r for r in rows if r["system"] == "floor"]

    assert len(floor) == 3
    assert all(r["retrieval_metrics"] == {} and r["retrieved_contexts"] == [] for r in floor)
    assert all(r["failure_type"] not in {"retrieval_miss", "possible_qrels_gap"} for r in floor)
    assert next(r for r in floor if r["question_id"] == "q_003")["failure_type"] == "no_failure", "refusing an unanswerable question is correct"
    assert not any("floor" in line for line in (mixed_run / "qrels_audit.csv").read_text().splitlines())


def test_the_ceiling_baseline_reads_the_whole_corpus_and_costs_more_to_ask(mixed_run):
    rows = [json.loads(line) for line in (mixed_run / "per_question_results.jsonl").read_text().splitlines() if line.strip()]
    contexts = {r["system"]: len(r["retrieved_contexts"]) for r in rows if r["question_id"] == "q_001"}

    assert contexts["ceiling"] == 3 and contexts["floor"] == 0 and 0 < contexts["window_x"] <= 3
