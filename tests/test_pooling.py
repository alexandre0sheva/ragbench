"""`ragbench label`: pooling, grading, never proposing labels for unjudged documents, and human labels staying authoritative."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from scripted_llm import ScriptedLLM
from typer.testing import CliRunner

from ragbench.cache import CacheRuntime, DiskCache, activate_cache
from ragbench.cli import app
from ragbench.config.schema import CacheConfig
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.pooling import (
    LabelError,
    build_pool,
    estimate_grading_cost,
    grade_pool,
    load_run,
    merge_qrels,
    parse_grade,
    proposed_rows,
    write_label_outputs,
)
from ragbench.evaluation.budget import BudgetGuard
from ragbench.models import cost as pricing
from ragbench.models.cached import CachedLLM
from ragbench.utils.jsonl import read_jsonl, write_jsonl

DOCS = {
    "doc_001": "# Pricing\n\nHarborShield costs $200 per month for the marine module.",
    "doc_002": "# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.",
    "doc_003": "# Support\n\nSupport is available around the clock for every plan.",
    "doc_004": "# Security\n\nAll data is encrypted at rest and in transit.",
    "doc_005": "# Billing\n\nInvoices are issued monthly and are payable within thirty days.",
}


def _row(system: str, question_id: str, docs: list[str], failure: str = "no_failure") -> dict:
    # Several chunks of the same document are common: the ranking must collapse them to one entry per document.
    contexts = [{"rank": rank, "doc_id": doc, "chunk_id": f"{doc}::{rank}", "score": 1.0} for rank, doc in enumerate(docs, start=1)]
    return {"system": system, "question_id": question_id, "error": None, "retrieved_contexts": contexts, "failure_type": failure}


def _make_run(tmp_path: Path, questions: list[dict], rows: list[dict], qrels: list[dict] | None = None) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    for doc_id, text in DOCS.items():
        (docs / f"{doc_id}.md").write_text(text, encoding="utf-8")
    write_jsonl(tmp_path / "questions.jsonl", questions)
    if qrels is not None:
        write_jsonl(tmp_path / "qrels.jsonl", qrels)
    config = {
        "run": {"name": "pooled", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(docs), "questions_path": str(tmp_path / "questions.jsonl"), **({"qrels_path": str(tmp_path / "qrels.jsonl")} if qrels is not None else {})},
        "systems": [{"type": "bm25", "name": "A"}, {"type": "bm25", "name": "B"}],
        "cache": {"enabled": False},
    }
    run = tmp_path / "results" / "pooled_run"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_jsonl(run / "per_question_results.jsonl", rows)
    return run


QUESTIONS = [
    {"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"]},
    {"id": "q2", "question": "When does ClaimPilot ship?", "answerable": True},  # no labels
    {"id": "q3", "question": "Who is the CEO of Atlantis?", "answerable": False},
]
ROWS = [
    _row("A", "q1", ["doc_001", "doc_001", "doc_002", "doc_003"]),
    _row("B", "q1", ["doc_003", "doc_004", "doc_005"]),
    _row("A", "q2", ["doc_002", "doc_004"]),
    _row("B", "q2", ["doc_002", "doc_005"], failure="possible_qrels_gap"),
    _row("A", "q3", ["doc_005"]),
]


def _grader(grades: dict[tuple[str, str], int] | None = None, *, default: int = 0, junk: set[str] | None = None) -> ScriptedLLM:
    """Grades each (question, document) from a table; a document id in `junk` never gets a valid reply."""

    def respond(prompt: str) -> str:
        question = re.search(r"^Question: (.*)$", prompt, flags=re.M)[1]  # type: ignore[index]
        doc_id = re.search(r"Document (doc_\d+) \(title", prompt)[1]  # type: ignore[index]
        if junk and doc_id in junk:
            return "not json at all"
        key = ("q1" if "HarborShield" in question else "q2", doc_id)
        return json.dumps({"grade": (grades or {}).get(key, default), "reason": f"scripted {key}"})

    return ScriptedLLM(respond)


@pytest.fixture(autouse=True)
def _clean_pricing():
    yield
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()


# -- pooling ---------------------------------------------------------------------------------------


def test_the_pool_is_the_union_of_every_systems_top_k_distinct_documents(tmp_path):
    plan = build_pool(load_run(_make_run(tmp_path, QUESTIONS, ROWS)), top_k=2)
    # A's top 2 distinct documents for q1 are doc_001 (its two chunks count once) and doc_002; B's are doc_003 and doc_004.
    assert plan.pool["q1"] == {"doc_001": {"A": 1}, "doc_002": {"A": 2}, "doc_003": {"B": 1}, "doc_004": {"B": 2}}
    assert plan.pool["q2"] == {"doc_002": {"A": 1, "B": 1}, "doc_004": {"A": 2}, "doc_005": {"B": 2}}
    assert [q.id for q in plan.questions] == ["q1", "q2"] and plan.skipped_unanswerable == 1 and plan.pairs == 7
    deeper = build_pool(load_run(tmp_path / "results" / "pooled_run"), top_k=10)
    assert set(deeper.pool["q1"]) == {"doc_001", "doc_002", "doc_003", "doc_004", "doc_005"}


def test_pooling_skips_failed_rows_and_questions_nobody_retrieved_for(tmp_path):
    rows = [*ROWS, {**_row("A", "q2", ["doc_005"]), "error": {"type": "X", "message": "boom"}}]
    run = load_run(_make_run(tmp_path, QUESTIONS, rows))
    assert run.rankings["q2"]["A"] == ["doc_002", "doc_004"]  # the failed row did not replace the real one
    assert build_pool(run, 5).pool["q2"]["doc_002"] == {"A": 1, "B": 1}
    with pytest.raises(LabelError, match="Nothing to grade"):
        build_pool(load_run(_make_run(tmp_path / "again", [QUESTIONS[2]], [ROWS[4]])), 5)


def test_load_run_explains_a_directory_that_is_not_a_run(tmp_path):
    with pytest.raises(LabelError, match="finished run"):
        load_run(tmp_path)


# -- grading ---------------------------------------------------------------------------------------


def test_grades_are_proposed_only_for_pooled_and_validly_graded_documents(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=2)
    llm = _grader({("q1", "doc_001"): 3, ("q1", "doc_002"): 1, ("q2", "doc_002"): 3}, junk={"doc_004"})
    graded = grade_pool(run, plan, llm, workers=1)

    assert ("q1", "doc_004") in graded.ungraded and ("q2", "doc_004") in graded.ungraded
    assert llm.calls == plan.pairs + 2  # the two doc_004 pairs were asked twice: the first try and one retry
    rows = {(row["query_id"], row["doc_id"]): row["relevance"] for row in proposed_rows(plan, graded)}
    assert rows[("q1", "doc_001")] == 3 and rows[("q2", "doc_002")] == 3 and rows[("q1", "doc_003")] == 0  # zeros are proposals too: judged, not relevant
    assert not any(doc == "doc_004" for _, doc in rows)  # never graded, so never proposed
    judged_pool = {(q, d) for q, docs in plan.pool.items() for d in docs} - set(graded.ungraded)
    assert set(rows) == judged_pool
    assert not any(query == "q3" for query, _ in rows) and not any(doc == "doc_005" and query == "q1" for query, doc in rows)


@pytest.mark.parametrize(
    ("reply", "grade"),
    [
        ('{"grade": 2, "reason": "ok"}', 2),
        ('```json\n{"grade": 3, "reason": "exact"}\n```', 3),
        ('{"grade": "1", "reason": "partial"}', 1),
        ('{"grade": 2.0}', 2),
        ('{"grade": 5}', None),
        ('{"grade": 2.5}', None),
        ('{"grade": true}', None),
        ('{"grade": -1}', None),
        ('{"reason": "no grade"}', None),
        ("2", None),
        ("not json", None),
    ],
)
def test_parse_grade_accepts_only_whole_grades_from_zero_to_three(reply, grade):
    judgment = parse_grade(reply)
    assert (judgment.grade if judgment else None) == grade


def test_a_reply_that_is_not_a_grade_is_retried_once_with_the_format_reminder(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=1)
    replies = {"count": 0}

    def respond(prompt: str) -> str:
        if "not a valid grade" not in prompt:
            replies["count"] += 1
            return "hmm, maybe"
        return '{"grade": 2, "reason": "second try"}'

    llm = ScriptedLLM(respond)
    graded = grade_pool(run, plan, llm, workers=1)
    assert graded.ungraded == [] and llm.calls == 2 * plan.pairs
    assert all(j.grade == 2 for judgments in graded.grades.values() for j in judgments.values())


def test_grading_does_not_depend_on_the_number_of_workers(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=10)
    table = {("q1", "doc_001"): 3, ("q2", "doc_002"): 2, ("q1", "doc_003"): 1}
    sequential = proposed_rows(plan, grade_pool(run, plan, _grader(table), workers=1))
    parallel = proposed_rows(plan, grade_pool(run, plan, _grader(table), workers=8))
    assert sequential == parallel and len(sequential) == plan.pairs


def test_grading_stops_at_the_spending_cap_and_says_so(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=10)
    llm = _grader()
    llm.price_per_call = 0.5
    graded = grade_pool(run, plan, llm, workers=1, budget=BudgetGuard(1.0))
    assert graded.stopped_by_budget and llm.calls == 2 and sum(len(v) for v in graded.grades.values()) == 2
    assert graded.cost_usd == pytest.approx(1.0)


def test_grading_goes_through_the_disk_cache_so_a_rerun_makes_no_new_calls(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=3)
    inner = _grader({("q1", "doc_001"): 3})
    llm = CachedLLM(inner)
    runtime = CacheRuntime(DiskCache(tmp_path / "cache.sqlite3"), CacheConfig())
    with activate_cache(runtime):
        first = grade_pool(run, plan, llm, workers=2)
        calls = inner.calls
        second = grade_pool(run, plan, llm, workers=2)
    assert calls == plan.pairs and inner.calls == calls
    assert proposed_rows(plan, first) == proposed_rows(plan, second)
    assert runtime.disk.stats()["hits"] == plan.pairs
    runtime.disk.close()


def test_the_grading_cost_estimate_uses_the_real_prompts_and_the_price_table(tmp_path):
    run = load_run(_make_run(tmp_path, QUESTIONS, ROWS))
    plan = build_pool(run, top_k=3)
    pricing.register_pricing({"fake-grader": {"input": 1.0, "output": 2.0}})
    assert estimate_grading_cost(run, plan, "fake-grader") > 0
    assert estimate_grading_cost(run, plan, "unpriced-model") == 0
    assert estimate_grading_cost(run, build_pool(run, top_k=1), "fake-grader") < estimate_grading_cost(run, plan, "fake-grader")


# -- merging and the review ------------------------------------------------------------------------


def test_apply_keeps_human_labels_authoritative_and_never_touches_the_users_files(tmp_path):
    qrels = [{"query_id": "q1", "doc_id": "doc_001", "relevance": 3}, {"query_id": "q1", "doc_id": "doc_005", "relevance": 2}]
    root = tmp_path / "w"
    root.mkdir()
    run_dir = _make_run(root, QUESTIONS, ROWS, qrels)
    questions_before = (root / "questions.jsonl").read_bytes()
    qrels_before = (root / "qrels.jsonl").read_bytes()
    run = load_run(run_dir)
    plan = build_pool(run, top_k=10)
    # The grader disagrees with the human on doc_001 (1 vs 3) and doc_005 (0 vs 2), and adds doc_002 and doc_003 for q1.
    graded = grade_pool(run, plan, _grader({("q1", "doc_001"): 1, ("q1", "doc_002"): 2, ("q1", "doc_003"): 3, ("q2", "doc_002"): 3}), workers=1)
    files = write_label_outputs(run_dir, run, plan, graded, top_k=10, judge_model="scripted", mock=False, apply=True)

    assert (root / "qrels.jsonl").read_bytes() == qrels_before and (root / "questions.jsonl").read_bytes() == questions_before
    assert files.merged is not None and files.merged.name == "qrels.merged.jsonl"
    merged = {(row["query_id"], row["doc_id"]): row["relevance"] for row in read_jsonl(files.merged)}
    assert merged[("q1", "doc_001")] == 3 and merged[("q1", "doc_005")] == 2  # human grades win, whatever the grader said
    assert merged[("q1", "doc_002")] == 2 and merged[("q1", "doc_003")] == 3  # documents the labels do not mention take the proposed grade
    assert merged[("q2", "doc_002")] == 3
    proposed = {(row["query_id"], row["doc_id"]): row["relevance"] for row in read_jsonl(files.proposed)}
    assert proposed[("q1", "doc_001")] == 1  # the proposal file is the grader's view; the merge is what resolves the conflict
    assert merge_qrels(run.dataset, plan, graded) == read_jsonl(files.merged)


def test_without_apply_no_merged_file_is_written(tmp_path):
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS)
    run = load_run(run_dir)
    plan = build_pool(run, top_k=3)
    graded = grade_pool(run, plan, _grader(), workers=1)
    files = write_label_outputs(run_dir, run, plan, graded, top_k=3, judge_model="scripted", mock=False, apply=False)
    assert files.merged is None and not (run_dir / "qrels.merged.jsonl").exists()
    assert files.proposed.exists() and files.review.exists()


def test_the_merged_qrels_label_questions_that_had_none(tmp_path):
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS)
    run = load_run(run_dir)
    plan = build_pool(run, top_k=10)
    graded = grade_pool(run, plan, _grader({("q2", "doc_002"): 3}), workers=1)
    files = write_label_outputs(run_dir, run, plan, graded, top_k=10, judge_model="scripted", mock=False, apply=True)
    after = load_dataset(tmp_path / "questions.jsonl", files.merged)
    q2 = next(q for q in after.questions if q.id == "q2")
    assert q2.is_answerable and after.qrels["q2"]["doc_002"] == 3
    assert next(q for q in after.questions if q.id == "q3").is_answerable is False  # explicit answerable: false is kept


def test_the_review_shows_where_the_grader_and_the_labels_disagree(tmp_path):
    qrels = [{"query_id": "q1", "doc_id": "doc_001", "relevance": 2}, {"query_id": "q1", "doc_id": "doc_005", "relevance": 2}]
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS, qrels)
    run = load_run(run_dir)
    plan = build_pool(run, top_k=2)  # doc_005 is only third for B on q1, so it is never pooled
    graded = grade_pool(run, plan, _grader({("q1", "doc_001"): 0, ("q1", "doc_003"): 3, ("q2", "doc_002"): 2}), workers=1)
    text = write_label_outputs(run_dir, run, plan, graded, top_k=2, judge_model="scripted-model", mock=False, apply=False).review.read_text()

    assert "Disputed" in text and "doc_001" in text  # a labeled document graded 0
    assert "Possible missing label" in text and "doc_003 (3)" in text  # graded 2+ and not labeled
    assert "Not judged" in text and "doc_005" in text  # labeled, but nobody retrieved it
    assert "possible_qrels_gap" in text and "B" in text  # the run's own flag for q2
    assert "Proposed labels for questions without labels" in text and "q2" in text
    assert "scripted-model" in text and "pooling" in text.lower() and "Mock grader" not in text
    assert "### q3" not in text  # the unanswerable question was not graded


def test_the_review_warns_when_the_grader_is_a_mock_and_about_questions_with_no_relevant_document(tmp_path):
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS)
    run = load_run(run_dir)
    plan = build_pool(run, top_k=10)
    graded = grade_pool(run, plan, _grader(default=0), workers=1)
    text = write_label_outputs(run_dir, run, plan, graded, top_k=10, judge_model="mock grader", mock=True, apply=False).review.read_text()
    assert "Mock grader" in text and "No relevant document found" in text and "answerable: true" in text


# -- the CLI ---------------------------------------------------------------------------------------


def test_cli_label_with_the_mock_grader_writes_the_proposals_and_never_overwrites_the_merge(tmp_path):
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS)
    runner = CliRunner()
    result = runner.invoke(app, ["label", "--run", str(run_dir), "--mock", "--apply", "--top-k", "3"])
    assert result.exit_code == 0, result.output
    assert "Mock grader" in result.output and "dataset.qrels_path" in result.output
    for name in ("qrels.proposed.jsonl", "qrels_review.md", "qrels.merged.jsonl"):
        assert (run_dir / name).exists(), name
    assert load_dataset(tmp_path / "questions.jsonl", run_dir / "qrels.merged.jsonl").questions[1].is_answerable
    again = runner.invoke(app, ["label", "--run", str(run_dir), "--mock", "--apply"])
    assert again.exit_code == 2 and "--force" in again.output
    assert runner.invoke(app, ["label", "--run", str(run_dir), "--mock", "--apply", "--force"]).exit_code == 0


def test_cli_label_explains_a_bad_run_directory(tmp_path):
    result = CliRunner().invoke(app, ["label", "--run", str(tmp_path), "--mock"])
    assert result.exit_code == 2 and "finished run" in result.output


def test_cli_label_refuses_an_expensive_live_grading_when_nobody_can_confirm(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, QUESTIONS, ROWS)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    config["pricing"] = {"gpt-6-luna": {"input": 5_000_000.0, "output": 5_000_000.0}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("CI", "1")
    result = CliRunner().invoke(app, ["label", "--run", str(run_dir), "--top-k", "3"])
    assert result.exit_code == 2 and "confirmation threshold" in result.output and "--yes" in result.output
    assert not (run_dir / "qrels.proposed.jsonl").exists()
