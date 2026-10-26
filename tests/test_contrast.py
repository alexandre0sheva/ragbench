"""Contrast sets: the questions where one system is right and another is not, with the first step at which their traces part."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.config.schema import ContrastConfig
from ragbench.evaluation.contrast import (
    CONTRASTS_FILE,
    SUMMARY_FILE,
    ContrastError,
    build_contrasts,
    first_divergence,
    read_summaries,
    write_contrasts,
)
from ragbench.evaluation.stats import CORRECT_THRESHOLD
from ragbench.utils.jsonl import read_jsonl, write_jsonl


def retrieve(docs: list[str], name: str = "bm25_search") -> dict[str, Any]:
    return {"kind": "retrieve", "name": name, "output_preview": f"{len(docs)} chunks", "metadata": {"doc_ids": docs}}


def rerank(docs: list[str]) -> dict[str, Any]:
    return {"kind": "rerank", "name": "cross_encoder", "output_preview": f"{len(docs)} chunks", "metadata": {"doc_ids": docs}}


def generate(text: str = "an answer") -> dict[str, Any]:
    return {"kind": "generate", "name": "answer", "output_preview": text, "metadata": {"context_chunks": 3}}


def row(system: str, question_id: str, score: float, *, steps: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "system": system,
        "question_id": question_id,
        "category": "direct_fact",
        "difficulty": "easy",
        "answer": f"{system} says {score}",
        "answer_judge": {"answer_score": score},
        "failure_type": "no_failure" if score >= 4 else "retrieval_miss",
        "steps": steps if steps is not None else [retrieve(["d1", "d2"]), generate()],
        "error": None,
    }
    return {**base, **extra}


def write_run(root: Path, rows: list[dict[str, Any]], *, pairwise: list[dict[str, Any]] | None = None) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "per_question_results.jsonl", rows)
    (run_dir / "metrics_summary.csv").write_text("system\nalpha\nbeta\n", encoding="utf-8")
    if pairwise is not None:
        write_jsonl(run_dir / "pairwise.jsonl", pairwise)
    return run_dir


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """alpha and beta over six questions: q1 alpha only, q2 beta only, q3 both right, q4 both wrong, q5 alpha only, q6 beta only."""
    scores = {"q1": (5, 1), "q2": (1, 5), "q3": (5, 4.5), "q4": (1, 2), "q5": (4.5, 2), "q6": (3, 4)}
    categories = {"q1": "direct_fact", "q2": "multi_hop", "q3": "direct_fact", "q4": "multi_hop", "q5": "multi_hop", "q6": "direct_fact"}
    rows = []
    for qid, (a, b) in scores.items():
        rows.append(row("alpha", qid, a, category=categories[qid], steps=[retrieve(["d1", "d2"]), generate("alpha")]))
        rows.append(row("beta", qid, b, category=categories[qid], steps=[retrieve(["d1", "d2"] if qid != "q1" else ["d9", "d8"], "vector_search"), generate("beta")]))
    return write_run(tmp_path, rows)


def only(contrasts, winner: str) -> list[str]:
    return sorted(c["question_id"] for c in contrasts.records if c["winner"] == winner)


def test_contrast_set_holds_exactly_the_questions_where_one_system_is_correct_and_the_other_is_not(run_dir):
    contrasts = build_contrasts(run_dir, [("alpha", "beta")])
    assert only(contrasts, "alpha") == ["q1", "q5"]
    assert only(contrasts, "beta") == ["q2", "q6"]
    summary = contrasts.summaries[0]
    assert (summary["pair"], summary["a_only"], summary["b_only"], summary["both_correct"], summary["both_wrong"]) == (["alpha", "beta"], 2, 2, 1, 1)
    assert summary["compared"] == 6 and summary["skipped"] == 0


def test_a_contrast_names_who_won_with_both_scores_and_the_loser_s_failure(run_dir):
    contrast = next(c for c in build_contrasts(run_dir, [("alpha", "beta")]).records if c["question_id"] == "q1")
    assert (contrast["winner"], contrast["loser"], contrast["source"]) == ("alpha", "beta", "score")
    assert contrast["scores"] == {"alpha": 5.0, "beta": 1.0} and contrast["gap"] == 4.0
    assert contrast["loser_failure_type"] == "retrieval_miss" and contrast["category"] == "direct_fact"


def test_the_threshold_decides_what_correct_means(run_dir):
    strict = build_contrasts(run_dir, [("alpha", "beta")], threshold=4.8)
    assert only(strict, "alpha") == ["q1", "q3"] and only(strict, "beta") == ["q2"]


def test_the_first_diverging_step_is_where_the_retrieved_documents_differ(run_dir):
    contrast = next(c for c in build_contrasts(run_dir, [("alpha", "beta")]).records if c["question_id"] == "q1")
    assert contrast["diverges_at"] == {"index": 0, "stage": "retrieve", "reason": "different_documents", "a": "retrieve:bm25_search", "b": "retrieve:vector_search"}
    assert [step["same"] for step in contrast["steps"]] == [False, False]  # documents differ, and so do the answers


def test_the_same_evidence_diverges_at_generation(run_dir):
    contrast = next(c for c in build_contrasts(run_dir, [("alpha", "beta")]).records if c["question_id"] == "q2")
    assert contrast["diverges_at"]["stage"] == "generate" and contrast["diverges_at"]["reason"] == "different_answer"
    assert contrast["diverges_at"]["index"] == 1
    assert [step["same"] for step in contrast["steps"]] == [True, False]


def test_traces_are_aligned_step_by_step_with_what_each_system_did():
    a = [retrieve(["d1", "d2"]), rerank(["d2", "d1"]), generate("x")]
    b = [retrieve(["d1", "d2"], "vector_search"), generate("y")]
    divergence, aligned = first_divergence(a, b)
    assert divergence == {"index": 1, "stage": "rerank", "reason": "different_stage", "a": "rerank:cross_encoder", "b": "generate:answer"}
    assert [(step["a"], step["b"]) for step in aligned] == [("retrieve:bm25_search", "retrieve:vector_search"), ("rerank:cross_encoder", "generate:answer"), ("generate:answer", None)]


def test_a_deeper_ranking_with_the_same_head_is_not_a_divergence():
    divergence, _ = first_divergence([retrieve(["d1", "d2", "d3", "d4"]), generate("x")], [retrieve(["d1", "d2"]), generate("y")])
    assert divergence["stage"] == "generate"


def test_a_system_with_more_steps_diverges_where_the_other_stopped():
    divergence, _ = first_divergence([retrieve(["d1"]), generate("x")], [retrieve(["d1"]), {"kind": "tool", "name": "search", "output_preview": "r"}, generate("x")])
    assert (divergence["index"], divergence["stage"], divergence["reason"]) == (1, "generate", "different_stage")


def test_steps_that_hold_no_documents_are_compared_by_their_output():
    llm = {"kind": "llm", "name": "rewrite", "output_preview": "query one"}
    divergence, _ = first_divergence([llm, retrieve(["d1"])], [{**llm, "output_preview": "query two"}, retrieve(["d1"])])
    assert divergence["index"] == 0 and divergence["reason"] == "different_output"


def test_a_system_without_a_trace_has_no_divergence_step():
    divergence, aligned = first_divergence([], [retrieve(["d1"]), generate()])
    assert divergence is None and aligned == []


def test_clusters_count_the_loser_s_failure_type_and_category_per_direction(run_dir):
    summary = build_contrasts(run_dir, [("alpha", "beta")]).summaries[0]
    assert summary["clusters"]["alpha"] == {"failure_type": {"retrieval_miss": 2}, "category": {"direct_fact": 1, "multi_hop": 1}, "diverges_at": {"retrieve": 1, "generate": 1}}
    assert summary["clusters"]["beta"]["failure_type"] == {"retrieval_miss": 2}


def test_a_failed_question_is_left_out_not_scored_as_zero(tmp_path):
    rows = [row("alpha", "q1", 5), row("beta", "q1", 0, error="boom", answer_judge=None), row("alpha", "q2", 5), row("beta", "q2", 1)]
    contrasts = build_contrasts(write_run(tmp_path, rows), [("alpha", "beta")])
    assert only(contrasts, "alpha") == ["q2"]
    assert contrasts.summaries[0]["skipped"] == 1 and contrasts.summaries[0]["compared"] == 1


def test_trials_are_averaged_before_the_threshold(tmp_path):
    rows = [row("alpha", "q1", 5, trial=0), row("alpha", "q1", 4, trial=1), row("beta", "q1", 5, trial=0), row("beta", "q1", 1, trial=1)]
    contrasts = build_contrasts(write_run(tmp_path, rows), [("alpha", "beta")])  # alpha 4.5, beta 3.0
    assert only(contrasts, "alpha") == ["q1"] and contrasts.records[0]["scores"] == {"alpha": 4.5, "beta": 3.0}


def test_pairwise_verdicts_can_decide_instead_of_the_score(tmp_path):
    rows = [row(s, q, 4.5) for s in ("alpha", "beta") for q in ("q1", "q2", "q3")]  # the scores see no difference
    verdicts = [
        {"question_id": "q1", "systems": ["alpha", "beta"], "judge": "j", "verdicts": ["x", "x"], "score": 1.0},
        {"question_id": "q2", "systems": ["beta", "alpha"], "judge": "j", "verdicts": ["x", "x"], "score": 1.0},  # beta is x here
        {"question_id": "q3", "systems": ["alpha", "beta"], "judge": "j", "verdicts": ["tie", "tie"], "score": 0.5},
    ]
    run = write_run(tmp_path, rows, pairwise=verdicts)
    assert build_contrasts(run, [("alpha", "beta")]).records == []
    by_verdict = build_contrasts(run, [("alpha", "beta")], source="verdicts")
    assert only(by_verdict, "alpha") == ["q1"] and only(by_verdict, "beta") == ["q2"]
    assert by_verdict.records[0]["source"] == "verdicts" and by_verdict.summaries[0]["both_equal"] == 1


def test_verdicts_need_the_pairwise_stage(run_dir):
    with pytest.raises(ContrastError, match="pairwise"):
        build_contrasts(run_dir, [("alpha", "beta")], source="verdicts")


def test_only_the_asked_questions_are_compared(run_dir):
    contrasts = build_contrasts(run_dir, [("alpha", "beta")], question_ids={"q1", "q2", "q3"})
    assert only(contrasts, "alpha") == ["q1"] and contrasts.summaries[0]["compared"] == 3


def test_an_unknown_system_gets_a_did_you_mean(run_dir):
    with pytest.raises(ContrastError, match="Did you mean 'alpha'"):
        build_contrasts(run_dir, [("alhpa", "beta")])


def test_a_system_cannot_be_contrasted_with_itself(run_dir):
    with pytest.raises(ContrastError, match="two different systems"):
        build_contrasts(run_dir, [("alpha", "alpha")])


def test_contrasts_round_trip_and_a_later_pair_replaces_only_its_own_lines(run_dir):
    first = build_contrasts(run_dir, [("alpha", "beta")])
    write_contrasts(run_dir, first)
    assert len(read_jsonl(run_dir / CONTRASTS_FILE)) == len(first.records) == 4
    assert json.loads((run_dir / SUMMARY_FILE).read_text())["pairs"][0]["pair"] == ["alpha", "beta"]
    other = build_contrasts(run_dir, [("alpha", "beta")], threshold=4.8)
    write_contrasts(run_dir, other, merge=True)
    assert len(read_jsonl(run_dir / CONTRASTS_FILE)) == len(other.records) == 3  # the pair was replaced, not duplicated
    assert [s["threshold"] for s in read_summaries(run_dir)] == [4.8]


# -- config, a run, the CLI and the reports ---------------------------------------------------------------------------------------------------

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
THREE = [{"type": "bm25", "name": "bm25"}, {"type": "no_retrieval", "name": "closed_book"}, {"type": "hybrid", "name": "hybrid"}]


def _config(root: Path, contrast: dict | None, systems: list[dict] = THREE, **evaluation: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    settings = {"max_questions": 20, "max_workers": 2, "latency_probe_questions": 0, "stats": {"n_boot": 200}, **evaluation}
    if contrast is not None:
        settings["contrast"] = contrast
    config = {
        "run": {"name": "disagree", "output_dir": str(root / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": systems,
        "evaluation": settings,
    }
    path = root / "disagree.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def mock_run_dir(tmp_path_factory) -> Path:
    from ragbench.evaluation.evaluator import run_benchmark

    return run_benchmark(_config(tmp_path_factory.mktemp("contrast"), {"enabled": True}), force_mock=True)


def test_contrast_is_off_by_default_and_a_run_without_it_writes_nothing_new(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    assert ContrastConfig().enabled is False
    out = run_benchmark(_config(tmp_path, None, THREE[:2]), force_mock=True)
    assert not (out / CONTRASTS_FILE).exists() and not (out / SUMMARY_FILE).exists()
    assert "Where systems disagree" not in (out / "leaderboard.md").read_text() + (out / "report.html").read_text(encoding="utf-8")


def test_the_default_threshold_is_the_one_cost_per_correct_uses():
    assert ContrastConfig().threshold == CORRECT_THRESHOLD


def test_contrast_needs_two_systems_and_pairwise_for_verdicts(tmp_path):
    with pytest.raises(ValidationError, match="at least two systems"):
        load_config(_config(tmp_path / "one", {"enabled": True}, THREE[:1]))
    with pytest.raises(ValidationError, match=r"source: verdicts.*pairwise"):
        load_config(_config(tmp_path / "verdicts", {"enabled": True, "source": "verdicts"}))
    load_config(_config(tmp_path / "ok", {"enabled": True, "source": "verdicts"}, pairwise={"enabled": True}))


def test_contrast_settings_do_not_invalidate_checkpoints(tmp_path):
    from ragbench.evaluation.checkpoint import context_hash

    plain, enabled = (load_config(_config(tmp_path / name, contrast)) for name, contrast in (("a", None), ("b", {"enabled": True, "threshold": 3.0})))
    assert context_hash(plain.systems[0], plain, "digest", "mock") == context_hash(enabled.systems[0], enabled, "digest", "mock")


def test_an_enabled_run_contrasts_every_system_with_the_baseline(mock_run_dir):
    summaries = read_summaries(mock_run_dir)
    baseline = json.loads((mock_run_dir / "stats.json").read_text())["baseline"]
    assert [s["pair"] for s in summaries] == [[baseline, other] for other in ("bm25", "closed_book", "hybrid") if other != baseline]
    records = read_jsonl(mock_run_dir / CONTRASTS_FILE)
    assert records and all(baseline in r["pair"] and r["winner"] in r["pair"] for r in records)
    assert all(s["compared"] == 20 and s["a_only"] + s["b_only"] + s["both_correct"] + s["both_wrong"] == 20 for s in summaries)
    assert json.loads((mock_run_dir / "run_summary.json").read_text())["outputs"]["contrasts"] == CONTRASTS_FILE


def test_the_leaderboard_and_the_report_have_a_section_where_systems_disagree(mock_run_dir):
    leaderboard = (mock_run_dir / "leaderboard.md").read_text()
    section = leaderboard.split("## Where systems disagree")[1]
    first = read_summaries(mock_run_dir)[0]
    assert f"{first['pair'][0]} vs {first['pair'][1]}" in section and "only" in section
    page = (mock_run_dir / "report.html").read_text(encoding="utf-8")
    assert "Where systems disagree" in page and "contrast.jsonl" not in page
    assert json.loads((mock_run_dir / "report_data.json").read_text())["contrast"]["pairs"][0]["pair"] == first["pair"]


def test_rebuilding_the_reports_keeps_the_section(mock_run_dir, tmp_path):
    from ragbench.reporting.regenerate import regenerate_reports

    copy = tmp_path / "copy"
    shutil.copytree(mock_run_dir, copy)
    (copy / "leaderboard.md").unlink()
    regenerate_reports(copy)
    assert "## Where systems disagree" in (copy / "leaderboard.md").read_text()


def test_the_cli_contrasts_two_systems_of_a_finished_run(mock_run_dir, tmp_path):
    from cli_support import invoke

    copy = tmp_path / "copy"
    shutil.copytree(mock_run_dir, copy)
    result = invoke("contrast", str(copy), "bm25", "closed_book", "--threshold", "3.5")
    assert result.exit_code == 0, result.output
    assert "bm25" in result.output and "closed_book" in result.output and "traces first part at" in result.output
    summaries = {tuple(s["pair"]): s for s in read_summaries(copy)}
    assert summaries[("bm25", "closed_book")]["threshold"] == 3.5  # the pair was recomputed in place
    assert "## Where systems disagree" in (copy / "leaderboard.md").read_text()  # and the reports were rebuilt with it


def test_the_cli_prints_json_and_names_a_wrong_system(mock_run_dir):
    from cli_support import invoke

    result = invoke("contrast", str(mock_run_dir), "bm25", "hybrid", "--json", "--no-write")
    payload = json.loads(result.stdout)
    assert payload["summaries"][0]["pair"] == ["bm25", "hybrid"] and isinstance(payload["contrasts"], list)
    bad = invoke("contrast", str(mock_run_dir), "bm52", "hybrid", "--no-write")
    assert bad.exit_code == 2 and "Did you mean 'bm25'" in bad.output
