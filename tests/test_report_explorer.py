"""The report's question explorer: the compact per-question data it embeds, its size cap, and that hostile text stays inert."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from dataset_support import write_tiny_dataset
from pydantic import ValidationError

from ragbench.config.schema import ExperimentConfig
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.reporting.explorer_data import DEFAULT_MAX_EMBEDDED_MB, DISAGREE_SPREAD, QUESTIONS_FILE, build_questions, script_json
from ragbench.reporting.html_report import write_report
from ragbench.reporting.report_data import build_report
from ragbench.utils.jsonl import read_jsonl, write_jsonl

JUDGE_KEYS = ("correctness", "faithfulness", "completeness", "relevance", "citation_quality")


def _row(system: str, qid: str, score: float | None = 4.0, **more) -> dict:
    """A `per_question_results.jsonl` row, with just what a real one carries."""
    error = more.pop("error", None)
    row = {
        "system": system,
        "system_type": "bm25",
        "question_id": qid,
        "question": f"Question {qid}?",
        "category": "fact",
        "difficulty": "easy",
        "answer_type": "single_fact",
        "reference_answer": f"Reference {qid}.",
        "relevant_doc_ids": ["doc_a"],
        "answer": f"{system} answers {qid}.",
        "retrieved_contexts": [
            {"rank": 1, "doc_id": "doc_a", "chunk_id": "doc_a::0", "score": 0.9, "in_context": True, "text_preview": "relevant text", "page": 3, "heading": "Pricing > Plans"},
            {"rank": 2, "doc_id": "doc_b", "chunk_id": "doc_b::0", "score": 0.5, "in_context": True, "text_preview": "other text"},
            {"rank": 3, "doc_id": "doc_c", "chunk_id": "doc_c::0", "score": 0.2, "in_context": False, "text_preview": "deep text"},
        ],
        "retrieval_metrics": {"recall@5": 1.0},
        "answer_judge": {**dict.fromkeys(JUDGE_KEYS, score), "answer_score": score, "reasoning": "because", "metadata": {"judge": "heuristic"}},
        "cost": {"total_cost": 0.002},
        "latency_ms": 12.34567,
        "failure_type": "no_failure",
        "error": error,
        "refused": False,
        "route": None,
        "agent": None,
        "steps": [
            {"kind": "retrieve", "name": "bm25_search", "latency_ms": 1.5, "cost": {"total_cost": 0.0}, "prompt_tokens": 0, "completion_tokens": 0, "input_preview": "q", "output_preview": "3 chunks", "metadata": {"top_k": 20, "nested": {"x": 1}}},
            {"kind": "tool", "name": "calculator", "latency_ms": 0.2, "cost": {"total_cost": 0.0}, "prompt_tokens": 0, "completion_tokens": 0, "input_preview": "2+2", "output_preview": "4", "metadata": {}},
        ],
    }
    row.update(more)
    return row


def _rows() -> list[dict]:
    return [
        _row("alpha", "q1", 5.0),
        _row("beta", "q1", 1.0, failure_type="partial_answer"),
        _row("alpha", "q2", 4.0),
        _row("beta", "q2", None, error="boom", failure_type="run_error", answer="", retrieved_contexts=[], steps=[]),
    ]


def _build(rows: list[dict], *, audit: list[dict] | None = None, cap: int = 4 * 1024 * 1024, order: list[str] | None = None):
    return build_questions(rows, order or ["alpha", "beta"], audit or [], cap)


# -- schema ------------------------------------------------------------------------------------------------------------------------------------


def test_one_entry_per_question_with_a_run_per_system_in_report_order():
    payload = _build(_rows(), order=["beta", "alpha"])
    page = payload.embedded
    assert payload.full is None and page["file"] is None
    assert page["systems"] == ["beta", "alpha"]
    assert (page["version"], page["detail"], page["total"], page["shown"]) == (1, "full", 2, 2)
    assert page["disagree_spread"] == DISAGREE_SPREAD
    assert page["failure_names"]["partial_answer"] and page["failure_names"]["run_error"]
    q1 = page["questions"][0]
    assert (q1["id"], q1["q"], q1["cat"], q1["ref"], q1["rel"]) == ("q1", "Question q1?", "fact", "Reference q1.", ["doc_a"])
    assert {run["s"] for run in q1["runs"]} == {0, 1} and [page["systems"][run["s"]] for run in q1["runs"]] == ["beta", "alpha"]  # runs follow the system order


def test_a_run_carries_the_answer_judgment_contexts_and_trace():
    run = next(r for r in _build(_rows()).embedded["questions"][0]["runs"] if r["s"] == 0)
    assert run["answer"] == "alpha answers q1." and run["score"] == 5.0 and run["failure"] == "no_failure" and run["error"] is None
    assert run["judge"] == dict.fromkeys(JUDGE_KEYS, 5.0) and run["reasoning"] == "because"
    assert run["latency_ms"] == 12.3 and run["cost"] == 0.002
    assert [(c["rank"], c["doc"], c["in"]) for c in run["ctx"]] == [(1, "doc_a", True), (2, "doc_b", True), (3, "doc_c", False)]
    assert run["ctx"][0]["page"] == 3 and run["ctx"][0]["heading"] == "Pricing > Plans" and run["ctx"][0]["text"] == "relevant text"
    first, tool = run["steps"]
    assert (first["kind"], first["name"], first["ms"], first["in"], first["out"]) == ("retrieve", "bm25_search", 1.5, "q", "3 chunks")
    assert first["meta"] == {"top_k": 20}  # only plain values: a nested structure is not a trace detail worth the bytes
    assert (tool["kind"], tool["in"], tool["out"]) == ("tool", "2+2", "4")


def test_labeled_documents_are_green_and_audited_unlabeled_ones_amber():
    audit = [{"system": "alpha", "question_id": "q1", "unlabeled_retrieved_doc_ids": "doc_b, doc_x"}]
    ctx = next(r for r in _build(_rows(), audit=audit).embedded["questions"][0]["runs"] if r["s"] == 0)["ctx"]
    assert [c["mark"] for c in ctx] == ["relevant", "judged_good", None]
    other = next(r for r in _build(_rows(), audit=audit).embedded["questions"][0]["runs"] if r["s"] == 1)["ctx"]
    assert [c["mark"] for c in other] == ["relevant", None, None]  # the audit named alpha's answer, not beta's


def test_a_question_that_raised_has_an_error_and_no_scores():
    run = next(r for r in _build(_rows()).embedded["questions"][1]["runs"] if r["s"] == 1)
    assert run["error"] == "boom" and run["score"] is None and run["judge"] is None and run["failure"] == "run_error" and run["ctx"] == []


def test_systems_disagree_when_the_answer_scores_are_far_apart():
    q1, q2 = _build(_rows()).embedded["questions"]
    assert q1["spread"] == 4.0 and q1["spread"] >= DISAGREE_SPREAD
    assert q2["spread"] is None  # one system answered, one raised: nothing to disagree about


def test_long_text_is_shortened_and_unlabeled_questions_have_no_relevant_documents():
    rows = [_row("alpha", "q1", answer="x" * 5000, relevant_doc_ids=[], reference_answer=None)]
    q = _build(rows, order=["alpha"]).embedded["questions"][0]
    assert q["rel"] == [] and q.get("ref") is None
    assert len(q["runs"][0]["answer"]) <= 1300 and q["runs"][0]["answer"].endswith("…")


# -- the size cap ------------------------------------------------------------------------------------------------------------------------------


def _thousands(n: int = 5000) -> list[dict]:
    rows = []
    for i in range(n):
        for system in ("alpha", "beta", "gamma"):
            rows.append(_row(system, f"q{i:05d}", answer="a long answer " * 60, failure_type="partial_answer" if i % 7 == 0 else "no_failure"))
    return rows


def test_five_thousand_questions_fit_the_cap_and_the_full_data_goes_to_a_file():
    cap = 4 * 1024 * 1024
    payload = _build(_thousands(), cap=cap, order=["alpha", "beta", "gamma"])
    page, full = payload.embedded, payload.full
    assert len(script_json(page).encode()) <= cap
    assert page["file"] == QUESTIONS_FILE and page["total"] == 5000 and page["detail"] in {"reduced", "scores"}
    assert full is not None and full["detail"] == "full" and full["shown"] == full["total"] == 5000 and full["file"] is None
    assert len(full["questions"][0]["runs"][0]["ctx"]) == 3 and len(full["questions"][0]["runs"][0]["steps"]) == 2


def test_when_even_scores_do_not_fit_the_questions_that_matter_most_stay():
    rows = _thousands(1000)
    payload = _build(rows, cap=60_000, order=["alpha", "beta", "gamma"])
    page = payload.embedded
    assert len(script_json(page).encode()) <= 60_000
    assert 0 < page["shown"] < page["total"] == 1000 and len(page["questions"]) == page["shown"]
    ids = [q["id"] for q in page["questions"]]
    assert ids == sorted(ids)  # still in the dataset's order
    assert ids[0] == "q00000"  # a failing question (every 7th, starting at 0) is kept before a clean one


def test_a_small_run_embeds_everything_and_names_no_file():
    page = _build(_rows(), cap=4 * 1024 * 1024).embedded
    assert page["file"] is None and page["detail"] == "full" and page["shown"] == page["total"]


def test_the_embedded_form_has_no_characters_that_could_end_a_script_element():
    rows = [_row("alpha", "q1", answer='</script><img src=x onerror="alert(1)"> & <!--  ')]
    text = script_json(_build(rows, order=["alpha"]).embedded)
    assert "<" not in text and ">" not in text and "&" not in text and " " not in text
    assert json.loads(text)["questions"][0]["runs"][0]["answer"].startswith("</script>")  # and it is still the same text once parsed


# -- the config key ----------------------------------------------------------------------------------------------------------------------------


def test_report_max_embedded_mb_defaults_to_four_and_rejects_nonsense():
    base = {"run": {"name": "r"}, "dataset": {"documents_path": "d", "questions_path": "q"}, "systems": [{"type": "bm25"}]}
    assert ExperimentConfig.model_validate(base).report.max_embedded_mb == DEFAULT_MAX_EMBEDDED_MB == 4.0
    assert ExperimentConfig.model_validate({**base, "report": {"max_embedded_mb": 0.5}}).report.max_embedded_mb == 0.5
    for bad in ({"max_embedded_mb": 0}, {"max_embedded_mb": -1}, {"max_embeded_mb": 1}):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate({**base, "report": bad})


# -- end to end --------------------------------------------------------------------------------------------------------------------------------


class Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.count += tag == "script"
        if dict(attrs).get("id"):
            self.ids.add(str(dict(attrs)["id"]))


def _mock_run(root: Path) -> Path:
    config = {
        "run": {"name": "explorer", "output_dir": str(root / "results")},
        "dataset": write_tiny_dataset(root),
        "systems": [{"type": "bm25", "name": "bm25", "chunker": {"type": "markdown"}}, {"type": "vector", "name": "vector", "chunker": {"type": "markdown"}}],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(root / "config.yaml", force_mock=True, max_workers=1)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    return _mock_run(tmp_path_factory.mktemp("explorer"))


def _questions_blob(html: str) -> dict:
    return json.loads(html.split('id="questions-data">')[1].split("</script>")[0])


def test_a_real_run_embeds_the_questions_and_the_report_has_the_section(run):
    html = (run / "report.html").read_text(encoding="utf-8")
    scripts = Scripts()
    scripts.feed(html)
    assert {"questions", "q-table", "report-data", "questions-data"} <= scripts.ids and scripts.count == 3  # data, questions, behavior
    page = _questions_blob(html)
    assert sorted(page["systems"]) == ["bm25", "vector"]
    assert [q["id"] for q in page["questions"]] == [f"q{i}" for i in range(1, 9)] and page["file"] is None
    assert not (run / QUESTIONS_FILE).exists()
    heading_runs = [c for q in page["questions"] for r in q["runs"] for c in r["ctx"] if "heading" in c]
    assert heading_runs and all(isinstance(c["heading"], str) for c in heading_runs)  # the markdown chunker's heading path reaches the explorer


def test_hostile_answers_and_questions_cannot_run_in_the_page(run, tmp_path):
    rows = read_jsonl(run / "per_question_results.jsonl")
    hostile = '<script>alert("xss")</script><img src=x onerror=alert(1)>'
    for row in rows:
        row["answer"], row["question"], row["answer_judge"]["reasoning"] = hostile, hostile, hostile
        row["retrieved_contexts"][0]["text_preview"] = hostile
        row["steps"] = [{"kind": "tool", "name": hostile, "latency_ms": 1, "cost": {"total_cost": 0}, "input_preview": hostile, "output_preview": hostile, "metadata": {"error": hostile}}]
    copy = tmp_path / "run"
    copy.mkdir()
    for path in run.iterdir():
        if path.is_file():
            (copy / path.name).write_bytes(path.read_bytes())
    write_jsonl(copy / "per_question_results.jsonl", rows)
    write_report(copy)
    html = (copy / "report.html").read_text(encoding="utf-8")
    scripts = Scripts()
    scripts.feed(html)
    assert scripts.count == 3 and "<img src=x" not in html and hostile not in html
    blob = html.split('id="questions-data">')[1].split("</script>")[0]
    assert "<" not in blob and ">" not in blob
    assert json.loads(blob)["questions"][0]["runs"][0]["answer"] == hostile  # kept verbatim as data; the page script only ever sets textContent
    js = (Path(__file__).resolve().parents[1] / "src/ragbench/reporting/templates/report.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js and "eval(" not in js


def test_a_tight_cap_trims_the_page_and_writes_the_full_questions_next_to_it(run, tmp_path):
    copy = tmp_path / "run"
    copy.mkdir()
    for path in run.iterdir():
        if path.is_file():
            (copy / path.name).write_bytes(path.read_bytes())
    rows = read_jsonl(run / "per_question_results.jsonl")
    write_jsonl(copy / "per_question_results.jsonl", [{**row, "question_id": f"{row['question_id']}_{i}"} for i in range(30) for row in rows])
    write_report(copy, max_embedded_mb=0.05)
    page = _questions_blob((copy / "report.html").read_text(encoding="utf-8"))
    assert page["file"] == QUESTIONS_FILE and page["total"] == 240 and (page["shown"] < 240 or page["detail"] != "full")  # trimmed one way or the other
    assert len(script_json(page).encode()) <= 0.05 * 1024 * 1024
    full = json.loads((copy / QUESTIONS_FILE).read_text(encoding="utf-8"))
    assert full["detail"] == "full" and len(full["questions"]) == full["total"] == 240
    write_report(copy)  # the default cap fits everything again, so the stale file must not linger
    assert not (copy / QUESTIONS_FILE).exists()


def test_the_report_data_json_only_summarizes_the_questions(run):
    data = json.loads((run / "report_data.json").read_text(encoding="utf-8"))
    assert data["questions"] == {"total": 8, "embedded": 8, "detail": "full", "file": None}
    assert "runs" not in json.dumps(data["questions"])  # the answers stay in per_question_results.jsonl, not duplicated here
    assert build_report(run).data["questions"]["total"] == 8


def test_a_run_without_per_question_rows_has_no_questions_section(run, tmp_path):
    copy = tmp_path / "run"
    copy.mkdir()
    for path in run.iterdir():
        if path.is_file() and path.name != "per_question_results.jsonl":
            (copy / path.name).write_bytes(path.read_bytes())
    write_report(copy)
    html = (copy / "report.html").read_text(encoding="utf-8")
    assert 'id="questions"' not in html and 'id="questions-data"' not in html
