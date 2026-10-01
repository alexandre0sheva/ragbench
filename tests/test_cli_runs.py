"""`runs`, `report` and `compare-runs`: a results folder as a history, and reports rebuilt from the files alone."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pytest
from cli_support import invoke, mock_run

from ragbench.reporting.history import resolve_run, scan_runs, sparkline_svg, write_index
from ragbench.utils.jsonl import read_jsonl, write_jsonl


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Path:
    """A results folder with two finished mock runs of the same two systems."""
    root = tmp_path_factory.mktemp("history")
    mock_run(root, root / "results" / "first_run", name="first")
    mock_run(root, root / "results" / "second_run", name="second")
    return root / "results"


def _copy(results: Path, tmp_path: Path) -> Path:
    target = tmp_path / "results"
    shutil.copytree(results, target)
    return target


# -- runs --------------------------------------------------------------------------------------------------------------------------------------


def test_runs_lists_every_run_newest_first_and_bare_runs_does_the_same(results):
    listing = invoke("runs", "list", "--results-dir", str(results))
    assert listing.exit_code == 0 and "first_run" in listing.output and "second_run" in listing.output and "mock" in listing.output
    assert listing.output.index("second_run") < listing.output.index("first_run")  # the later run is on top
    assert invoke("runs", "--results-dir", str(results)).output == listing.output


def test_runs_list_json_has_the_columns_of_the_table(results):
    rows = json.loads(invoke("runs", "list", "--results-dir", str(results), "--json").stdout)
    assert {r["name"] for r in rows} == {"first_run", "second_run"}
    row = rows[0]
    assert row["status"] == "complete" and row["mode"] == "mock" and row["systems"] == ["bm25", "vector"] and row["questions"] == 163
    assert row["winner"] in {"bm25", "vector"} and row["has_report"] is True and set(row["scores"]) == {"bm25", "vector"}
    assert [r["started"] for r in rows] == sorted((r["started"] for r in rows), reverse=True)  # newest first


def test_an_empty_or_missing_results_folder_is_not_an_error(tmp_path):
    result = invoke("runs", "list", "--results-dir", str(tmp_path / "nothing"))
    assert result.exit_code == 0 and "No runs" in result.output
    assert json.loads(invoke("runs", "list", "--results-dir", str(tmp_path), "--json").stdout) == []


def test_runs_show_resolves_names_fragments_and_latest(results):
    for ref in ("second_run", "second", str(results / "second_run"), "latest"):
        shown = invoke("runs", "show", ref, "--results-dir", str(results))
        assert shown.exit_code == 0 and "Leaderboard" in shown.output and "bm25" in shown.output, ref
    payload = json.loads(invoke("runs", "show", "first", "--results-dir", str(results), "--json").stdout)
    assert payload["run_id"] and payload["files"]["report"].endswith("report.html") and len(payload["systems"]) == 2


def test_an_unknown_or_ambiguous_run_is_a_usage_error_that_lists_what_exists(results):
    missing = invoke("runs", "show", "nope", "--results-dir", str(results))
    assert missing.exit_code == 2 and "No run 'nope'" in missing.output and "first_run" in missing.output
    ambiguous = invoke("runs", "show", "_run", "--results-dir", str(results))
    assert ambiguous.exit_code == 2 and "matches 2 runs" in ambiguous.output
    with pytest.raises(Exception, match="No runs found"):
        resolve_run("latest", results / "empty")


def test_runs_rm_deletes_only_with_consent_and_only_run_directories(results, tmp_path):
    folder = _copy(results, tmp_path)
    refused = invoke("runs", "rm", "first_run", "--results-dir", str(folder))
    assert refused.exit_code == 2 and "--yes" in refused.output and (folder / "first_run").exists()  # no one to ask, so nothing is deleted
    plain = folder / "notes"
    plain.mkdir()
    (plain / "keep.txt").write_text("precious")
    not_a_run = invoke("runs", "rm", "notes", "--results-dir", str(folder), "--yes")
    assert not_a_run.exit_code == 2 and "not a run directory" in not_a_run.output and (plain / "keep.txt").exists()
    deleted = invoke("runs", "rm", "first_run", "--results-dir", str(folder), "--yes")
    assert deleted.exit_code == 0 and not (folder / "first_run").exists() and (folder / "second_run").exists()


def test_runs_rm_never_deletes_the_folder_you_are_standing_in(results, tmp_path, monkeypatch):
    folder = _copy(results, tmp_path)
    monkeypatch.chdir(folder / "second_run")
    result = invoke("runs", "rm", str(folder / "second_run"), "--yes")
    assert result.exit_code == 2 and "Refusing" in result.output and (folder / "second_run").exists()


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.hrefs += [value for key, value in attrs if key in ("href", "src") and value]


def test_runs_index_writes_a_static_page_with_relative_links(results, tmp_path):
    folder = _copy(results, tmp_path)
    result = invoke("runs", "index", "--results-dir", str(folder))
    assert result.exit_code == 0
    html = (folder / "index.html").read_text(encoding="utf-8")
    page = _Page()
    page.feed(html)
    assert page.hrefs == ["second_run/report.html", "first_run/report.html"]  # newest first, each relative to the page
    assert not any(ref.startswith(("/", "http")) or str(tmp_path) in ref for ref in page.hrefs)  # moving the folder must not break it
    assert "<svg" in html and html.count('class="spark"') == 2  # one trend line per system name (bm25, vector)
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "<script" not in html
    for name in ("first_run", "second_run", "bm25", "vector"):
        assert name in html


def test_the_index_escapes_hostile_run_and_system_names(tmp_path):
    results = tmp_path / "results"
    hostile = results / "evil<img src=x onerror=alert(1)>"
    hostile.mkdir(parents=True)
    (hostile / "metrics_summary.csv").write_text('system,answer_score\n"<script>alert(1)</script>",4.0\n', encoding="utf-8")
    (hostile / "report.html").write_text("<html></html>")
    html = write_index(results).read_text(encoding="utf-8")
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "evil%3Cimg" in html  # shown as text, and percent-encoded in the link


def test_an_incomplete_run_is_listed_but_marked(results, tmp_path):
    folder = _copy(results, tmp_path)
    (folder / "crashed").mkdir()
    (folder / "crashed" / "config.yaml").write_text("run: {name: x}\n")
    records = {r.name: r for r in scan_runs(folder)}
    assert records["crashed"].status == "incomplete" and records["first_run"].status == "complete"
    assert "(incomplete)" in invoke("runs", "list", "--results-dir", str(folder)).output
    assert "INCOMPLETE" in write_index(folder).read_text(encoding="utf-8")


def test_sparklines_are_valid_svg_for_any_number_of_points():
    for values in ([], [3.0], [1.0, 4.5, 2.0], [0.0, 5.0]):
        svg = sparkline_svg(values, label='a "quoted" <label>')
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg") and "<label>" not in svg


# -- report ------------------------------------------------------------------------------------------------------------------------------------


GENERATED = ("leaderboard.md", "failures.md", "qrels_audit.md", "recommendation.md", "recommendation.json", "winner.yaml", "report.html", "report_data.json")


def test_report_rebuilds_the_reports_byte_for_byte_from_the_files(results, tmp_path):
    folder = _copy(results, tmp_path)
    run = folder / "second_run"
    before = {name: (run / name).read_bytes() for name in GENERATED if (run / name).exists()}
    assert {"leaderboard.md", "failures.md", "report.html"} <= set(before)
    for name in before:
        (run / name).unlink()
    result = invoke("report", "second_run", "--results-dir", str(folder))
    assert result.exit_code == 0 and "Rebuilt" in result.output
    after = {name: (run / name).read_bytes() for name in before}
    for name in ("leaderboard.md", "failures.md", "qrels_audit.md", "recommendation.md", "recommendation.json", "winner.yaml"):
        if name in before:
            assert after[name] == before[name], f"{name} changed when it was rebuilt"
    assert after["report_data.json"] == before["report_data.json"]
    assert after["report.html"] == before["report.html"]


def test_report_works_from_a_path_and_from_latest(results, tmp_path):
    folder = _copy(results, tmp_path)
    assert invoke("report", str(folder / "first_run")).exit_code == 0
    assert invoke("report", "latest", "--results-dir", str(folder)).exit_code == 0


def test_report_on_a_directory_without_results_is_a_usage_error(tmp_path):
    empty = tmp_path / "results" / "half"
    empty.mkdir(parents=True)
    (empty / "config.yaml").write_text("run: {name: x}\n")
    result = invoke("report", "half", "--results-dir", str(tmp_path / "results"))
    assert result.exit_code == 2 and "metrics_summary.csv" in result.output and "Traceback" not in result.output


def test_report_notes_what_it_could_not_rewrite(results, tmp_path):
    folder = _copy(results, tmp_path)
    (folder / "first_run" / "config.yaml").unlink()
    result = invoke("report", "first_run", "--results-dir", str(folder))
    assert result.exit_code == 0 and "recommendation not rewritten" in result.output and (folder / "first_run" / "report.html").exists()


# -- compare-runs ------------------------------------------------------------------------------------------------------------------------------


def _degrade(source: Path, target: Path, system: str, by: float) -> Path:
    """A copy of a run in which `system` answers `by` points worse on every question (the summary is kept consistent)."""
    shutil.copytree(source, target)
    rows = read_jsonl(target / "per_question_results.jsonl")
    for row in rows:
        if row["system"] == system and row["error"] is None:
            row["answer_judge"]["answer_score"] = max(0.0, row["answer_judge"]["answer_score"] - by)
    write_jsonl(target / "per_question_results.jsonl", rows)
    summary = (target / "metrics_summary.csv").read_text(encoding="utf-8").splitlines()
    header = summary[0].split(",")
    column = header.index("answer_score")
    out = [summary[0]]
    for line in summary[1:]:
        cells = line.split(",")
        if cells[header.index("system")] == system:
            scores = [r["answer_judge"]["answer_score"] for r in rows if r["system"] == system and r["error"] is None]
            cells[column] = repr(sum(scores) / len(scores))
        out.append(",".join(cells))
    (target / "metrics_summary.csv").write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


def test_comparing_a_run_with_itself_flags_nothing(results):
    result = invoke("compare-runs", "first_run", "first_run", "--results-dir", str(results), "--fail-on-regression")
    assert result.exit_code == 0 and "▼" not in result.output and "Regressed" not in result.output
    payload = json.loads(invoke("compare-runs", "first_run", "first_run", "--results-dir", str(results), "--json").stdout)
    assert payload["regressions"] == [] and all(s["verdict"] == "no clear change" for s in payload["systems"])


def test_a_significant_drop_is_flagged_and_can_fail_the_command(results, tmp_path):
    worse = _degrade(results / "first_run", tmp_path / "worse", "bm25", by=1.5)
    plain = invoke("compare-runs", str(results / "first_run"), str(worse))
    assert plain.exit_code == 0 and "regression" in plain.output and "Regressed: bm25" in plain.output
    failing = invoke("compare-runs", str(results / "first_run"), str(worse), "--fail-on-regression")
    assert failing.exit_code == 1
    payload = json.loads(invoke("compare-runs", str(results / "first_run"), str(worse), "--json").stdout)
    by_name = {s["system"]: s for s in payload["systems"]}
    assert by_name["bm25"]["verdict"] == "regression" and by_name["bm25"]["metrics"]["answer_score"]["delta"] < -0.5
    assert by_name["bm25"]["paired_questions"] == 163 and by_name["bm25"]["answer_score_delta_ci"][1] < 0  # tested on the same questions
    assert by_name["vector"]["verdict"] == "no clear change" and payload["regressions"] == ["bm25"]


def test_an_improvement_is_not_a_regression(results, tmp_path):
    better = _degrade(results / "first_run", tmp_path / "better", "vector", by=1.0)
    payload = json.loads(invoke("compare-runs", str(better), str(results / "first_run"), "--json").stdout)  # the degraded run is the baseline
    assert {s["system"]: s["verdict"] for s in payload["systems"]}["vector"] == "improvement" and payload["regressions"] == []


def test_runs_with_different_questions_fall_back_to_a_plain_threshold(results, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(results / "first_run", other)
    rows = read_jsonl(other / "per_question_results.jsonl")
    write_jsonl(other / "per_question_results.jsonl", [{**r, "question_id": "x" + r["question_id"]} for r in rows])  # no question in common
    worse = _degrade(other, tmp_path / "worse", "bm25", by=1.0)
    payload = json.loads(invoke("compare-runs", str(results / "first_run"), str(worse), "--json").stdout)
    by_name = {s["system"]: s for s in payload["systems"]}
    assert by_name["bm25"]["verdict"] == "drop (not tested)" and by_name["bm25"]["paired_questions"] == 0 and payload["regressions"] == ["bm25"]
    assert any("not tested" in note for note in payload["notes"])
    lenient = json.loads(invoke("compare-runs", str(results / "first_run"), str(worse), "--json", "--min-drop", "5").stdout)
    assert lenient["regressions"] == []


def test_compare_runs_names_systems_that_are_in_only_one_run(results, tmp_path):
    narrower = tmp_path / "narrower"
    shutil.copytree(results / "first_run", narrower)
    lines = (narrower / "metrics_summary.csv").read_text(encoding="utf-8").splitlines()
    (narrower / "metrics_summary.csv").write_text("\n".join(line for line in lines if not line.startswith("vector,")) + "\n", encoding="utf-8")
    payload = json.loads(invoke("compare-runs", str(results / "first_run"), str(narrower), "--json").stdout)
    assert payload["only_in_a"] == ["vector"] and [s["system"] for s in payload["systems"]] == ["bm25"]
