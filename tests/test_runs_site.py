"""`ragbench runs site`: many run directories as one static, filterable cross-run leaderboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from cli_support import invoke

from ragbench.reporting.site import SiteError, leaderboard_markdown, site_rows, write_site

ROOT = Path(__file__).resolve().parents[1]


def make_run(root: Path, name: str, systems: dict[str, dict[str, Any]], *, mode: str = "live", started: str = "2026-10-01T10:00:00Z", report: bool = True, questions: int = 100) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    columns = ["system", "system_type", "n_ok", "n_error", "retrieval_hit@5", "answer_score", "answer_score_ci_lo", "answer_score_ci_hi", "avg_cost_per_question", "latency_ms_p50"]
    lines = [",".join(columns)]
    for system, values in systems.items():
        base = {"system": system, "system_type": values.get("type", "bm25"), "n_ok": questions, "n_error": 0, **{k: v for k, v in values.items() if k != "type"}}
        lines.append(",".join("" if base.get(c) is None else str(base[c]) for c in columns))
    (run_dir / "metrics_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps({"run_id": name, "mode": mode, "primary_k": 5, "dataset": {"questions": questions}, "cache": {"charged_cost_usd": 1.5}}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"started_utc": started, "mode": mode}), encoding="utf-8")
    if report:
        (run_dir / "report.html").write_text(f"<html>report of {name}</html>", encoding="utf-8")
    return run_dir


@pytest.fixture
def results(tmp_path) -> Path:
    root = tmp_path / "results"
    make_run(root, "first", {"bm25": {"retrieval_hit@5": 0.8, "answer_score": 3.1, "answer_score_ci_lo": 2.9, "answer_score_ci_hi": 3.3, "avg_cost_per_question": 0.001, "latency_ms_p50": 40}, "vector": {"type": "vector", "answer_score": 2.5}}, started="2026-10-01T10:00:00Z")
    make_run(root, "second", {"bm25": {"answer_score": 3.4, "avg_cost_per_question": 0.0}}, mode="mock", started="2026-10-02T10:00:00Z", report=False)
    (root / "half_done").mkdir()
    (root / "half_done" / "config.yaml").write_text("run: {}\n", encoding="utf-8")  # a run that never reached a leaderboard
    (root / "notes").mkdir()
    return root


def embedded(page: str) -> dict[str, Any]:
    match = re.search(r'<script id="site-data" type="application/json">(.*?)</script>', page, re.S)
    assert match, "the page carries its data as JSON"
    return json.loads(match.group(1))


def test_one_row_per_run_and_system_with_the_numbers_of_the_run(results, tmp_path):
    rows = site_rows(results)
    assert [(r["run"], r["system"]) for r in rows] == [("second", "bm25"), ("first", "bm25"), ("first", "vector")]  # newest run first, then the run's own order
    first = next(r for r in rows if (r["run"], r["system"]) == ("first", "bm25"))
    assert first["score"] == 3.1 and (first["ci_lo"], first["ci_hi"]) == (2.9, 3.3) and first["hit"] == 0.8 and first["cost"] == 0.001 and first["p50"] == 40
    assert first["mode"] == "live" and first["questions"] == 100 and first["started"] == "2026-10-01T10:00:00Z"


def test_a_missing_number_is_missing_never_zero(results):
    vector = next(r for r in site_rows(results) if r["system"] == "vector")
    assert vector["hit"] is None and vector["cost"] is None and vector["p50"] is None and vector["ci_lo"] is None
    zero = next(r for r in site_rows(results) if r["run"] == "second")
    assert zero["cost"] == 0.0  # a free run is a zero; an absent column is not


def test_the_page_is_one_file_with_its_data_filters_and_sorting(results, tmp_path):
    out = tmp_path / "site"
    page_path = write_site(results, out)
    assert page_path == out / "index.html"
    page = page_path.read_text(encoding="utf-8")
    data = embedded(page)
    assert len(data["rows"]) == 3 and data["skipped"] == ["half_done"]
    for control in ("filter-run", "filter-mode", "filter-type", "filter-search"):
        assert f'id="{control}"' in page
    assert "data-sort" in page and "<script" in page and "http://" not in page and "https://" not in page  # offline: no CDN, no fonts, no tracking


def test_mock_runs_are_marked_and_off_until_asked_for_when_a_live_run_exists(results, tmp_path):
    data = embedded(write_site(results, tmp_path / "site").read_text(encoding="utf-8"))
    assert data["default_mode"] == "live" and {r["mode"] for r in data["rows"]} == {"live", "mock"}
    only_mock = tmp_path / "only_mock"
    make_run(only_mock, "m", {"bm25": {"answer_score": 1.0}}, mode="mock")
    assert embedded(write_site(only_mock, tmp_path / "s2").read_text(encoding="utf-8"))["default_mode"] == ""


def test_reports_are_copied_next_to_the_page_and_linked_relatively(results, tmp_path):
    out = tmp_path / "site"
    data = embedded(write_site(results, out).read_text(encoding="utf-8"))
    links = {r["run"]: r["report"] for r in data["rows"]}
    assert links == {"first": "runs/first/report.html", "second": None}
    assert (out / "runs" / "first" / "report.html").read_text(encoding="utf-8") == "<html>report of first</html>"


def test_a_name_cannot_break_out_of_the_data_block(tmp_path):
    root = tmp_path / "results"
    make_run(root, "evil", {"</script><b>x": {"answer_score": 2.0}})
    page = write_site(root, tmp_path / "site").read_text(encoding="utf-8")
    assert "</script><b>" not in page
    assert embedded(page)["rows"][0]["system"] == "</script><b>x"


def test_writing_twice_gives_the_same_page(results, tmp_path):
    assert write_site(results, tmp_path / "a").read_bytes() == write_site(results, tmp_path / "b").read_bytes()


def test_a_folder_without_complete_runs_is_an_error_that_says_what_to_do(tmp_path):
    with pytest.raises(SiteError, match="No complete runs") as empty:
        write_site(tmp_path, tmp_path / "site")
    assert "ragbench run" in empty.value.hint
    with pytest.raises(SiteError, match="does not exist"):
        write_site(tmp_path / "missing", tmp_path / "site")


def test_the_command_writes_the_site_into_the_folder_by_default(results):
    result = invoke("runs", "site", str(results))
    assert result.exit_code == 0, result.output
    assert (results / "site" / "index.html").is_file() and "index.html" in result.output
    again = invoke("runs", "site", str(results), "--json")  # the site folder is not mistaken for a run on the second call
    assert again.exit_code == 0 and json.loads(again.stdout)["rows"] == 3


def test_the_command_takes_an_output_folder_a_title_and_fails_on_a_missing_folder(results, tmp_path):
    out = tmp_path / "elsewhere"
    assert invoke("runs", "site", str(results), "--out", str(out), "--title", "Our benchmark").exit_code == 0
    assert "Our benchmark" in (out / "index.html").read_text(encoding="utf-8")
    assert invoke("runs", "site", str(tmp_path / "nope")).exit_code == 2


# -- the Leaderboard page of the docs site: built from the committed result files, never typed -----------------------------------------------


def test_the_leaderboard_has_a_table_per_committed_run_with_the_numbers_of_its_csv():
    import csv

    text = leaderboard_markdown(ROOT / "docs" / "benchmarks")
    for run in sorted((ROOT / "docs" / "benchmarks").glob("*-run")):
        assert f"`{run.name}`" in text
        with (run / "metrics_summary.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        best = max((r for r in rows if r["answer_score"]), key=lambda r: float(r["answer_score"]))
        line = next(line for line in text.splitlines() if line.startswith(f"| {best['system']} |") and f"{float(best['answer_score']):.2f}" in line)
        assert line
        assert text.count("| ---") >= 1


def test_the_leaderboard_orders_by_answer_score_marks_mock_and_dashes_what_is_missing(tmp_path):
    bench = tmp_path / "benchmarks"
    make_run(bench, "alpha-run", {"low": {"answer_score": 1.0}, "high": {"answer_score": 4.0, "avg_cost_per_question": 0.002}, "unscored": {"type": "vector"}}, mode="mock")
    text = leaderboard_markdown(bench)
    assert text.index("| high |") < text.index("| low |") < text.index("| unscored |")
    unscored = next(line for line in text.splitlines() if line.startswith("| unscored |"))
    assert "—" in unscored and "0.00" not in unscored
    assert "mock" in text.lower() and "$0.00200" in text


def test_the_leaderboard_says_so_when_there_is_nothing_to_show(tmp_path):
    assert "No benchmark runs" in leaderboard_markdown(tmp_path)
