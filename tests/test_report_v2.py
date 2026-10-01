"""The HTML report: built from a run directory, self-contained, accessible, escaped, and honest about missing data."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from dataset_support import write_tiny_dataset
from paid_fakes import EMBEDDING, GENERATOR, JUDGE, PRICING
from paid_fakes import install as install_paid_fakes

from ragbench.evaluation.evaluator import run_benchmark
from ragbench.reporting.html_report import write_report
from ragbench.reporting.report_data import ReportError, build_report
from ragbench.utils.jsonl import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
WORD_CHUNKER = {"type": "word", "chunk_size": 40, "chunk_overlap": 0}


def _run(root: Path, systems: list[dict], *, category: str = "fact", **extra) -> Path:
    config = {
        "run": {"name": "rep", "output_dir": str(root / "results")},
        "dataset": write_tiny_dataset(root, category=category),
        "systems": systems,
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0, **extra.pop("evaluation", {})},
        **extra,
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(path, force_mock=True, max_workers=1)


CHUNKED = {"bm25", "vector", "hybrid", "rerank", "hyde", "contextual", "rag_fusion", "decompose", "agent_search"}


def _system(kind: str, name: str | None = None, **more) -> dict:
    block: dict = {"type": kind, "name": name or kind, **more}
    if kind in CHUNKED:
        block["chunker"] = WORD_CHUNKER
    return block


ELEVEN = ["bm25", "vector", "hybrid", "rerank", "parent_doc", "hyde", "contextual", "hierarchical", "sentence_window", "rag_fusion", "decompose"]


@pytest.fixture(scope="module")
def eleven(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("eleven"), [_system(kind) for kind in ELEVEN])


@pytest.fixture(scope="module")
def single(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("single"), [_system("bm25")])


class Scan(HTMLParser):
    """Collects what the page loads, and its scripts, ids and heading levels."""

    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []
        self.scripts = 0
        self.ids: set[str] = set()
        self.headings: list[str] = []
        self.svgs: list[dict[str, str | None]] = []
        self.marks: list[dict[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for key in ("src", "href", "data", "poster", "srcset"):
            if attributes.get(key):
                self.refs.append(str(attributes[key]))
        if tag == "script":
            self.scripts += 1
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
        if tag in {"h1", "h2", "h3"}:
            self.headings.append(tag)
        if tag == "svg" and "chart" in (attributes.get("class") or "").split():
            self.svgs.append(attributes)
        if "mark" in (attributes.get("class") or "").split():
            self.marks.append(attributes)


def _html(run: Path) -> str:
    return (run / "report.html").read_text(encoding="utf-8")


def _scan(html: str) -> Scan:
    scan = Scan()
    scan.feed(html)
    return scan


# -- renders for the shapes of run that matter -----------------------------------------------------


def test_a_single_system_run_renders_with_every_core_section(single):
    html = _html(single)
    scan = _scan(html)
    assert {"recommendation", "tradeoff", "leaderboard-section", "costs", "repro"} <= scan.ids
    assert "agents" not in scan.ids  # no agentic systems, so no agent panel
    assert (single / "report_data.json").exists()
    assert 'class="hero-name">bm25<' in html.replace("\n", "")


def test_a_run_of_eleven_systems_renders_every_chart_with_a_mark_per_system(eleven):
    scan = _scan(_html(eleven))
    scatter = [m for m in scan.marks if "mk-" in (m.get("class") or "")]
    assert len(scatter) == 11
    assert len(scan.svgs) >= 5  # scatter, heatmap, latency, failures (and the cost chart when there is cost)
    data = json.loads((eleven / "report_data.json").read_text())
    assert len(data["leaderboard"]["rows"]) == 11 and len(data["heatmap"]["systems"]) == 11
    assert [row["system"] for row in data["leaderboard"]["rows"]][0] == data["recommendation"]["winner"]  # ordered as the recommendation ranks them


def test_a_system_that_retrieves_nothing_has_blank_retrieval_cells_not_zeros(tmp_path):
    run = _run(tmp_path, [_system("bm25"), _system("no_retrieval", "closed_book")])
    html = _html(run)
    data = json.loads((run / "report_data.json").read_text())
    row = next(r for r in data["leaderboard"]["rows"] if r["system"] == "closed_book")
    recall = next(c for c in row["cells"] if c["key"].startswith("retrieval_recall@"))
    assert recall["raw"] is None and recall["text"] == "—" and recall["ci"] is None
    assert data["heatmap"]["recall"] is not None and any(v is None for line in data["heatmap"]["recall"] for v in line)
    heat = html.split('data-heat="recall"')[1].split("</svg>")[0]
    assert "cell na" in heat  # drawn as an empty cell, not a 0
    assert "closed_book" in html


def test_a_mock_run_says_so_and_a_live_one_does_not(single, tmp_path):
    assert 'class="pill mock"' in _html(single) and "Mock run" in _html(single)
    live = tmp_path / "live"
    live.mkdir()
    for path in single.iterdir():
        if path.is_file():
            (live / path.name).write_bytes(path.read_bytes())
    summary = json.loads((live / "run_summary.json").read_text())
    summary.update(mode="live", notices=[])
    (live / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_report(live)
    html = _html(live)
    assert 'class="pill "' in html and "Mock run" not in html and "pill mock" not in html


def test_a_run_without_failures_says_so_instead_of_drawing_an_empty_chart(single, tmp_path):
    run = tmp_path / "clean"
    run.mkdir()
    for path in single.iterdir():
        if path.is_file():
            (run / path.name).write_bytes(path.read_bytes())
    rows = read_jsonl(run / "per_question_results.jsonl")
    for row in rows:
        row["failure_type"] = "no_failure"
    write_jsonl(run / "per_question_results.jsonl", rows)
    write_report(run)
    html = _html(run)
    assert "No classified failures" in html and "stacked" not in html.split('id="failures"')[1].split('id="audit"')[0]


def test_agentic_systems_get_an_agent_panel_with_tool_usage(tmp_path):
    run = _run(tmp_path, [_system("bm25"), _system("agent_search", "agent_plain", tools=[])])
    scan = _scan(_html(run))
    assert "agents" in scan.ids
    data = json.loads((run / "report_data.json").read_text())
    assert data["agents"]["rows"] and any(row["avg_steps"] is not None for row in data["agents"]["rows"])


def test_a_run_with_cost_draws_the_stage_chart_and_puts_cost_on_the_axis(tmp_path, monkeypatch):
    install_paid_fakes(monkeypatch)
    models = {"generator": GENERATOR, "embedding": EMBEDDING}
    config = {
        "run": {"name": "paid", "output_dir": str(tmp_path / "results")},
        "dataset": write_tiny_dataset(tmp_path),
        "systems": [
            {"type": "bm25", "name": "bm25", "models": models, "chunker": WORD_CHUNKER},
            {"type": "vector", "name": "vector", "models": models, "chunker": WORD_CHUNKER, "retrieval": {"vector_store": "numpy"}},
            {"type": "hyde", "name": "hyde", "models": models, "chunker": WORD_CHUNKER, "retrieval": {"vector_store": "numpy"}},
        ],
        "evaluation": {"max_workers": 1, "judge_model": JUDGE, "latency_probe_questions": 0},
        "pricing": PRICING,
        "cache": {"enabled": False},
    }
    path = tmp_path / "paid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = run_benchmark(path, max_workers=1)
    data = json.loads((run / "report_data.json").read_text())
    html = _html(run)
    assert data["scatter"]["x_metric"] == "cost" and data["stages"] and not data["stages"]["all_zero"]
    assert "Quality against cost" in html and "Where the money goes" in html and "No cost was recorded" not in html
    assert data["run"]["mode"] == "live" and 'class="pill "' in html and data["cost"]["charged_usd"] > 0


def test_the_demo_report_stays_small(tmp_path):
    run = _run_demo(tmp_path)
    html = _html(run)
    explorer = html.split('id="questions-data">')[1].split("</script>")[0]
    assert len(html.encode()) - len(explorer.encode()) < 600 * 1024  # the report itself; the question explorer's data is capped separately (report.max_embedded_mb)
    assert len(explorer.encode()) < 4 * 1024 * 1024


def _run_demo(tmp_path: Path) -> Path:
    config = {
        "run": {"name": "demo", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [_system(kind) for kind in ELEVEN],
        "evaluation": {"max_workers": 4, "latency_probe_questions": 0},
    }
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(path, force_mock=True)


# -- self-contained, escaped, accessible -----------------------------------------------------------


def test_the_page_loads_nothing_from_the_network_and_has_exactly_three_scripts(eleven):
    html = _html(eleven)
    scan = _scan(html)
    assert [ref for ref in scan.refs if not ref.startswith("#")] == []  # no src, no outgoing href
    stripped = html.replace("http://www.w3.org/2000/svg", "")
    assert not re.search(r"https?://", stripped), "no URL at all, other than the SVG namespace"
    assert not re.search(r"url\(\s*[\"']?(https?:|//)|@import", html)
    assert scan.scripts == 3  # the report data, the question data and the behavior; nothing else runs


def test_hostile_text_in_names_and_categories_is_escaped_everywhere(tmp_path):
    hostile = '<script>alert("xss")</script>'
    systems = [_system("bm25", 'evil"><img src=x onerror=alert(1)>'), _system("vector", hostile)]
    run = _run(tmp_path, systems, category=hostile)
    html = _html(run)
    scan = _scan(html)
    assert scan.scripts == 3 and "<img src=x" not in html and hostile not in html
    assert "&lt;script&gt;" in html  # shown, not executed
    blob = html.split('id="report-data">')[1].split("</script>")[0]
    assert "<" not in blob and ">" not in blob and json.loads(blob)["schema"] == 1  # safe inside a script element, and still valid JSON
    for svg in html.split("<svg")[1:]:
        assert "<script" not in svg.split("</svg>")[0]


def test_every_chart_is_labelled_every_mark_is_focusable_and_every_chart_has_a_table_view(eleven):
    html = _html(eleven)
    scan = _scan(html)
    assert all(svg.get("role") == "img" and svg.get("aria-label") for svg in scan.svgs)
    assert scan.marks and all(m.get("tabindex") == "0" and m.get("data-tip") for m in scan.marks)
    assert html.count("<summary>Table view</summary>") >= len(scan.svgs) - 1  # the scatter, heatmap, latency, failures (and cost) all have a twin
    assert scan.headings.count("h1") == 1 and 'lang="en"' in html and 'class="skip"' in html


def test_the_stylesheet_covers_dark_print_reduced_motion_and_forced_colors(single):
    css = _html(single).split("<style>")[1].split("</style>")[0]
    for needle in ("prefers-color-scheme: dark", '[data-theme="dark"]', "@media print", "prefers-reduced-motion", "forced-colors"):
        assert needle in css, needle
    assert re.search(r"--s[1-8]:", css) and "--c0:" in css  # colors are tokens


# -- data and behavior of the builder --------------------------------------------------------------


def test_the_report_is_built_only_from_files_so_it_can_be_regenerated_byte_for_byte(single):
    before = (single / "report.html").read_bytes()
    data_before = (single / "report_data.json").read_bytes()
    write_report(single)
    assert (single / "report.html").read_bytes() == before and (single / "report_data.json").read_bytes() == data_before


def test_the_embedded_data_matches_the_json_written_next_to_the_page(single):
    blob = _html(single).split('id="report-data">')[1].split("</script>")[0]
    assert json.loads(blob) == json.loads((single / "report_data.json").read_text())


def test_a_run_missing_its_optional_files_still_reports(tmp_path, single):
    run = tmp_path / "bare"
    run.mkdir()
    (run / "metrics_summary.csv").write_bytes((single / "metrics_summary.csv").read_bytes())
    write_report(run)
    scan = _scan(_html(run))
    assert "recommendation" not in scan.ids and {"tradeoff", "leaderboard-section", "repro"} <= scan.ids
    assert "failures" not in scan.ids and "categories" not in scan.ids


def test_an_empty_run_directory_is_an_error_not_an_empty_page(tmp_path):
    with pytest.raises(ReportError, match="metrics_summary.csv"):
        build_report(tmp_path)


def test_notices_and_the_synthetic_question_flag_reach_the_banner(tmp_path):
    root = tmp_path
    data = write_tiny_dataset(root)
    rows = read_jsonl(root / "questions.jsonl")
    for row in rows:
        row["metadata"] = {"synthetic": True, "needs_review": True, "generator": "mock-template", "mock": True}
    write_jsonl(root / "questions.jsonl", rows)
    config = {"run": {"name": "syn", "output_dir": str(root / "results")}, "dataset": data, "systems": [_system("bm25")], "evaluation": {"max_workers": 1}}
    (root / "c.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    html = _html(run_benchmark(root / "c.yaml", force_mock=True, max_workers=1))
    assert "Synthetic questions — review recommended" in html and "8 of 8 questions" in html and "mock templates" in html


def test_the_heatmap_and_leaderboard_values_agree_with_the_csvs(eleven):
    import pandas as pd

    data = json.loads((eleven / "report_data.json").read_text())
    summary = pd.read_csv(eleven / "metrics_summary.csv").set_index("system")
    for row in data["leaderboard"]["rows"]:
        answer = next(c for c in row["cells"] if c["key"] == "answer_score")
        assert answer["raw"] == pytest.approx(summary.loc[row["system"], "answer_score"])
    answers = pd.read_csv(eleven / "answer_metrics.csv")
    expected = answers.groupby(["system", "category"])["answer_score"].mean()
    heat = data["heatmap"]
    for i, system in enumerate(heat["systems"]):
        for j, category in enumerate(heat["categories"]):
            assert heat["answer_score"][i][j] == pytest.approx(expected[(system, category)])
