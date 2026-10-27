"""scripts/check_results.py: the published benchmark pages are rendered from the committed result files and fail CI when they drift."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml
from dataset_support import write_tiny_dataset

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_results = _load("check_results")

PAGE = """# Demo benchmark

| System | Answer score | $ / question |
| --- | --- | --- |
{% for row in run.by("answer_score") %}
| `{{ row.system }}` | {{ ci(row, "answer_score") }} | {{ row.avg_cost_per_question | money }} |
{% endfor %}

{% set test = run.test("bm25", "vector") or run.test("vector", "bm25") %}
Paired: {{ test.mean_diff | signed }} ({{ test.n }} questions).
"""
README = "# Project\n\n<!-- results:demo:start -->\nold numbers 1.23\n<!-- results:demo:end -->\n\nMore text.\n"
BLOCK = "Best: `{{ run.by('answer_score')[0].system }}` at {{ run.by('answer_score')[0].answer_score | score }}.\n"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    from ragbench.evaluation.evaluator import run_benchmark

    root = tmp_path_factory.mktemp("repo")
    config = {
        "run": {"name": "demo", "output_dir": str(root / "results")},
        "dataset": write_tiny_dataset(root / "data"),
        "systems": [{"type": "bm25", "name": "bm25"}, {"type": "vector", "name": "vector"}],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    run = run_benchmark(root / "config.yaml", force_mock=True)
    published = root / "docs" / "benchmarks" / "demo-run"
    published.mkdir(parents=True)
    for name in ("metrics_summary.csv", "significance.csv", "run_summary.json", "run_manifest.json", "config.yaml"):
        (published / name).write_bytes((run / name).read_bytes())
    templates = root / "scripts" / "benchmark_templates"
    templates.mkdir(parents=True)
    (templates / "demo.md.j2").write_text(PAGE, encoding="utf-8")
    (templates / "demo.readme.md.j2").write_text(BLOCK, encoding="utf-8")
    (root / "README.md").write_text(README, encoding="utf-8")
    return root


def test_unwritten_pages_fail_and_writing_them_makes_the_check_pass(root, capsys):
    assert check_results.main(["--root", str(root)]) == 1
    assert "docs/benchmarks/demo.md does not match" in capsys.readouterr().err
    assert check_results.main(["--root", str(root), "--write"]) == 0
    assert check_results.main(["--root", str(root)]) == 0
    page = (root / "docs" / "benchmarks" / "demo.md").read_text(encoding="utf-8")
    assert "| `bm25` |" in page and "| `vector` |" in page and "Paired: " in page
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "old numbers" not in readme and "Best: `" in readme and readme.endswith("More text.\n")


def test_a_number_edited_by_hand_fails_the_check(root, capsys):
    check_results.main(["--root", str(root), "--write"])
    page = root / "docs" / "benchmarks" / "demo.md"
    text = page.read_text(encoding="utf-8")
    first = check_results.read_csv(root / "docs" / "benchmarks" / "demo-run" / "metrics_summary.csv")[0]["avg_cost_per_question"]
    page.write_text(text.replace(check_results.money(first), "$9.99999", 1), encoding="utf-8")
    assert check_results.main(["--root", str(root)]) == 1
    assert "demo.md does not match" in capsys.readouterr().err
    check_results.main(["--root", str(root), "--write"])


def test_a_benchmark_page_without_a_template_is_refused(root, capsys):
    check_results.main(["--root", str(root), "--write"])
    stray = root / "docs" / "benchmarks" / "typed.md"
    stray.write_text("# Typed by hand\n\nbm25 scored 4.99.\n", encoding="utf-8")
    try:
        assert check_results.main(["--root", str(root)]) == 1
        assert "typed.md has no template" in capsys.readouterr().err
    finally:
        stray.unlink()


def test_the_formatters():
    row = {"answer_score": 4.8123, "answer_score_ci_lo": 4.7, "answer_score_ci_hi": 4.9}
    assert check_results.ci(row, "answer_score") == "4.81 [4.70, 4.90]"
    assert check_results.ci({"answer_score": None}, "answer_score") == "—"
    assert check_results.signed(-0.037) == "−0.04" and check_results.signed(3.15) == "+3.15"
    assert check_results.money(0.000494) == "$0.00049" and check_results.ms(1371.4) == "1371 ms"


def test_the_repositorys_published_results_match():
    assert check_results.problems(ROOT) == []


def test_publish_copies_only_the_small_files_and_the_report_rebuilds_from_them(root, tmp_path, monkeypatch):
    publish = _load("publish_benchmark")
    run = next((root / "results").iterdir())
    dest = tmp_path / "demo-run"
    copied = publish.copy(run, dest, publish.RUN_FILES)
    assert "metrics_summary.csv" in copied and not (dest / "per_question_results.jsonl").exists()
    publish.rebuild(dest)
    assert (dest / "leaderboard.md").is_file() and (dest / "report.html").is_file() and (dest / "recommendation.md").is_file()


def test_publish_writes_pattern_aggregates_but_no_question_text(root, tmp_path):
    import json

    publish = _load("publish_benchmark")
    run = next((root / "results").iterdir())
    rows = [json.loads(line) for line in (run / "per_question_results.jsonl").read_text(encoding="utf-8").splitlines()]
    dest = tmp_path / "demo-run"
    dest.mkdir()
    publish.write_patterns(run, dest)
    text = (dest / "patterns.json").read_text(encoding="utf-8")
    patterns = json.loads(text)
    assert all(row["question"] not in text and (row["answer"] or "§") not in text for row in rows)
    questions = {row["question_id"] for row in rows}
    assert sum(patterns["questions"].values()) == len(questions)
    assert set(patterns["answerable_score"]) == {"bm25", "vector"}
    assert sum(patterns["failures"].values()) == len(rows)
    assert patterns["hard"]["threshold"] == 4.0 and patterns["hard"]["none"] <= len(questions)
    assert [bucket["recall"] for bucket in patterns["recall_buckets"]] == ["none", "some", "all"]
    assert sum(bucket["answers"] for bucket in patterns["recall_buckets"]) == sum(
        1 for row in rows if row["answerable"] and "recall@5" in (row.get("retrieval_metrics") or {}) and row.get("answer_judge")
    )


def test_publish_skips_patterns_when_the_raw_rows_are_gone(tmp_path):
    publish = _load("publish_benchmark")
    publish.write_patterns(tmp_path, tmp_path)
    assert not (tmp_path / "patterns.json").exists()


def test_rank_correlation():
    rows = [{"x": 1, "y": 10}, {"x": 2, "y": 20}, {"x": 3, "y": 30}, {"x": None, "y": 5}]
    assert check_results.rank_correlation(rows, "x", "y") == pytest.approx(1.0)
    assert check_results.rank_correlation([{"x": row["x"], "y": -row["y"]} for row in rows], "x", "y") == pytest.approx(-1.0)
    tied = [{"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 3}]
    assert check_results.rank_correlation(tied, "x", "y") == pytest.approx(0.866, abs=1e-3)


def test_a_republish_keeps_the_hand_recorded_files(root, tmp_path):
    publish = _load("publish_benchmark")
    run = next((root / "results").iterdir())
    dest = tmp_path / "demo-run"
    dest.mkdir()
    (dest / "report-desktop.png").write_bytes(b"png")
    (dest / "corrections.json").write_text("{}\n", encoding="utf-8")
    (dest / "stale.csv").write_text("old\n", encoding="utf-8")
    publish.copy(run, dest, publish.RUN_FILES)
    assert (dest / "report-desktop.png").is_file() and (dest / "corrections.json").is_file()
    assert not (dest / "stale.csv").exists()
