"""The shape of `report.html` and `report_data.json`, pinned: which sections, headings, charts, tables and data keys a report has. Numbers and
styling are not pinned (they change on purpose or with timing); a section that disappears or a data key that is renamed fails here.

Update the snapshot after an intended change with `RAGBENCH_UPDATE_SNAPSHOTS=1 pytest tests/test_report_snapshot.py`."""

from __future__ import annotations

import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

from ragbench.evaluation.evaluator import run_benchmark

SNAPSHOT = Path(__file__).parent / "golden" / "report_structure_v1.json"
WORD = {"type": "word", "chunk_size": 30, "chunk_overlap": 0}


class Structure(HTMLParser):
    """Collects what a reader would call the layout of the page, and nothing that depends on the run's numbers."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[str] = []
        self.headings: list[str] = []
        self.tables: list[str] = []
        self.charts: list[str] = []
        self.scripts: list[str] = []
        self.details: list[str] = []
        self.sort_headers: list[str] = []
        self.buttons: list[str] = []
        self._capture: str | None = None
        self._text = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "section" and a.get("id"):
            self.sections.append(a["id"])
        if tag in ("h2", "h3"):
            self._capture, self._text = tag, ""
        if tag == "table":
            self.tables.append(a.get("id") or "table")
        if tag == "svg" and "chart" in classes:
            self.charts.append(a.get("aria-label") or "chart")
        if tag == "script":
            self.scripts.append(a.get("id") or "behavior")
        if tag == "summary":
            self._capture, self._text = "summary", ""
        if tag == "button" and "data-sort" in a:
            self._capture, self._text = "sort", ""
        if tag == "button" and a.get("id"):
            self.buttons.append(a["id"])

    def handle_data(self, data):
        if self._capture:
            self._text += data

    def handle_endtag(self, tag):
        if self._capture in ("h2", "h3") and tag == self._capture:
            self.headings.append(f"{tag}: {' '.join(self._text.split())}")
            self._capture = None
        elif self._capture == "summary" and tag == "summary":
            self.details.append(" ".join(self._text.split()))
            self._capture = None
        elif self._capture == "sort" and tag == "button":
            self.sort_headers.append(" ".join(self._text.split()))
            self._capture = None


def _keys(value: Any, depth: int = 2) -> Any:
    """The key structure of a JSON value to `depth` levels (lists by their first element); leaves are named by type."""
    if isinstance(value, dict):
        return {k: _keys(v, depth - 1) if depth > 0 else "…" for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_keys(value[0], depth - 1)] if value and depth > 0 else ("list" if not value else "[…]")
    return type(value).__name__ if value is not None else "null"


def snapshot_of(run: Path) -> dict[str, Any]:
    page = Structure()
    page.feed((run / "report.html").read_text(encoding="utf-8"))
    data = json.loads((run / "report_data.json").read_text(encoding="utf-8"))
    structure = {
        "sections": page.sections,
        "headings": page.headings,
        "tables": page.tables,
        "charts": page.charts,
        "scripts": page.scripts,
        "details": page.details,
        "leaderboard_headers": page.sort_headers,
        "buttons": page.buttons,
        "data_keys": _keys(data),
    }
    return json.loads(json.dumps(structure))


def _run(root: Path, dataset: dict[str, str]) -> Path:
    config = {
        "run": {"name": "snapshot", "output_dir": str(root / "results")},
        "dataset": dataset,
        "systems": [
            {"type": "bm25", "name": "bm25", "chunker": WORD},
            {"type": "vector", "name": "vector", "chunker": WORD},
            {"type": "agent_search", "name": "agent", "chunker": WORD, "tools": ["calculator", "date_calc"]},
        ],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    path = root / "snapshot.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return run_benchmark(path, force_mock=True, max_workers=1, use_cache=False, run_dir=root / "results" / "snapshot")


def test_the_report_keeps_its_structure(tmp_path, tiny_dataset):
    current = snapshot_of(_run(tmp_path, tiny_dataset))
    if os.environ.get("RAGBENCH_UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for key in expected:
        assert current[key] == expected[key], f"report structure changed in '{key}' (RAGBENCH_UPDATE_SNAPSHOTS=1 pytest tests/test_report_snapshot.py to accept)"
    assert current.keys() == expected.keys()


def test_the_structure_snapshot_describes_a_full_report():
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert {"recommendation", "tradeoff", "leaderboard-section", "categories", "costs", "agents", "failures", "audit", "questions", "repro"} <= set(expected["sections"])
    assert expected["scripts"] == ["report-data", "questions-data", "behavior"]
