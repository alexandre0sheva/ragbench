"""Past runs: find them in a results folder, describe them, and write `results/index.html`, a static history page linking each run's report.

A run is a directory the benchmark wrote (`metrics_summary.csv`, `run_summary.json`, `report.html`, ...). Everything here only reads those files,
so it works on runs from any machine and on runs that stopped early (they are listed as `incomplete`).
"""

from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from jinja2 import Environment, select_autoescape

from ragbench.errors import RunNotFoundError

RUN_MARKERS = ("metrics_summary.csv", "run_manifest.json", "run_summary.json", "auto.json", "config.yaml")
INDEX_NAME = "index.html"
MIN_SPARK_SPAN = 1.0  # answer-score points: the least range a sparkline is stretched over
_ENV = Environment(autoescape=select_autoescape(default=True, default_for_string=True))


@dataclass
class RunRecord:
    path: Path
    name: str
    status: str  # "complete" (it has a leaderboard) or "incomplete" (it stopped early or is still running)
    started: str | None = None
    mode: str | None = None
    systems: list[str] = field(default_factory=list)
    questions: int | None = None
    winner: str | None = None
    charged_usd: float | None = None
    scores: dict[str, float] = field(default_factory=dict)  # system -> mean answer score
    has_report: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "status": self.status,
            "started": self.started,
            "mode": self.mode,
            "systems": self.systems,
            "questions": self.questions,
            "winner": self.winner,
            "charged_usd": self.charged_usd,
            "scores": self.scores,
            "has_report": self.has_report,
        }


def _json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _clean(value: Any) -> Any:
    """A pandas/numpy value as plain JSON data: NaN and infinity become None."""
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else value


def _summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics_summary.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path)
    except (ValueError, pd.errors.EmptyDataError):
        return []
    return [{str(key): _clean(value) for key, value in row.items()} for row in frame.to_dict("records")]


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).exists() for marker in RUN_MARKERS)


def load_record(run_dir: Path) -> RunRecord:
    rows = _summary_rows(run_dir)
    summary, manifest = _json(run_dir / "run_summary.json"), _json(run_dir / "run_manifest.json")
    started = manifest.get("started_utc")
    if not started:
        try:
            started = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            started = None
    dataset = summary.get("dataset") or {}
    questions = dataset.get("questions")
    if questions is None and summary.get("num_question_rows") and summary.get("num_systems"):
        questions = int(summary["num_question_rows"]) // int(summary["num_systems"])
    return RunRecord(
        path=run_dir,
        name=run_dir.name,
        status="complete" if rows else "incomplete",
        started=started,
        mode=summary.get("mode") or manifest.get("mode"),
        systems=[str(row["system"]) for row in rows],
        questions=questions,
        winner=_json(run_dir / "recommendation.json").get("winner"),
        charged_usd=(summary.get("cache") or {}).get("charged_cost_usd"),
        scores={str(row["system"]): float(row["answer_score"]) for row in rows if row.get("answer_score") is not None},
        has_report=(run_dir / "report.html").is_file(),
    )


def scan_runs(results_dir: Path) -> list[RunRecord]:
    """Every run directory directly under `results_dir`, oldest first."""
    if not results_dir.is_dir():
        return []
    records = [load_record(child) for child in sorted(results_dir.iterdir()) if not child.name.startswith(".") and is_run_dir(child)]
    return sorted(records, key=lambda record: (record.started or "", record.name))


def resolve_run(ref: str | Path, results_dir: Path) -> Path:
    """The run directory a user means: a path, a name under `results_dir`, `latest`, or a name fragment that matches exactly one run."""
    candidate = Path(ref)
    if is_run_dir(candidate):
        return candidate
    if str(ref) == "latest":
        runs = scan_runs(results_dir)
        if not runs:
            raise RunNotFoundError(f"No runs found in {results_dir}.", hint="Pass --results-dir, or start one with `ragbench run`.")
        return runs[-1].path
    if is_run_dir(results_dir / candidate):
        return results_dir / candidate
    names = [record.name for record in scan_runs(results_dir)]
    fragment = [name for name in names if str(ref) in name]
    if len(fragment) == 1:
        return results_dir / fragment[0]
    if len(fragment) > 1:
        raise RunNotFoundError(f"'{ref}' matches {len(fragment)} runs in {results_dir}: {', '.join(fragment[-5:])}. Give more of the name.")
    shown = f" Runs there: {', '.join(names[-5:])}." if names else ""
    raise RunNotFoundError(f"No run '{ref}' (looked at that path and in {results_dir}).{shown}", hint="`ragbench runs` lists them.")


def describe_run(run_dir: Path) -> dict[str, Any]:
    """A finished run as one JSON-ready dictionary (what `ragbench run --json` prints): identity, per-system numbers, the recommendation, files."""
    record = load_record(run_dir)
    summary = _json(run_dir / "run_summary.json")
    files = {
        key: str(run_dir / name)
        for key, name in (
            ("report", "report.html"),
            ("leaderboard", "leaderboard.md"),
            ("recommendation", "recommendation.md"),
            ("winner", "winner.yaml"),
            ("metrics", "metrics_summary.csv"),
            ("per_question", "per_question_results.jsonl"),
        )
        if (run_dir / name).is_file()
    }
    return {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id") or record.name,
        "status": record.status,
        "mode": record.mode,
        "started": record.started,
        "questions": record.questions,
        "errors": summary.get("num_errors", 0),
        "charged_usd": record.charged_usd,
        "real_spend_usd": (summary.get("cache") or {}).get("real_spend_usd"),
        "winner": record.winner,
        "systems": _summary_rows(run_dir),
        "recommendation": _json(run_dir / "recommendation.json") or None,
        "files": files,
    }


# -- the history page ---------------------------------------------------------------------------


def sparkline_svg(values: list[float], *, label: str, width: int = 120, height: int = 28) -> str:
    """A tiny line of `values` (answer scores, oldest first) as inline SVG, scaled to its own range but never to less than one point, so noise does
    not look like a trend. One value is a dot; none is an empty box."""
    pad = 3.0
    inner_w, inner_h = width - 2 * pad, height - 2 * pad
    lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
    if hi - lo < MIN_SPARK_SPAN:
        middle = (hi + lo) / 2
        lo, hi = middle - MIN_SPARK_SPAN / 2, middle + MIN_SPARK_SPAN / 2
    points = [(pad + (inner_w * i / (len(values) - 1) if len(values) > 1 else inner_w / 2), pad + inner_h * (1 - (v - lo) / (hi - lo))) for i, v in enumerate(values)]
    body = ""
    if len(points) > 1:
        body += f'<polyline class="line" fill="none" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}"/>'
    if points:
        x, y = points[-1]
        body += f'<circle class="last" cx="{x:.1f}" cy="{y:.1f}" r="2.6"/>'
    title = html.escape(label)
    return f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{title}"><title>{title}</title>{body}</svg>'


def _relative(target: Path, start: Path) -> str:
    return "/".join(quote(part) for part in Path(os.path.relpath(target, start)).parts)


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}" if value >= 1 else f"${value:.4f}" if value else "$0"


def _when(stamp: str | None) -> str:
    try:
        return datetime.strptime(stamp or "", "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stamp or "—"


def index_context(page_dir: Path, results_dir: Path, records: list[RunRecord]) -> dict[str, Any]:
    """What the history page shows. Links are relative to `page_dir`, where the page is written."""
    runs = []
    for record in reversed(records):  # newest first
        link = _relative(record.path / "report.html", page_dir) if record.has_report else None
        runs.append(
            {
                "name": record.name,
                "link": link,
                                "when": _when(record.started),
                "mode": record.mode or "—",
                "status": record.status,
                "systems": len(record.systems),
                "questions": record.questions if record.questions is not None else "—",
                "winner": record.winner,
                "cost": _money(record.charged_usd),
            }
        )
    series: dict[str, list[tuple[str, float]]] = {}
    for record in records:  # oldest first, so a series reads left to right in time
        for system, score in record.scores.items():
            series.setdefault(system, []).append((record.name, score))
    trends = []
    for system, points in sorted(series.items()):
        values = [score for _, score in points]
        label = f"{system}: answer score over {len(values)} run(s), latest {values[-1]:.2f}"
        trends.append({"system": system, "runs": len(values), "latest": f"{values[-1]:.2f}", "first": f"{values[0]:.2f}", "svg": sparkline_svg(values, label=label)})
    return {"runs": runs, "trends": trends, "folder": results_dir.name or str(results_dir), "count": len(records)}


def write_index(results_dir: Path, path: Path | None = None) -> Path:
    """Write the history page (default `results_dir/index.html`): every run with its winner and link, and a score-over-time line per system name."""
    records = scan_runs(results_dir)
    template = (resources.files("ragbench.reporting") / "templates" / "runs_index.html.j2").read_text(encoding="utf-8")
    target = path or results_dir / INDEX_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    context = index_context(target.parent, results_dir, records)
    target.write_text(_ENV.from_string(template).render(**context), encoding="utf-8")
    return target
