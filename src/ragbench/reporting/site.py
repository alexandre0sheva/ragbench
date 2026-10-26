"""Many runs as one static page: `ragbench runs site DIR` and the docs site's Leaderboard.

Everything is read from the files of finished runs (`metrics_summary.csv`, `run_summary.json`, `run_manifest.json`), so it works on runs from
any machine and nothing is re-run. A number a run did not measure stays missing (`None`, shown `—`), never zero. The page is one HTML file
with its data embedded as JSON and no network requests: filter by run, mode, system type or name, sort by any column, move the folder anywhere.
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

from ragbench.errors import UsageError
from ragbench.reporting.history import RunRecord, _summary_rows, load_record, scan_runs
from ragbench.reporting.history import _json as read_json

PAGE_NAME = "index.html"
REPORTS_DIR = "runs"
DEFAULT_PRIMARY_K = 5
_ENV = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
_MOCK_NOTE = "Mock run: scores validate the pipeline only (hashing embeddings, scripted LLM, heuristic judge). They say nothing about real-world quality."


class SiteError(UsageError):
    """There is nothing to build a site from."""


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _complete_runs(results_dir: Path) -> tuple[list[RunRecord], list[str]]:
    """The complete runs under `results_dir`, newest first, and the names of the folders that look like runs but have no leaderboard."""
    if not results_dir.is_dir():
        raise SiteError(f"{results_dir} does not exist.", hint="Pass the folder that holds your run directories (the --results-dir of `ragbench run`).")
    records = scan_runs(results_dir)
    complete = [record for record in records if record.status == "complete"]
    return list(reversed(complete)), [record.name for record in records if record.status != "complete"]


def _run_rows(record: RunRecord) -> list[dict[str, Any]]:
    summary = read_json(record.path / "run_summary.json")
    primary_k = int(summary.get("primary_k") or DEFAULT_PRIMARY_K)
    rows = []
    for row in _summary_rows(record.path):
        rows.append(
            {
                "run": record.name,
                "started": record.started,
                "mode": record.mode,
                "questions": record.questions,
                "system": str(row["system"]),
                "type": row.get("system_type"),
                "score": _number(row.get("answer_score")),
                "ci_lo": _number(row.get("answer_score_ci_lo")),
                "ci_hi": _number(row.get("answer_score_ci_hi")),
                "hit": _number(row.get(f"retrieval_hit@{primary_k}")),
                "hit_k": primary_k,
                "cost": _number(row.get("avg_cost_per_question")),
                "p50": _number(row.get("latency_ms_p50")),
                "errors": _number(row.get("n_error")),
                "report": f"{REPORTS_DIR}/{record.name}/report.html" if record.has_report else None,
            }
        )
    return rows


def site_rows(results_dir: Path) -> list[dict[str, Any]]:
    """One row per (run, system) of every complete run, newest run first, each run's systems in the order of its leaderboard."""
    records, _ = _complete_runs(results_dir)
    return [row for record in records for row in _run_rows(record)]


def write_site(results_dir: Path, out_dir: Path | None = None, *, title: str | None = None) -> Path:
    """Write `index.html` (default into `results_dir/site`) and copy each run's `report.html` next to it; return the page."""
    records, skipped = _complete_runs(results_dir)
    if not records:
        raise SiteError(f"No complete runs in {results_dir}.", hint="Start one with `ragbench run --config configs/all.yaml --mock`.")
    target = out_dir or results_dir / "site"
    rows = [row for record in records for row in _run_rows(record)]
    for record in records:
        if record.has_report:
            copy = target / REPORTS_DIR / record.name / "report.html"
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.path / "report.html", copy)
    modes = {row["mode"] for row in rows}
    data = {
        "rows": rows,
        "skipped": skipped,
        "runs": [record.name for record in records],
        "modes": sorted(str(m) for m in modes if m),
        "types": sorted({str(row["type"]) for row in rows if row["type"]}),
        "default_mode": "live" if "live" in modes and "mock" in modes else "",  # mock scores are not comparable with live ones: hidden until asked for
    }
    template = (resources.files("ragbench.reporting") / "templates" / "runs_site.html.j2").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")  # cannot close the script block
    page = _ENV.from_string(template).render(title=title or "RAGBench benchmark", data_json=payload, runs=len(records), count=len(rows), skipped=skipped)
    target.mkdir(parents=True, exist_ok=True)
    (target / PAGE_NAME).write_text(page, encoding="utf-8")
    return target / PAGE_NAME


# -- the docs site's Leaderboard ----------------------------------------------------------------------------------------------------------


def _cell(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def _score(row: dict[str, Any]) -> str:
    if row["score"] is None:
        return "—"
    interval = f" [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]" if row["ci_lo"] is not None and row["ci_hi"] is not None else ""
    return f"{row['score']:.2f}{interval}"


def leaderboard_markdown(benchmarks_dir: Path, link_prefix: str = "benchmarks") -> str:
    """The Leaderboard page of the docs site: a table per committed run under `benchmarks_dir/*-run`, systems by answer score. Built from the files only."""
    runs = [load_record(path) for path in sorted(benchmarks_dir.glob("*-run")) if path.is_dir()] if benchmarks_dir.is_dir() else []
    runs = sorted([run for run in runs if run.status == "complete"], key=lambda run: (run.started or "", run.name), reverse=True)
    lines = ["# Leaderboard", ""]
    if not runs:
        return "\n".join([*lines, "No benchmark runs are committed yet. They appear here, as tables built from their result files, once a run is added under `docs/benchmarks/`.", ""])
    lines += [
        "Every table below is generated from the result files of a run committed under `docs/benchmarks/`; nothing here is typed by hand. "
        "Systems are ordered by answer score; `—` means the run did not measure it, which is not the same as zero. "
        "How to read the numbers, and what they cannot tell you: [methodology](methodology.md).",
        "",
    ]
    for run in runs:
        rows = sorted(_run_rows(run), key=lambda row: (row["score"] is None, -(row["score"] or 0.0)))
        hit_k = rows[0]["hit_k"] if rows else DEFAULT_PRIMARY_K
        facts = [f"{run.mode} run" if run.mode else "run", (run.started or "")[:10] or None, f"{run.questions} questions" if run.questions is not None else None, f"{len(rows)} systems"]
        lines += [f"## `{run.name}`", "", ", ".join(fact for fact in facts if fact) + f". Files: [metrics_summary.csv]({link_prefix}/{run.name}/metrics_summary.csv).", ""]
        if run.mode == "mock":
            lines += [f"> {_MOCK_NOTE}", ""]
        lines += [f"| System | Type | Answer score | Hit@{hit_k} | $/question | Latency p50 |", "| --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            cost = "—" if row["cost"] is None else f"${row['cost']:.5f}"
            latency = "—" if row["p50"] is None else f"{row['p50']:.0f} ms"
            lines.append(f"| {row['system']} | {row['type'] or '—'} | {_score(row)} | {_cell(row['hit'], '.2f')} | {cost} | {latency} |")
        lines.append("")
    return "\n".join(lines)
