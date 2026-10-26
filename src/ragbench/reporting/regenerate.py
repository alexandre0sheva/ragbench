"""Rebuild every report of a finished run from the files in its directory, without running anything.

The run directory is the source of truth: `leaderboard.md`, `failures.md`, `qrels_audit.md`, the recommendation and `report.html` are all functions of
the CSV, JSON and JSONL files the run wrote. `ragbench report RUN` calls this, so a change to a report (or a lost file) never needs a re-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ragbench.evaluation.contrast import read_summaries
from ragbench.reporting.explorer_data import DEFAULT_MAX_EMBEDDED_MB
from ragbench.reporting.html_report import write_report
from ragbench.reporting.markdown_report import STAGES_FILE, stage_summary, write_failures, write_leaderboard, write_qrels_audit
from ragbench.reporting.report_data import ReportError, _read_csv, _read_json, _read_jsonl


def _max_embedded_mb(run_dir: Path) -> float:
    path = run_dir / "config.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        value = ((raw or {}).get("report") or {}).get("max_embedded_mb")
        return float(value) if value else DEFAULT_MAX_EMBEDDED_MB
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return DEFAULT_MAX_EMBEDDED_MB


def _stage_rows(run_dir: Path, per_question: list[dict[str, Any]], systems: list[str], notes: list[str]) -> list[dict[str, Any]]:
    """The stage tables: from the per-question rows, else from the `stages.json` the run wrote (a published run folder keeps no
    per-question rows, and rebuilding its leaderboard must not drop the tables)."""
    if per_question:
        return stage_summary(per_question, systems)
    saved = _read_json_list(run_dir / STAGES_FILE)
    if saved is None:
        notes.append(f"stage tables left out of leaderboard.md: the run has neither per_question_results.jsonl nor {STAGES_FILE}")
        return []
    return [row for row in saved if isinstance(row, dict) and row.get("system") in systems]


def _read_json_list(path: Path) -> list[Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except ValueError:
        return None
    return value if isinstance(value, list) else None


def regenerate_reports(run_dir: Path) -> tuple[list[Path], list[str]]:
    """Write the reports of `run_dir` again. Returns the files written and a note for each one that could not be written.

    Raises `ReportError` when the directory has no `metrics_summary.csv` (nothing to report on).
    """
    summary = _read_csv(run_dir / "metrics_summary.csv")
    if not summary:
        raise ReportError(f"{run_dir} has no metrics_summary.csv with results to report on.")
    run_summary, stats = _read_json(run_dir / "run_summary.json"), _read_json(run_dir / "stats.json")
    per_question = _read_jsonl(run_dir / "per_question_results.jsonl")
    primary_k = run_summary.get("primary_k")
    written: list[Path] = []
    notes: list[str] = []

    write_leaderboard(
        run_dir / "leaderboard.md",
        summary,
        notices=run_summary.get("notices") or None,
        primary_k=primary_k,
        stage_rows=_stage_rows(run_dir, per_question, [str(row["system"]) for row in summary], notes),
        significance_rows=_read_csv(run_dir / "significance.csv"),
        stats_info=stats or None,
        pairwise=run_summary.get("pairwise"),
        split=(run_summary.get("dataset") or {}).get("split"),
        contrast=read_summaries(run_dir),
    )
    written.append(run_dir / "leaderboard.md")
    if per_question:
        write_failures(run_dir / "failures.md", per_question)
        written.append(run_dir / "failures.md")
    if (run_dir / "qrels_audit.csv").exists():
        write_qrels_audit(run_dir / "qrels_audit.md", _read_csv(run_dir / "qrels_audit.csv"), primary_k=primary_k or 5)
        written.append(run_dir / "qrels_audit.md")

    # The recommendation is decided again from the run's own `selection:` settings (it is seeded, so it comes out the same).
    if (run_dir / "config.yaml").exists():
        try:
            from ragbench.selection.recommend import recommend, selection_of_run, write_recommendation

            selection = selection_of_run(run_dir)
            paths = write_recommendation(run_dir, recommend(run_dir, constraints=selection.constraints, weights=selection.weights, profile=selection.profile))
            written.extend(paths.values())
        except Exception as exc:  # noqa: BLE001 (the other reports are still good)
            notes.append(f"recommendation not rewritten: {type(exc).__name__}: {exc}")
    else:
        notes.append("recommendation not rewritten: the run has no config.yaml")

    written.append(write_report(run_dir, max_embedded_mb=_max_embedded_mb(run_dir)))
    return written, notes
