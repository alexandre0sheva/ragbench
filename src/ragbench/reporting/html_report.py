"""The self-contained HTML report of a run: `write_report(run_dir)` turns the files a run wrote into `report.html` (and `report_data.json`).

The page is a Jinja template (`templates/report.html.j2`) with its stylesheet and script inlined, charts pre-rendered as SVG (`charts.py`), and the
data it was built from embedded as JSON. It loads nothing from the network. All numbers come from `report_data.build_report`.
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

from ragbench.reporting import charts
from ragbench.reporting.columns import MISSING
from ragbench.reporting.explorer_data import DEFAULT_MAX_EMBEDDED_MB, QUESTIONS_FILE, script_json
from ragbench.reporting.report_data import FAILURE_NAMES, FAILURE_OTHER, ReportBuild, build_report, money

TEMPLATES = "ragbench.reporting"
_ENV = Environment(autoescape=select_autoescape(default=True, default_for_string=True), trim_blocks=False, lstrip_blocks=False)


def _asset(name: str) -> str:
    return (resources.files(TEMPLATES) / "templates" / name).read_text(encoding="utf-8")


def _num(value: float | None, fmt: str = "{:.2f}") -> str:
    return MISSING if value is None else fmt.format(value)


def _date(stamp: str | None) -> str:
    if not stamp:
        return "—"
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").strftime("%-d %b %Y, %H:%M UTC")
    except ValueError:
        return stamp


CONSTRAINT_LABELS = {
    "max_cost_per_question": lambda v: f"≤ {money(v)} per question",
    "max_latency_ms_p95": lambda v: f"p95 ≤ {v:.0f} ms",
    "min_faithfulness": lambda v: f"faithfulness ≥ {v:g}",
    "min_answer_score": lambda v: f"answer score ≥ {v:g}",
    "max_ingestion_cost": lambda v: f"indexing ≤ {money(v)}",
    "require_local_models": lambda v: "local models only",
    "require_no_network": lambda v: "no data leaves the machine",
}


def _view(build: ReportBuild) -> dict[str, Any]:
    data, view = build.data, build.view
    scatter = data["scatter"]
    cost_axis = scatter["x_metric"] == "cost"
    points = [charts.Point(p["name"], p["x"], p["y"], p["size"], p["kind"], p["tip"]) for p in scatter["points"]]
    scatter_svg = charts.scatter_svg(
        points,
        x_label="Cost per question (USD)" if cost_axis else "Average latency (ms)",
        y_label="Answer score (0-5)",
        x_fmt=(lambda v: money(v)) if cost_axis else (lambda v: f"{v:.0f}"),
        y_fmt=lambda v: f"{v:.1f}",
        width=900,
        height=440,
    )
    latency_by_name = {row["system"]: row for row in data["leaderboard"]["rows"]}
    avg_latency = {name: next((c["raw"] for c in row["cells"] if c["key"] == "avg_latency_ms"), None) for name, row in latency_by_name.items()}
    scatter_table = [
        {
            "name": p["name"],
            "kind": {"winner": "recommended", "tie": "tied", "pareto": "Pareto-optimal", "other": "—"}[p["kind"]],
            "quality": _num(p["y"]),
            "x": money(p["x"]) if cost_axis else _num(p["x"], "{:.0f}"),
            "latency": _num(avg_latency.get(p["name"]), "{:.0f}"),
        }
        for p in scatter["points"]
    ]

    heat = data["heatmap"]
    heat_ctx = heat_table = None
    if heat:
        def make(grid: list[list[float | None]], name: str) -> tuple[str, str, str]:
            flat = [v for row in grid for v in row if v is not None]
            lo, hi = (min(flat), max(flat)) if flat else (0.0, 0.0)
            svg = charts.heatmap_svg(heat["systems"], heat["categories"], grid, col_notes=[str(heat["question_counts"][c]) for c in heat["categories"]], value_name=name)
            return svg, _num(lo if flat else None), _num(hi if flat else None)

        answer_svg, a_lo, a_hi = make(heat["answer_score"], "answer score")
        recall_svg = r_lo = r_hi = None
        if heat["recall"] and any(v is not None for row in heat["recall"] for v in row):
            recall_svg, r_lo, r_hi = make(heat["recall"], f"recall@{heat['recall_k']}")
        heat_ctx = {"answer_svg": answer_svg, "answer_lo": a_lo, "answer_hi": a_hi, "recall_svg": recall_svg, "recall_lo": r_lo, "recall_hi": r_hi, "recall_k": heat["recall_k"]}
        heat_table = {
            "categories": [f"{c} (n={heat['question_counts'][c]})" for c in heat["categories"]],
            "rows": [{"system": s, "cells": [_num(v) for v in heat["answer_score"][i]]} for i, s in enumerate(heat["systems"])],
        }

    stages = data["stages"]
    stage_svg = None
    stage_legend: list[dict[str, str]] = []
    stage_table: list[dict[str, Any]] = []
    if stages and not stages["all_zero"]:
        keys = stages["keys"]
        stage_legend = [{"slot": view["stage_slots"][k], "name": view["stage_names"][k], "key": k} for k in keys]
        stage_svg = charts.stacked_bars_svg(
            [(s["system"], s["cost"]) for s in stages["systems"]], keys, view["stage_slots"], fmt=money, key_names=view["stage_names"], value_name="cost per question", width=520
        )
        stage_table = [{"system": s["system"], "cells": [money(s["cost"][k]) for k in keys], "total": money(sum(s["cost"][k] for k in keys))} for s in stages["systems"]]

    latency_svg = charts.latency_bars_svg([(row["system"], row["p50"], row["p95"]) for row in data["latency"]], width=520 if stage_svg else 900)
    latency_table = [{"system": r["system"], "p50": _num(r["p50"], "{:.0f}"), "p95": _num(r["p95"], "{:.0f}"), "source": r["source"] or MISSING} for r in data["latency"]]

    agents = data["agents"]
    agent_headers = ["Steps", "LLM calls", "Tool calls", "Tool errors", "Required tool used", "Budget hit", "Route accuracy"]
    agent_table: list[dict[str, Any]] = []
    if agents and agents["rows"]:
        keys = ["avg_steps", "avg_llm_calls", "avg_tool_calls", "tool_error_rate", "required_tool_used_rate", "budget_exhausted_rate", "route_accuracy"]
        formats = ["{:.1f}", "{:.1f}", "{:.1f}", "{:.0%}", "{:.0%}", "{:.0%}", "{:.0%}"]
        peaks = {k: max((r[k] or 0 for r in agents["rows"]), default=0) or 1 for k in keys[:3]}
        for row in agents["rows"]:
            cells = [{"text": _num(row[k], f), "bar": round(48 * (row[k] or 0) / peaks[k]) if k in peaks and row[k] is not None else None} for k, f in zip(keys, formats, strict=True)]
            agent_table.append({"system": row["system"], "cells": cells})

    failures = data["failures"]
    failure_svg = None
    failure_legend: list[dict[str, str]] = []
    failure_table: list[dict[str, Any]] = []
    if failures:
        keys = failures["keys"]
        failure_legend = [{"slot": view["failure_slots"][k], "name": view["failure_names"][k]} for k in keys]
        if keys:
            failure_svg = charts.stacked_bars_svg(
                [(s["system"], {k: float(s["counts"].get(k, 0)) for k in keys}) for s in failures["systems"]],
                keys,
                view["failure_slots"],
                fmt=lambda v: f"{int(round(v))}",
                key_names=view["failure_names"],
                value_name="failed questions",
                width=900,
            )
        failure_table = [{"system": s["system"], "questions": s["questions"], "cells": [s["counts"].get(k, 0) for k in keys], "total": s["total"]} for s in failures["systems"]]

    rec = data["recommendation"]
    constraint_labels = [CONSTRAINT_LABELS[k](v) for k, v in (rec["constraints"].items() if rec else []) if k in CONSTRAINT_LABELS]

    sections = []
    if rec:
        sections.append(("recommendation", "Recommendation"))
    sections.append(("tradeoff", "Quality vs cost" if cost_axis else "Quality vs latency"))
    sections.append(("leaderboard-section", "Leaderboard"))
    if heat:
        sections.append(("categories", "Categories"))
    sections.append(("costs", "Cost and latency"))
    if agents:
        sections.append(("agents", "Agents and tools"))
    if failures:
        sections += [("failures", "Failures"), ("audit", "Label audit")]
    if build.questions:
        sections.append(("questions", "Questions"))
    sections.append(("repro", "Reproducibility"))
    numbers = {key: f"{n:02d}" for n, (key, _) in enumerate(sections, start=1)}
    sec = {"recommendation": numbers.get("recommendation"), "tradeoff": numbers["tradeoff"], "leaderboard": numbers["leaderboard-section"], "categories": numbers.get("categories"),
           "costs": numbers["costs"], "agents": numbers.get("agents"), "failures": numbers.get("failures"), "audit": numbers.get("audit"), "questions": numbers.get("questions"), "repro": numbers["repro"]}
    return {
        "css": _asset("report.css"),
        "js": _asset("report.js"),
        "run": data["run"],
        "started": _date(data["run"]["started_utc"]),
        "notices": data["notices"],
        "dataset": data["dataset"],
        "cost": data["cost"],
        "money": money,
        "rec": rec,
        "constraint_labels": constraint_labels,
        "scatter": scatter,
        "scatter_svg": scatter_svg,
        "scatter_table": scatter_table,
        "leaderboard": data["leaderboard"],
        "whiskers": view["whiskers"],
        "baseline": data["baseline"],
        "heat": heat_ctx,
        "heat_table": heat_table,
        "stage_svg": stage_svg,
        "stage_legend": stage_legend,
        "stage_table": stage_table,
        "latency_svg": latency_svg,
        "latency_table": latency_table,
        "agents": agents,
        "agent_headers": agent_headers,
        "agent_table": agent_table,
        "failures": failures,
        "failure_svg": failure_svg,
        "failure_legend": failure_legend,
        "failure_table": failure_table,
        "other_types": ", ".join(FAILURE_NAMES.get(k, k).lower() for k in FAILURE_OTHER),
        "audit": data["qrels_audit"],
        "config_yaml": data["reproducibility"]["config_yaml"],
        "sections": [{"id": key, "title": title, "number": numbers[key]} for key, title in sections],
        "sec": sec,
        "repo_text": view["repo_text"],
        "blob": script_json(data),
        "questions": build.questions.embedded if build.questions else None,
        "questions_blob": script_json(build.questions.embedded) if build.questions else "",
    }


def render_report(build: ReportBuild) -> str:
    return _ENV.from_string(_asset("report.html.j2")).render(**_view(build))


def write_report(run_dir: Path, path: Path | None = None, max_embedded_mb: float = DEFAULT_MAX_EMBEDDED_MB) -> Path:
    """Write `report.html` (default: in `run_dir`) and `report_data.json` next to it, from the files of the finished run in `run_dir`.

    The question explorer's data is embedded up to `max_embedded_mb`; beyond that the page keeps a reduced copy and the complete data is written to
    `report_questions.json` beside it (a stale one from an earlier build is removed when everything fits).
    """
    build = build_report(run_dir, max_embedded_mb)
    target = path or run_dir / "report.html"
    target.write_text(render_report(build), encoding="utf-8")
    (target.parent / "report_data.json").write_text(json.dumps(build.data, ensure_ascii=False, indent=1), encoding="utf-8")
    questions_file = target.parent / QUESTIONS_FILE
    if build.questions and build.questions.full is not None:
        questions_file.write_text(json.dumps(build.questions.full, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    else:
        questions_file.unlink(missing_ok=True)
    return target
