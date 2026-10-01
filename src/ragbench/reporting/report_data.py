"""Everything the HTML report shows, computed from a finished run directory.

`build_report(run_dir)` reads the files a run wrote (`metrics_summary.csv`, `per_question_results.jsonl`, `recommendation.json`, `stats.json`,
`significance.csv`, ...) and returns two things: `data`, a JSON-safe dictionary (embedded in the page and written as `report_data.json` for other
tools), and `view`, the same facts as the page needs them (pre-rendered SVG and HTML-ready strings). Because it only reads files, a report can be
rebuilt from any past run. Every file but `metrics_summary.csv` is optional: a missing one just leaves its section out.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ragbench.rag_systems.trace import STAGE_KEYS, UNTRACKED
from ragbench.reporting import charts
from ragbench.reporting.columns import MISSING, Column, extra_columns, format_cell, is_missing, leaderboard_columns, to_float
from ragbench.reporting.explorer_data import DEFAULT_MAX_EMBEDDED_MB, QuestionsPayload, build_questions
from ragbench.reporting.notices import build_notices

SCHEMA_VERSION = 1
REPO_TEXT = "github.com/alexandre0sheva/ragbench"  # shown as text: the report makes no outgoing links and loads nothing from the network

# Fixed color slots (not rank-based): a stage or failure type keeps its color in every run. Segments are drawn in slot order, the order the palette was validated in.
STAGE_SLOTS = {"retrieve": "s1", "rerank": "s2", "llm": "s3", "tool": "s4", "embed": "s5", "route": "s6", "grade": "s7", "generate": "s8", UNTRACKED: "sgray"}
STAGE_NAMES = {"retrieve": "Retrieve", "rerank": "Rerank", "llm": "LLM calls", "tool": "Tools", "embed": "Embed", "route": "Route", "grade": "Grade", "generate": "Generate", UNTRACKED: "Untracked"}
FAILURE_SLOTS = {
    "retrieval_miss": "s1",
    "bad_reranking": "s2",
    "insufficient_context": "s3",
    "partial_answer": "s4",
    "wrong_entity": "s5",
    "over_refusal": "s6",
    "wrong_date": "s7",
    "answer_hallucination": "s8",
}
FAILURE_NAMES = {
    "retrieval_miss": "Retrieval miss",
    "bad_reranking": "Bad reranking",
    "insufficient_context": "Insufficient context",
    "partial_answer": "Partial answer",
    "wrong_entity": "Wrong entity",
    "over_refusal": "Over-refusal",
    "wrong_date": "Wrong date",
    "answer_hallucination": "Hallucination",
    "other": "Other",
}
FAILURE_OTHER = ("possible_qrels_gap", "format_error", "run_error")
SIGNIFICANCE_SYMBOL = {"better": "▲", "worse": "▼"}


class ReportError(ValueError):
    """The run directory has nothing to report on."""


@dataclass
class ReportBuild:
    data: dict[str, Any]
    view: dict[str, Any]
    questions: QuestionsPayload | None = None  # the explorer's data: the page's copy, and the complete one when the page's had to be cut down


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path)
    except (ValueError, pd.errors.EmptyDataError):
        return []
    return [{key: (None if is_missing(value) and not isinstance(value, str) else value) for key, value in row.items()} for row in frame.to_dict("records")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def money(value: float | None) -> str:
    if value is None:
        return MISSING
    return f"${value:,.2f}" if value >= 1 else f"${value:.5f}" if value else "$0"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _order(summary: list[dict[str, Any]], recommendation: dict[str, Any]) -> list[str]:
    names = [str(row["system"]) for row in summary]
    ranked = [str(item["system"]) for item in recommendation.get("ranking", []) if str(item["system"]) in names]
    return [*ranked, *[name for name in names if name not in ranked]]


def _recommendation(recommendation: dict[str, Any], winner_yaml: str, summary: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not recommendation:
        return None
    winner = recommendation.get("winner")
    constraints = {key: value for key, value in (recommendation.get("constraints") or {}).items() if value not in (None, False)}
    rationale = [str(line).replace("`", "") for line in recommendation.get("rationale", [])]
    mock_lines = [line for line in rationale if line.startswith("Mock run")]
    reasons = [line for line in rationale if line not in mock_lines]
    row = summary.get(str(winner), {}) if winner else {}
    return {
        "winner": winner,
        "profile": recommendation.get("profile"),
        "weights": recommendation.get("weights"),
        "constraints": constraints,
        "tied_with_winner": recommendation.get("tied_with_winner", []),
        "pareto": recommendation.get("pareto", []),
        "summary": reasons[0] if reasons else "",
        "reasons": reasons,
        "infeasible": recommendation.get("infeasible", {}),
        "closest_miss": recommendation.get("closest_miss"),
        "quality_metric": recommendation.get("quality_metric"),
        "winner_facts": {
            "answer_score": to_float(row.get("answer_score")),
            "cost_per_question": to_float(row.get("avg_cost_per_question")),
            "latency_ms_p95": to_float(row.get("latency_ms_p95")),
            "faithfulness": to_float(row.get("faithfulness")),
        },
        "winner_yaml": winner_yaml,
    }


def _significance(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    found = {}
    for row in rows:
        verdict = str(row.get("verdict") or "")
        symbol = SIGNIFICANCE_SYMBOL.get(verdict, "≈")
        tip = (
            f"vs {row['baseline']}: {float(row['mean_diff']):+.3f} (95% CI {float(row['ci_lo']):+.3f} to {float(row['ci_hi']):+.3f}); "
            f"Holm-adjusted p = {float(row['p_holm']):.3f}; {verdict}"
        )
        found[(str(row["system"]), str(row["metric"]))] = {"symbol": symbol, "verdict": verdict, "tip": tip, "p_holm": to_float(row.get("p_holm"))}
    return found


def _leaderboard(summary: list[dict[str, Any]], order: list[str], columns: list[Column], extras: list[Column], badges: dict[str, list[str]], significance: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["system"]): row for row in summary}
    best: dict[str, float] = {}
    for column in [*columns, *extras]:
        values = [v for row in summary if (v := to_float(row.get(column.key))) is not None]
        if values and len(set(values)) > 1:  # a column where every system is the same has no best
            best[column.key] = max(values) if column.higher_is_better else min(values)
    scales: dict[str, tuple[float, float]] = {}
    for column in columns:
        if column.interval:
            lows = [v for row in summary if (v := to_float(row.get(f"{column.key}_ci_lo"))) is not None]
            highs = [v for row in summary if (v := to_float(row.get(f"{column.key}_ci_hi"))) is not None]
            if lows and highs:
                scales[column.key] = (min(lows), max(highs))
    rows, view_cells = [], {}
    for name in order:
        row = by_name[name]
        cells = []
        for column in [*columns, *extras]:
            number = to_float(row.get(column.key))
            lo, hi = to_float(row.get(f"{column.key}_ci_lo")), to_float(row.get(f"{column.key}_ci_hi"))
            cells.append(
                {
                    "key": column.key,
                    "raw": number,
                    "text": charts_text(column, row),
                    "ci": [lo, hi] if column.interval and lo is not None and hi is not None else None,
                    "best": number is not None and number == best.get(column.key),
                    "sig": significance.get((name, column.key)),
                }
            )
            if column.interval and column.key in scales and lo is not None:
                view_cells[(name, column.key)] = charts.whisker_svg(number, lo, hi, *scales[column.key])
        rows.append({"system": name, "type": row.get("system_type"), "badges": badges.get(name, []), "cells": cells})
    return {
        "columns": [{"key": c.key, "header": c.header, "higher_is_better": c.higher_is_better, "interval": c.interval, "default": True} for c in columns]
        + [{"key": c.key, "header": c.header, "higher_is_better": c.higher_is_better, "interval": False, "default": False} for c in extras],
        "rows": rows,
        "_whiskers": view_cells,
    }


def charts_text(column: Column, row: dict[str, Any]) -> str:
    """The cell text without its interval (the interval is drawn as a whisker and listed in the table view)."""
    return format_cell(Column(column.key, column.header, column.fmt, column.higher_is_better, interval=False), row)


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("error") is None:
            grouped[row["system"]].append(row)
    return grouped


def _heatmaps(grouped: dict[str, list[dict[str, Any]]], order: list[str], primary_k: int | None) -> dict[str, Any] | None:
    if not grouped:
        return None
    counts: dict[str, set[str]] = defaultdict(set)
    answer: dict[tuple[str, str], list[float]] = defaultdict(list)
    recall: dict[tuple[str, str], list[float]] = defaultdict(list)
    for system, rows in grouped.items():
        for row in rows:
            category = str(row.get("category") or "unknown")
            counts[category].add(row["question_id"])
            score = to_float((row.get("answer_judge") or {}).get("answer_score"))
            if score is not None:
                answer[(system, category)].append(score)
            if primary_k is not None and (value := to_float((row.get("retrieval_metrics") or {}).get(f"recall@{primary_k}"))) is not None:
                recall[(system, category)].append(value)
    categories = sorted(counts, key=lambda c: (-len(counts[c]), c))
    systems = [name for name in order if name in grouped]

    def grid(source: dict[tuple[str, str], list[float]]) -> list[list[float | None]]:
        return [[_mean(source.get((system, category), [])) for category in categories] for system in systems]

    return {
        "categories": categories,
        "question_counts": {category: len(counts[category]) for category in categories},
        "systems": systems,
        "answer_score": grid(answer),
        "recall": grid(recall) if recall else None,
        "recall_k": primary_k,
    }


def _stage_costs(grouped: dict[str, list[dict[str, Any]]], order: list[str]) -> dict[str, Any] | None:
    systems: list[dict[str, Any]] = []
    for name in order:
        rows = grouped.get(name)
        if not rows:
            continue
        totals = dict.fromkeys(STAGE_KEYS, 0.0)
        for row in rows:
            for step in row.get("steps") or []:
                kind = UNTRACKED if step.get("name") == UNTRACKED else step.get("kind")
                if kind in totals:
                    totals[kind] += float((step.get("cost") or {}).get("total_cost", 0.0))
        systems.append({"system": name, "cost": {kind: total / len(rows) for kind, total in totals.items()}})
    if not systems:
        return None
    present = [key for key in STAGE_KEYS if any(s["cost"][key] > 0 for s in systems)]
    return {"keys": present, "systems": systems, "all_zero": not present}


def _failures(grouped: dict[str, list[dict[str, Any]]], order: list[str]) -> dict[str, Any] | None:
    if not grouped:
        return None
    systems: list[dict[str, Any]] = []
    for name in order:
        rows = grouped.get(name)
        if rows is None:
            continue
        counts: Counter[str] = Counter()
        for row in rows:
            kind = str(row.get("failure_type") or "no_failure")
            if kind != "no_failure":
                counts["other" if kind not in FAILURE_SLOTS else kind] += 1
        systems.append({"system": name, "questions": len(rows), "counts": dict(counts), "total": sum(counts.values())})
    present = [key for key in [*FAILURE_SLOTS, "other"] if any(s["counts"].get(key) for s in systems)]
    return {"keys": present, "systems": systems}


def _failure_detail(rows_all: list[dict[str, Any]]) -> Counter[str]:
    """How the folded-in `other` failures break down (shown in the table view, so nothing is hidden by the grouping)."""
    detail: Counter[str] = Counter()
    for row in rows_all:
        kind = str(row.get("failure_type") or "")
        if kind in FAILURE_OTHER:
            detail[kind] += 1
    return detail


def _agents(summary: list[dict[str, Any]], order: list[str], tools: list[dict[str, Any]], routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    keys = ("avg_steps", "avg_llm_calls", "avg_tool_calls", "tool_error_rate", "required_tool_used_rate", "budget_exhausted_rate", "route_accuracy")
    by_name = {str(row["system"]): row for row in summary}
    rows = []
    for name in order:
        row = by_name[name]
        values = {key: to_float(row.get(key)) for key in keys}
        if any(value is not None for value in values.values()):
            rows.append({"system": name, **values})
    if not rows and not routes:
        return None
    return {"rows": rows, "tools": tools, "routes": routes}


def _qrels_audit(rows: list[dict[str, Any]], systems: list[str]) -> dict[str, Any]:
    by_system = {name: {"high": 0, "medium": 0} for name in systems}
    for row in rows:
        entry = by_system.setdefault(str(row.get("system")), {"high": 0, "medium": 0})
        entry[str(row.get("severity") or "medium")] = entry.get(str(row.get("severity") or "medium"), 0) + 1
    questions = {str(row.get("question_id")) for row in rows}
    return {"candidates": len(rows), "questions": len(questions), "by_system": by_system}


def _scatter(summary: list[dict[str, Any]], order: list[str], kinds: dict[str, str]) -> dict[str, Any]:
    by_name = {str(row["system"]): row for row in summary}
    costs = [to_float(row.get("avg_cost_per_question")) for row in summary]
    use_cost = any(c is not None and c > 0 for c in costs)
    points = []
    for name in order:
        row = by_name[name]
        quality, cost, latency = to_float(row.get("answer_score")), to_float(row.get("avg_cost_per_question")), to_float(row.get("avg_latency_ms"))
        x = cost if use_cost else latency
        tip = f"{name} · answer {quality:.2f}" if quality is not None else f"{name} · no answer score"
        tip += f" · {money(cost)}/question" + (f" · {latency:.0f} ms" if latency is not None else "")
        points.append({"name": name, "x": x, "y": quality, "size": latency if use_cost else None, "kind": kinds[name], "tip": tip})
    return {"x_metric": "cost" if use_cost else "latency", "points": points}


def build_report(run_dir: Path, max_embedded_mb: float = DEFAULT_MAX_EMBEDDED_MB) -> ReportBuild:
    summary_path = run_dir / "metrics_summary.csv"
    summary = _read_csv(summary_path)
    if not summary:
        raise ReportError(f"{run_dir} has no metrics_summary.csv with results to report on.")
    run_summary, manifest = _read_json(run_dir / "run_summary.json"), _read_json(run_dir / "run_manifest.json")
    recommendation, stats = _read_json(run_dir / "recommendation.json"), _read_json(run_dir / "stats.json")
    rows_all = _read_jsonl(run_dir / "per_question_results.jsonl")
    grouped = _group_rows(rows_all)
    winner_yaml = (run_dir / "winner.yaml").read_text(encoding="utf-8") if (run_dir / "winner.yaml").exists() else ""
    config_text = (run_dir / "config.yaml").read_text(encoding="utf-8") if (run_dir / "config.yaml").exists() else ""
    primary_k = run_summary.get("primary_k")
    by_name = {str(row["system"]): row for row in summary}
    order = _order(summary, recommendation)

    columns = leaderboard_columns({key for row in summary for key in row}, primary_k)
    extras = extra_columns({key for row in summary for key in row}, columns)
    winner = recommendation.get("winner")
    ties, pareto = set(recommendation.get("tied_with_winner", [])), {str(n) for n in recommendation.get("pareto", [])} | {str(r["system"]) for r in summary if r.get("pareto_optimal") is True}
    baseline = stats.get("baseline")
    badges: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for name in order:
        marks = (["recommended"] if name == winner else []) + (["tie"] if name in ties else []) + (["pareto"] if name in pareto else []) + (["baseline"] if name == baseline else [])
        marks += ["failed"] if (to_float(by_name[name].get("n_ok")) or 0) == 0 and by_name[name].get("n_ok") is not None else []
        badges[name] = marks
        kinds[name] = "winner" if name == winner else "tie" if name in ties else "pareto" if name in pareto else "other"

    significance = _significance(_read_csv(run_dir / "significance.csv"))
    leaderboard = _leaderboard(summary, order, columns, extras, badges, significance)
    whiskers = leaderboard.pop("_whiskers")
    scatter = _scatter(summary, order, kinds)
    heat = _heatmaps(grouped, order, primary_k)
    stages = _stage_costs(grouped, order)
    failures = _failures(grouped, order)
    tools, routes = _read_csv(run_dir / "tool_usage.csv"), _read_csv(run_dir / "routes.csv")
    agents = _agents(summary, order, tools, routes)
    audit_rows = _read_csv(run_dir / "qrels_audit.csv")
    audit = _qrels_audit(audit_rows, order)
    questions = build_questions(rows_all, order, audit_rows, int(max_embedded_mb * 1024 * 1024)) if rows_all else None

    cost_rows = _read_csv(run_dir / "cost_breakdown.csv")
    documents = max((int(r["num_documents"]) for r in cost_rows if r.get("stage") == "ingestion" and r.get("num_documents") is not None), default=None)
    dataset_info = run_summary.get("dataset", {})
    categories: Counter[str] = Counter()
    first = next(iter(grouped.values()), [])
    for row in first:
        categories[str(row.get("category") or "unknown")] += 1
    cache = run_summary.get("cache", {})
    notices = run_summary.get("notices")
    if notices is None:
        notices = build_notices(run_summary.get("mode"), run_summary.get("unknown_priced_models"))
    run = {
        "id": run_summary.get("run_id") or recommendation.get("run_id") or run_dir.name,
        "mode": run_summary.get("mode") or manifest.get("mode") or "unknown",
        "started_utc": manifest.get("started_utc"),
        "finished_utc": manifest.get("finished_utc"),
        "wall_time_s": (run_summary.get("run_wall_time_ms") or 0) / 1000,
        "ragbench_version": manifest.get("ragbench_version"),
        "python": manifest.get("python"),
        "git_sha": manifest.get("git_sha"),
        "git_dirty": manifest.get("git_dirty"),
        "config_hash": manifest.get("config_hash"),
        "dataset_hash": manifest.get("dataset_hash"),
        "models_used": run_summary.get("models_used") or manifest.get("models_used") or [],
        "pricing_as_of": run_summary.get("pricing_as_of"),
        "systems": len(summary),
        "errors": run_summary.get("num_errors", 0),
        "latency_source": (run_summary.get("execution") or {}).get("latency_source"),
        "primary_k": primary_k,
    }
    dataset = {
        "questions": dataset_info.get("questions") or (len({r["question_id"] for r in first}) if first else None),
        "labeled_questions": dataset_info.get("labeled_questions"),
        "label_free": bool(dataset_info.get("label_free")),
        "documents": documents,
        "categories": [{"name": name, "count": count} for name, count in sorted(categories.items(), key=lambda item: (-item[1], item[0]))],
        "synthetic": dataset_info.get("synthetic_questions", 0),
        "needs_review": dataset_info.get("needs_review_questions", 0),
        "mock_generated": dataset_info.get("mock_questions", 0),
        "warnings": run_summary.get("dataset_warnings", []),
    }
    budget = run_summary.get("budget") or {}
    cost = {
        "charged_usd": cache.get("charged_cost_usd"),
        "real_spend_usd": cache.get("real_spend_usd"),
        "saved_usd": (cache.get("saved_cost_usd") or 0.0) + (run_summary.get("embedding_cache") or {}).get("saved_cost_usd", 0.0),
        "cache_hit_rate": cache.get("hit_rate"),
        "cache_hits": (cache.get("hits") or 0) + (run_summary.get("embedding_cache") or {}).get("hits", 0),
        "budget": budget,
    }
    data = {
        "schema": SCHEMA_VERSION,
        "run": run,
        "notices": notices,
        "dataset": dataset,
        "cost": cost,
        "recommendation": _recommendation(recommendation, winner_yaml, by_name),
        "leaderboard": leaderboard,
        "baseline": baseline,
        "scatter": scatter,
        "heatmap": heat,
        "stages": stages,
        "latency": [{"system": name, "p50": to_float(by_name[name].get("latency_ms_p50")), "p95": to_float(by_name[name].get("latency_ms_p95")), "source": by_name[name].get("latency_source")} for name in order],
        "agents": agents,
        "failures": failures,
        "failure_other_detail": dict(_failure_detail(rows_all)),
        "qrels_audit": audit,
        "questions": {"total": questions.embedded["total"], "embedded": questions.embedded["shown"], "detail": questions.embedded["detail"], "file": questions.embedded["file"]} if questions else None,
        "reproducibility": {"manifest": manifest, "config_yaml": config_text},
    }
    view = {"whiskers": whiskers, "repo_text": REPO_TEXT, "stage_slots": STAGE_SLOTS, "stage_names": STAGE_NAMES, "failure_slots": {**FAILURE_SLOTS, "other": "sgray"}, "failure_names": FAILURE_NAMES}
    return ReportBuild(_json_safe(data), view, questions)
