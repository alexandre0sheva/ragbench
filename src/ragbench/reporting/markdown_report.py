from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.rag_systems.trace import STAGE_KEYS
from ragbench.reporting.columns import format_value, leaderboard_columns


def _best_for(system_type: str) -> str:
    import ragbench.rag_systems  # noqa: F401  (registers the built-in systems)
    from ragbench.registry import SYSTEMS

    cls = SYSTEMS.mapping.get(system_type)
    spec = getattr(cls, "spec", None)
    return spec.best_for if spec is not None else "Custom comparison"


def write_leaderboard(
    path: Path,
    summary_rows: list[dict[str, Any]],
    notices: list[str] | None = None,
    primary_k: int | None = None,
    stage_rows: list[dict[str, Any]] | None = None,
) -> None:
    columns = leaderboard_columns({key for row in summary_rows for key in row}, primary_k)
    headers = ["System", *(column.header for column in columns), "Errors", "Wall Time", "Best For"]
    rows = []
    for row in summary_rows:
        system_type = row.get("system_type", row.get("system", ""))
        cells = {"System": row["system"]}
        cells.update({column.header: format_value(column, row.get(column.key)) for column in columns})
        cells["Errors"] = f"{row.get('n_error', 0)}/{row.get('n_ok', 0) + row.get('n_error', 0)}" if row.get("n_error") else "0"
        cells["Wall Time"] = f"{row.get('system_wall_time_ms', 0):.0f} ms"
        cells["Best For"] = _best_for(str(system_type))
        rows.append(cells)
    content = ["# RAGBench Leaderboard", ""]
    if notices:
        content.extend([*(f"> **Note:** {notice}" for notice in notices), ""])
    content.extend([_markdown_table(headers, rows), ""])
    content.extend(_stage_sections(stage_rows or []))
    path.write_text("\n".join(content), encoding="utf-8")


def _stage_sections(stage_rows: list[dict[str, Any]]) -> list[str]:
    """"Cost by stage" and "Latency by stage" tables; a stage column appears only when some system spent anything there."""
    lines: list[str] = []
    for title, field, fmt, unit in (
        ("Cost by stage", "cost", "${:.6f}", "$ per question, answering only (the judge's cost is excluded)"),
        ("Latency by stage", "latency_ms", "{:.1f} ms", "ms per question, answering only"),
    ):
        used = [key for key in STAGE_KEYS if any((row[field].get(key) or 0) > 0 for row in stage_rows)]
        if not used:
            continue
        table_rows = [{"System": row["system"], **{key: fmt.format(row[field][key]) for key in used}} for row in stage_rows]
        lines.extend([f"## {title}", "", f"Mean {unit}. `untracked` is cost or time that no recorded step accounts for.", "", _markdown_table(["System", *used], table_rows), ""])
    return lines


def write_failures(path: Path, failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Failure Analysis",
        "",
        "`no_failure` means the answer passed the current heuristic/LLM checks and no failure class was assigned. It is summarized below but omitted from failure-detail tables.",
        "",
    ]
    by_system: dict[str, dict[str, int]] = {}
    examples: dict[tuple[str, str], str] = {}
    for row in failure_rows:
        system = row["system"]
        failure_type = row["failure_type"]
        by_system.setdefault(system, {})[failure_type] = by_system.setdefault(system, {}).get(failure_type, 0) + 1
        examples.setdefault((system, failure_type), row.get("question", ""))
    summary_rows = []
    for system, counts in sorted(by_system.items()):
        no_failure = counts.get("no_failure", 0)
        classified = sum(value for key, value in counts.items() if key != "no_failure")
        summary_rows.append({"System": system, "No Classified Failure": str(no_failure), "Classified Failures": str(classified)})
    lines.extend(["## Summary", "", _markdown_table(["System", "No Classified Failure", "Classified Failures"], summary_rows), ""])
    for system, counts in sorted(by_system.items()):
        lines.extend([f"## {system}", ""])
        rows = [
            {"Failure Type": key, "Count": str(value), "Example Question": examples.get((system, key), "")}
            for key, value in sorted(counts.items())
            if key != "no_failure"
        ]
        if rows:
            lines.extend([_markdown_table(["Failure Type", "Count", "Example Question"], rows), ""])
        else:
            lines.extend(["No classified failures.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_qrels_audit(path: Path, audit_rows: list[dict[str, Any]], primary_k: int = 5) -> None:
    lines = [
        "# Qrels Audit",
        "",
        "This report lists cases where the answer judge rated an answer highly even though retrieval did not find all labeled relevant documents. These are candidates for reviewing qrels, not automatic qrel changes.",
        "",
    ]
    if not audit_rows:
        lines.append("No qrels audit candidates found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    rows = [
        {
            "System": row["system"],
            "Question": row["question_id"],
            "Severity": row["severity"],
            f"Recall@{primary_k}": f"{row['recall']:.2f}",
            "Labeled Docs": row["labeled_relevant_doc_ids"],
            "Unlabeled Retrieved Docs": row["unlabeled_retrieved_doc_ids"],
        }
        for row in audit_rows
    ]
    lines.append(_markdown_table(["System", "Question", "Severity", f"Recall@{primary_k}", "Labeled Docs", "Unlabeled Retrieved Docs"], rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(columns: list[str], rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])
