from __future__ import annotations

from pathlib import Path
from typing import Any

LEADERBOARD_COLUMNS = [
    "System",
    "Recall@5",
    "MRR@10",
    "nDCG@10",
    "Answer Score",
    "Faithfulness",
    "Avg Cost / Question",
    "Avg Latency",
    "Wall Time",
    "Best For",
]


BEST_FOR = {
    "bm25": "Cheap lexical baseline",
    "vector": "Semantic baseline",
    "hybrid": "Balanced lexical + semantic search",
    "rerank": "Higher precision retrieval",
    "parent_doc": "Small-to-big context",
    "llm_heavy": "Quality-oriented expensive pipeline",
}


def write_leaderboard(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in summary_rows:
        system_type = row.get("system_type", row.get("system", ""))
        rows.append(
            {
                "System": row["system"],
                "Recall@5": f"{row.get('retrieval_recall@5', 0):.3f}",
                "MRR@10": f"{row.get('retrieval_mrr@10', 0):.3f}",
                "nDCG@10": f"{row.get('retrieval_ndcg@10', 0):.3f}",
                "Answer Score": f"{row.get('answer_score', 0):.2f}",
                "Faithfulness": f"{row.get('faithfulness', 0):.2f}",
                "Avg Cost / Question": f"${row.get('avg_cost_per_question', 0):.6f}",
                "Avg Latency": f"{row.get('avg_latency_ms', 0):.0f} ms",
                "Wall Time": f"{row.get('system_wall_time_ms', 0):.0f} ms",
                "Best For": BEST_FOR.get(str(system_type), "Custom comparison"),
            }
        )
    content = ["# RAGBench Leaderboard", "", _markdown_table(LEADERBOARD_COLUMNS, rows), ""]
    path.write_text("\n".join(content), encoding="utf-8")


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


def write_qrels_audit(path: Path, audit_rows: list[dict[str, Any]]) -> None:
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
            "Recall@5": f"{row['recall@5']:.2f}",
            "Labeled Docs": row["labeled_relevant_doc_ids"],
            "Unlabeled Retrieved Docs": row["unlabeled_retrieved_doc_ids"],
        }
        for row in audit_rows
    ]
    lines.append(_markdown_table(["System", "Question", "Severity", "Recall@5", "Labeled Docs", "Unlabeled Retrieved Docs"], rows))
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(columns: list[str], rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])
