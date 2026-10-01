"""Leaderboard columns shared by the console, Markdown, and HTML reports.

Retrieval columns follow the configured `k_values`: recall at the headline `primary_k`, and
MRR / nDCG at the deepest k that was measured. A column whose value is missing renders as an
em dash, never as a fake 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ragbench.config.schema import default_primary_k

MISSING = "—"
_RECALL_KEY = re.compile(r"^retrieval_recall@(\d+)$")


@dataclass(frozen=True)
class Column:
    key: str
    header: str
    fmt: str
    higher_is_better: bool
    interval: bool = False  # headline metric with `<key>_ci_lo` / `<key>_ci_hi` columns: cells can show `mean [lo, hi]`


def metric_ks(keys: Iterable[str]) -> list[int]:
    """The k cut-offs present in a summary, parsed from `retrieval_recall@k` column names."""
    return sorted({int(match.group(1)) for key in keys if (match := _RECALL_KEY.match(key))})


def leaderboard_columns(keys: Iterable[str], primary_k: int | None = None) -> list[Column]:
    keys = set(keys)
    ks = metric_ks(keys)
    columns: list[Column] = []

    def headline(key: str, header: str, fmt: str) -> Column:
        return Column(key, header, fmt, True, interval=f"{key}_ci_lo" in keys and f"{key}_ci_hi" in keys)

    if ks:
        cutoff = primary_k if primary_k in ks else default_primary_k(ks)
        deepest = max(ks)
        columns += [
            headline(f"retrieval_recall@{cutoff}", f"Recall@{cutoff}", "{:.3f}"),
            headline(f"retrieval_mrr@{deepest}", f"MRR@{deepest}", "{:.3f}"),
            headline(f"retrieval_ndcg@{deepest}", f"nDCG@{deepest}", "{:.3f}"),
        ]
    columns += [
        headline("answer_score", "Answer", "{:.2f}"),
        headline("faithfulness", "Faithful", "{:.2f}"),
    ]
    # Deterministic and context metrics (see docs/methodology.md); older runs do not have them.
    if "token_f1" in keys:
        columns.append(Column("token_f1", "F1", "{:.2f}", True))
    if "context_recall" in keys:
        columns.append(Column("context_recall", "Ctx recall", "{:.2f}", True))
    columns.append(Column("avg_cost_per_question", "$/Q", "${:.5f}", False))
    if "cost_per_correct" in keys:  # total spend / answers judged correct; blank when a system had none
        columns.append(Column("cost_per_correct", "$/correct", "${:.5f}", False))
    columns.append(Column("avg_latency_ms", "Latency", "{:.0f} ms", False))
    if "latency_ms_p95" in keys:
        columns.append(Column("latency_ms_p95", "p95", "{:.0f} ms", False))
    return columns


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def to_float(value: Any) -> float | None:
    return None if is_missing(value) else float(value)


def format_value(column: Column, value: Any) -> str:
    number = to_float(value)
    return MISSING if number is None else column.fmt.format(number)


def format_cell(column: Column, row: Mapping[str, Any], sep: str = " ") -> str:
    """A leaderboard cell: the value, followed by its confidence interval `[lo, hi]` when the run computed one."""
    text = format_value(column, row.get(column.key))
    if column.interval and text != MISSING:
        lo, hi = to_float(row.get(f"{column.key}_ci_lo")), to_float(row.get(f"{column.key}_ci_hi"))
        if lo is not None and hi is not None:
            text += f"{sep}[{column.fmt.format(lo)}, {column.fmt.format(hi)}]"
    return text
