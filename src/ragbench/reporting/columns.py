"""Leaderboard columns shared by the console, Markdown, and HTML reports.

Retrieval columns follow the configured `k_values`: recall at the headline `primary_k`, and
MRR / nDCG at the deepest k that was measured. A column whose value is missing renders as an
em dash, never as a fake 0.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
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


def metric_ks(keys: Iterable[str]) -> list[int]:
    """The k cut-offs present in a summary, parsed from `retrieval_recall@k` column names."""
    return sorted({int(match.group(1)) for key in keys if (match := _RECALL_KEY.match(key))})


def leaderboard_columns(keys: Iterable[str], primary_k: int | None = None) -> list[Column]:
    keys = set(keys)
    ks = metric_ks(keys)
    columns: list[Column] = []
    if ks:
        headline = primary_k if primary_k in ks else default_primary_k(ks)
        deepest = max(ks)
        columns += [
            Column(f"retrieval_recall@{headline}", f"Recall@{headline}", "{:.3f}", True),
            Column(f"retrieval_mrr@{deepest}", f"MRR@{deepest}", "{:.3f}", True),
            Column(f"retrieval_ndcg@{deepest}", f"nDCG@{deepest}", "{:.3f}", True),
        ]
    columns += [
        Column("answer_score", "Answer", "{:.2f}", True),
        Column("faithfulness", "Faithful", "{:.2f}", True),
        Column("avg_cost_per_question", "$/Q", "${:.5f}", False),
        Column("avg_latency_ms", "Latency", "{:.0f} ms", False),
    ]
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
