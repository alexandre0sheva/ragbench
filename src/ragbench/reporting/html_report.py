from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

# Color palette cycled across systems so every chart uses consistent colors.
SYSTEM_PALETTE = ["#6366f1", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f43f5e", "#84cc16", "#64748b"]

HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAGBench Report · {{ run_id }}</title>
<style>
  :root {
    --bg: #f8fafc; --panel: #ffffff; --ink: #0f172a; --muted: #64748b;
    --line: #e2e8f0; --accent: #6366f1; --good: #059669; --track: #eef2f7;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0b1120; --panel: #111a2e; --ink: #e2e8f0; --muted: #94a3b8;
      --line: #1e293b; --accent: #818cf8; --good: #34d399; --track: #1a2438;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; background: var(--bg); color: var(--ink); line-height: 1.5;
  }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
  header h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: -0.02em; }
  header .meta { color: var(--muted); font-size: 13px; }
  h2 { font-size: 18px; margin: 40px 0 12px; letter-spacing: -0.01em; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 24px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  }
  .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .card .value { font-size: 20px; font-weight: 700; margin-top: 2px; }
  .card .detail { font-size: 13px; color: var(--muted); }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--panel); border-radius: 12px; overflow: hidden; }
  th, td { border-bottom: 1px solid var(--line); padding: 8px 12px; text-align: left; vertical-align: top; }
  th { background: transparent; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; user-select: none; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
  td.best { color: var(--good); font-weight: 700; }
  .tablebox { border: 1px solid var(--line); border-radius: 12px; overflow-x: auto; }
  th.sortable { cursor: pointer; }
  th.sortable:hover { color: var(--accent); }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
  .chart { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .chart h3 { margin: 0 0 12px; font-size: 14px; }
  .bar-row { display: grid; grid-template-columns: 130px 1fr 70px; gap: 8px; align-items: center; margin: 6px 0; font-size: 12.5px; }
  .bar-row .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { background: var(--track); border-radius: 6px; height: 14px; }
  .bar-fill { height: 100%; border-radius: 6px; min-width: 2px; }
  .bar-row .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  pre {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px; overflow-x: auto; font-size: 12.5px;
  }
  .footnote { color: var(--muted); font-size: 12.5px; margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>RAGBench Evaluation Report</h1>
    <div class="meta">
      Run <strong>{{ run_id }}</strong>
      · {{ run_meta.num_systems }} systems · {{ run_meta.num_questions }} questions
      · wall time {{ "%.1f" | format(run_meta.run_wall_time_ms / 1000) }}s
      {% if run_meta.cache_hits %} · embedding cache reused {{ run_meta.cache_hits }} embeddings (~${{ "%.4f" | format(run_meta.cache_saved_usd) }} saved){% endif %}
    </div>
  </header>

  <div class="cards">
    {% for card in cards %}
    <div class="card">
      <div class="label">{{ card.label }}</div>
      <div class="value">{{ card.value }}</div>
      <div class="detail">{{ card.detail }}</div>
    </div>
    {% endfor %}
  </div>

  <h2>Leaderboard</h2>
  <div class="tablebox">
    <table id="leaderboard">
      <thead>
        <tr>
          <th class="sortable" data-col="0" data-kind="text">System</th>
          {% for col in leaderboard_columns %}
          <th class="sortable num" data-col="{{ loop.index }}" data-kind="num">{{ col.header }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in leaderboard_rows %}
        <tr>
          <td><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{{ row.color }};margin-right:7px"></span>{{ row.system }}</td>
          {% for cell in row.cells %}
          <td class="num{{ ' best' if cell.best }}" data-value="{{ cell.raw }}">{{ cell.text }}</td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <div class="footnote">Click a column header to sort. Green marks the best value in each column.</div>

  <h2>Comparison Charts</h2>
  <div class="charts">
    {% for chart in charts %}
    <div class="chart">
      <h3>{{ chart.title }}</h3>
      {% for bar in chart.bars %}
      <div class="bar-row">
        <div class="name">{{ bar.name }}</div>
        <div class="bar-track"><div class="bar-fill" style="width: {{ bar.pct }}%; background: {{ bar.color }}"></div></div>
        <div class="val">{{ bar.value }}</div>
      </div>
      {% endfor %}
    </div>
    {% endfor %}
  </div>

  {% if category_rows %}
  <h2>Per-Category Answer Quality</h2>
  <div class="tablebox">
    <table>
      <thead><tr><th>System</th><th>Category</th><th class="num">Answer Score</th><th class="num">Faithfulness</th></tr></thead>
      <tbody>
        {% for row in category_rows %}
        <tr><td>{{ row.system }}</td><td>{{ row.category }}</td><td class="num">{{ "%.2f" | format(row.answer_score) }}</td><td class="num">{{ "%.2f" | format(row.faithfulness) }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  <h2>Cost Breakdown</h2>
  <div class="tablebox">
    <table>
      <thead><tr><th>System</th><th class="num">Ingestion $</th><th class="num">Query $</th><th class="num">Judge $</th><th class="num">Total $</th></tr></thead>
      <tbody>
        {% for row in cost_rows %}
        <tr><td>{{ row.system }}</td><td class="num">{{ "%.6f" | format(row.ingestion_cost) }}</td><td class="num">{{ "%.6f" | format(row.query_cost) }}</td><td class="num">{{ "%.6f" | format(row.judge_cost) }}</td><td class="num">{{ "%.6f" | format(row.total_cost) }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  <div class="footnote">Per-system cost is charged at standalone prices even when the shared embedding cache avoided an API call, so systems stay comparable.</div>

  <h2>Failure Analysis</h2>
  {% if failure_rows %}
  <div class="tablebox">
    <table>
      <thead><tr><th>System</th><th>Failure Type</th><th class="num">Count</th></tr></thead>
      <tbody>
        {% for row in failure_rows %}
        <tr><td>{{ row.system }}</td><td>{{ row.failure_type }}</td><td class="num">{{ row.count }}</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p class="footnote">No classified failures.</p>
  {% endif %}

  <h2>Configuration</h2>
  <pre>{{ config_text }}</pre>
</div>
<script>
document.querySelectorAll("#leaderboard th.sortable").forEach(function (th) {
  th.addEventListener("click", function () {
    var table = th.closest("table");
    var tbody = table.querySelector("tbody");
    var col = parseInt(th.dataset.col, 10);
    var kind = th.dataset.kind;
    var asc = th.dataset.asc !== "true";
    table.querySelectorAll("th").forEach(function (other) { delete other.dataset.asc; });
    th.dataset.asc = asc ? "true" : "false";
    Array.from(tbody.rows)
      .sort(function (a, b) {
        var av, bv;
        if (kind === "num") {
          av = parseFloat(a.cells[col].dataset.value);
          bv = parseFloat(b.cells[col].dataset.value);
        } else {
          av = a.cells[col].textContent.trim().toLowerCase();
          bv = b.cells[col].textContent.trim().toLowerCase();
        }
        return (av < bv ? -1 : av > bv ? 1 : 0) * (asc ? 1 : -1);
      })
      .forEach(function (row) { tbody.appendChild(row); });
  });
});
</script>
</body>
</html>
"""
)

LEADERBOARD_COLUMNS: list[dict[str, Any]] = [
    {"key": "retrieval_recall@5", "header": "Recall@5", "fmt": "{:.3f}", "higher_is_better": True},
    {"key": "retrieval_mrr@10", "header": "MRR@10", "fmt": "{:.3f}", "higher_is_better": True},
    {"key": "retrieval_ndcg@10", "header": "nDCG@10", "fmt": "{:.3f}", "higher_is_better": True},
    {"key": "answer_score", "header": "Answer", "fmt": "{:.2f}", "higher_is_better": True},
    {"key": "faithfulness", "header": "Faithful", "fmt": "{:.2f}", "higher_is_better": True},
    {"key": "avg_cost_per_question", "header": "$/Q", "fmt": "{:.5f}", "higher_is_better": False},
    {"key": "avg_latency_ms", "header": "Latency ms", "fmt": "{:.0f}", "higher_is_better": False},
]

CHART_SPECS: list[dict[str, Any]] = [
    {"key": "answer_score", "title": "Answer score (0–5)", "fmt": "{:.2f}"},
    {"key": "retrieval_recall@5", "title": "Recall@5", "fmt": "{:.3f}"},
    {"key": "avg_cost_per_question", "title": "Cost per question (USD)", "fmt": "${:.5f}"},
    {"key": "avg_latency_ms", "title": "Avg latency (ms)", "fmt": "{:.0f}"},
]


def write_html_report(
    path: Path,
    run_id: str,
    summary_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    config_text: str,
    run_meta: dict[str, Any],
) -> None:
    colors = {row["system"]: SYSTEM_PALETTE[idx % len(SYSTEM_PALETTE)] for idx, row in enumerate(summary_rows)}
    html = HTML_TEMPLATE.render(
        run_id=run_id,
        run_meta=run_meta,
        cards=_build_cards(summary_rows),
        leaderboard_columns=LEADERBOARD_COLUMNS,
        leaderboard_rows=_build_leaderboard_rows(summary_rows, colors),
        charts=_build_charts(summary_rows, colors),
        category_rows=category_rows,
        cost_rows=cost_rows,
        failure_rows=failure_rows,
        config_text=config_text,
    )
    path.write_text(html, encoding="utf-8")


def _winner(summary_rows: list[dict[str, Any]], key: str, higher_is_better: bool = True) -> dict[str, Any] | None:
    rows = [row for row in summary_rows if key in row]
    if not rows:
        return None
    return max(rows, key=lambda row: row[key]) if higher_is_better else min(rows, key=lambda row: row[key])


def _build_cards(summary_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    specs = [
        ("Best answers", "answer_score", True, "{:.2f} / 5"),
        ("Best retrieval (nDCG@10)", "retrieval_ndcg@10", True, "{:.3f}"),
        ("Cheapest", "avg_cost_per_question", False, "${:.5f} / question"),
        ("Fastest", "avg_latency_ms", False, "{:.0f} ms / question"),
    ]
    for label, key, higher, fmt in specs:
        row = _winner(summary_rows, key, higher)
        if row is None:
            continue
        cards.append({"label": label, "value": str(row["system"]), "detail": fmt.format(float(row[key]))})
    return cards


def _build_leaderboard_rows(summary_rows: list[dict[str, Any]], colors: dict[str, str]) -> list[dict[str, Any]]:
    best: dict[str, float] = {}
    for col in LEADERBOARD_COLUMNS:
        values = [float(row.get(col["key"], 0.0)) for row in summary_rows]
        if values:
            best[col["key"]] = max(values) if col["higher_is_better"] else min(values)
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        cells = []
        for col in LEADERBOARD_COLUMNS:
            value = float(row.get(col["key"], 0.0))
            cells.append(
                {
                    "raw": value,
                    "text": col["fmt"].format(value),
                    "best": len(summary_rows) > 1 and value == best.get(col["key"]),
                }
            )
        rows.append({"system": row["system"], "color": colors[row["system"]], "cells": cells})
    return rows


def _build_charts(summary_rows: list[dict[str, Any]], colors: dict[str, str]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for spec in CHART_SPECS:
        values = [float(row.get(spec["key"], 0.0)) for row in summary_rows]
        peak = max(values) if values else 0.0
        bars = [
            {
                "name": row["system"],
                "pct": round(100.0 * float(row.get(spec["key"], 0.0)) / peak, 1) if peak else 0.0,
                "value": spec["fmt"].format(float(row.get(spec["key"], 0.0))),
                "color": colors[row["system"]],
            }
            for row in summary_rows
        ]
        charts.append({"title": spec["title"], "bars": bars})
    return charts
