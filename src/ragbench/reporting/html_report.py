from __future__ import annotations

from pathlib import Path

from jinja2 import Template

HTML_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAGBench Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }
    h1, h2 { color: #102a43; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0 32px; font-size: 14px; }
    th, td { border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f0f4f8; }
    .muted { color: #627d98; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 12px; }
  </style>
</head>
<body>
  <h1>RAGBench Evaluation Report</h1>
  <p class="muted">Run ID: {{ run_id }}</p>

  <h2>Leaderboard</h2>
  {{ leaderboard_html }}

  <h2>Per-Category Performance</h2>
  {{ category_html }}

  <h2>Cost Breakdown</h2>
  {{ cost_html }}

  <h2>Failure Analysis</h2>
  {{ failures_html }}

  <h2>Configuration Summary</h2>
  <pre>{{ config_text }}</pre>
</body>
</html>
"""
)


def write_html_report(
    path: Path,
    run_id: str,
    leaderboard_html: str,
    category_html: str,
    cost_html: str,
    failures_html: str,
    config_text: str,
) -> None:
    html = HTML_TEMPLATE.render(
        run_id=run_id,
        leaderboard_html=leaderboard_html,
        category_html=category_html,
        cost_html=cost_html,
        failures_html=failures_html,
        config_text=config_text,
    )
    path.write_text(html, encoding="utf-8")

