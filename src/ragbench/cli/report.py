"""`ragbench report`, `ragbench compare-runs`, `ragbench contrast` and `ragbench judge-audit`: work on finished runs without running anything."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail, json_output, print_json
from ragbench.cli.runs import RESULTS_DIR
from ragbench.evaluation.compare_runs import (
    DEFAULT_MIN_DROP,
    IMPROVEMENT,
    REGRESSION,
    UNTESTED_DROP,
    RunComparison,
    compare_runs,
    comparison_to_dict,
)
from ragbench.evaluation.contrast import ContrastError, build_contrasts, write_contrasts
from ragbench.evaluation.judge_audit import REPORT_FILE, score_audit, write_audit_sheet
from ragbench.evaluation.stats import CORRECT_THRESHOLD
from ragbench.reporting.history import resolve_run
from ragbench.reporting.markdown_report import top_groups
from ragbench.reporting.regenerate import regenerate_reports
from ragbench.reporting.report_data import ReportError
from ragbench.utils.text import truncate

commands = typer.Typer()


@commands.command("report")
def report_command(
    run: str = typer.Argument(..., help="A run directory, its name under --results-dir, `latest`, or part of a name."),
    results_dir: Path = RESULTS_DIR,
    open_report: bool = typer.Option(False, "--open", help="Open report.html in your browser when done."),
) -> None:
    """Rebuild every report of a finished run (leaderboard.md, failures.md, qrels_audit.md, the recommendation, report.html) from its files.

    Nothing is re-run and nothing is paid for: a run directory holds everything its reports are made of.
    """
    run_dir = resolve_run(run, results_dir)
    try:
        written, notes = regenerate_reports(run_dir)
    except ReportError as exc:
        raise fail(str(exc)) from exc
    console.print(f"[green]Rebuilt {len(written)} files in {run_dir}[/green]", soft_wrap=True)
    for path in written:
        console.print(f"  {path.name}")
    for note in notes:
        console.print(f"  [yellow]•[/yellow] {escape(note)}", soft_wrap=True)
    if open_report:
        webbrowser.open((run_dir / "report.html").resolve().as_uri())


def _cell(delta: float | None, a: float | None, b: float | None, fmt: str, higher_is_better: bool) -> str:
    if a is None or b is None:
        return "—"
    text = f"{format(a, fmt)} → {format(b, fmt)}"
    if delta is None or delta == 0:
        return text
    good = (delta > 0) == higher_is_better
    return f"{text} [{'green' if good else 'red'}]({delta:+{fmt}})[/{'green' if good else 'red'}]"


def _print_comparison(comparison: RunComparison) -> None:
    first = comparison.systems[0] if comparison.systems else None
    table = Table(title=f"{comparison.a.name} → {comparison.b.name}", title_justify="left")
    table.add_column("System", style="bold", overflow="fold")
    formats = {"answer_score": ".2f", "faithfulness": ".2f", "avg_cost_per_question": ".5f", "avg_latency_ms": ".0f"}
    for metric in first.metrics if first else []:
        table.add_column(metric.header, justify="right")
    table.add_column("Verdict")
    for row in comparison.systems:
        verdict = {REGRESSION: "[bold red]▼ regression[/bold red]", IMPROVEMENT: "[green]▲ improvement[/green]", UNTESTED_DROP: "[red]▼ drop (not tested)[/red]"}.get(row.verdict, row.verdict)
        cells = [_cell(m.delta, m.a, m.b, formats.get(m.key, ".3f"), m.higher_is_better) for m in row.metrics]
        table.add_row(escape(row.system), *cells, verdict)
    console.print(table)
    console.print("[dim]Answer score, faithfulness and recall: higher is better; cost and latency: lower. Verdicts are the paired bootstrap on the questions both runs answered (Holm-adjusted).[/dim]")
    for label, names in (("Only in the first run", comparison.only_a), ("Only in the second run", comparison.only_b)):
        if names:
            console.print(f"{label}: {escape(', '.join(names))}", soft_wrap=True)
    for note in comparison.notes:
        console.print(f"[yellow]• {escape(note)}[/yellow]", soft_wrap=True)


@commands.command("judge-audit")
def judge_audit_command(
    run: str = typer.Argument(..., help="A run directory, its name under --results-dir, `latest`, or part of a name."),
    results_dir: Path = RESULTS_DIR,
    sample: int = typer.Option(50, "--sample", min=1, help="Answers to put on the sheet, drawn evenly from each system and judge-score band."),
    seed: int = typer.Option(0, "--seed", help="Random seed of the draw: the same seed draws the same sheet."),
    labels: Path | None = typer.Option(None, "--labels", exists=True, dir_okay=False, help="The filled-in sheet: report how well the judge agrees with it."),
    force: bool = typer.Option(False, "--force", help="Draw a new sheet over an existing one."),
) -> None:
    """Calibrate the judge against a human: draw a blind sheet of answers to score by hand, then compare it with the judge.

    Without --labels it writes judge_audit_sheet.csv into the run (fill in the five 0-5 axes; blank cells are skipped). With
    --labels it writes judge_audit.md (judge-human agreement per axis, bias per system) and links it from the report. Nothing is paid for.
    """
    run_dir = resolve_run(run, results_dir)
    if labels is None:
        sheet = write_audit_sheet(run_dir, sample=sample, seed=seed, overwrite=force)
        console.print(f"[green]Wrote {sheet.n} answers to score into {sheet.sheet_path}[/green]", soft_wrap=True)
        console.print(f"  Score each answer 0-5 on correctness, faithfulness, completeness, relevance and citation_quality, then run\n  ragbench judge-audit {run_dir} --labels {sheet.sheet_path}", soft_wrap=True)
        return
    result = score_audit(run_dir, labels)
    answer = result["axes"]["answer_score"]
    alpha = "—" if answer["alpha"] is None else f"{answer['alpha']:.2f}"
    console.print(f"[green]{result['n_labeled']} labeled answers; judge-human alpha on the answer score: {alpha} (n={answer['n']})[/green]", soft_wrap=True)
    try:
        regenerate_reports(run_dir)
    except ReportError as exc:
        console.print(f"  [yellow]•[/yellow] report not rebuilt: {escape(str(exc))}", soft_wrap=True)
    console.print(f"  {run_dir / REPORT_FILE}", soft_wrap=True)


@commands.command("compare-runs")
def compare_runs_command(
    a: str = typer.Argument(..., help="The earlier (baseline) run: a directory, a name under --results-dir, or `latest`."),
    b: str = typer.Argument(..., help="The later run."),
    results_dir: Path = RESULTS_DIR,
    min_drop: float = typer.Option(DEFAULT_MIN_DROP, "--min-drop", min=0, help="When two runs share too few questions to test, flag an answer-score drop larger than this."),
    fail_on_regression: bool = typer.Option(False, "--fail-on-regression", help="Exit with status 1 when any system regressed (for CI)."),
    as_json: bool = typer.Option(False, "--json", help="Print the comparison as JSON on stdout."),
) -> None:
    """How a later run moved against an earlier one, per system: metric changes and whether an answer-score drop is real."""
    with json_output(as_json):
        try:
            comparison = compare_runs(resolve_run(a, results_dir), resolve_run(b, results_dir), min_drop=min_drop)
        except ValueError as exc:
            raise fail(str(exc)) from exc
        if as_json:
            print_json(comparison_to_dict(comparison))
        else:
            _print_comparison(comparison)
        if comparison.regressions:
            console.print(f"[red]Regressed: {escape(', '.join(comparison.regressions))}[/red]", soft_wrap=True)
            if fail_on_regression:
                raise typer.Exit(1)


@commands.command("contrast")
def contrast_command(
    run: str = typer.Argument(..., help="A run directory, its name under --results-dir, `latest`, or part of a name."),
    a: str = typer.Argument(..., help="The first system."),
    b: str = typer.Argument(..., help="The second system."),
    results_dir: Path = RESULTS_DIR,
    source: str = typer.Option("score", "--source", help="What makes a system right: `score` (its answer score reaches --threshold) or `verdicts` (it wins the arena's verdict; needs evaluation.pairwise)."),
    threshold: float = typer.Option(CORRECT_THRESHOLD, "--threshold", min=0, max=5, help="Answer score (0-5) from which an answer counts as correct, for --source score."),
    limit: int = typer.Option(5, "--limit", min=0, help="Contrast questions to list per direction."),
    no_write: bool = typer.Option(False, "--no-write", help="Only print: leave contrasts.jsonl and the reports of the run as they are."),
    as_json: bool = typer.Option(False, "--json", help="Print the summary and every contrast as JSON on stdout."),
) -> None:
    """Where two systems disagree: the questions one answers well and the other does not, and the first step at which their traces part.

    Computed from the run's files (nothing is re-run or paid for). Unless --no-write, the pair is written into contrasts.jsonl and
    contrast_summary.json (replacing that pair's earlier lines) and the run's reports are rebuilt with a "Where systems disagree" section.
    """
    if source not in ("score", "verdicts"):
        raise fail(f"--source must be 'score' or 'verdicts', not '{source}'.")
    with json_output(as_json):
        run_dir = resolve_run(run, results_dir)
        try:
            contrasts = build_contrasts(run_dir, [(a, b)], source=source, threshold=threshold)  # type: ignore[arg-type]
        except ContrastError as exc:
            raise fail(str(exc)) from exc
        summary = contrasts.summaries[0]
        if not no_write:
            write_contrasts(run_dir, contrasts, merge=True)
            try:
                regenerate_reports(run_dir)
            except ReportError as exc:
                console.print(f"  [yellow]•[/yellow] report not rebuilt: {escape(str(exc))}", soft_wrap=True)
        if as_json:
            print_json({"summaries": contrasts.summaries, "contrasts": contrasts.records})
            return
        _print_contrast(summary, contrasts.records, limit)
        if not no_write:
            console.print(f"[dim]Written to {run_dir / 'contrasts.jsonl'} and the report.[/dim]", soft_wrap=True)


def _print_contrast(summary: dict[str, Any], records: list[dict[str, Any]], limit: int) -> None:
    first, second = summary["pair"]
    rule = "won the arena's verdict" if summary["source"] == "verdicts" else f"scored at least {summary['threshold']:g}"
    console.print(f"[bold]{escape(first)} vs {escape(second)}[/bold]  [dim]right = {rule}; {summary['compared']} questions compared, {summary['skipped']} left out[/dim]", soft_wrap=True)
    console.print(f"  only {escape(first)} right: {summary['a_only']}   only {escape(second)} right: {summary['b_only']}   " + (f"equal: {summary['both_equal']}" if summary["source"] == "verdicts" else f"both right: {summary['both_correct']}   both wrong: {summary['both_wrong']}"), soft_wrap=True)
    for winner in summary["pair"]:
        cluster = summary["clusters"][winner]
        won = [r for r in records if r["winner"] == winner]
        if not won:
            continue
        loser = second if winner == first else first
        console.print(f"\n[green]{escape(winner)} right, {escape(loser)} not[/green] ({len(won)})")
        console.print(f"  {escape(loser)} failed as {escape(top_groups(cluster['failure_type']))}; categories {escape(top_groups(cluster['category']))}", soft_wrap=True)
        console.print(f"  traces first part at {escape(top_groups(cluster['diverges_at']))}", soft_wrap=True)
        for record in sorted(won, key=lambda r: -r["gap"])[:limit]:
            step = record["diverges_at"]
            where = f"{step['reason'].replace('_', ' ')} at step {step['index']} ({step['a']} | {step['b']})" if step else "no trace to align"
            console.print(f"  [dim]{escape(record['question_id'])}[/dim] {escape(truncate(str(record['question']), 70))}  {record['scores'][winner]:.1f} vs {record['scores'][loser]:.1f}; {escape(where)}", soft_wrap=True)
