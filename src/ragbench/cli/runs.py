"""`ragbench runs`: list, inspect, remove and index the runs in a results folder."""

from __future__ import annotations

import shutil
import webbrowser
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail, is_interactive, json_output, money, print_json
from ragbench.cli.run import print_run_summary
from ragbench.errors import UsageError
from ragbench.reporting.history import INDEX_NAME, RunRecord, describe_run, is_run_dir, resolve_run, scan_runs, write_index

RESULTS_DIR = typer.Option(Path("results"), "--results-dir", envvar="RAGBENCH_RESULTS_DIR", help="The folder that holds the run directories.")
AS_JSON = typer.Option(False, "--json", help="Print JSON on stdout instead of a table.")

runs_app = typer.Typer(help="List, inspect, remove and index the runs in a results folder.", invoke_without_command=True)


def _when(record: RunRecord) -> str:
    return (record.started or "—").replace("T", " ").removesuffix("Z")[:16]


def _list(results_dir: Path, as_json: bool) -> None:
    records = list(reversed(scan_runs(results_dir)))  # newest first
    if as_json:
        print_json([record.as_dict() for record in records])
        return
    if not records:
        console.print(f"No runs in {results_dir}. Start one with `ragbench run --config configs/all.yaml --mock`.")
        return
    table = Table(title=f"Runs in {results_dir}", title_justify="left")
    longest = max(len(record.name) for record in records)
    for column, justify in (("Date (UTC)", "left"), ("Run", "left"), ("Mode", "left"), ("Systems", "right"), ("Questions", "right"), ("Winner", "left"), ("Cost", "right")):
        # The run name is what you type back into `show` and `report`, so it keeps a readable width; the short columns never wrap.
        table.add_column(column, justify=justify, overflow="fold", no_wrap=column in ("Mode", "Systems", "Questions", "Cost"), min_width=min(longest, 24) if column == "Run" else None)  # type: ignore[arg-type]
    for record in records:
        name = escape(record.name) + ("" if record.status == "complete" else " [yellow](incomplete)[/yellow]")
        table.add_row(
            _when(record),
            name,
            record.mode or "—",
            str(len(record.systems)),
            "—" if record.questions is None else str(record.questions),
            escape(record.winner) if record.winner else "—",
            "—" if record.charged_usd is None else money(record.charged_usd),
        )
    console.print(table)
    console.print("[dim]`ragbench runs show RUN` for one run, `ragbench report RUN` to rebuild its reports, `ragbench runs index` for a history page.[/dim]")


@runs_app.callback()
def runs_main(ctx: typer.Context, results_dir: Path = RESULTS_DIR, as_json: bool = AS_JSON) -> None:
    """Without a subcommand, lists the runs (the same as `runs list`)."""
    if ctx.invoked_subcommand is None:
        with json_output(False):
            _list(results_dir, as_json)


@runs_app.command("list")
def list_runs(results_dir: Path = RESULTS_DIR, as_json: bool = AS_JSON) -> None:
    """A table of runs, newest first: date, name, mode, systems, questions, winner, cost."""
    _list(results_dir, as_json)


@runs_app.command("show")
def show_run(
    run: str = typer.Argument(..., help="A run directory, its name under --results-dir, `latest`, or part of a name."),
    results_dir: Path = RESULTS_DIR,
    as_json: bool = typer.Option(False, "--json", help="Print the run as JSON on stdout."),
) -> None:
    """One run in detail: its leaderboard, recommendation and files."""
    run_dir = resolve_run(run, results_dir)
    if as_json:
        print_json(describe_run(run_dir))
        return
    info = describe_run(run_dir)
    console.print(f"[bold]{escape(run_dir.name)}[/bold]  [dim]{info['status']} · {info['mode'] or 'unknown mode'} · {info['started'] or ''}[/dim]")
    console.print(f"Directory: {run_dir}", soft_wrap=True)
    print_run_summary(run_dir)
    for label, path in info["files"].items():
        console.print(f"  {label}: {path}", soft_wrap=True)


@runs_app.command("rm")
def remove_run(
    run: str = typer.Argument(..., help="A run directory or its name under --results-dir (a part of a name is not enough for a delete)."),
    results_dir: Path = RESULTS_DIR,
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without asking."),
) -> None:
    """Delete a run directory. Only a directory that looks like a run is ever removed."""
    candidate = Path(run) if Path(run).is_dir() else results_dir / run
    if not is_run_dir(candidate):
        raise fail(f"{candidate} is not a run directory (no metrics_summary.csv, run_manifest.json or config.yaml in it). Nothing was deleted.")
    target = candidate.resolve()
    here = Path.cwd().resolve()
    if target == here or target in here.parents or target == Path.home().resolve():
        raise fail(f"Refusing to delete {target}: it contains the folder you are in.")
    if not yes:
        if not is_interactive():
            raise fail(f"Pass --yes to delete {target} (there is no one to ask).")
        typer.confirm(f"Delete {target} and everything in it?", abort=True)
    shutil.rmtree(target)
    console.print(f"[green]Deleted {target}[/green]", soft_wrap=True)
    if (target.parent / INDEX_NAME).exists():
        console.print("[dim]`ragbench runs index` refreshes the history page.[/dim]")


@runs_app.command("index")
def index_runs(
    results_dir: Path = RESULTS_DIR,
    open_page: bool = typer.Option(False, "--open", help="Open the page in your browser."),
) -> None:
    """Write results/index.html: every run with its winner and a link to its report, and a score-over-time line per system."""
    if not results_dir.is_dir():
        raise UsageError(f"{results_dir} does not exist.", hint="Pass --results-dir, or start a run with `ragbench run`.")
    page = write_index(results_dir)
    console.print(f"[green]Wrote {page}[/green] ({len(scan_runs(results_dir))} runs)", soft_wrap=True)
    if open_page:
        webbrowser.open(page.resolve().as_uri())
