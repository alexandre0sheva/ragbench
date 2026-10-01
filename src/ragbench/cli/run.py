"""`run`, `compare`, `estimate` and `recommend`: running benchmarks and deciding from them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragbench.cli.common import (
    console,
    duration,
    fail,
    in_ci,
    is_interactive,
    json_output,
    load_experiment,
    mock_warning,
    money,
    print_json,
    resolve_config,
)
from ragbench.config.presets import PRESET_NAMES
from ragbench.config.schema import ExperimentConfig
from ragbench.documents.loaders import DocumentLoadError
from ragbench.errors import CommandFailed, ConfigError, config_issues
from ragbench.evaluation.budget import BudgetExceededError
from ragbench.evaluation.estimate import Estimate, confirmation_decision, estimate_run
from ragbench.evaluation.evaluator import BenchmarkRunError, ProgressListener, run_benchmark
from ragbench.models.refs import resolve_run_mode
from ragbench.reporting.columns import format_cell, is_missing, leaderboard_columns
from ragbench.reporting.history import describe_run
from ragbench.utils.env import load_project_env

commands = typer.Typer()


def print_estimate(estimate: Estimate) -> None:
    table = Table(title="Estimate", title_justify="left")
    table.add_column("System", style="bold", overflow="fold")
    for header in ("Ingestion", "Query", "Judge", "Total"):
        table.add_column(header, justify="right")
    for row in estimate.systems:
        if row.error is not None:
            table.add_row(row.system, "[red]could not be estimated[/red]", "", "", "")
            continue
        table.add_row(row.system + (" ~" if row.agentic else ""), money(row.ingestion_usd), money(row.query_usd), money(row.judge_usd), money(row.total_usd))
    table.add_section()
    table.add_row("Total", "", "", "", f"[bold]{money(estimate.total_usd)}[/bold]")
    console.print(table)
    console.print(
        f"{estimate.n_questions} questions over {estimate.n_documents} documents (~{estimate.corpus_tokens:,} tokens) · "
        f"about {duration(estimate.wall_seconds)} at the configured concurrency (rough) · prices as of {estimate.pricing_as_of}"
    )
    for warning in estimate.warnings:
        console.print(f"[yellow]• {escape(warning)}[/yellow]")
    console.print(
        "[dim]Measured by running each system on the offline mock models; prompts, call counts and embedding volume are exact, output lengths are the mock's "
        "(real answers are usually longer). Assumes no cache hits. ~ marks agentic systems.[/dim]"
    )


def confirm_cost(config: ExperimentConfig, yes: bool) -> None:
    """Before a live run: project its cost, and ask (or refuse, or just go) when it is above `evaluation.cost_confirm_threshold_usd`."""
    try:
        with console.status("Estimating the cost…"):
            estimate = estimate_run(config)
    except Exception as exc:  # noqa: BLE001 (the estimate is advice; a failure in it must not stop the run itself)
        console.print(f"[yellow]Could not estimate the cost ({escape(str(exc))}); continuing. Set `evaluation.max_cost_usd` to cap the spending.[/yellow]")
        return
    threshold = config.evaluation.cost_confirm_threshold_usd
    cap = config.evaluation.max_cost_usd
    decision = confirmation_decision(estimate.total_usd, threshold, yes=yes, interactive=is_interactive(), ci=in_ci())
    console.print(f"[dim]Estimated cost: about {money(estimate.total_usd)} (confirmation threshold {money(threshold)}; see `ragbench estimate`).[/dim]")
    if cap is not None and estimate.total_usd > cap:
        console.print(f"[yellow]The estimate is above your `evaluation.max_cost_usd` of {money(cap)}: the run will stop early.[/yellow]")
    if decision == "proceed":
        return
    print_estimate(estimate)
    if decision == "refuse":
        raise fail(
            f"The estimated cost {money(estimate.total_usd)} is above the {money(threshold)} confirmation threshold and there is no one to ask. "
            "Re-run with --yes to go ahead, or raise `evaluation.cost_confirm_threshold_usd`."
        )
    if not typer.confirm(f"Run it for about {money(estimate.total_usd)}?", default=False):
        raise fail("Cancelled. Nothing was spent.", code=1)


class _RichProgress(ProgressListener):
    """Live per-system progress: spinner while ingesting, bar over questions."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self.tasks: dict[str, Any] = {}
        self.finished: dict[str, str] = {}
        self.num_systems = 0

    def run_started(self, num_systems: int, num_questions: int) -> None:
        self.num_systems = num_systems

    def system_started(self, name: str, index: int, total: int) -> None:
        self.tasks[name] = self.progress.add_task(f"[bold]{name}[/bold] · ingesting…", total=None)

    def ingestion_finished(self, name: str, num_chunks: int, latency_ms: float) -> None:
        self.progress.update(self.tasks[name], description=f"[bold]{name}[/bold] · {num_chunks} chunks · answering")

    def question_finished(self, name: str, done: int, total: int) -> None:
        self.progress.update(self.tasks[name], total=total, completed=done)

    def system_finished(self, name: str, wall_time_ms: float) -> None:
        self.finished[name] = f"{wall_time_ms / 1000:.1f}s"
        self.progress.update(self.tasks[name], description=f"[bold]{name}[/bold] · done in {self.finished[name]}")

    def latency_probe(self, name: str, done: int, total: int) -> None:
        suffix = f"measuring latency {done}/{total}" if done < total else "latency measured"
        self.progress.update(self.tasks[name], description=f"[bold]{name}[/bold] · done in {self.finished.get(name, '?')} · {suffix}")

    def system_restored(self, name: str, num_questions: int) -> None:
        self.tasks[name] = self.progress.add_task(f"[bold]{name}[/bold] · restored from the earlier attempt", total=num_questions, completed=num_questions)


def execute_benchmark(
    config: Path,
    mock: bool,
    max_workers: int | None,
    done_message: str,
    use_cache: bool = True,
    system_workers: int | None = None,
    raw_config: dict[str, Any] | None = None,
    yes: bool = False,
    run_dir: Path | None = None,
) -> Path:
    load_project_env(config)
    loaded = load_experiment(config, raw_config)
    mock_warning(loaded, force_mock=mock)
    if resolve_run_mode(loaded, mock) == "live":
        confirm_cost(loaded, yes)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    try:
        with progress:
            output_dir = run_benchmark(
                config,
                force_mock=mock,
                max_workers=max_workers,
                progress=_RichProgress(progress),
                use_cache=use_cache,
                system_workers=system_workers,
                raw_config=raw_config,
                run_dir=run_dir,
            )
    except (BudgetExceededError, BenchmarkRunError) as exc:
        print_run_summary(exc.output_dir)  # what finished is still worth seeing
        raise CommandFailed(str(exc)) from exc
    except ValidationError as exc:
        raise ConfigError(f"config {config}", config_issues(exc)) from exc
    print_run_summary(output_dir)
    console.print(f"[green]{done_message}[/green] Results: [bold]{output_dir}[/bold]", soft_wrap=True)
    return output_dir


def recommendation_panel(data: dict[str, Any], footer: str | None = None) -> Panel:
    """The decision as a panel: the winner (or why there is none), the statistical ties and the reasons."""
    winner = data.get("winner")
    if winner:
        headline = f"[bold green]Deploy {escape(winner)}[/bold green]  [dim](profile {escape(str(data.get('profile')))})[/dim]"
        ties = data.get("tied_with_winner") or []
        if ties:
            headline += f"\n[dim]Statistically tied on quality: {escape(', '.join(ties))}[/dim]"
    else:
        headline = "[bold red]No system meets your constraints[/bold red]"
    body = "\n".join([headline, "", *(f"• {escape(line.replace('`', ''))}" for line in data.get("rationale", []))])
    return Panel(body, title="Recommendation", title_align="left", subtitle=footer, subtitle_align="left", expand=False)


def print_run_summary(output_dir: Path) -> None:
    summary_path = output_dir / "metrics_summary.csv"
    if not summary_path.exists():
        return
    import pandas as pd

    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    run_summary_path = output_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    columns = leaderboard_columns(summary.columns, run_summary.get("primary_k"))
    table = Table(title="Leaderboard", title_justify="left")
    table.add_column("System", style="bold")
    for column in columns:
        table.add_column(column.header, justify="right")
    best: dict[str, float] = {}
    for column in columns:
        if column.key in summary.columns and summary[column.key].notna().any():
            best[column.key] = float(summary[column.key].max() if column.higher_is_better else summary[column.key].min())
    for _, row in summary.iterrows():
        cells = [str(row["system"])]
        for column in columns:
            value = row.get(column.key)
            text = escape(format_cell(column, row, sep="\n"))
            cells.append(f"[bold green]{text}[/bold green]" if not is_missing(value) and float(value) == best.get(column.key) else text)
        table.add_row(*cells)
    console.print(table)
    if run_summary:
        num_errors = int(run_summary.get("num_errors", 0))
        if num_errors:
            details = ", ".join(f"{name}: {count}" for name, count in run_summary.get("errors_by_system", {}).items())
            console.print(f"[red]{num_errors} question(s) failed and are excluded from the scores above ({details}). See failures.md.[/red]")
        cache_stats = run_summary.get("embedding_cache", {})
        saved = float(cache_stats.get("saved_cost_usd", 0.0))
        hits = int(cache_stats.get("hits", 0))
        if hits:
            console.print(f"[dim]Embedding cache: {hits} reused embeddings, ~${saved:.4f} of API spend avoided.[/dim]")
    recommendation_path = output_dir / "recommendation.json"
    if recommendation_path.exists():
        files = "recommendation.md" + (" · winner.yaml" if (output_dir / "winner.yaml").exists() else "")
        console.print(recommendation_panel(json.loads(recommendation_path.read_text(encoding="utf-8")), footer=files))
        disk = run_summary.get("cache", {})
        if disk.get("enabled"):
            lookups = int(disk["hits"]) + int(disk["misses"])
            console.print(
                f"[dim]Disk cache: {disk['hits']}/{lookups} lookups hit ({disk['hit_rate']:.0%}), ~${disk['saved_cost_usd']:.4f} of API spend avoided. "
                f"Systems were charged ${disk['charged_cost_usd']:.4f} at standalone prices; real spend ≈ ${disk['real_spend_usd']:.4f}.[/dim]"
            )
        unknown_priced = run_summary.get("unknown_priced_models", [])
        if unknown_priced:
            console.print(
                f"[yellow]No price registered for {', '.join(unknown_priced)}: reported cost is understated. Add it under `pricing:` in the config.[/yellow]"
            )
        warnings = run_summary.get("dataset_warnings", [])
        if warnings:
            console.print(f"[yellow]Dataset issues detected ({len(warnings)}). Run `ragbench inspect-dataset` for details:[/yellow]")
            for warning in warnings[:5]:
                console.print(f"  [yellow]•[/yellow] {warning}")
            if len(warnings) > 5:
                console.print(f"  [yellow]… and {len(warnings) - 5} more.[/yellow]")


# Where a run's config comes from, shared by `run`, `compare`, `evaluate` and `estimate`.
_CONFIG_HELP = "YAML config to run. Systems with a `sweep:` are expanded into one system per combination."
_PRESET_HELP = f"Use a ready-made list of systems ({', '.join(PRESET_NAMES)}) instead of the config's own; the dataset comes from --docs/--questions or the config."
_ONLY_HELP = "Only these systems (names or sweep base names, comma-separated or repeated), after sweeps are expanded."
_SKIP_HELP = "Leave these systems out (names or sweep base names, comma-separated or repeated). Applied after --only."
_JSON_HELP = "Print one JSON document on stdout when the run ends (progress and messages go to stderr), for scripts and CI."
_DONE = {"run": "Run complete.", "compare": "Comparison complete.", "evaluate": "Evaluation complete."}


def run(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    only: list[str] | None = typer.Option(None, "--only", "--systems", help=_ONLY_HELP),
    skip: list[str] | None = typer.Option(None, "--skip", help=_SKIP_HELP),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above evaluation.cost_confirm_threshold_usd."),
    as_json: bool = typer.Option(False, "--json", help=_JSON_HELP),
) -> None:
    """Run a config: every system on the same questions, then the report and a recommendation."""
    if ctx.info_name == "evaluate":
        console.print("[yellow]`ragbench evaluate` is deprecated and will be removed; use `ragbench run` (same options).[/yellow]")
    with json_output(as_json):
        path, raw = resolve_config(config, preset, docs, questions, qrels, only, skip)
        output_dir = execute_benchmark(path, mock, max_workers, _DONE.get(ctx.info_name or "run", "Run complete."), use_cache=not no_cache, system_workers=system_workers, raw_config=raw, yes=yes)
    if as_json:
        print_json(describe_run(output_dir))


commands.command("run")(run)
commands.command("compare", help="Run multiple systems from one config (the same as `run`).")(run)
commands.command("evaluate", hidden=True, help="Deprecated alias of `run`.")(run)


@commands.command()
def estimate(
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    only: list[str] | None = typer.Option(None, "--only", "--systems", help=_ONLY_HELP),
    skip: list[str] | None = typer.Option(None, "--skip", help=_SKIP_HELP),
) -> None:
    """Project what a run would cost and how long it would take, without spending anything.

    Runs every system on the offline mock models (corpus indexing in full, a sample of the questions) to measure prompts and call
    counts, then prices them with the configured models. Warns when the price table is old or a model has no price.
    """
    path, raw = resolve_config(config, preset, docs, questions, qrels, only, skip)
    load_project_env(path)
    loaded = load_experiment(path, raw)
    try:
        with console.status("Measuring each system on the mock models…") as status:
            result = estimate_run(loaded, progress=lambda name: status.update(f"Measuring {name}…"))
    except (FileNotFoundError, DocumentLoadError, ValueError) as exc:
        raise fail(str(exc)) from exc
    print_estimate(result)


@commands.command("recommend")
def recommend_command(
    run: Path = typer.Option(..., "--run", help="A finished run directory, e.g. results/<run>."),
    profile: str | None = typer.Option(None, "--profile", help="balanced, max_quality, cheapest_acceptable or lowest_latency. Default: the run's `selection.profile`."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Highest acceptable mean cost per question, in dollars."),
    max_latency: float | None = typer.Option(None, "--max-latency", min=0, help="Highest acceptable p95 latency, in milliseconds."),
    min_faithfulness: float | None = typer.Option(None, "--min-faithfulness", min=0, max=5, help="Lowest acceptable mean faithfulness (0-5)."),
    min_answer_score: float | None = typer.Option(None, "--min-answer-score", min=0, max=5, help="Lowest acceptable mean answer score (0-5)."),
    max_ingestion_cost: float | None = typer.Option(None, "--max-ingestion-cost", min=0, help="Highest acceptable one-off indexing cost, in dollars."),
    local_models: bool = typer.Option(False, "--local-models", help="Only systems whose models all run on this machine."),
    no_network: bool = typer.Option(False, "--no-network", help="Only systems that send no data off this machine (local models, no network tools)."),
    export: Path | None = typer.Option(None, "--export", help="Write the winner's runnable config to this file."),
    as_json: bool = typer.Option(False, "--json", help="Print the recommendation as JSON on stdout instead of the panel (the exit status is the same)."),
) -> None:
    """Which system to deploy, from a finished run: ranked by your constraints and priorities, with the reasons.

    Starts from the run's own `selection:` settings; the options here override them. Exits with status 1 when no system qualifies.
    """
    from ragbench.selection.recommend import recommend, selection_of_run, winner_yaml_text

    with json_output(as_json):
        try:
            base = selection_of_run(run)
            overrides = {
                "max_cost_per_question": max_cost,
                "max_latency_ms_p95": max_latency,
                "min_faithfulness": min_faithfulness,
                "min_answer_score": min_answer_score,
                "max_ingestion_cost": max_ingestion_cost,
                "require_local_models": True if local_models else None,
                "require_no_network": True if no_network else None,
            }
            constraints = base.constraints.model_copy(update={key: value for key, value in overrides.items() if value is not None})
            recommendation = recommend(run, constraints=constraints, weights=base.weights, profile=profile or base.profile)
        except (FileNotFoundError, ValueError) as exc:
            raise fail(str(exc)) from exc
        if as_json:
            print_json(recommendation.to_dict())
        else:
            console.print(recommendation_panel(recommendation.to_dict()))
        if recommendation.winner is None:
            raise typer.Exit(1)
        if export is not None:
            export.write_text(winner_yaml_text(run, recommendation.winner), encoding="utf-8")
            console.print(f"[green]Wrote the winner's config to {export}[/green]")
