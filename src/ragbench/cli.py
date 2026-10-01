from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragbench import __version__
from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.presets import PRESET_NAMES, apply_preset
from ragbench.config.schema import ExperimentConfig
from ragbench.config.sweep import expand_sweeps, select_systems, split_system_names
from ragbench.datasets.demo_generator import write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import DocumentLoadError, load_documents
from ragbench.documents.preview import DEFAULT_MIN_TOKENS, chunk_stats, stats_as_dict
from ragbench.documents.tokenizer import count_tokens
from ragbench.evaluation.budget import BudgetExceededError, BudgetGuard
from ragbench.evaluation.estimate import Estimate, confirmation_decision, estimate_run
from ragbench.evaluation.evaluator import BenchmarkRunError, ProgressListener, run_benchmark
from ragbench.models.errors import MissingExtraError, ModelInitError, RagbenchModelError
from ragbench.models.refs import resolve_run_mode
from ragbench.reporting.columns import format_cell, is_missing, leaderboard_columns
from ragbench.utils.env import load_project_env
from ragbench.utils.text import estimate_tokens

app = typer.Typer(help="RAGBench: evaluation-first RAG benchmark framework.")
console = Console()


def _mock_warning(config: ExperimentConfig, force_mock: bool = False) -> None:
    if force_mock:
        console.print("[yellow]Forced mock mode enabled. Scores are for pipeline validation only.[/yellow]")
        return
    if resolve_run_mode(config, force_mock=False) == "mock":
        console.print(
            "[yellow]No API key found for the configured models (OPENAI_API_KEY / ANTHROPIC_API_KEY). Running in mock mode. "
            "Scores are for pipeline validation only.[/yellow]"
        )


def _fail(message: str, code: int = 2) -> typer.Exit:
    console.print(f"[red]{escape(message)}[/red]")
    return typer.Exit(code)


def _resolve_config(
    config: Path | None, preset: str | None, docs: Path | None, questions: Path | None, qrels: Path | None, systems: list[str] | None
) -> tuple[Path, dict[str, Any] | None]:
    """Where a run's config comes from: a file as written, or (a preset, `--systems`) a config assembled here and passed to the run as data.

    Returns the path (used to find `.env` and to name things) and the assembled config, None when the file is to be used as it is.
    """
    if config is None and preset is None:
        raise _fail("Pass --config (a YAML config) or --preset (quick, standard, thorough or agentic, with --docs and --questions).")
    if preset is None and (docs or questions or qrels):
        raise _fail("--docs, --questions and --qrels only apply with --preset; put the dataset in your config otherwise.")
    try:
        raw: dict[str, Any] | None = load_config_dict(config) if config is not None else None
        if preset is not None:
            raw = expand_sweeps(apply_preset(raw, preset, docs=docs, questions=questions, qrels=qrels))
        if systems:
            assert raw is not None
            raw = select_systems(raw, split_system_names(systems))
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    path = config if config is not None else Path.cwd() / f"preset-{preset}.yaml"
    return path, (raw if preset is not None or systems else None)


def _load_experiment(path: Path, raw: dict[str, Any] | None) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(raw) if raw is not None else load_config(path)
    except ValidationError as exc:
        console.print(f"[red]Invalid config {path}:[/red]")
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            console.print(f"  [red]•[/red] {where + ': ' if where else ''}{escape(str(error['msg']).removeprefix('Value error, '))}", soft_wrap=True)
        raise typer.Exit(2) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _in_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in {"", "0", "false", "no"}


def _money(value: float) -> str:
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _print_estimate(estimate: Estimate) -> None:
    table = Table(title="Estimate", title_justify="left")
    table.add_column("System", style="bold", overflow="fold")
    for header in ("Ingestion", "Query", "Judge", "Total"):
        table.add_column(header, justify="right")
    for row in estimate.systems:
        if row.error is not None:
            table.add_row(row.system, "[red]could not be estimated[/red]", "", "", "")
            continue
        table.add_row(row.system + (" ~" if row.agentic else ""), _money(row.ingestion_usd), _money(row.query_usd), _money(row.judge_usd), _money(row.total_usd))
    table.add_section()
    table.add_row("Total", "", "", "", f"[bold]{_money(estimate.total_usd)}[/bold]")
    console.print(table)
    console.print(
        f"{estimate.n_questions} questions over {estimate.n_documents} documents (~{estimate.corpus_tokens:,} tokens) · "
        f"about {_duration(estimate.wall_seconds)} at the configured concurrency (rough) · prices as of {estimate.pricing_as_of}"
    )
    for warning in estimate.warnings:
        console.print(f"[yellow]• {escape(warning)}[/yellow]")
    console.print(
        "[dim]Measured by running each system on the offline mock models; prompts, call counts and embedding volume are exact, output lengths are the mock's "
        "(real answers are usually longer). Assumes no cache hits. ~ marks agentic systems.[/dim]"
    )


def _confirm_cost(config: ExperimentConfig, yes: bool) -> None:
    """Before a live run: project its cost, and ask (or refuse, or just go) when it is above `evaluation.cost_confirm_threshold_usd`."""
    try:
        with console.status("Estimating the cost…"):
            estimate = estimate_run(config)
    except Exception as exc:  # noqa: BLE001 (the estimate is advice; a failure in it must not stop the run itself)
        console.print(f"[yellow]Could not estimate the cost ({escape(str(exc))}); continuing. Set `evaluation.max_cost_usd` to cap the spending.[/yellow]")
        return
    threshold = config.evaluation.cost_confirm_threshold_usd
    cap = config.evaluation.max_cost_usd
    decision = confirmation_decision(estimate.total_usd, threshold, yes=yes, interactive=_is_interactive(), ci=_in_ci())
    console.print(f"[dim]Estimated cost: about {_money(estimate.total_usd)} (confirmation threshold {_money(threshold)}; see `ragbench estimate`).[/dim]")
    if cap is not None and estimate.total_usd > cap:
        console.print(f"[yellow]The estimate is above your `evaluation.max_cost_usd` of {_money(cap)}: the run will stop early.[/yellow]")
    if decision == "proceed":
        return
    _print_estimate(estimate)
    if decision == "refuse":
        raise _fail(
            f"The estimated cost {_money(estimate.total_usd)} is above the {_money(threshold)} confirmation threshold and there is no one to ask. "
            "Re-run with --yes to go ahead, or raise `evaluation.cost_confirm_threshold_usd`."
        )
    if not typer.confirm(f"Run it for about {_money(estimate.total_usd)}?", default=False):
        raise _fail("Cancelled. Nothing was spent.", code=1)


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


def _execute_benchmark(
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
    loaded = _load_experiment(config, raw_config)
    _mock_warning(loaded, force_mock=mock)
    if resolve_run_mode(loaded, mock) == "live":
        _confirm_cost(loaded, yes)
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
    except BudgetExceededError as exc:
        _print_run_summary(exc.output_dir)
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except BenchmarkRunError as exc:
        _print_run_summary(exc.output_dir)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except ValidationError as exc:
        console.print(f"[red]Invalid config {config}:[/red]")
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            console.print(f"  [red]•[/red] {where + ': ' if where else ''}{str(error['msg']).removeprefix('Value error, ')}", soft_wrap=True)
        raise typer.Exit(2) from exc
    except (FileNotFoundError, DocumentLoadError, ModelInitError, MissingExtraError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")  # messages name extras like ragbench[anthropic], which Rich would read as markup
        raise typer.Exit(2) from exc
    _print_run_summary(output_dir)
    console.print(f"[green]{done_message}[/green] Results: [bold]{output_dir}[/bold]", soft_wrap=True)
    return output_dir


def _recommendation_panel(data: dict[str, Any], footer: str | None = None) -> Panel:
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


def _print_run_summary(output_dir: Path) -> None:
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
        console.print(_recommendation_panel(json.loads(recommendation_path.read_text(encoding="utf-8")), footer=files))
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


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ragbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(None, "--version", help="Show version and exit.", callback=_version_callback, is_eager=True),
) -> None:
    load_project_env()


@app.command()
def demo(
    output: Path = typer.Option(Path("data/demo"), "--output", "-o", help="Directory where the demo dataset is written."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite demo files that differ from the bundled copy (by default they are kept and reported)."),
) -> None:
    """Create or verify the bundled demo dataset."""
    stats = write_demo_dataset(output, overwrite=overwrite)
    console.print(f"[green]Demo dataset ready at {output}[/green]")
    console.print(f"Documents: {stats['documents']} | Questions: {stats['questions']} | Qrels: {stats['qrels']}")
    if stats["modified"]:
        console.print(f"[yellow]{stats['modified']} file(s) differ from the bundled copy and were kept; use --overwrite to restore them.[/yellow]")


def _distribution(values: Any, unit: str = "") -> str:
    return f"min {values.min:,.0f} · median {values.p50:,.0f} · p95 {values.p95:,.0f} · max {values.max:,.0f}{unit}"


def _print_profile(profile: Any, label_free: bool) -> None:
    from ragbench.datasets.profile import DatasetProfile

    assert isinstance(profile, DatasetProfile)
    table = Table(title="Dataset Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    documents, questions, qrels = profile.documents, profile.questions, profile.qrels
    table.add_row("Documents", str(documents.count))
    table.add_row("Corpus size", f"~{documents.total_tokens:,} tokens")
    table.add_row("Tokens per document", _distribution(documents.tokens))
    language = f"{documents.language} ({documents.language_confidence:.0%})" if documents.language != "unknown" else "unknown"
    table.add_row("Language (guess)", language)
    table.add_row("Questions", str(questions.count))
    table.add_row("Words per question", _distribution(questions.words))
    table.add_row("Answerable questions", f"{questions.answerable} ({questions.answerable_ratio:.0%})")
    table.add_row("Not-in-context questions", str(questions.unanswerable))
    table.add_row("Qrel rows", str(qrels.qrel_rows))
    console.print(table)

    coverage = Table(title="Qrels coverage")
    coverage.add_column("Metric")
    coverage.add_column("Value", justify="right")
    coverage.add_row("Questions with relevance labels", f"{qrels.labeled_questions} of {questions.count} ({qrels.labeled_share:.0%})")
    coverage.add_row("Answerable, no labels (retrieval not scored)", str(qrels.unlabeled_answerable))
    coverage.add_row("Documents some question points to", f"{qrels.documents_referenced} of {documents.count} ({qrels.corpus_coverage:.0%})")
    console.print(coverage)
    if label_free:
        console.print("[yellow]Label-free dataset: retrieval metrics will be blank; answers are judged against the reference answer or the retrieved context.[/yellow]")

    for title, column, counts in (
        ("Categories", "Category", questions.categories),
        ("Difficulty", "Difficulty", questions.difficulties),
        ("Answer types", "Answer type", questions.answer_types),
    ):
        if title != "Categories" and set(counts) <= {"unknown"}:
            continue
        balance = Table(title=title)
        balance.add_column(column)
        balance.add_column("Count", justify="right")
        for name, count in counts.items():
            balance.add_row(name, str(count))
        console.print(balance)

    if profile.chunk_sizes:
        console.print(f"[bold]Suggested chunk sizes[/bold] (tokens): {', '.join(map(str, profile.chunk_sizes))}. {escape(profile.chunk_size_note)}")
        console.print("[dim]Compare them with a sweep: `sweep: {chunker.chunk_size: [...]}` (docs/configuration.md#sweeps).[/dim]")


@app.command("inspect-dataset")
def inspect_dataset(
    docs: Path = typer.Option(..., "--docs", help="Document folder."),
    questions: Path = typer.Option(..., "--questions", help="Questions JSONL file."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL file."),
    include: list[str] | None = typer.Option(None, "--include", help="Only load files matching this glob (repeatable)."),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Skip files matching this glob (repeatable)."),
    on_error: str = typer.Option("raise", "--on-error", help="`raise` (fail on a file that cannot be loaded) or `skip` (warn and go on)."),
    estimate_cost: bool = typer.Option(True, "--estimate/--no-estimate", help="Also project the cost of the `standard` preset (runs each system once on the mock models; a few seconds)."),
) -> None:
    """Profile a dataset: sizes and balance, label coverage, suggested chunk sizes, projected cost and likely problems."""
    from ragbench.config.schema import DatasetConfig
    from ragbench.datasets.profile import profile_dataset, projected_standard_cost

    if on_error not in ("raise", "skip"):
        console.print("[red]--on-error must be 'raise' or 'skip'.[/red]")
        raise typer.Exit(2)
    load_warnings: list[str] = []
    try:
        documents = load_documents(docs, include=include, exclude=exclude, on_error=on_error, warnings=load_warnings)  # type: ignore[arg-type]
    except (DocumentLoadError, ValueError, FileNotFoundError, MissingExtraError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc
    try:
        dataset = load_dataset(questions, qrels)
    except (FileNotFoundError, ValueError) as exc:  # ValidationError is a ValueError
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc
    profile = profile_dataset(documents, dataset)
    _print_profile(profile, dataset.label_free)
    if load_warnings:
        console.print(f"[yellow]{len(load_warnings)} document loading warning(s):[/yellow]")
        for message in load_warnings:
            console.print(f"  [yellow]•[/yellow] {escape(message)}", soft_wrap=True)
    warnings = [*validate_dataset(documents, dataset), *profile.warnings]
    if warnings:
        console.print(f"[yellow]Found {len(warnings)} dataset issue(s):[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    else:
        console.print("[green]No dataset issues found.[/green]")
    if estimate_cost:
        try:
            with console.status("Projecting the cost of the standard preset…"):
                projected = projected_standard_cost(
                    DatasetConfig(documents_path=docs, questions_path=questions, qrels_path=qrels, include=include, exclude=exclude, on_error=on_error)
                )
        except Exception as exc:  # noqa: BLE001 (advice only: a failure here must not hide the profile above)
            console.print(f"[yellow]Could not project the cost: {escape(str(exc))}[/yellow]")
        else:
            console.print("\n[bold]Projected cost[/bold] of `--preset standard` with the default models:")
            _print_estimate(projected)


def _confirm_spend(total_usd: float, what: str, yes: bool, threshold: float) -> None:
    """Before a paid command: ask (or refuse, or just go) when its estimated cost is above the confirmation threshold; the rule `run` uses."""
    decision = confirmation_decision(total_usd, threshold, yes=yes, interactive=_is_interactive(), ci=_in_ci())
    console.print(f"[dim]Estimated cost of {what}: about {_money(total_usd)} (confirmation threshold {_money(threshold)}).[/dim]")
    if decision == "proceed":
        return
    if decision == "refuse":
        raise _fail(
            f"The estimated cost {_money(total_usd)} is above the {_money(threshold)} confirmation threshold and there is no one to ask. "
            "Re-run with --yes to go ahead, or pass --max-cost to cap the spending."
        )
    if not typer.confirm(f"Go ahead for about {_money(total_usd)}?", default=False):
        raise _fail("Cancelled. Nothing was spent.", code=1)


def _load_optional_config(config: Path | None) -> ExperimentConfig | None:
    """A config given only for its `providers:`, `pricing:` and `cache:` sections (the dataset and systems in it are not used)."""
    return _load_experiment(config, None) if config is not None else None


def _run_generation(
    documents: list[Any],
    *,
    n: int,
    shares: dict[str, float],
    ref: str,
    seed: int,
    loaded: ExperimentConfig | None,
    mock: bool,
    max_cost: float | None,
    no_cache: bool,
    yes: bool,
) -> tuple[Any, bool, float]:
    """Write synthetic questions: with the model `ref` (estimate, confirm, cache) or, in mock mode / without a key, from templates.

    Returns the `SynthesisResult`, whether it ran live, and what the cache avoided. Shared by `generate-questions` and `auto`.
    """
    from ragbench.cache import activate_cache, open_cache_runtime
    from ragbench.config.schema import CacheConfig, EvaluationConfig
    from ragbench.datasets.synthesis import allocate, estimate_generation_cost, generate_questions
    from ragbench.models import cost as pricing
    from ragbench.models.llms import create_llm
    from ragbench.models.refs import parse_model_ref, provider_reachable
    from ragbench.runtime import RuntimeContext, activate_runtime

    try:
        live = not mock and provider_reachable(parse_model_ref(ref)[0])
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    providers = loaded.providers if loaded else {}
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()
    pricing.register_pricing({name: price.model_dump() for name, price in loaded.pricing.items()} if loaded else {})
    cache_runtime = None
    try:
        if live:
            from ragbench.datasets.synthesis import allocate

            estimate = estimate_generation_cost(documents, n, shares, ref)
            if ref in pricing.unknown_priced_models() or estimate == 0:
                console.print(f"[yellow]No price is registered for {escape(ref)}: the cost is shown as $0. Add one under `pricing:` (pass --config).[/yellow]")
            _confirm_spend(estimate, f"writing {n} questions ({', '.join(f'{count} {name}' for name, count in allocate(n, shares).items())})", yes, (loaded.evaluation if loaded else EvaluationConfig()).cost_confirm_threshold_usd)
            cache_config = loaded.cache if loaded else CacheConfig()
            cache_runtime = open_cache_runtime(cache_config) if cache_config.enabled and not no_cache else None
            with activate_cache(cache_runtime), activate_runtime(RuntimeContext(providers=providers)):
                try:
                    llm = create_llm(ref, providers=providers)
                    with console.status(f"Writing {n} questions with {ref}…"):
                        result = generate_questions(documents, n=n, mix=shares, llm=llm, seed=seed, budget=BudgetGuard(max_cost))
                except (RagbenchModelError, MissingExtraError) as exc:
                    raise _fail(f"The model failed: {exc}") from exc
        else:
            console.print("[yellow]Mock mode: template questions, no model called. They validate the pipeline and say nothing about real quality.[/yellow]")
            result = generate_questions(documents, n=n, mix=shares, llm=None, seed=seed)
        saved = cache_runtime.disk.stats()["saved_cost_usd"] if cache_runtime is not None else 0.0
    finally:
        if cache_runtime is not None:
            cache_runtime.disk.close()
        pricing.clear_pricing_overrides()
    return result, live, saved


@app.command("generate-questions")
def generate_questions_command(
    docs: Path = typer.Option(..., "--docs", help="Your documents: a folder, or one file."),
    out: Path = typer.Option(..., "--out", help="Questions JSONL to write, e.g. questions.jsonl."),
    n: int = typer.Option(100, "--n", min=1, help="How many questions to write."),
    mix: str | None = typer.Option(None, "--mix", help="Category shares, e.g. single_hop=0.4,multi_hop=0.2,paraphrase=0.15,numeric=0.1,unanswerable=0.15 (the default)."),
    model: str | None = typer.Option(None, "--model", help="Model ref that writes the questions (default: the default generator model)."),
    seed: int = typer.Option(0, "--seed", help="Same seed, same documents sampled, same questions."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Read `providers:`, `pricing:` and `cache:` from this config (needed for openai_compatible: models)."),
    mock: bool = typer.Option(False, "--mock", help="Write template questions with no model (pipeline validation only). Also used when the model's API key is missing."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Stop writing once this many dollars were charged (partial results are kept)."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above the confirmation threshold."),
    force: bool = typer.Option(False, "--force", help="Replace --out if it exists (it may hold questions you reviewed)."),
) -> None:
    """Write questions for documents that have none: single-hop, multi-hop, paraphrased, numeric and unanswerable, each flagged `needs_review`."""
    from ragbench.datasets.synthesis import MOCK_GENERATOR, parse_mix
    from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
    from ragbench.utils.jsonl import write_jsonl

    load_project_env(docs)
    if out.exists() and not force:
        raise _fail(f"{out} already exists. Use --force to replace it (reviewed questions would be lost), or pick another --out.")
    try:
        shares = parse_mix(mix)
        documents = load_documents(docs)
        loaded = _load_optional_config(config)
        ref = model or DEFAULT_GENERATOR_MODEL
    except (ValueError, FileNotFoundError, DocumentLoadError, MissingExtraError) as exc:
        raise _fail(str(exc)) from exc
    result, live, saved = _run_generation(documents, n=n, shares=shares, ref=ref, seed=seed, loaded=loaded, mock=mock, max_cost=max_cost, no_cache=no_cache, yes=yes)
    if not result.rows:
        raise _fail("No question could be written. " + " ".join(result.warnings))
    write_jsonl(out, result.rows)

    table = Table(title=f"Questions written to {out}", title_justify="left")
    table.add_column("Category", style="bold")
    table.add_column("Asked for", justify="right")
    table.add_column("Written", justify="right")
    for name, wanted in result.requested.items():
        table.add_row(name, str(wanted), str(result.made.get(name, 0)))
    table.add_section()
    table.add_row("Total", str(sum(result.requested.values())), str(len(result.rows)))
    console.print(table)
    if live:
        spent = f"{_money(result.cost_usd)} charged at standalone prices over {result.calls} model calls" + (f"; the cache avoided {_money(saved)}" if saved else "")
        console.print(f"[dim]Generator {escape(result.generator)} · {spent}.[/dim]")
    else:
        console.print(f"[dim]Generator {MOCK_GENERATOR} (mock).[/dim]")
    for warning in result.warnings:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    console.print(
        f"[bold]Review them:[/bold] every row is flagged `metadata.needs_review`. Read the file, fix or delete bad questions, then check it with "
        f"`ragbench inspect-dataset --docs {docs} --questions {out}`.",
        soft_wrap=True,
    )
    if result.stopped_by_budget:
        console.print(f"[red]Stopped at --max-cost {_money(max_cost or 0)}: {len(result.rows)} of {n} questions were written.[/red]")
        raise typer.Exit(1)


@app.command("label")
def label_command(
    run: Path = typer.Option(..., "--run", help="A finished run directory, e.g. results/<run>."),
    top_k: int = typer.Option(10, "--top-k", min=1, help="Pool the first K distinct documents each system retrieved, per question."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model ref that grades relevance (default: the run's judge model)."),
    out: Path | None = typer.Option(None, "--out", help="Directory for the outputs (default: the run directory)."),
    apply: bool = typer.Option(False, "--apply", help="Also write qrels.merged.jsonl: your labels plus the proposed grades for documents they do not mention."),
    mock: bool = typer.Option(False, "--mock", help="Grade with a word-overlap stand-in, no model (pipeline validation only). Also used when the model's API key is missing."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Stop grading once this many dollars were charged (partial proposals are written)."),
    max_workers: int = typer.Option(4, "--max-workers", min=1, help="Documents graded at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above the confirmation threshold."),
    force: bool = typer.Option(False, "--force", help="With --apply: replace an existing qrels.merged.jsonl."),
) -> None:
    """Propose relevance labels for a run by pooling: every system's top documents, graded 0-3 by an LLM. Never changes your qrels."""
    from ragbench.cache import activate_cache, open_cache_runtime
    from ragbench.datasets.pooling import (
        MERGED_NAME,
        LabelError,
        build_pool,
        estimate_grading_cost,
        grade_pool,
        load_run,
        write_label_outputs,
    )
    from ragbench.models import cost as pricing
    from ragbench.models.llms import create_llm
    from ragbench.models.refs import parse_model_ref, provider_reachable
    from ragbench.runtime import RuntimeContext, activate_runtime

    load_project_env(run)
    destination = out or run
    if apply and (destination / MERGED_NAME).exists() and not force:
        raise _fail(f"{destination / MERGED_NAME} already exists. Use --force to replace it, or pick another --out.")
    try:
        data = load_run(run)
        plan = build_pool(data, top_k)
        ref = judge_model or data.config.evaluation.judge_model
        live = not mock and provider_reachable(parse_model_ref(ref)[0])
    except (LabelError, ValueError, ValidationError) as exc:
        raise _fail(str(exc)) from exc
    pricing.clear_pricing_overrides()
    pricing.reset_unknown_priced_models()
    pricing.register_pricing({name: price.model_dump() for name, price in data.config.pricing.items()})
    console.print(f"{len(plan.questions)} questions · {plan.pairs} (question, document) pairs pooled from {len({s for r in data.rankings.values() for s in r})} systems at top {top_k}")
    if plan.skipped_unanswerable:
        console.print(f"[dim]{plan.skipped_unanswerable} unanswerable question(s) skipped.[/dim]")
    cache_runtime = None
    try:
        if live:
            estimate = estimate_grading_cost(data, plan, ref)
            if estimate == 0:
                console.print(f"[yellow]No price is registered for {escape(ref)}: the cost is shown as $0. Add one under `pricing:` in the run's config.[/yellow]")
            _confirm_spend(estimate, f"grading {plan.pairs} documents with {ref}", yes, data.config.evaluation.cost_confirm_threshold_usd)
            cache_runtime = open_cache_runtime(data.config.cache) if data.config.cache.enabled and not no_cache else None
        else:
            console.print("[yellow]Mock grader: word overlap, no model called. The proposals only validate the pipeline; do not use them as labels.[/yellow]")
        progress = Progress(SpinnerColumn(), TextColumn("Grading"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console, transient=True)
        with activate_cache(cache_runtime), activate_runtime(RuntimeContext(providers=data.config.providers)), progress:
            task = progress.add_task("grade", total=plan.pairs)
            try:
                llm = create_llm(ref, force_mock=not live, providers=data.config.providers)
                graded = grade_pool(data, plan, llm, workers=max_workers, budget=BudgetGuard(max_cost) if max_cost is not None else None, progress=lambda done, total: progress.update(task, completed=done))
            except (RagbenchModelError, MissingExtraError) as exc:
                raise _fail(f"The grader failed: {exc}") from exc
        saved = cache_runtime.disk.stats()["saved_cost_usd"] if cache_runtime is not None else 0.0
    finally:
        if cache_runtime is not None:
            cache_runtime.disk.close()
        pricing.clear_pricing_overrides()
    files = write_label_outputs(destination, data, plan, graded, top_k=top_k, judge_model=ref if live else "mock grader", mock=not live, apply=apply)

    distribution = {grade: sum(1 for judgments in graded.grades.values() for j in judgments.values() if j.grade == grade) for grade in range(4)}
    console.print(f"[green]Graded {sum(distribution.values())} pairs[/green] (0: {distribution[0]} · 1: {distribution[1]} · 2: {distribution[2]} · 3: {distribution[3]})")
    if live:
        console.print(f"[dim]{_money(graded.cost_usd)} charged at standalone prices over {graded.calls} calls" + (f"; the cache avoided {_money(saved)}" if saved else "") + ".[/dim]")
    for pair in graded.ungraded[:5]:
        console.print(f"  [yellow]•[/yellow] {pair[0]}/{pair[1]}: no valid grade; nothing is proposed for it", soft_wrap=True)
    console.print(f"Wrote [bold]{files.proposed}[/bold] and [bold]{files.review}[/bold]" + (f" and [bold]{files.merged}[/bold]" if files.merged else ""))
    if files.merged:
        console.print(f"Use the merged labels: set `dataset.qrels_path: {files.merged}` in your config. Your own files were not touched.")
    else:
        console.print("[dim]Read the review, then add --apply to write a merged qrels file (your labels stay authoritative).[/dim]")
    if graded.stopped_by_budget:
        console.print(f"[red]Stopped at --max-cost {_money(max_cost or 0)}: some pairs were not graded.[/red]")
        raise typer.Exit(1)


@app.command()
def auto(
    docs: Path | None = typer.Option(None, "--docs", help="Your documents: a folder, or one file."),
    questions: Path | None = typer.Option(None, "--questions", help="Your questions JSONL. Without it, questions are written from your documents (flagged needs_review)."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --questions."),
    preset: str = typer.Option("standard", "--preset", help=f"Which systems to compare: {', '.join(PRESET_NAMES)}."),
    profile: str = typer.Option("balanced", "--profile", help="What the recommendation optimizes: balanced, max_quality, cheapest_acceptable or lowest_latency."),
    n_questions: int = typer.Option(50, "--n-questions", min=1, help="How many questions to write when you have none."),
    seed: int = typer.Option(0, "--seed", help="Seed for writing questions."),
    max_cost: float | None = typer.Option(None, "--max-cost", min=0, help="Total dollars to spend (writing questions plus the run). The run stops once it is reached."),
    model: str | None = typer.Option(None, "--model", help="Model ref that writes the questions (default: the default generator model)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Take models, providers, pricing, evaluation and selection settings from this config (its systems and dataset are not used)."),
    output_dir: Path = typer.Option(Path("results"), "--output-dir", help="Where the run directory is created."),
    resume: Path | None = typer.Option(None, "--resume", help="Continue an earlier auto run in this directory: finished systems are kept, the rest are run."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode: nothing is paid for and the scores only validate the pipeline."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before spending more than evaluation.cost_confirm_threshold_usd."),
    open_report: bool = typer.Option(False, "--open", help="Open report.html in your browser when done."),
) -> None:
    """From a folder of documents to a decision: profile them, write questions if you have none, run the preset, and recommend a system.

    Writes everything to one run directory (questions, results, recommendation.md, winner.yaml, report.html). If it stops (budget, a crash, Ctrl-C),
    `--resume RUN_DIR` continues without paying again for the systems that finished.
    """
    import webbrowser

    from ragbench.datasets.profile import profile_dataset
    from ragbench.datasets.synthesis import parse_mix
    from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
    from ragbench.utils.jsonl import write_jsonl
    from ragbench.workflows import auto as workflow

    load_project_env(resume or docs)
    try:
        if resume is not None:
            given = [flag for flag, value in (("--docs", docs), ("--questions", questions), ("--qrels", qrels), ("--config", config)) if value is not None]
            if given:
                raise workflow.AutoError(f"{', '.join(given)} come from the run being resumed; drop them (only --max-cost, --mock, --yes, --no-cache and --open can change).")
            settings, generation_cost = workflow.read_state(resume)
            run_dir = resume
            if max_cost is not None:
                settings.max_cost = max_cost
            settings.mock = settings.mock or mock
        else:
            if docs is None:
                raise workflow.AutoError("Pass --docs (a folder of documents) to start, or --resume RUN_DIR to continue a run.")
            workflow.check_choices(preset, profile)
            settings = workflow.AutoSettings(
                docs=str(docs.resolve()),
                questions=str(questions.resolve()) if questions else None,
                qrels=str(qrels.resolve()) if qrels else None,
                preset=preset,
                profile=profile,
                n_questions=n_questions,
                seed=seed,
                max_cost=max_cost,
                model=model,
                config=str(config.resolve()) if config else None,
                mock=mock,
            )
            generation_cost = 0.0
            run_dir = None  # created below, once the inputs are known to be readable
        documents = load_documents(Path(settings.docs))
        loaded = _load_optional_config(Path(settings.config)) if settings.config else None
        if settings.questions is not None and not Path(settings.questions).is_file():
            raise workflow.AutoError(f"Questions file not found: {settings.questions}")
        shares = parse_mix(None)
    except (workflow.AutoError, ValueError, FileNotFoundError, DocumentLoadError, MissingExtraError) as exc:
        raise _fail(str(exc)) from exc
    if run_dir is None:
        run_dir = workflow.new_run_dir(output_dir)
    workflow.write_state(run_dir, settings, generation_cost)
    console.print(f"Run directory: [bold]{run_dir}[/bold]" + ("  [dim](resuming)[/dim]" if resume else ""), soft_wrap=True)
    total_tokens = sum(estimate_tokens(document.text) for document in documents)
    console.print(f"{len(documents)} documents, ~{total_tokens:,} tokens")

    # 1. Questions: yours, ones an earlier attempt already wrote, or new synthetic ones.
    if settings.questions is not None:
        questions_path = Path(settings.questions)
    else:
        questions_path = run_dir / workflow.QUESTIONS_NAME
        if questions_path.exists():
            console.print(f"[dim]Using the questions already written to {questions_path}.[/dim]")
        else:
            result, live, _ = _run_generation(
                documents,
                n=settings.n_questions,
                shares=shares,
                ref=settings.model or DEFAULT_GENERATOR_MODEL,
                seed=settings.seed,
                loaded=loaded,
                mock=settings.mock,
                max_cost=settings.max_cost,
                no_cache=no_cache,
                yes=yes,
            )
            if not result.rows:
                raise _fail("No question could be written. " + " ".join(result.warnings))
            write_jsonl(questions_path, result.rows)
            generation_cost = result.cost_usd
            workflow.write_state(run_dir, settings, generation_cost)
            console.print(f"Wrote {len(result.rows)} questions to {questions_path} ({'live' if live else 'mock templates'}, {_money(result.cost_usd)}); read them: every one is flagged `needs_review`.")
            for warning in result.warnings:
                console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
            if result.stopped_by_budget:
                raise _fail(f"Stopped at --max-cost while writing questions. Raise it and run `ragbench auto --resume {run_dir}`.", code=1)

    # 2. What the data looks like, and what is likely to distort the comparison.
    try:
        dataset = load_dataset(questions_path, Path(settings.qrels) if settings.qrels else None)
    except (FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    profile_result = profile_dataset(documents, dataset)
    console.print(
        f"{profile_result.questions.count} questions ({profile_result.questions.answerable} answerable) · {profile_result.qrels.labeled_questions} with relevance labels"
        + (" · [yellow]label-free: retrieval metrics will be blank[/yellow]" if dataset.label_free else "")
    )
    for warning in [*validate_dataset(documents, dataset), *profile_result.warnings][:6]:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)

    # 3. The run: estimate, confirm, run every system, recommend. Systems an earlier attempt finished are not run again.
    try:
        existing = run_dir / "config.yaml"
        if resume is not None and existing.exists():
            raw = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
            if max_cost is not None:
                raw["evaluation"] = {**(raw.get("evaluation") or {}), "max_cost_usd": workflow.remaining_budget(settings.max_cost, generation_cost)}
        else:
            raw = workflow.build_config(settings, run_dir, questions_path, max_cost_usd=workflow.remaining_budget(settings.max_cost, generation_cost))
    except (workflow.AutoError, ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    try:
        finished = _execute_benchmark(existing, settings.mock, None, "Auto run complete.", use_cache=not no_cache, raw_config=raw, yes=yes, run_dir=run_dir)
    except typer.Exit as stopped:
        console.print(f"[bold]Resume with:[/bold] ragbench auto --resume {run_dir}" + (" --max-cost <more>" if settings.max_cost else ""), soft_wrap=True)
        raise stopped

    # 4. The deliverables.
    files = workflow.winner_files(finished)
    recommendation_path = finished / "recommendation.json"
    winner = json.loads(recommendation_path.read_text(encoding="utf-8")).get("winner") if recommendation_path.exists() else None
    console.print(f"\n[bold]{'Deploy ' + escape(winner) if winner else 'No system meets your constraints'}[/bold]")
    for label, key in (("Runnable config", "winner"), ("Report", "report"), ("Reasons", "recommendation")):
        if key in files:
            console.print(f"  {label}: {files[key]}", soft_wrap=True)
    if open_report and "report" in files:
        webbrowser.open(files["report"].resolve().as_uri())


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to write ragbench.yaml (and questions.jsonl if you have none yet)."),
    docs: Path = typer.Option(..., "--docs", help="Your documents: a folder, or one file."),
    questions: Path | None = typer.Option(None, "--questions", help="Your questions JSONL. Without it a questions.jsonl template is created in DIRECTORY."),
    preset: str = typer.Option("standard", "--preset", help=f"Which systems to compare: {', '.join(PRESET_NAMES)}."),
    force: bool = typer.Option(False, "--force", help="Replace an existing ragbench.yaml (questions files are never replaced)."),
) -> None:
    """Start a benchmark of your own documents: write a ready-to-run config and a questions file to edit."""
    from ragbench.datasets.scaffold import ScaffoldError, scaffold_project

    try:
        result = scaffold_project(directory, docs, questions, preset=preset, force=force)
    except ScaffoldError as exc:
        raise _fail(str(exc)) from exc
    console.print(f"[green]Wrote {result.config_path}[/green] ({result.n_documents} documents, preset `{result.preset}`)")
    if result.created_questions:
        console.print(f"[green]Wrote {result.questions_path}[/green]: placeholder questions, so the project runs; replace them with real ones.")
    console.print("\n[bold]Next steps[/bold]")
    steps = []
    if result.created_questions:
        steps.append(f"Put your real questions in {result.questions_path} (docs/dataset-format.md), or bring existing ones with `ragbench import`.")
    steps += [
        f"ragbench inspect-dataset --docs {docs} --questions {result.questions_path}   # check the data and see the projected cost",
        f"ragbench run --config {result.config_path} --mock   # a free pipeline check",
        f"ragbench run --config {result.config_path}   # the real run (asks before spending more than the confirmation threshold)",
    ]
    for number, step in enumerate(steps, start=1):
        console.print(f"  {number}. {escape(step)}", soft_wrap=True)
    console.print("[dim]Paths in the config are relative to the directory you run ragbench from.[/dim]")


@app.command("import")
def import_command(
    fmt: str = typer.Option(..., "--format", help="csv (question,answer,doc_ids), beir (corpus.jsonl + queries.jsonl + qrels/), or qa-md (Markdown `Q:` / `A:` pairs)."),
    source: Path = typer.Option(..., "--input", help="The file (csv, qa-md) or folder (beir, qa-md) to import."),
    output: Path = typer.Option(..., "--output", help="Directory to write questions.jsonl (and qrels.jsonl, docs/ for beir) into."),
    docs: Path | None = typer.Option(None, "--docs", help="Your documents folder, to check the imported doc ids against (csv, qa-md)."),
    split: str = typer.Option("test", "--split", help="beir: which qrels/<split>.tsv to use."),
    force: bool = typer.Option(False, "--force", help="Replace the files of an earlier import into --output (for beir, its docs/ folder too)."),
) -> None:
    """Convert a CSV, BEIR dataset or Markdown Q/A file into RAGBench's questions (and qrels, documents)."""
    from ragbench.datasets.importers import ImportDatasetError, import_dataset

    try:
        result = import_dataset(fmt, source, output, docs=docs, split=split, force=force)
    except (ImportDatasetError, DocumentLoadError, MissingExtraError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    console.print(f"[green]Imported {result.n_questions} questions[/green] to {result.questions_path}")
    if result.docs_dir is not None:
        console.print(f"Documents: {result.n_documents} in {result.docs_dir}" + (f" · qrels: {result.n_qrels} rows in {result.qrels_path}" if result.qrels_path else ""))
    for warning in result.warnings[:10]:
        console.print(f"  [yellow]•[/yellow] {escape(warning)}", soft_wrap=True)
    if len(result.warnings) > 10:
        console.print(f"  [yellow]… and {len(result.warnings) - 10} more.[/yellow]")
    documents = result.docs_dir or docs or Path("YOUR_DOCS")
    console.print(f"\nNext: [bold]ragbench init {output} --docs {documents} --questions {result.questions_path}[/bold]", soft_wrap=True)


def parse_chunker_spec(text: str) -> dict[str, Any]:
    """`--chunker` value: a YAML or JSON mapping such as `{type: markdown, chunk_size: 300}`; empty means all defaults."""
    parsed = yaml.safe_load(text) if text.strip() else {}
    parsed = {} if parsed is None else parsed
    if not isinstance(parsed, dict):
        raise ValueError(f"--chunker must be a YAML/JSON mapping like '{{type: markdown}}', got: {text!r}")
    return parsed


@app.command("chunk-preview")
def chunk_preview(
    docs: Path = typer.Option(..., "--docs", help="Document file or folder."),
    chunker: str = typer.Option("{}", "--chunker", help="Chunker section as YAML or JSON, e.g. '{type: markdown, chunk_size: 300}'."),
    doc: str | None = typer.Option(None, "--doc", help="Only this document id."),
    limit: int = typer.Option(5, "--limit", "-n", min=0, help="How many chunks to show."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of tables."),
    mock: bool = typer.Option(False, "--mock", help="Use mock hashing embeddings for chunkers that embed text (`semantic`)."),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Embedding model ref for `semantic` (default: the benchmark default)."),
) -> None:
    """Show how a chunker cuts your documents: size statistics plus the first chunks. Nothing is indexed or paid for (except `semantic`'s sentence embeddings)."""
    from ragbench.config.schema import ChunkerConfig
    from ragbench.rag_systems.components import build_chunker, chunk_documents

    load_project_env(docs)
    try:
        spec = parse_chunker_spec(chunker)
        config = ChunkerConfig.model_validate(spec)
        documents = load_documents(docs)
        if doc is not None:
            documents = [d for d in documents if d.doc_id == doc]
            if not documents:
                known = ", ".join(d.doc_id for d in load_documents(docs)[:10])
                raise ValueError(f"No document with id {doc!r} under {docs} (first ids: {known})")
        models = {"embedding": embedding_model} if embedding_model else {}
        chunks, cost = chunk_documents(build_chunker(config, models=models, force_mock=mock), documents)
    except ValidationError as exc:
        console.print("[red]Invalid --chunker:[/red]")
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            console.print(f"  [red]•[/red] {escape(where + ': ' if where else '')}{escape(str(error['msg']).removeprefix('Value error, '))}", soft_wrap=True)
        raise typer.Exit(2) from exc
    except (ValueError, FileNotFoundError, ModelInitError, MissingExtraError, yaml.YAMLError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc

    threshold = config_min_tokens(config)
    stats = chunk_stats(chunks, documents=len(documents), threshold_tokens=threshold)
    shown = chunks[:limit]
    if as_json:
        payload = {
            "chunker": config.model_dump(exclude_none=True),
            "stats": stats_as_dict(stats),
            "embedding_cost_usd": cost.total_cost,
            "chunks": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "tokens": count_tokens(c.text), "text": c.text, "metadata": c.metadata} for c in shown
            ],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
        return

    table = Table(title=f"Chunking: {config.type}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Documents", str(stats.documents))
    table.add_row("Chunks", str(stats.chunks))
    table.add_row(f"Tokens per chunk ({stats.tokenizer})", f"mean {stats.mean_tokens:.0f} · median {stats.median_tokens:.0f} · p95 {stats.p95_tokens:.0f} · max {stats.max_tokens}")
    table.add_row(f"Chunks under {stats.threshold_tokens} tokens", f"{stats.percent_below_threshold:.1f}%")
    if cost.total_cost:
        table.add_row("Embedding cost", f"${cost.total_cost:.4f}")
    console.print(table)
    if shown:
        chunk_table = Table(title=f"First {len(shown)} chunk(s)")
        for column in ("#", "Doc", "Tokens", "Section"):
            chunk_table.add_column(column)
        chunk_table.add_column("Text")
        for chunk in shown:
            path = chunk.metadata.get("heading_path")
            chunk_table.add_row(
                str(chunk.metadata["chunk_index"]),
                chunk.doc_id,
                str(count_tokens(chunk.text)),
                " > ".join(path) if path else "",
                escape(" ".join(chunk.text.split())[:160]),
            )
        console.print(chunk_table)


def config_min_tokens(config: Any) -> int:
    """The "small chunk" threshold for the preview: the chunker's `min_chunk_size` (or its class default), else 50 tokens."""
    from ragbench.registry import CHUNKERS

    if config.min_chunk_size is not None:
        return int(config.min_chunk_size)
    return int(CHUNKERS.get(config.type).default_min_chunk_size or DEFAULT_MIN_TOKENS)


@app.command("list-systems")
def list_systems() -> None:
    """Print available RAG systems with their cost and latency profile."""
    from ragbench.rag_systems import SYSTEMS, all_specs

    table = Table(title="Available RAG Systems")
    table.add_column("Type", style="bold", no_wrap=True)
    table.add_column("Cost")
    table.add_column("Latency")
    table.add_column("LLM", header_style="bold", justify="center")
    table.add_column("Best for")
    described = set()
    for spec in all_specs():
        described.add(spec.type)
        table.add_row(spec.type, spec.cost_profile, spec.latency_profile, "yes" if spec.requires_llm else "no", spec.best_for)
    for system_type in sorted(set(SYSTEMS.names()) - described):  # custom/plugin systems without a spec
        table.add_row(system_type, "?", "?", "?", "Custom system")
    console.print(table)
    console.print("[dim]LLM = the retrieval side calls an LLM. Options for each system: docs/systems.md[/dim]")


# Where a run's config comes from, shared by `run`, `compare`, `evaluate` and `estimate`.
_CONFIG_HELP = "YAML config to run. Systems with a `sweep:` are expanded into one system per combination."
_PRESET_HELP = f"Use a ready-made list of systems ({', '.join(PRESET_NAMES)}) instead of the config's own; the dataset comes from --docs/--questions or the config."
_SYSTEMS_HELP = "Only these systems (names or sweep base names, comma-separated or repeated), after sweeps are expanded."


@app.command()
def run(
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    systems: list[str] | None = typer.Option(None, "--systems", help=_SYSTEMS_HELP),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above evaluation.cost_confirm_threshold_usd."),
) -> None:
    """Run a single config."""
    path, raw = _resolve_config(config, preset, docs, questions, qrels, systems)
    _execute_benchmark(path, mock, max_workers, "Run complete.", use_cache=not no_cache, system_workers=system_workers, raw_config=raw, yes=yes)


@app.command()
def compare(
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    systems: list[str] | None = typer.Option(None, "--systems", help=_SYSTEMS_HELP),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above evaluation.cost_confirm_threshold_usd."),
) -> None:
    """Run multiple systems from one config."""
    path, raw = _resolve_config(config, preset, docs, questions, qrels, systems)
    _execute_benchmark(path, mock, max_workers, "Comparison complete.", use_cache=not no_cache, system_workers=system_workers, raw_config=raw, yes=yes)


@app.command()
def estimate(
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    systems: list[str] | None = typer.Option(None, "--systems", help=_SYSTEMS_HELP),
) -> None:
    """Project what a run would cost and how long it would take, without spending anything.

    Runs every system on the offline mock models (corpus indexing in full, a sample of the questions) to measure prompts and call
    counts, then prices them with the configured models. Warns when the price table is old or a model has no price.
    """
    path, raw = _resolve_config(config, preset, docs, questions, qrels, systems)
    load_project_env(path)
    loaded = _load_experiment(path, raw)
    try:
        with console.status("Measuring each system on the mock models…") as status:
            result = estimate_run(loaded, progress=lambda name: status.update(f"Measuring {name}…"))
    except (FileNotFoundError, DocumentLoadError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    _print_estimate(result)


@app.command("recommend")
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
) -> None:
    """Which system to deploy, from a finished run: ranked by your constraints and priorities, with the reasons.

    Starts from the run's own `selection:` settings; the options here override them. Exits with status 1 when no system qualifies.
    """
    from ragbench.selection.recommend import recommend, selection_of_run, winner_yaml_text

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
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc
    console.print(_recommendation_panel(recommendation.to_dict()))
    if recommendation.winner is None:
        raise typer.Exit(1)
    if export is not None:
        export.write_text(winner_yaml_text(run, recommendation.winner), encoding="utf-8")
        console.print(f"[green]Wrote the winner's config to {export}[/green]")


@app.command()
def evaluate(
    config: Path | None = typer.Option(None, "--config", "-c", help=_CONFIG_HELP),
    preset: str | None = typer.Option(None, "--preset", help=_PRESET_HELP),
    docs: Path | None = typer.Option(None, "--docs", help="Documents folder for --preset."),
    questions: Path | None = typer.Option(None, "--questions", help="Questions JSONL for --preset."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL for --preset."),
    systems: list[str] | None = typer.Option(None, "--systems", help=_SYSTEMS_HELP),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before a live run whose estimated cost is above evaluation.cost_confirm_threshold_usd."),
) -> None:
    """Alias for run/compare."""
    path, raw = _resolve_config(config, preset, docs, questions, qrels, systems)
    _execute_benchmark(path, mock, max_workers, "Evaluation complete.", use_cache=not no_cache, system_workers=system_workers, raw_config=raw, yes=yes)


cache_app = typer.Typer(help="Inspect or clear the persistent cache of LLM responses and corpus embeddings.")
app.add_typer(cache_app, name="cache")


def _cache_path(cache_dir: Path | None) -> tuple[Path, Path]:
    directory = cache_dir or Path(os.environ.get("RAGBENCH_CACHE_DIR") or ".ragbench_cache")
    return directory, directory / "cache.sqlite3"


def _human_bytes(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


@cache_app.command("stats")
def cache_stats(
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Cache directory (default: $RAGBENCH_CACHE_DIR or .ragbench_cache)."),
) -> None:
    """Show what the persistent cache holds."""
    from ragbench.cache import DiskCache

    directory, path = _cache_path(cache_dir)
    info = DiskCache(path).describe()
    if not info:
        console.print(f"The cache is empty (or does not exist):\n{directory}")
        return
    table = Table(title="Persistent cache", title_justify="left", caption=str(directory))
    table.add_column("Namespace", style="bold")
    table.add_column("Entries", justify="right")
    table.add_column("Stored data", justify="right")
    for namespace, row in info.items():
        table.add_row(namespace, str(row["entries"]), _human_bytes(row["bytes"]))
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Cache directory (default: $RAGBENCH_CACHE_DIR or .ragbench_cache)."),
    namespace: str | None = typer.Option(None, "--namespace", help="Only clear this namespace: `llm` or `embeddings`."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask for confirmation."),
) -> None:
    """Delete cached entries. Re-running a live benchmark afterwards pays for them again."""
    from ragbench.cache import DiskCache

    directory, path = _cache_path(cache_dir)
    what = f"the `{namespace}` cache" if namespace else "the whole cache"
    if not yes:
        typer.confirm(f"Delete {what} in {directory}?", abort=True)
    removed = DiskCache(path).clear(namespace)
    console.print(f"[green]Removed {removed} cached entries.[/green]")


if __name__ == "__main__":
    app()
