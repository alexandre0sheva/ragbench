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
from ragbench.evaluation.budget import BudgetExceededError
from ragbench.evaluation.estimate import Estimate, confirmation_decision, estimate_run
from ragbench.evaluation.evaluator import BenchmarkRunError, ProgressListener, run_benchmark
from ragbench.models.errors import MissingExtraError, ModelInitError
from ragbench.models.refs import resolve_run_mode
from ragbench.reporting.columns import format_cell, is_missing, leaderboard_columns
from ragbench.utils.env import load_project_env

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


def _execute_benchmark(
    config: Path,
    mock: bool,
    max_workers: int | None,
    done_message: str,
    use_cache: bool = True,
    system_workers: int | None = None,
    raw_config: dict[str, Any] | None = None,
    yes: bool = False,
) -> None:
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
    console.print(f"[green]{done_message}[/green] Results: [bold]{output_dir}[/bold]")


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


@app.command("inspect-dataset")
def inspect_dataset(
    docs: Path = typer.Option(..., "--docs", help="Document folder."),
    questions: Path = typer.Option(..., "--questions", help="Questions JSONL file."),
    qrels: Path | None = typer.Option(None, "--qrels", help="Optional qrels JSONL file."),
    include: list[str] | None = typer.Option(None, "--include", help="Only load files matching this glob (repeatable)."),
    exclude: list[str] | None = typer.Option(None, "--exclude", help="Skip files matching this glob (repeatable)."),
    on_error: str = typer.Option("raise", "--on-error", help="`raise` (fail on a file that cannot be loaded) or `skip` (warn and go on)."),
) -> None:
    """Show dataset counts, categories, answerability, and qrels coverage."""
    if on_error not in ("raise", "skip"):
        console.print("[red]--on-error must be 'raise' or 'skip'.[/red]")
        raise typer.Exit(2)
    load_warnings: list[str] = []
    try:
        documents = load_documents(docs, include=include, exclude=exclude, on_error=on_error, warnings=load_warnings)  # type: ignore[arg-type]
    except (DocumentLoadError, ValueError, FileNotFoundError, MissingExtraError) as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from exc
    dataset = load_dataset(questions, qrels)
    categories: dict[str, int] = {}
    for question in dataset.questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    answerable = sum(1 for question in dataset.questions if question.is_answerable)
    qrel_count = sum(len(v) for v in dataset.qrels.values())
    table = Table(title="Dataset Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Documents", str(len(documents)))
    table.add_row("Questions", str(len(dataset.questions)))
    table.add_row("Answerable questions", str(answerable))
    table.add_row("Not-in-context questions", str(len(dataset.questions) - answerable))
    table.add_row("Qrel rows", str(qrel_count))
    console.print(table)
    cat_table = Table(title="Categories")
    cat_table.add_column("Category")
    cat_table.add_column("Count", justify="right")
    for category, count in sorted(categories.items()):
        cat_table.add_row(category, str(count))
    console.print(cat_table)
    if load_warnings:
        console.print(f"[yellow]{len(load_warnings)} document loading warning(s):[/yellow]")
        for message in load_warnings:
            console.print(f"  [yellow]•[/yellow] {escape(message)}", soft_wrap=True)
    warnings = validate_dataset(documents, dataset)
    if warnings:
        console.print(f"[yellow]Found {len(warnings)} dataset issue(s):[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")
    else:
        console.print("[green]No dataset issues found.[/green]")


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
