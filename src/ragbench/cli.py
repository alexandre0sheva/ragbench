from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragbench import __version__
from ragbench.config.loader import load_config
from ragbench.datasets.demo_generator import write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import DocumentLoadError, load_documents
from ragbench.documents.preview import DEFAULT_MIN_TOKENS, chunk_stats, stats_as_dict
from ragbench.documents.tokenizer import count_tokens
from ragbench.evaluation.evaluator import BenchmarkRunError, ProgressListener, run_benchmark
from ragbench.models.errors import MissingExtraError, ModelInitError
from ragbench.models.refs import resolve_run_mode
from ragbench.reporting.columns import format_value, is_missing, leaderboard_columns
from ragbench.utils.env import load_project_env

app = typer.Typer(help="RAGBench: evaluation-first RAG benchmark framework.")
console = Console()


def _mock_warning(config: Path, force_mock: bool = False) -> None:
    if force_mock:
        console.print("[yellow]Forced mock mode enabled. Scores are for pipeline validation only.[/yellow]")
        return
    try:
        mock = resolve_run_mode(load_config(config), force_mock=False) == "mock"
    except Exception:
        return  # an unreadable or invalid config is reported by the run itself
    if mock:
        console.print(
            "[yellow]No API key found for the configured models (OPENAI_API_KEY / ANTHROPIC_API_KEY). Running in mock mode. "
            "Scores are for pipeline validation only.[/yellow]"
        )


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
    config: Path, mock: bool, max_workers: int | None, done_message: str, use_cache: bool = True, system_workers: int | None = None
) -> None:
    load_project_env(config)
    _mock_warning(config, force_mock=mock)
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
            output_dir = run_benchmark(config, force_mock=mock, max_workers=max_workers, progress=_RichProgress(progress), use_cache=use_cache, system_workers=system_workers)
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
            text = format_value(column, value)
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
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing demo document files."),
) -> None:
    """Create or verify the bundled demo dataset."""
    stats = write_demo_dataset(output, overwrite=overwrite)
    console.print(f"[green]Demo dataset ready at {output}[/green]")
    console.print(f"Documents: {stats['documents']} | Questions: {stats['questions']} | Qrels: {stats['qrels']}")


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


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config to run."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
) -> None:
    """Run a single config."""
    _execute_benchmark(config, mock, max_workers, "Run complete.", use_cache=not no_cache, system_workers=system_workers)


@app.command()
def compare(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config containing multiple systems."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
) -> None:
    """Run multiple systems from one config."""
    _execute_benchmark(config, mock, max_workers, "Comparison complete.", use_cache=not no_cache, system_workers=system_workers)


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config to evaluate."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", min=1, help="Override evaluation.max_workers: questions answered at the same time within a system."),
    system_workers: int | None = typer.Option(None, "--system-workers", min=1, help="Override evaluation.system_workers: systems evaluated at the same time."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not read or write the persistent disk cache for this run."),
) -> None:
    """Alias for run/compare."""
    _execute_benchmark(config, mock, max_workers, "Evaluation complete.", use_cache=not no_cache, system_workers=system_workers)


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
