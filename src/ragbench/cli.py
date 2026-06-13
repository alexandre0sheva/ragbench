from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragbench import __version__
from ragbench.datasets.demo_generator import write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents
from ragbench.evaluation.evaluator import ProgressListener, run_benchmark
from ragbench.rag_systems import SYSTEM_REGISTRY
from ragbench.utils.env import has_openai_key, load_project_env

app = typer.Typer(help="RAGBench: evaluation-first RAG benchmark framework.")
console = Console()


def _mock_warning(force_mock: bool = False) -> None:
    if force_mock:
        console.print("[yellow]Forced mock mode enabled. Scores are for pipeline validation only.[/yellow]")
    elif not has_openai_key():
        console.print("[yellow]No OpenAI API key found. Running in mock mode. Scores are for pipeline validation only.[/yellow]")


class _RichProgress(ProgressListener):
    """Live per-system progress: spinner while ingesting, bar over questions."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self.tasks: dict[str, Any] = {}
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
        self.progress.update(self.tasks[name], description=f"[bold]{name}[/bold] · done in {wall_time_ms / 1000:.1f}s")


def _execute_benchmark(config: Path, mock: bool, max_workers: int | None, done_message: str) -> None:
    load_project_env(config)
    _mock_warning(force_mock=mock)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        output_dir = run_benchmark(config, force_mock=mock, max_workers=max_workers, progress=_RichProgress(progress))
    _print_run_summary(output_dir)
    console.print(f"[green]{done_message}[/green] Results: [bold]{output_dir}[/bold]")


_LEADERBOARD_COLUMNS: list[tuple[str, str, str, bool]] = [
    # (csv column, header, format, higher_is_better)
    ("retrieval_recall@5", "Recall@5", "{:.3f}", True),
    ("retrieval_mrr@10", "MRR@10", "{:.3f}", True),
    ("retrieval_ndcg@10", "nDCG@10", "{:.3f}", True),
    ("answer_score", "Answer", "{:.2f}", True),
    ("faithfulness", "Faithful", "{:.2f}", True),
    ("avg_cost_per_question", "$/Q", "${:.5f}", False),
    ("avg_latency_ms", "Latency", "{:.0f} ms", False),
]


def _print_run_summary(output_dir: Path) -> None:
    summary_path = output_dir / "metrics_summary.csv"
    if not summary_path.exists():
        return
    import pandas as pd

    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    table = Table(title="Leaderboard", title_justify="left")
    table.add_column("System", style="bold")
    for _, header, _, _ in _LEADERBOARD_COLUMNS:
        table.add_column(header, justify="right")
    best: dict[str, float] = {}
    for col, _, _, higher in _LEADERBOARD_COLUMNS:
        if col in summary.columns:
            best[col] = float(summary[col].max() if higher else summary[col].min())
    for _, row in summary.iterrows():
        cells = [str(row["system"])]
        for col, _, fmt, _ in _LEADERBOARD_COLUMNS:
            if col not in summary.columns:
                cells.append("—")
                continue
            value = float(row[col])
            text = fmt.format(value)
            cells.append(f"[bold green]{text}[/bold green]" if value == best.get(col) else text)
        table.add_row(*cells)
    console.print(table)
    run_summary_path = output_dir / "run_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        cache_stats = run_summary.get("embedding_cache", {})
        saved = float(cache_stats.get("saved_cost_usd", 0.0))
        hits = int(cache_stats.get("hits", 0))
        if hits:
            console.print(f"[dim]Embedding cache: {hits} reused embeddings, ~${saved:.4f} of API spend avoided.[/dim]")
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
) -> None:
    """Show dataset counts, categories, answerability, and qrels coverage."""
    documents = load_documents(docs)
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
    warnings = validate_dataset(documents, dataset)
    if warnings:
        console.print(f"[yellow]Found {len(warnings)} dataset issue(s):[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")
    else:
        console.print("[green]No dataset issues found.[/green]")


@app.command("list-systems")
def list_systems() -> None:
    """Print available RAG systems."""
    table = Table(title="Available RAG Systems")
    table.add_column("Type")
    table.add_column("Class")
    for system_type, cls in sorted(SYSTEM_REGISTRY.items()):
        table.add_row(system_type, cls.__name__)
    console.print(table)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config to run."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Override evaluation.max_workers for per-system question parallelism."),
) -> None:
    """Run a single config."""
    _execute_benchmark(config, mock, max_workers, "Run complete.")


@app.command()
def compare(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config containing multiple systems."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Override evaluation.max_workers for per-system question parallelism."),
) -> None:
    """Run multiple systems from one config."""
    _execute_benchmark(config, mock, max_workers, "Comparison complete.")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config to evaluate."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Override evaluation.max_workers for per-system question parallelism."),
) -> None:
    """Alias for run/compare."""
    _execute_benchmark(config, mock, max_workers, "Evaluation complete.")


if __name__ == "__main__":
    app()
