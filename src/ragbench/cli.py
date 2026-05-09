from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragbench import __version__
from ragbench.datasets.demo_generator import write_demo_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.documents.loaders import load_documents
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.rag_systems import SYSTEM_REGISTRY
from ragbench.utils.env import has_openai_key, load_project_env

app = typer.Typer(help="RAGBench: evaluation-first RAG benchmark framework.")
console = Console()


def _mock_warning(force_mock: bool = False) -> None:
    if force_mock:
        console.print("[yellow]Forced mock mode enabled. Scores are for pipeline validation only.[/yellow]")
    elif not has_openai_key():
        console.print("[yellow]No OpenAI API key found. Running in mock mode. Scores are for pipeline validation only.[/yellow]")


@app.callback()
def main(version: bool | None = typer.Option(None, "--version", help="Show version and exit.")) -> None:
    load_project_env()
    if version:
        console.print(f"ragbench {__version__}")
        raise typer.Exit()


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
    load_project_env(config)
    _mock_warning(force_mock=mock)
    output_dir = run_benchmark(config, force_mock=mock, max_workers=max_workers)
    console.print(f"[green]Run complete.[/green] Results: {output_dir}")


@app.command()
def compare(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config containing multiple systems."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Override evaluation.max_workers for per-system question parallelism."),
) -> None:
    """Run multiple systems from one config."""
    load_project_env(config)
    _mock_warning(force_mock=mock)
    output_dir = run_benchmark(config, force_mock=mock, max_workers=max_workers)
    console.print(f"[green]Comparison complete.[/green] Results: {output_dir}")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config to evaluate."),
    mock: bool = typer.Option(False, "--mock", help="Force local mock mode even if OPENAI_API_KEY is set."),
    max_workers: int | None = typer.Option(None, "--max-workers", help="Override evaluation.max_workers for per-system question parallelism."),
) -> None:
    """Alias for run/compare."""
    load_project_env(config)
    _mock_warning(force_mock=mock)
    output_dir = run_benchmark(config, force_mock=mock, max_workers=max_workers)
    console.print(f"[green]Evaluation complete.[/green] Results: {output_dir}")


if __name__ == "__main__":
    app()
