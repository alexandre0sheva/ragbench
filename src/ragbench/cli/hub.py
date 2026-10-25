"""`ragbench datasets`: public benchmarks fetched on demand."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail, in_ci, is_interactive, print_json

datasets_app = typer.Typer(help="Public benchmarks on demand: list them, pull one (checksum-verified, converted, optionally a stratified subset).")


def _entries() -> list:
    from ragbench.datasets.hub.registry import HUB, get_dataset

    get_dataset("multihop-rag")  # registers the built-ins
    return sorted(HUB.values(), key=lambda entry: entry.name)


@datasets_app.command("list")
def list_datasets(as_json: bool = typer.Option(False, "--json", help="Print the datasets as JSON on stdout.")) -> None:
    """The datasets `pull` can fetch, with download size, license and kind of corpus."""
    from ragbench.datasets.hub.pull import human_size

    entries = _entries()
    if as_json:
        print_json(
            [
                {
                    "name": entry.name,
                    "title": entry.title,
                    "description": entry.description,
                    "license": entry.license,
                    "license_url": entry.license_url,
                    "download_bytes": entry.total_bytes,
                    "shared_corpus": entry.shared_corpus,
                    "revision": entry.revision,
                    "source_url": entry.source_url,
                }
                for entry in entries
            ]
        )
        return
    table = Table(title="Public benchmarks (ragbench datasets pull NAME)")
    for column in ("Name", "Dataset", "Download", "License", "Corpus"):
        table.add_column(column)
    for entry in entries:
        table.add_row(entry.name, entry.title, human_size(entry.total_bytes), entry.license, "shared" if entry.shared_corpus else "per question")
    console.print(table)
    console.print("[dim]Nothing is downloaded until you pull, and then only after you confirm. Terms and citations are written to SOURCE.md next to the data.[/dim]")


@datasets_app.command("pull")
def pull_dataset(
    name: str = typer.Argument(..., help="A dataset from `ragbench datasets list`."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Where to write the dataset folder (default: data/NAME)."),
    subset: int | None = typer.Option(None, "--subset", min=1, help="Keep a stratified sample of N questions: every category and the unanswerable share are kept."),
    seed: int = typer.Option(42, "--seed", help="Seed of the subset sample."),
    force: bool = typer.Option(False, "--force", help="Replace an earlier pull in the output folder."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Download without asking first."),
) -> None:
    """Download a public benchmark, verify its checksum, convert it to RAGBench files and write `docs/`, `questions.jsonl`, `qrels.jsonl` and `SOURCE.md`."""
    from ragbench.datasets.hub.pull import pull
    from ragbench.datasets.hub.registry import get_dataset

    entry = get_dataset(name)
    destination = output or Path("data") / entry.name

    def confirm(message: str) -> bool:
        if yes:
            return True
        if not is_interactive() or in_ci():
            raise fail(f"{message} There is no one to ask: re-run with --yes to go ahead.", code=1)
        return typer.confirm(message, default=False)

    result = pull(entry, destination, subset=subset, seed=seed, force=force, confirm=confirm, on_download=lambda file: console.print(f"[dim]Downloading {escape(file)} ...[/dim]"))
    shown = "" if result.subset is None else f" (stratified subset, seed {result.subset['seed']})"
    console.print(f"[green]Wrote {destination}[/green]: {result.n_questions} questions{shown}, {result.n_documents} documents, {result.n_qrels} qrels.")
    console.print(f"License: {entry.license}. Read {destination / 'SOURCE.md'} for the citation and caveats.")
    for warning in result.warnings[:10]:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    if len(result.warnings) > 10:
        console.print(f"[yellow]... and {len(result.warnings) - 10} more warnings.[/yellow]")
    config = f"configs/{entry.config_name}.yaml"
    console.print("\n[bold]Next[/bold]")
    steps = [f"ragbench estimate --config {config}   # projected cost of a live run", f"ragbench run --config {config} --mock   # a free pipeline check"]
    if destination != Path("data") / entry.name:
        steps.insert(0, f"Edit {config}: its dataset paths expect data/{entry.name}/ (yours is {destination}).")
    for number, step in enumerate(steps, start=1):
        console.print(f"  {number}. {escape(step)}", soft_wrap=True)
