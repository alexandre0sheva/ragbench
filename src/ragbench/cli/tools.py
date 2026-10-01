"""Smaller utilities: previewing chunking, listing systems, the persistent cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError
from rich.markup import escape
from rich.table import Table

from ragbench.cli.common import console, fail
from ragbench.documents.loaders import load_documents
from ragbench.documents.preview import DEFAULT_MIN_TOKENS, chunk_stats, stats_as_dict
from ragbench.documents.tokenizer import count_tokens
from ragbench.errors import ConfigError, config_issues
from ragbench.utils.env import load_project_env

commands = typer.Typer()


def parse_chunker_spec(text: str) -> dict[str, Any]:
    """`--chunker` value: a YAML or JSON mapping such as `{type: markdown, chunk_size: 300}`; empty means all defaults."""
    parsed = yaml.safe_load(text) if text.strip() else {}
    parsed = {} if parsed is None else parsed
    if not isinstance(parsed, dict):
        raise ValueError(f"--chunker must be a YAML/JSON mapping like '{{type: markdown}}', got: {text!r}")
    return parsed


@commands.command("chunk-preview")
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
        raise ConfigError("--chunker", config_issues(exc)) from exc
    except (ValueError, FileNotFoundError, yaml.YAMLError) as exc:
        raise fail(str(exc)) from exc

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


@commands.command("list-systems")
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


cache_app = typer.Typer(help="Inspect or clear the persistent cache of LLM responses and corpus embeddings.")
commands.add_typer(cache_app, name="cache")


def cache_path(cache_dir: Path | None) -> tuple[Path, Path]:
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

    directory, path = cache_path(cache_dir)
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

    directory, path = cache_path(cache_dir)
    what = f"the `{namespace}` cache" if namespace else "the whole cache"
    if not yes:
        typer.confirm(f"Delete {what} in {directory}?", abort=True)
    removed = DiskCache(path).clear(namespace)
    console.print(f"[green]Removed {removed} cached entries.[/green]")


completion_app = typer.Typer(help="Shell completion for the `ragbench` command.")
commands.add_typer(completion_app, name="completion")
SHELLS = ("bash", "zsh", "fish", "powershell", "pwsh")


def _shell(name: str | None) -> str:
    chosen = name or Path(os.environ.get("SHELL", "")).name
    if chosen not in SHELLS:
        raise fail(f"Cannot tell which shell to use ({chosen or 'none detected'}). Pass --shell {'|'.join(SHELLS)}.")
    return chosen


@completion_app.command("show")
def completion_show(shell: str | None = typer.Option(None, "--shell", help=f"{', '.join(SHELLS)} (default: the shell in $SHELL).")) -> None:
    """Print the completion script, to review it or to source it yourself."""
    from typer._completion_shared import get_completion_script

    typer.echo(get_completion_script(prog_name="ragbench", complete_var="_RAGBENCH_COMPLETE", shell=_shell(shell)))


@completion_app.command("install")
def completion_install(shell: str | None = typer.Option(None, "--shell", help=f"{', '.join(SHELLS)} (default: the shell in $SHELL).")) -> None:
    """Install tab completion for your shell (writes a script and, for bash and zsh, a line in your shell's startup file)."""
    from typer._completion_shared import install

    installed, path = install(shell=_shell(shell), prog_name="ragbench", complete_var="_RAGBENCH_COMPLETE")
    console.print(f"[green]Installed {installed} completion at {path}[/green]", soft_wrap=True)
    console.print("Restart your shell (or `source` that file) for it to take effect.")
