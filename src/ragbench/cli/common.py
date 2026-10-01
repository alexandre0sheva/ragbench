"""Helpers shared by the command modules: the console, failing with a message and an exit code, loading a config, asking before spending."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.presets import apply_preset
from ragbench.config.schema import ExperimentConfig
from ragbench.config.sweep import expand_sweeps, select_systems, skip_systems, split_system_names
from ragbench.errors import CommandFailed, ConfigError, RagbenchError, UsageError, config_issues
from ragbench.evaluation.estimate import confirmation_decision
from ragbench.models.refs import resolve_run_mode

console = Console()
error_console = Console(stderr=True)  # errors go to stderr, so a script reading stdout (`--json`) never mistakes one for data


class State:
    """Process-wide switches set by the root command's options."""

    debug = False


STATE = State()


def render_error(exc: RagbenchError) -> None:
    """An error as a message: a config's problems as a list (where, what, what to do), anything else as one red line, then its hint."""
    if isinstance(exc, ConfigError):
        error_console.print(f"[red]{escape(str(exc))}:[/red]", soft_wrap=True)
        for issue in exc.issues:
            where = f"{issue.path}: " if issue.path else ""
            error_console.print(f"  [red]•[/red] {escape(where + issue.message)}", soft_wrap=True)
            if issue.suggestion:
                error_console.print(f"      [dim]→[/dim] {escape(issue.suggestion)}", soft_wrap=True)
    else:
        error_console.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
    if exc.hint:
        error_console.print(f"[dim]{escape(exc.hint)}[/dim]", soft_wrap=True)


def mock_warning(config: ExperimentConfig, force_mock: bool = False) -> None:
    if force_mock:
        console.print("[yellow]Forced mock mode enabled. Scores are for pipeline validation only.[/yellow]")
        return
    if resolve_run_mode(config, force_mock=False) == "mock":
        console.print(
            "[yellow]No API key found for the configured models (OPENAI_API_KEY / ANTHROPIC_API_KEY). Running in mock mode. "
            "Scores are for pipeline validation only.[/yellow]"
        )


def fail(message: str, code: int = 2) -> RagbenchError:
    """The error to `raise` for a problem with the input (exit code 2) or, with `code=1`, a command that ran and ended badly."""
    return UsageError(message) if code == 2 else CommandFailed(message, exit_code=code)


@contextmanager
def json_output(enabled: bool) -> Iterator[None]:
    """With `--json`, everything the command prints for people (progress, tables, errors) goes to stderr so stdout holds only the JSON."""
    if not enabled:
        yield
        return
    console.stderr = True
    try:
        yield
    finally:
        console.stderr = False


def print_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def resolve_config(
    config: Path | None,
    preset: str | None,
    docs: Path | None,
    questions: Path | None,
    qrels: Path | None,
    systems: list[str] | None,
    skip: list[str] | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    """Where a run's config comes from: a file as written, or (a preset, `--only`, `--skip`) a config assembled here and passed to the run as data.

    Returns the path (used to find `.env` and to name things) and the assembled config, None when the file is to be used as it is.
    """
    if config is None and preset is None:
        raise fail("Pass --config (a YAML config) or --preset (quick, standard, thorough or agentic, with --docs and --questions).")
    if preset is None and (docs or questions or qrels):
        raise fail("--docs, --questions and --qrels only apply with --preset; put the dataset in your config otherwise.")
    try:
        raw: dict[str, Any] | None = load_config_dict(config) if config is not None else None
        if preset is not None:
            raw = expand_sweeps(apply_preset(raw, preset, docs=docs, questions=questions, qrels=qrels))
        if systems:
            assert raw is not None
            raw = select_systems(raw, split_system_names(systems))
        if skip:
            assert raw is not None
            raw = skip_systems(raw, split_system_names(skip))
    except (FileNotFoundError, ValueError) as exc:
        raise fail(str(exc)) from exc
    path = config if config is not None else Path.cwd() / f"preset-{preset}.yaml"
    return path, (raw if preset is not None or systems or skip else None)


def load_experiment(path: Path, raw: dict[str, Any] | None) -> ExperimentConfig:
    try:
        return ExperimentConfig.model_validate(raw) if raw is not None else load_config(path)
    except ValidationError as exc:
        raise ConfigError(f"config {path}", config_issues(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise fail(str(exc)) from exc


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def in_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in {"", "0", "false", "no"}


def money(value: float) -> str:
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def confirm_spend(total_usd: float, what: str, yes: bool, threshold: float) -> None:
    """Before a paid command: ask (or refuse, or just go) when its estimated cost is above the confirmation threshold; the rule `run` uses."""
    decision = confirmation_decision(total_usd, threshold, yes=yes, interactive=is_interactive(), ci=in_ci())
    console.print(f"[dim]Estimated cost of {what}: about {money(total_usd)} (confirmation threshold {money(threshold)}).[/dim]")
    if decision == "proceed":
        return
    if decision == "refuse":
        raise fail(
            f"The estimated cost {money(total_usd)} is above the {money(threshold)} confirmation threshold and there is no one to ask. "
            "Re-run with --yes to go ahead, or cap the spending first (`evaluation.max_cost_usd` in the config, or `--max-cost` on commands that have it)."
        )
    if not typer.confirm(f"Go ahead for about {money(total_usd)}?", default=False):
        raise fail("Cancelled. Nothing was spent.", code=1)


def load_optional_config(config: Path | None) -> ExperimentConfig | None:
    """A config given only for its `providers:`, `pricing:` and `cache:` sections (the dataset and systems in it are not used)."""
    return load_experiment(config, None) if config is not None else None
