"""`ragbench library list|show|export|clear`: the lessons `ragbench optimize` keeps about document collections."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ragbench.cli.common import console, fail, in_ci, is_interactive, json_output, load_experiment, print_json
from ragbench.optimize.library import Entry, Library, LibraryError, resolve_library
from ragbench.utils.text import truncate

library_app = typer.Typer(help="The experience library: what `ragbench optimize` learned on earlier collections. Offline; nothing here spends money.")

PATH = typer.Option(None, "--path", help="Library folder (default: `library.path` of --config, else RAGBENCH_LIBRARY, else ~/.ragbench/library).")
CONFIG = typer.Option(None, "--config", "-c", help="Take `library.path` from this config.")
FINGERPRINT = typer.Option(None, "--fingerprint", help="Only the entries of this collection key, or the start of one (`show` and `--json` print it).")


def _library(path: Path | None, config: Path | None) -> Library:
    if path is not None:
        return Library(path)
    return resolve_library(load_experiment(config, None).library.path if config is not None else None)[0]


def _gain(entry: Entry) -> str:
    gain = entry.gain
    return f"{gain['mean_diff']:+.2f} ({gain['ci_lo']:+.2f} to {gain['ci_hi']:+.2f})"


def _entries(library: Library, fingerprint: str | None) -> list[Entry]:
    try:
        return [entry for entry in library.entries() if fingerprint is None or entry.fingerprint.startswith(fingerprint)]
    except LibraryError as exc:
        raise fail(str(exc)) from exc


@library_app.command("list")
def list_entries(path: Path | None = PATH, config: Path | None = CONFIG, fingerprint: str | None = FINGERPRINT, as_json: bool = typer.Option(False, "--json", help="Print the entries as JSON on stdout.")) -> None:
    """List the entries, oldest first: id, date, the system, its verified gain on held-out questions and the lesson."""
    with json_output(as_json):
        library = _library(path, config)
        entries = _entries(library, fingerprint)
        if as_json:
            print_json([entry.to_dict() for entry in entries])
            return
        if not entries:
            console.print(f"The library at {library.path} is empty.", soft_wrap=True)
            return
        console.print(f"Experience library: {library.path}", soft_wrap=True)
        table = Table()
        table.add_column("Id", no_wrap=True)
        table.add_column("Date", no_wrap=True)
        table.add_column("System", no_wrap=True)
        table.add_column("Gain", justify="right", no_wrap=True)
        table.add_column("Lesson", overflow="fold")
        for entry in entries:
            table.add_row(entry.id + ("*" if entry.mock else ""), entry.created_at[:10], entry.system_type, f"{entry.gain['mean_diff']:+.2f}", truncate(entry.lesson, 90))
        console.print(table)
        console.print("[dim]Gain: the verified change in mean answer rating on held-out questions (`show` has the interval). * marks entries made by a mock run.[/dim]")


@library_app.command("show")
def show(
    entry_id: str = typer.Argument(..., help="An entry id, or the start of one."),
    path: Path | None = PATH,
    config: Path | None = CONFIG,
    as_json: bool = typer.Option(False, "--json", help="Print the entry as JSON on stdout."),
) -> None:
    """Print one entry in full: the collection's features, the adopted harness, the verified gain and the lesson."""
    with json_output(as_json):
        try:
            entry = _library(path, config).find(entry_id)
        except LibraryError as exc:
            raise fail(str(exc)) from exc
        if as_json:
            print_json(entry.to_dict())
            return
        gain = entry.gain
        console.print(f"[bold]{entry.id}[/bold]{' (mock)' if entry.mock else ''}  {entry.created_at}  collection {entry.fingerprint}", soft_wrap=True)
        console.print(f"System: {entry.system} (type {entry.system_type}), from {entry.run}", soft_wrap=True)
        console.print(f"Collection: {json.dumps(entry.features, ensure_ascii=False)}", soft_wrap=True, markup=False)
        console.print(
            f"Verified gain on {gain.get('n', '?')} held-out questions: {_gain(entry)}, Holm-adjusted p = {gain.get('p_holm', float('nan')):.4f}"
            f" (base {gain.get('base_mean', float('nan')):.3f}, adopted {gain.get('pick_mean', float('nan')):.3f})",
            soft_wrap=True,
        )
        console.print(f"\nLesson: {entry.lesson}", soft_wrap=True, markup=False)
        console.print(f"\nAdopted harness:\n{json.dumps(entry.harness, indent=2, ensure_ascii=False)}", soft_wrap=True, markup=False)


@library_app.command("export")
def export(
    path: Path | None = PATH,
    config: Path | None = CONFIG,
    fingerprint: str | None = FINGERPRINT,
    out: Path | None = typer.Option(None, "--out", help="Write the JSONL here instead of stdout."),
) -> None:
    """Write the entries as JSONL (one entry per line, oldest first) to stdout, or to --out."""
    entries = _entries(_library(path, config), fingerprint)
    text = "".join(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries)
    if out is None:
        typer.echo(text, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    console.print(f"Wrote {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to {out}", soft_wrap=True)


@library_app.command("clear")
def clear(
    path: Path | None = PATH,
    config: Path | None = CONFIG,
    fingerprint: str | None = typer.Option(None, "--fingerprint", help="Remove only the entries of this collection key; without it, every entry."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask first."),
) -> None:
    """Delete entries for good (the library is append-only otherwise). Asks first; with no terminal, or in CI, it needs --yes."""
    library = _library(path, config)
    entries = _entries(library, fingerprint)
    if not entries:
        console.print("Nothing to remove.")
        return
    what = f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} from {library.path}"
    if not yes:
        if not is_interactive() or in_ci():
            raise fail(f"This would remove {what}, and there is no one to ask. Re-run with --yes to go ahead.")
        if not typer.confirm(f"Remove {what}?", default=False):
            raise fail("Cancelled. Nothing was removed.", code=1)
    try:
        removed = library.clear(fingerprint)
    except LibraryError as exc:
        raise fail(str(exc)) from exc
    console.print(f"Removed {removed} entr{'y' if removed == 1 else 'ies'} from {library.path}", soft_wrap=True)
