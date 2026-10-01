"""`ragbench doctor`: check that this machine can run a benchmark, and say what to fix. It never prints a secret."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path

import typer
from rich.markup import escape
from rich.table import Table

from ragbench import __version__
from ragbench.cli.common import console, fail, json_output, print_json
from ragbench.config.schema import ExperimentConfig
from ragbench.models import cost as pricing
from ragbench.models.errors import MissingExtraError  # noqa: F401  (keeps the extras list below next to the error it explains)
from ragbench.utils.env import has_api_key, load_project_env
from ragbench.utils.extras import extra_installed

commands = typer.Typer()

OK, INFO, WARN, FAIL = "ok", "info", "warn", "fail"
REQUIRED_PACKAGES = ("typer", "pydantic", "pyyaml", "rich", "pandas", "numpy", "scikit-learn", "rank-bm25", "tiktoken", "openai", "jinja2", "markdown")
# extra -> (import name, distribution name)
OPTIONAL_EXTRAS = {
    "anthropic": ("anthropic", "anthropic"),
    "pdf": ("pypdf", "pypdf"),
    "docx": ("docx", "python-docx"),
    "chroma": ("chromadb", "chromadb"),
    "faiss": ("faiss", "faiss-cpu"),
    "qdrant": ("qdrant_client", "qdrant-client"),
    "local": ("sentence_transformers", "sentence-transformers"),
}
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
MIN_PYTHON = (3, 11)


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str
    fix: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"group": self.group, "name": self.name, "status": self.status, "detail": self.detail, "fix": self.fix}


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _folder_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _size(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _writable(directory: Path) -> tuple[bool, str]:
    """Whether a run could write into `directory` (or create it): tried for real, with a file that is removed again."""
    probe = directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        with tempfile.NamedTemporaryFile(dir=probe, prefix=".ragbench-doctor-"):
            pass
    except OSError as exc:
        return False, f"{probe} is not writable ({exc.strerror or exc})"
    return True, "exists and is writable" if directory.exists() else f"will be created under {probe}"


def environment_checks(results_dir: Path, cache_dir: Path, today: date | None = None) -> list[Check]:
    checks: list[Check] = []
    python = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= MIN_PYTHON
    checks.append(Check("Environment", "Python", OK if ok else FAIL, f"{python} ({platform.python_implementation()}, {platform.system()} {platform.machine()})", None if ok else "ragbench needs Python 3.11 or newer."))
    checks.append(Check("Environment", "ragbench", OK, __version__))

    missing = [name for name in REQUIRED_PACKAGES if _version(name) is None]
    checks.append(Check("Packages", "required", FAIL if missing else OK, f"missing: {', '.join(missing)}" if missing else f"all {len(REQUIRED_PACKAGES)} installed", "pip install --upgrade ragbench" if missing else None))
    for extra, (module, distribution) in OPTIONAL_EXTRAS.items():
        installed = extra_installed(module)
        version = _version(distribution)
        checks.append(Check("Optional extras", extra, OK if installed else INFO, f"{distribution} {version or ''}".strip() if installed else "not installed", None if installed else f"pip install 'ragbench[{extra}]'"))

    for variable in KEYS:
        present = has_api_key(variable)  # presence only: the value is never read into a message
        checks.append(Check("Credentials", variable, OK if present else INFO, "set" if present else "not set", None if present else "Without it, models of that provider run as offline mocks (`--mock` behavior)."))

    ok, detail = _writable(results_dir)
    checks.append(Check("Folders", f"results ({results_dir})", OK if ok else FAIL, detail, None if ok else "Pass --results-dir, or fix the permissions."))
    if cache_dir.exists():
        checks.append(Check("Folders", f"cache ({cache_dir})", OK, f"{_size(_folder_size(cache_dir))} on disk", "`ragbench cache clear` empties it."))
    else:
        checks.append(Check("Folders", f"cache ({cache_dir})", INFO, "empty (created on the first live run)"))

    today = today or date.today()
    age = (today - date.fromisoformat(pricing.PRICING_AS_OF)).days
    from ragbench.evaluation.estimate import STALE_PRICES_AFTER_DAYS

    stale = age > STALE_PRICES_AFTER_DAYS
    checks.append(
        Check(
            "Prices",
            "price table",
            WARN if stale else OK,
            f"as of {pricing.PRICING_AS_OF} ({age} days ago)",
            f"Older than {STALE_PRICES_AFTER_DAYS} days: costs may be off. Add current prices under `pricing:` in your config, or update ragbench." if stale else None,
        )
    )
    return checks


def config_checks(config: ExperimentConfig) -> list[Check]:
    """Every model the config calls: does it have a price, and could a live run reach it."""
    from ragbench.models.refs import collect_model_refs, missing_credentials

    checks: list[Check] = []
    pricing.clear_pricing_overrides()
    pricing.register_pricing({name: price.model_dump() for name, price in config.pricing.items()})
    try:
        refs = collect_model_refs(config)
        for kind, models in (("generator/judge", refs.llm), ("embedding", refs.embedding)):
            for ref in models:
                known = pricing.price_known(ref)
                checks.append(
                    Check(
                        "Models in the config",
                        f"{ref} ({kind})",
                        OK if known else WARN,
                        "price known" if known else "no price: its cost would be reported as $0",
                        None if known else f"Add it under `pricing:` in the config ({{{ref}: {{input: ..., output: ...}}}}, USD per 1M tokens).",
                    )
                )
    finally:
        pricing.clear_pricing_overrides()
    problems = missing_credentials(config)
    checks.append(Check("Models in the config", "credentials", WARN if problems else OK, "; ".join(problems) if problems else "every hosted model has its key", "A live run refuses to start until they are set; `--mock` needs none." if problems else None))
    return checks


MARKS = {OK: "[green]ok[/green]", INFO: "[dim]info[/dim]", WARN: "[yellow]warn[/yellow]", FAIL: "[red]FAIL[/red]"}


@commands.command("doctor")
def doctor(
    config: Path | None = typer.Option(None, "--config", "-c", help="Also check every model this config calls against the price table and the credentials it needs."),
    results_dir: Path = typer.Option(Path("results"), "--results-dir", envvar="RAGBENCH_RESULTS_DIR", help="The folder runs are written to."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Cache directory (default: $RAGBENCH_CACHE_DIR or .ragbench_cache)."),
    as_json: bool = typer.Option(False, "--json", help="Print the checks as JSON on stdout."),
) -> None:
    """Check Python, packages, optional extras, API keys (never shown), folders and the price table. Exits with status 1 if something is broken."""
    from ragbench.cli.tools import cache_path

    load_project_env(config)
    with json_output(as_json):
        checks = environment_checks(results_dir, cache_path(cache_dir)[0])
        if config is not None:
            from ragbench.cli.common import load_experiment

            checks += config_checks(load_experiment(config, None))
        broken = [check for check in checks if check.status == FAIL]
        if as_json:
            print_json({"ok": not broken, "checks": [check.as_dict() for check in checks]})
        else:
            group = ""
            table = Table(show_header=False, box=None, pad_edge=False)
            for check in checks:
                if check.group != group:
                    group = check.group
                    table.add_row("", f"[bold]{group}[/bold]", "")
                table.add_row(MARKS[check.status], escape(check.name), escape(check.detail))
                if check.fix and check.status != OK:
                    table.add_row("", "", f"[dim]→ {escape(check.fix)}[/dim]")
            console.print(table)
            warnings = sum(1 for check in checks if check.status == WARN)
            console.print(f"\n{'[red]' if broken else '[green]'}{len(broken)} problem(s)[/]{'' if not warnings else f', [yellow]{warnings} warning(s)[/yellow]'}")
        if broken:
            raise fail("Fix the problems marked FAIL above.", code=1)
