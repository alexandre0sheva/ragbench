"""The `ragbench` application: the root command, its commands, and what happens to an error that reaches the top."""

from __future__ import annotations

from typing import Any

import typer
from pydantic import ValidationError
from typer.core import TyperGroup
from typer.exceptions import TyperException

from ragbench import __version__
from ragbench.cli import auto, dataset, doctor, report, run, runs, tools
from ragbench.cli.common import STATE, console, render_error
from ragbench.documents.loaders import DocumentLoadError
from ragbench.errors import EXIT_FAILED, EXIT_USAGE, ConfigError, RagbenchError, config_issues
from ragbench.models.errors import MissingExtraError, ModelInitError
from ragbench.utils.env import load_project_env


class RagbenchGroup(TyperGroup):
    """The root command group. An error that reaches here is shown as a message with the right exit code; `--debug` shows the traceback too."""

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except (TyperException, typer.Exit, typer.Abort):
            raise  # the framework's own: usage errors, `typer.Exit`, Ctrl-C
        except Exception as exc:  # noqa: BLE001 (this is the one place every failure is turned into a message)
            if STATE.debug:
                raise
            code = self.report(exc)
            raise typer.Exit(code) from None

    @staticmethod
    def report(exc: Exception) -> int:
        """Print `exc` as a message and return the exit code for it."""
        if isinstance(exc, RagbenchError):
            render_error(exc)
            return exc.exit_code
        if isinstance(exc, ValidationError):
            render_error(ConfigError("config", config_issues(exc)))
            return EXIT_USAGE
        if isinstance(exc, MissingExtraError | ModelInitError | FileNotFoundError | DocumentLoadError):
            render_error(RagbenchError(str(exc)))
            return EXIT_USAGE
        render_error(RagbenchError(f"Unexpected {type(exc).__name__}: {exc}", hint="Run again with `ragbench --debug ...` for the traceback."))
        return EXIT_FAILED


app = typer.Typer(
    cls=RagbenchGroup,
    help=(
        "RAGBench: evaluation-first RAG benchmark framework.\n\n"
        "Exit codes: 0 success; 1 the command ran and ended badly (a run stopped, no system qualifies, you declined); "
        "2 it could not start because of its input or setup (a flag, file, config, dataset, missing package or API key). "
        "Messages go to stderr, so a command with --json prints only JSON on stdout. "
        "`--debug` (before the command) shows the traceback of a failure."
    ),
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ragbench {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(None, "--version", help="Show version and exit.", callback=_version_callback, is_eager=True),
    debug: bool = typer.Option(False, "--debug", envvar="RAGBENCH_DEBUG", help="Show the full traceback when a command fails (put it before the command: `ragbench --debug run ...`)."),
) -> None:
    STATE.debug = debug
    load_project_env()


for module in (run, auto, dataset, tools, report, doctor):
    app.add_typer(module.commands)
app.add_typer(runs.runs_app, name="runs")
