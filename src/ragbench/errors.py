"""Errors the command line turns into a message and an exit code instead of a traceback.

Exit codes: 1 = the command ran and the outcome is a failure (a run stopped, no system qualifies, the user said no), 2 = it could not start
because of its input or setup (a flag, a file, a config, a dataset, a missing optional package or API key). 0 is success.
`ragbench --debug` (or `RAGBENCH_DEBUG=1`) shows the traceback as well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EXIT_FAILED = 1
EXIT_USAGE = 2


class RagbenchError(Exception):
    """Base of every error the command line reports on its own: `str(error)` is the message, `exit_code` the process status."""

    exit_code = EXIT_FAILED

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class UsageError(RagbenchError):
    """A flag, file, config or dataset is wrong. Nothing was run or spent."""

    exit_code = EXIT_USAGE


class RunNotFoundError(UsageError):
    """`RUN` names no run directory."""


class CommandFailed(RagbenchError):
    """The command ran to a negative outcome (the run stopped, nothing qualified); `exit_code` can be set per instance."""

    def __init__(self, message: str, *, exit_code: int = EXIT_FAILED, hint: str | None = None):
        super().__init__(message, hint=hint)
        self.exit_code = exit_code


@dataclass(frozen=True)
class ConfigIssue:
    """One thing wrong in a config: where (`systems[0].retrieval.top_k`), what, and what to do about it when the validator knew."""

    path: str
    message: str
    suggestion: str | None = None


class ConfigError(UsageError):
    """A config (or a section such as `--chunker`) that failed validation, as a list of issues rather than a wall of text."""

    def __init__(self, source: str, issues: list[ConfigIssue]):
        super().__init__(f"Invalid {source}")
        self.source = source
        self.issues = issues


_SUGGESTION = re.compile(r"\s*((?:Did you mean|Install it with)\b.*)$", re.DOTALL)


def config_issues(error: object) -> list[ConfigIssue]:
    """The issues in a pydantic `ValidationError`: the location as a dotted path, the message without pydantic's prefix, and a trailing
    "Did you mean ...?" sentence split off as the suggestion."""
    issues: list[ConfigIssue] = []
    for item in error.errors():  # type: ignore[attr-defined]
        path = ""
        for part in item["loc"]:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        message = str(item["msg"]).removeprefix("Value error, ").strip()
        match = _SUGGESTION.search(message)
        if match and match.start() > 0:
            issues.append(ConfigIssue(path, message[: match.start()].rstrip(), match.group(1)))
        else:
            issues.append(ConfigIssue(path, message))
    return issues
