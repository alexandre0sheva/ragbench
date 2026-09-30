"""What every loader shares: the loading context, text decoding, and the file filters (`.ragbenchignore`, include / exclude)."""

from __future__ import annotations

import codecs
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ragbench.config.schema import TabularConfig


class DocumentLoadError(Exception):
    """A document could not be loaded (corrupt file, too large, bad option). The message names the file."""


@dataclass
class LoadContext:
    root: Path  # the documents folder: doc ids and ignore rules are relative to it
    tabular: TabularConfig
    warn: Callable[[str], None]


# --- decoding -----------------------------------------------------------------------------------


def decode_bytes(data: bytes, name: str, warn: Callable[[str], None]) -> str:
    """Text from `data`: BOMs (UTF-8/16/32) are honoured, UTF-8 is tried first, then Windows-1252 (which covers legacy
    Western-European files), then UTF-8 with replacement characters. Anything but clean UTF-8 is reported through `warn`.
    Newlines are normalised to `\\n` like Python's text mode did.
    """
    if data.startswith(codecs.BOM_UTF8):
        text = data[len(codecs.BOM_UTF8) :].decode("utf-8", errors="replace")
    elif data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        text = data.decode("utf-32", errors="replace")
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16", errors="replace")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("cp1252")
                warn(f"{name}: not valid UTF-8; decoded as cp1252 (Windows Western European). Re-save it as UTF-8 if characters look wrong.")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
                warn(f"{name}: contained bytes that are not valid text in any supported encoding; undecodable bytes were replaced with U+FFFD.")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# --- gitignore-style patterns -------------------------------------------------------------------


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif char == "*":
            out.append("[^/]*")
            i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "[" and (end := pattern.find("]", i + 2)) > 0:
            out.append(pattern[i : end + 1])
            i = end + 1
        else:
            out.append(re.escape(char))
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class Pattern:
    regex: re.Pattern[str]
    negate: bool
    directory_only: bool

    @classmethod
    def parse(cls, line: str) -> Pattern | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        negate = line.startswith("!")
        line = line.removeprefix("!") if negate else line
        directory_only = line.endswith("/")
        line = line.rstrip("/")
        anchored = "/" in line  # a slash at the start or middle ties the pattern to the root, like gitignore
        body = _glob_to_regex(line.lstrip("/"))
        return cls(re.compile(f"^{body}$" if anchored else f"^(?:.*/)?{body}$"), negate, directory_only)

    def matches(self, relative_path: str, is_directory: bool) -> bool:
        return (is_directory or not self.directory_only) and self.regex.match(relative_path) is not None


class IgnoreRules:
    """`.ragbenchignore`: gitignore-style patterns (`#` comments, `!` negation, `dir/`, leading `/` anchors, `*`, `**`, `?`)."""

    def __init__(self, lines: Iterable[str] = ()):
        self.patterns = [pattern for line in lines if (pattern := Pattern.parse(line)) is not None]

    def _verdict(self, relative_path: str, is_directory: bool) -> bool | None:
        verdict: bool | None = None
        for pattern in self.patterns:  # the last matching pattern wins
            if pattern.matches(relative_path, is_directory):
                verdict = not pattern.negate
        return verdict

    def ignores(self, relative_path: str) -> bool:
        """Whether the file is ignored. Like git, a file cannot be re-included once a parent directory is excluded."""
        parts = relative_path.split("/")
        for depth in range(1, len(parts)):
            if self._verdict("/".join(parts[:depth]), True):
                return True
        return bool(self._verdict(relative_path, False))


def matches_any(relative_path: str, patterns: list[str]) -> bool:
    """Whether the path matches any `include` / `exclude` glob (same syntax as ignore patterns, no negation)."""
    parsed = [Pattern.parse(pattern) for pattern in patterns]
    parts = relative_path.split("/")
    candidates = ["/".join(parts[:depth]) for depth in range(1, len(parts) + 1)]  # a glob naming a directory covers its contents
    return any(p is not None and any(p.matches(c, c != relative_path) for c in candidates) for p in parsed)
