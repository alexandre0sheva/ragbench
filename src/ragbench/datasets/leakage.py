"""The leakage guard: a tripwire between held-out questions and anything that adapts a system.

Whatever an optimizer, proposer or experience library writes (an evidence folder, a prompt, a harness file, a library entry) is checked
here before a model sees it. It must not hold a held-out (by default: test) question's id, its text or its reference answer, because a
method that has read them is no longer measured by them. The check is deliberately blunt: a false alarm costs a rerun, a miss costs the
benchmark's meaning.

What it matches: ids as whole tokens (at least 4 characters, so a short id like `7` cannot be told from the number); question texts and
reference answers of at least 4 words, ignoring case, punctuation, spacing and JSON `\\uXXXX` escapes. A one-word answer cannot be
recognised in running text, so it is not searched for; the question text and id of that question still are.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ragbench.datasets.schema import Dataset, Question
from ragbench.datasets.splits import Splits
from ragbench.errors import UsageError

MIN_ID_CHARS = 4
MIN_TEXT_WORDS = 4
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_NOT_WORD = re.compile(r"[\W_]+")
KIND_LABELS = {"id": "question id", "question": "question text", "reference_answer": "reference answer"}


@dataclass(frozen=True)
class Leak:
    kind: str  # "id", "question" or "reference_answer"
    question_id: str
    source: str  # where it was found: a relative path, or the label the caller gave the text


class LeakageError(UsageError):
    """Held-out content reached something a method may read. The message names where and which question, never the content."""

    def __init__(self, leaks: list[Leak]):
        shown = "; ".join(f"{leak.source}: {KIND_LABELS[leak.kind]} of {leak.question_id}" for leak in leaks[:8])
        more = f" (and {len(leaks) - 8} more)" if len(leaks) > 8 else ""
        super().__init__(f"Held-out content found where a method could read it: {shown}{more}.", hint="Remove it, or rebuild the artifact from the train split only.")
        self.leaks = leaks


def _normalize(text: str) -> str:
    text = _UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)
    return _NOT_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _id_pattern(question_id: str) -> re.Pattern[str]:
    # A whole token: not inside a longer identifier (`q_test_0178`, `xq_test_017`), though `-` ends one (`q_test_017-failed`).
    return re.compile(rf"(?<![A-Za-z0-9_])(?<![A-Za-z0-9]-){re.escape(question_id)}(?![A-Za-z0-9_])(?!-[A-Za-z0-9])")


class LeakageGuard:
    def __init__(self, held_out: Iterable[Question]):
        self._ids: list[tuple[str, re.Pattern[str]]] = []
        self._phrases: list[tuple[str, str, str]] = []  # (kind, question id, normalized text padded with spaces for whole-word matching)
        for question in held_out:
            if len(question.id) >= MIN_ID_CHARS:
                self._ids.append((question.id, _id_pattern(question.id)))
            for kind, text in (("question", question.question), ("reference_answer", question.reference_answer)):
                normalized = _normalize(text or "")
                if len(normalized.split()) >= MIN_TEXT_WORDS:
                    self._phrases.append((kind, question.id, f" {normalized} "))

    @classmethod
    def for_questions(cls, held_out: Iterable[Question]) -> LeakageGuard:
        return cls(held_out)

    @classmethod
    def for_splits(cls, dataset: Dataset, splits: Splits, held_out: tuple[str, ...] = ("test",)) -> LeakageGuard:
        """A guard over the questions of the named splits (default: test) that `dataset` holds."""
        hidden = set().union(*(splits.ids(name) for name in held_out))
        return cls(question for question in dataset.questions if question.id in hidden)

    def scan_text(self, text: str, source: str = "text") -> list[Leak]:
        leaks = [Leak("id", question_id, source) for question_id, pattern in self._ids if pattern.search(text)]
        padded = f" {_normalize(text)} "
        leaks.extend(Leak(kind, question_id, source) for kind, question_id, phrase in self._phrases if phrase in padded)
        return leaks

    def scan_path(self, path: Path) -> list[Leak]:
        """Every leak in a file, or in the text files under a directory (files that are not UTF-8 text are skipped), by path relative to `path`."""
        if path.is_file():
            return self._scan_file(path, path.name)
        leaks: list[Leak] = []
        for file in sorted(item for item in path.rglob("*") if item.is_file()):
            leaks.extend(self._scan_file(file, file.relative_to(path).as_posix()))
        return leaks

    def _scan_file(self, file: Path, source: str) -> list[Leak]:
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []
        return self.scan_text(text, source)

    def assert_text_clean(self, text: str, source: str = "text") -> None:
        leaks = self.scan_text(text, source)
        if leaks:
            raise LeakageError(leaks)

    def assert_path_clean(self, path: Path) -> None:
        leaks = self.scan_path(path)
        if leaks:
            raise LeakageError(leaks)
