"""The experience library: what an optimization learned on one document collection, kept for later work on similar ones.

An entry is written only when `ragbench optimize` *adopted* a harness (the held-out test split confirmed the gain), and holds that harness,
the verified gain with its confidence interval and one lesson. Entries are keyed by a **corpus fingerprint** computed from the dataset
profile (`datasets/profile.py`): the language, the size of the corpus and of its documents, how long the visible questions are, and whether
they carry relevance labels. The key is coarse (a corpus a few percent larger has the same key); `similarity` compares the features
themselves, so a later run finds the *nearest* entries rather than only identical collections.

The file is append-only (`entries.jsonl`, one JSON object per line, in the order written), and a run reads only entries older than its
own start: what it learns itself, or what a parallel run wrote meanwhile, never reaches it. Mock runs and live runs do not see each
other's entries. Nothing from the dev or test questions is written: the features describe the train questions, and an entry is passed
through the leakage guard before it is stored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ragbench.errors import UsageError
from ragbench.utils.text import truncate

if TYPE_CHECKING:
    from ragbench.datasets.leakage import LeakageGuard
    from ragbench.datasets.profile import DatasetProfile

ENV_VAR = "RAGBENCH_LIBRARY"
DEFAULT_PATH = Path("~/.ragbench/library")
FILE_NAME = "entries.jsonl"
VERSION = 1
MIN_SIMILARITY = 0.75  # nearer than this and an entry is read; below it the collections are too different for a lesson to carry over
MAX_MATCHES = 3
VALUE_CHARS = 1500  # a harness value longer than this is cut when an entry is shown to a proposer

# How much each feature counts towards `similarity`.
WEIGHTS = {"language": 2.0, "tokens": 2.0, "documents": 1.0, "doc_tokens_p50": 1.0, "question_words_p50": 1.0, "label_free": 1.0, "answerable_ratio": 0.5, "categories": 1.0}
DECADES_APART = 2.0  # sizes this many powers of ten apart count as unrelated


class LibraryError(UsageError):
    """The library file cannot be read; the message names the line."""


# -- the fingerprint ------------------------------------------------------------------------------------------------------------------------


def features_of(profile: DatasetProfile) -> dict[str, Any]:
    """What `similarity` compares: the corpus and the (visible) questions as the dataset profile describes them."""
    return {
        "language": profile.documents.language,
        "documents": profile.documents.count,
        "tokens": profile.documents.total_tokens,
        "doc_tokens_p50": round(profile.documents.tokens.p50, 1),
        "question_words_p50": round(profile.questions.words.p50, 1),
        "label_free": profile.qrels.label_free,
        "answerable_ratio": round(profile.questions.answerable_ratio, 2),
        "categories": sorted(profile.questions.categories),
    }


def _octave(value: float) -> int:
    return round(math.log2(max(float(value), 1.0)))


def fingerprint_of(features: dict[str, Any]) -> str:
    """The key of a collection: a hash of its language, label-freeness and the power-of-two bucket of each size, so near-identical shapes share it."""
    coarse = {
        "language": features["language"],
        "label_free": bool(features["label_free"]),
        "documents": _octave(features["documents"]),
        "tokens": _octave(features["tokens"]),
        "doc_tokens_p50": _octave(features["doc_tokens_p50"]),
        "question_words_p50": _octave(features["question_words_p50"]),
    }
    return hashlib.sha256(json.dumps(coarse, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _scale(a: float, b: float) -> float:
    """1 for equal sizes, falling to 0 at `DECADES_APART` powers of ten."""
    return max(0.0, 1.0 - abs(math.log10(max(float(a), 1.0) / max(float(b), 1.0))) / DECADES_APART)


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """How alike two collections are, 0 to 1: a weighted mean of per-feature similarities (1.0 only for identical features)."""
    language = 1.0 if a["language"] == b["language"] else 0.5 if "unknown" in (a["language"], b["language"]) else 0.0
    union = set(a["categories"]) | set(b["categories"])
    parts = {
        "language": language,
        "tokens": _scale(a["tokens"], b["tokens"]),
        "documents": _scale(a["documents"], b["documents"]),
        "doc_tokens_p50": _scale(a["doc_tokens_p50"], b["doc_tokens_p50"]),
        "question_words_p50": _scale(a["question_words_p50"], b["question_words_p50"]),
        "label_free": 1.0 if bool(a["label_free"]) == bool(b["label_free"]) else 0.0,
        "answerable_ratio": 1.0 - abs(float(a["answerable_ratio"]) - float(b["answerable_ratio"])),
        "categories": len(set(a["categories"]) & set(b["categories"])) / len(union) if union else 1.0,
    }
    return sum(WEIGHTS[name] * value for name, value in parts.items()) / sum(WEIGHTS.values())


# -- entries --------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Entry:
    fingerprint: str
    features: dict[str, Any]
    system: str  # the system's name in the config it was optimized in
    system_type: str
    harness: dict[str, Any]  # the adopted `harness:` patch
    gain: dict[str, Any]  # the verification on the test split: base_mean, pick_mean, mean_diff, ci_lo, ci_hi, p_holm, n, trials
    lesson: str
    changed: dict[str, dict[str, Any]] = field(default_factory=dict)  # field -> {"from", "to"} (effective values)
    mock: bool = False
    rationale: str = ""  # the proposer's reason for the adopted candidate, from train evidence
    run: str = ""  # the optimization folder
    id: str = ""
    created_at: str = ""
    version: int = VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class Match:
    entry: Entry
    similarity: float


def _describe(name: str, before: Any, after: Any) -> str:
    new = repr(after) if len(repr(after)) <= 120 else f"a rewritten text of {len(str(after))} characters"
    old = f" (was {before!r})" if before is not None and len(repr(before)) <= 40 else ""
    return f"`{name}` to {new}{old}"


def build_entry(
    *,
    features: dict[str, Any],
    system: str,
    system_type: str,
    base_harness: dict[str, Any],
    adopted_harness: dict[str, Any],
    patch: dict[str, Any],
    rationale: str,
    verification: dict[str, Any],
    mock: bool,
    run: str,
) -> Entry:
    """The entry for an adopted harness. The lesson is stated from what was measured (which fields changed, the gain and its interval on held-out
    questions) and the proposer's own reason for the change, which it gave after reading train evidence."""
    changed = {name: {"from": base_harness.get(name), "to": adopted_harness.get(name)} for name in adopted_harness if adopted_harness.get(name) != base_harness.get(name)}
    what = ", ".join(_describe(name, change["from"], change["to"]) for name, change in changed.items()) or "the harness"
    lesson = (
        f"On a collection like this, a `{system_type}` system improved by setting {what}: "
        f"the average answer rating rose by {verification['mean_diff']:+.2f} on held-out questions (95% CI {verification['ci_lo']:+.2f} to {verification['ci_hi']:+.2f})."
    )
    if rationale:
        lesson += f" Why, from the train evidence: {rationale.strip()}"
    gain: dict[str, Any] = {key: verification[key] for key in ("base_mean", "pick_mean", "mean_diff", "ci_lo", "ci_hi", "p", "p_holm", "n", "trials") if key in verification}
    return Entry(fingerprint_of(features), features, system, system_type, dict(patch), gain, lesson, changed, mock, rationale, run)


def _describe_features(features: dict[str, Any]) -> str:
    labels = "no relevance labels" if features["label_free"] else "relevance labels"
    return f"{features['language']} documents ({features['documents']:,} documents, ~{features['tokens']:,} tokens), questions of ~{features['question_words_p50']:g} words, {labels}"


def lesson_markdown(entry: Entry, score: float) -> str:
    """One entry as a note for the proposer (`library/<id>.md` of an evidence tree)."""
    shown = {name: (truncate(value, VALUE_CHARS) if isinstance(value, str) else value) for name, value in entry.harness.items()}
    return "\n".join(
        [
            f"# Lesson from earlier work ({entry.id})",
            "",
            f"A similar collection (similarity {score:.2f}: {_describe_features(entry.features)}) was optimized on {entry.created_at[:10]}.",
            "",
            entry.lesson,
            "",
            f"Harness that was adopted (JSON): {json.dumps(shown, ensure_ascii=False, default=str)}",
            "",
            "This is a hint from another collection, not a rule: check it against the rollouts before copying it.",
        ]
    )


# -- the store ------------------------------------------------------------------------------------------------------------------------------


def _moment(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


class Library:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()

    @property
    def file(self) -> Path:
        return self.path / FILE_NAME

    def _read(self) -> Iterator[Entry]:
        if not self.file.is_file():
            return
        for number, line in enumerate(self.file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield Entry.from_dict(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise LibraryError(f"{self.file} line {number} is not a library entry ({exc}). Fix or remove the line, or start over with `ragbench library clear`.") from exc

    def entries(self, before: datetime | None = None, mock: bool | None = None) -> list[Entry]:
        """The entries in the order they were written. `before` keeps those created strictly earlier; `mock` keeps mock (True) or live (False) ones."""
        found = [entry for entry in self._read() if mock is None or entry.mock == mock]
        if before is not None:
            moment = before if before.tzinfo is not None else before.replace(tzinfo=UTC)
            found = [entry for entry in found if _moment(entry.created_at) < moment]
        return sorted(found, key=lambda entry: _moment(entry.created_at))

    def append(self, entry: Entry, guard: LeakageGuard | None = None) -> Entry:
        """Store `entry` (stamped with the time and an id unless it has them) at the end of the file; with a `guard`, an entry quoting held-out text
        raises `LeakageError` and nothing is written."""
        created = entry.created_at or datetime.now(UTC).isoformat(timespec="microseconds")
        stored = replace(entry, created_at=created, id=entry.id or hashlib.sha1(f"{created}{entry.fingerprint}{entry.lesson}".encode()).hexdigest()[:8])
        text = json.dumps(stored.to_dict(), ensure_ascii=False, default=str)
        if guard is not None:
            guard.assert_text_clean(text, "library entry")
        self.path.mkdir(parents=True, exist_ok=True)
        with self.file.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        return stored

    def nearest(self, features: dict[str, Any], before: datetime | None = None, mock: bool | None = None, *, limit: int = MAX_MATCHES, min_similarity: float = MIN_SIMILARITY) -> list[Match]:
        """The entries most like `features`, most similar first (a tie goes to the newer one), at least `min_similarity` alike."""
        matches = [Match(entry, similarity(features, entry.features)) for entry in self.entries(before, mock)]
        matches = [match for match in matches if match.similarity >= min_similarity]
        matches.sort(key=lambda match: (-match.similarity, -_moment(match.entry.created_at).timestamp()))
        return matches[:limit]

    def find(self, prefix: str) -> Entry:
        found = [entry for entry in self._read() if entry.id.startswith(prefix)]
        if not found:
            raise LibraryError(f"No library entry with id '{prefix}'. `ragbench library list` shows them.")
        if len(found) > 1:
            raise LibraryError(f"'{prefix}' matches {len(found)} entries ({', '.join(entry.id for entry in found)}): give more of the id.")
        return found[0]

    def clear(self, fingerprint: str | None = None) -> int:
        """Remove the entries of one collection (`fingerprint`, or the start of one), or all of them; returns how many."""
        everything = list(self._read())
        keep = [entry for entry in everything if fingerprint is not None and not entry.fingerprint.startswith(fingerprint)]
        if len(keep) == len(everything):
            return 0
        if keep:
            self.file.write_text("".join(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n" for entry in keep), encoding="utf-8")
        else:
            self.file.unlink()
        return len(everything) - len(keep)


def resolve_library(configured: Path | str | None = None) -> tuple[Library, bool]:
    """The library to use and whether it was named: `library.path` of the config, else `RAGBENCH_LIBRARY`, else `~/.ragbench/library`.
    A run that is only a rehearsal (mock) writes just to a named one, never to the default in the home directory."""
    if configured:
        return Library(configured), True
    named = os.environ.get(ENV_VAR)
    if named:
        return Library(named), True
    return Library(DEFAULT_PATH), False
