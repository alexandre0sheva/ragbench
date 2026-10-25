"""Train / dev / test splits: a frozen assignment of question ids that tuning features read and reports are scored on.

`ragbench split` writes the assignment once (`splits.json`, never overwritten, because a re-split would let questions a method has
already seen end up in the test set); `dataset.split` then selects one part of it for a run.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from ragbench.datasets.schema import Dataset, Question
from ragbench.errors import UsageError

SPLITS_NAME = "splits.json"
SplitName = Literal["train", "dev", "test", "all"]
FORMAT_VERSION = 1


class SplitError(UsageError, ValueError):
    """A split that cannot be made, read or applied."""


class Splits(BaseModel):
    version: int = FORMAT_VERSION
    seed: int
    train: list[str]
    dev: list[str]
    test: list[str]

    @property
    def digest(self) -> str:
        """Identifies the assignment (not the seed that made it): two files with the same ids in the same parts are the same split."""
        payload = json.dumps({name: sorted(getattr(self, name)) for name in ("train", "dev", "test")}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ids(self, name: str) -> set[str]:
        return set(getattr(self, name))


def _counts(size: int, shares: tuple[float, float, float]) -> list[int]:
    """Split `size` items by `shares` (largest remainder), so the parts add up to `size` exactly."""
    exact = [size * share for share in shares]
    counts = [int(value) for value in exact]
    leftover = size - sum(counts)
    for index in sorted(range(len(shares)), key=lambda i: (counts[i] - exact[i], i))[:leftover]:
        counts[index] += 1
    return counts


def make_splits(questions: list[Question], *, train: float, dev: float, seed: int) -> Splits:
    """Assign every question to train, dev or test (test takes what train and dev leave), stratified by category and answerability.

    Each (category, answerable) group is split on its own, so every part keeps the dataset's mix of question kinds. The result depends on
    the question ids and the seed only, not on the order of the questions."""
    if not (0 < train < 1 and 0 <= dev < 1 and train + dev < 1):
        raise SplitError(f"Cannot split with train={train} and dev={dev}: train must be above 0, dev at least 0 and together they must leave a test share.")
    ids = [question.id for question in questions]
    if len(set(ids)) != len(ids):
        raise SplitError("Cannot split a dataset with duplicate question ids.")
    shares = (train, dev, 1.0 - train - dev)
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for question in questions:
        groups[(question.category, question.is_answerable)].append(question.id)
    parts: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for key in sorted(groups):
        members = sorted(groups[key])
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(members)
        start = 0
        for name, count in zip(parts, _counts(len(members), shares), strict=True):
            parts[name].extend(members[start : start + count])
            start += count
    return Splits(seed=seed, **{name: sorted(members) for name, members in parts.items()})


def write_splits(path: Path, splits: Splits) -> None:
    if path.exists():
        raise SplitError(f"{path} already exists. A split is made once so that no question a method has seen moves into the test set; delete the file to start over.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.model_dump(), indent=2) + "\n", encoding="utf-8")


def load_splits(path: Path) -> Splits:
    if not path.exists():
        raise SplitError(f"Splits file not found: {path}. Create it with `ragbench split` or point `dataset.splits_path` at it.")
    try:
        splits = Splits.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        raise SplitError(f"Cannot read {path}: {exc}") from exc
    if splits.version != FORMAT_VERSION:
        raise SplitError(f"{path} has format version {splits.version}; this RAGBench reads version {FORMAT_VERSION}.")
    return splits


def apply_split(dataset: Dataset, splits: Splits, name: SplitName) -> Dataset:
    """The questions (and their qrels) of one split, in dataset order. `all` returns the dataset itself.

    Refuses a dataset with a question the splits do not list: it would be in no split, so it could not be kept out of the test set."""
    if name == "all":
        return dataset
    known = splits.ids("train") | splits.ids("dev") | splits.ids("test")
    unknown = [question.id for question in dataset.questions if question.id not in known]
    if unknown:
        shown = ", ".join(unknown[:5]) + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else "")
        raise SplitError(f"The questions changed since the split was made: {shown} are not in the splits file. Re-split into a new file, or remove them.")
    keep = splits.ids(name)
    questions = [question for question in dataset.questions if question.id in keep]
    return dataset.model_copy(update={"questions": questions, "qrels": {qid: grades for qid, grades in dataset.qrels.items() if qid in keep}})
