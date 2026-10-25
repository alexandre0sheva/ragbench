"""Stratified subsets: a smaller dataset that still has every kind of question in the original's proportions."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


class SubsetError(ValueError):
    """The subset asked for cannot keep every stratum."""


def _quotas(sizes: dict[tuple[str, bool], int], wanted: int) -> dict[tuple[str, bool], int]:
    """How many questions each stratum keeps: one at least, the rest in proportion to its size (largest remainder), never more than it has."""
    total = sum(sizes.values())
    quotas = dict.fromkeys(sizes, 1)
    spare = wanted - len(sizes)
    exact = {key: (wanted - len(sizes)) * size / total for key, size in sizes.items()}
    for key in sizes:
        extra = min(int(exact[key]), sizes[key] - 1)
        quotas[key] += extra
        spare -= extra
    for key in sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), str(k))):
        if spare <= 0:
            break
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            spare -= 1
    for key in sorted(sizes, key=lambda k: (-sizes[k], str(k))):  # whatever the caps left over goes where there is room
        while spare > 0 and quotas[key] < sizes[key]:
            quotas[key] += 1
            spare -= 1
    return quotas


def stratified_subset(rows: Sequence[dict[str, Any]], wanted: int, *, seed: int) -> list[dict[str, Any]]:
    """`wanted` question rows, in their original order, keeping every (category, answerable) stratum and its share of the whole.

    The same rows and seed always give the same subset. Asking for as many rows as there are returns them all."""
    if wanted >= len(rows):
        return list(rows)
    strata: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        strata[(str(row.get("category", "unknown")), row.get("answerable") is not False)].append(position)
    if wanted < len(strata):
        raise SubsetError(f"A subset of {wanted} cannot keep every category and the unanswerable share: this dataset has {len(strata)} strata, so ask for at least {len(strata)}.")
    quotas = _quotas({key: len(members) for key, members in strata.items()}, wanted)
    chosen: list[int] = []
    for key in sorted(strata, key=str):
        chosen.extend(random.Random(f"{seed}:{key}").sample(strata[key], quotas[key]))
    return [rows[position] for position in sorted(chosen)]
