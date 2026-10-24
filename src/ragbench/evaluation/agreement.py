"""Inter-rater agreement: Krippendorff's alpha, for judge panels and for judge-human audits."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def krippendorff_alpha(units: Sequence[Sequence[float | None]]) -> float | None:
    """Krippendorff's alpha with the interval metric: 1 is perfect agreement, 0 is chance, below 0 is systematic disagreement.

    `units` holds one list per rated item with one value per rater; a rater who did not rate an item gives None or NaN. Items with
    fewer than two values carry no information about agreement and are skipped. None when alpha is undefined: no item has two
    values, or every value is the same (no disagreement is expected, so none can be compared with it).
    """
    pairable = [values for values in ([float(v) for v in unit if v is not None and not math.isnan(float(v))] for unit in units) if len(values) >= 2]
    if not pairable:
        return None
    pooled = np.concatenate([np.asarray(values) for values in pairable])
    n = len(pooled)
    # Sum of squared differences over ordered pairs of distinct positions: 2n·Σv² − 2(Σv)², within each unit and over the pool.
    observed = sum((2 * len(values) * np.sum(np.square(values)) - 2 * np.sum(values) ** 2) / (len(values) - 1) for values in map(np.asarray, pairable)) / n
    expected = (2 * n * np.sum(np.square(pooled)) - 2 * np.sum(pooled) ** 2) / (n * (n - 1))
    if expected <= 1e-12:
        return None
    return float(1 - observed / expected)
