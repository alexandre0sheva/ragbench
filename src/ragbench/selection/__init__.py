"""Turning a finished run into a decision: which system to deploy under your constraints and priorities.

`Constraints`, `Weights` and `PROFILES` are plain config models; the engine itself is `ragbench.selection.recommend.recommend`.
(It is not imported here because the config schema imports this package.)
"""

from __future__ import annotations

from ragbench.selection.constraints import Constraints
from ragbench.selection.scoring import PROFILES, Profile, Weights

__all__ = ["PROFILES", "Constraints", "Profile", "Weights"]
