"""The spending cap (`evaluation.max_cost_usd`): count what a run is charged and stop it from starting more work once the cap is reached."""

from __future__ import annotations

import threading
from pathlib import Path


class BudgetGuard:
    """A thread-safe running total of charged cost. With no cap it only counts."""

    def __init__(self, cap_usd: float | None):
        self.cap_usd = cap_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    def charge(self, usd: float) -> None:
        with self._lock:
            self._spent += usd

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def exhausted(self) -> bool:
        """The cap is reached: no new question or system should be started. Work already in flight finishes, so the total can end slightly over."""
        return self.cap_usd is not None and self.spent >= self.cap_usd


class BudgetExceededError(Exception):
    """Raised after a run stopped at its budget and wrote the results of the systems that finished."""

    def __init__(
        self,
        output_dir: Path,
        cap_usd: float,
        spent_usd: float,
        completed: list[str],
        incomplete: dict[str, dict[str, int]],
    ):
        self.output_dir = output_dir
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.completed = completed
        self.incomplete = incomplete
        done = ", ".join(completed) if completed else "none"
        unfinished = ", ".join(
            f"{name} ({counts['answered']}/{counts['total']} questions answered)" if counts["answered"] else f"{name} (not started)"
            for name, counts in incomplete.items()
        )
        super().__init__(
            f"Stopped at the budget: ${spent_usd:.4f} was charged, reaching `evaluation.max_cost_usd` of ${cap_usd:.4f}. "
            f"Finished systems (results written to {output_dir}): {done}. Not finished: {unfinished}. "
            "The answers given by unfinished systems are in per_question_partial.jsonl; they are left out of the leaderboard and the recommendation, "
            "because a mean over some of the questions cannot be compared with a mean over all of them. "
            "Raise `max_cost_usd`, run fewer systems (`--systems`) or questions, or check the cost first with `ragbench estimate`."
        )
