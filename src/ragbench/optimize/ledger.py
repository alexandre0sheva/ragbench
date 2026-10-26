"""The optimization's ledger: every step, what it cost, and Harness²'s budget identity.

Both modes make exactly `4k` train probes (`k` base rollouts, `k` INV and `k` DOM candidates, `k` compositions) and `3k` proposer stages
(`k` INV, `k` DOM, `k` compositions). A proposer stage's one retry is part of its stage, a candidate whose harness was already probed
(an inherited proposal, a duplicate) is counted and read back for free, and dev selection and test verification are outside the identity.
The identity is asserted when the loop ends, unless the spending cap stopped it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragbench.evaluation.budget import BudgetGuard

PROBE_STEPS = ("base", "probe")
PROPOSER_STEPS = ("propose", "compose")


@dataclass
class Entry:
    step: str  # base | propose | probe | compose | select | verify
    label: str
    cost_usd: float = 0.0
    rollouts: int = 1  # probes this entry counts for: the parallel base run is k rollouts in one run (k trials)
    reused: bool = False  # this harness was already probed in this attempt: read back, nothing run
    restored: bool = False  # done in an earlier attempt: read back from disk (its cost still counts against the cap)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        detail = data.pop("detail")
        return {**data, **detail}


class Ledger:
    def __init__(self, k: int, mode: str, cap_usd: float | None):
        self.k = k
        self.mode = mode
        self.budget = BudgetGuard(cap_usd)
        self.entries: list[Entry] = []
        self.real_spend_usd = 0.0
        self.stopped: str | None = None  # why the loop stopped early (the spending cap)

    def add(self, entry: Entry, real_spend_usd: float | None = None) -> Entry:
        self.entries.append(entry)
        self.budget.charge(entry.cost_usd)
        self.real_spend_usd += entry.cost_usd if real_spend_usd is None else real_spend_usd
        return entry

    @property
    def remaining(self) -> float | None:
        cap = self.budget.cap_usd
        return None if cap is None else max(0.0, cap - self.budget.spent)

    @property
    def probes(self) -> int:
        return sum(entry.rollouts for entry in self.entries if entry.step in PROBE_STEPS)

    @property
    def proposer_calls(self) -> int:
        return sum(1 for entry in self.entries if entry.step in PROPOSER_STEPS)

    @property
    def expected(self) -> dict[str, int]:
        return {"probes": 4 * self.k, "proposer_calls": 3 * self.k}

    def assert_identity(self) -> None:
        if self.stopped is None and (self.probes, self.proposer_calls) != (4 * self.k, 3 * self.k):
            raise AssertionError(f"budget identity broken: {self.probes} probes and {self.proposer_calls} proposer stages, expected {4 * self.k} and {3 * self.k}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k": self.k,
            "mode": self.mode,
            "expected": self.expected,
            "probes": self.probes,
            "proposer_calls": self.proposer_calls,
            "max_cost_usd": self.budget.cap_usd,
            "spent_usd": self.budget.spent,
            "real_spend_usd": self.real_spend_usd,
            "stopped": self.stopped,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
