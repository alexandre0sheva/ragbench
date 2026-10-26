"""Probing a harness: one ordinary RAGBench run of one or more candidate harnesses of a system on one split, read back as scores per question.

A probe run lives in its own directory under the optimization's `probes/` with a `probe.json` stamp (each candidate's harness fingerprint,
the split, the trials, and `finished` once the run succeeded). A later attempt reuses a finished run whose stamp matches, picks up the
checkpoints of an unfinished or failed one, and starts over when the stamp differs: that is how `ragbench optimize` resumes by harness
fingerprint.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.evaluation.budget import BudgetExceededError
from ragbench.evaluation.evaluator import BenchmarkRunError, run_benchmark
from ragbench.harness.apply import harness_fingerprint
from ragbench.runtime.progress import ProgressListener
from ragbench.utils.jsonl import read_jsonl

STAMP = "probe.json"


class ProbeFailed(Exception):
    """A probe run failed (too many of its questions errored, e.g. its indexing hit a rate limit); `spent_usd` is what it charged."""

    def __init__(self, message: str, spent_usd: float):
        super().__init__(message)
        self.spent_usd = spent_usd


class ProbeStopped(Exception):
    """A probe run reached the spending cap it was given; `spent_usd` is what it charged."""

    def __init__(self, spent_usd: float):
        super().__init__(f"the probe stopped at the spending cap after ${spent_usd:.4f}")
        self.spent_usd = spent_usd


@dataclass
class ProbeResult:
    """One candidate's answers in a probe run: the judge's answer score of each answered question, one per trial."""

    label: str
    split: str
    run_dir: Path
    scores: dict[str, list[float]] = field(default_factory=dict)  # question id -> answer score per trial (failed or unscored answers absent)
    arena_score: float | None = None  # set when the run judged pairwise (`--select arena`)

    def trial(self, index: int) -> ProbeResult:
        """The same candidate seen through one trial only (`scores` keep that trial's score)."""
        rows = [row for row in read_jsonl(self.run_dir / "per_question_results.jsonl") if row["system"] == self.label and int(row.get("trial", 0)) == index]
        return ProbeResult(self.label, self.split, self.run_dir, _scores(rows))

    @property
    def mean(self) -> float | None:
        """Mean over questions of each question's mean over trials; None when nothing was scored."""
        means = [sum(values) / len(values) for values in self.scores.values() if values]
        return sum(means) / len(means) if means else None


@dataclass
class ProbeRun:
    results: dict[str, ProbeResult]  # by label
    run_dir: Path
    cost_usd: float  # what the run charged (standalone prices, as the spending cap counts them)
    real_spend_usd: float | None  # after cache savings; None in mock runs
    restored: bool  # finished in an earlier attempt and read back, nothing spent now


def _scores(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for row in rows:
        value = (row.get("answer_judge") or {}).get("answer_score")
        if row.get("error") is None and isinstance(value, int | float):
            scores.setdefault(str(row["question_id"]), []).append(float(value))
    return scores


def probe_config(base_raw: dict[str, Any], systems: dict[str, dict[str, Any]], split: str, *, trials: int, cap_usd: float | None, arena: bool = False) -> dict[str, Any]:
    """The config of a probe run: the user's config with only `systems` (label -> raw system config, named by its label), on `split`,
    `trials` times, capped at `cap_usd`, without the latency probe or contrast stage; with `arena`, pairwise judging against the first."""
    raw = copy.deepcopy(base_raw)
    raw["systems"] = [{**config, "name": label} for label, config in systems.items()]
    raw["dataset"] = {**raw.get("dataset", {}), "split": split}
    evaluation = {key: value for key, value in (raw.get("evaluation") or {}).items() if key not in ("max_cost_usd", "contrast", "pairwise")}
    evaluation.update({"trials": trials, "latency_probe_questions": 0})
    if cap_usd is not None:
        evaluation["max_cost_usd"] = max(cap_usd, 1e-9)
    if arena:
        evaluation["pairwise"] = {"enabled": True, "pairs": "vs_baseline"}
        evaluation["stats"] = {**(evaluation.get("stats") or {}), "baseline": next(iter(systems))}
    raw["evaluation"] = evaluation
    return raw


def _stamp(systems: dict[str, dict[str, Any]], split: str, trials: int, arena: bool) -> dict[str, Any]:
    fingerprints = {label: harness_fingerprint(SystemConfig.model_validate({**config, "name": label})) for label, config in systems.items()}
    return {"fingerprints": fingerprints, "split": split, "trials": trials, "arena": arena}


def _read(run_dir: Path, labels: list[str], split: str) -> dict[str, ProbeResult]:
    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    results = {label: ProbeResult(label, split, run_dir, _scores([row for row in rows if row["system"] == label])) for label in labels}
    summary_path = run_dir / "metrics_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        if "arena_score" in summary.columns:
            for _, row in summary.iterrows():
                if row["system"] in results and pd.notna(row["arena_score"]):
                    results[row["system"]].arena_score = float(row["arena_score"])
    return results


def _spend(run_dir: Path) -> tuple[float, float | None]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    return float((summary.get("budget") or {}).get("spent_usd") or 0.0), (summary.get("cache") or {}).get("real_spend_usd")


def run_probe(
    config_path: Path,
    base_raw: dict[str, Any],
    systems: dict[str, dict[str, Any]],
    split: str,
    run_dir: Path,
    *,
    trials: int = 1,
    cap_usd: float | None = None,
    arena: bool = False,
    force_mock: bool = False,
    max_workers: int | None = None,
    progress: ProgressListener | None = None,
) -> ProbeRun:
    """Run (or read back) one probe. Raises `ProbeStopped` when the run reaches `cap_usd`, `ProbeFailed` when it fails."""
    stamp = _stamp(systems, split, trials, arena)
    stamp_path = run_dir / STAMP
    if run_dir.exists():
        previous = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.is_file() else None
        finished = bool(previous and previous.pop("finished", False))
        if previous == stamp and finished:
            cost, real = _spend(run_dir)
            return ProbeRun(_read(run_dir, list(systems), split), run_dir, cost, real, restored=True)
        if previous != stamp:
            shutil.rmtree(run_dir)  # another harness was probed here: its checkpoints must not be resumed
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    raw = probe_config(base_raw, systems, split, trials=trials, cap_usd=cap_usd, arena=arena)
    ExperimentConfig.model_validate(raw)  # a broken candidate config fails here, before any run starts
    try:
        run_benchmark(config_path, force_mock=force_mock, max_workers=max_workers, progress=progress, raw_config=raw, run_dir=run_dir)
    except BudgetExceededError as exc:
        raise ProbeStopped(exc.spent_usd) from exc
    except BenchmarkRunError as exc:
        spent = _spend(run_dir)[0] if (run_dir / "run_summary.json").is_file() else 0.0
        raise ProbeFailed(str(exc), spent) from exc
    stamp_path.write_text(json.dumps({**stamp, "finished": True}, indent=2), encoding="utf-8")
    cost, real = _spend(run_dir)
    return ProbeRun(_read(run_dir, list(systems), split), run_dir, cost, real, restored=False)
