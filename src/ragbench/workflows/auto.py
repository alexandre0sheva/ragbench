"""`ragbench auto`: the parts of "documents in, decision out" that are not already a command of their own.

Everything with a model or a metric in it (profiling, question generation, the run, the recommendation) is reused from the packages that own it;
this module only decides *where* an auto run lives, remembers how it was started so `--resume RUN_DIR` can pick it up, and assembles the config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ragbench.config.loader import load_config_dict
from ragbench.config.presets import apply_preset
from ragbench.config.sweep import expand_sweeps
from ragbench.selection.scoring import resolve_profile

STATE_NAME = "auto.json"
QUESTIONS_NAME = "questions.jsonl"


class AutoError(ValueError):
    """The auto run cannot start or be resumed; the message says what to change."""


@dataclass
class AutoSettings:
    """How an auto run was started: what `--resume` needs besides the run directory. Paths are absolute so a resume can start from anywhere."""

    docs: str
    questions: str | None = None
    qrels: str | None = None
    preset: str = "standard"
    profile: str = "balanced"
    n_questions: int = 50
    seed: int = 0
    max_cost: float | None = None
    model: str | None = None
    config: str | None = None
    mock: bool = False


def new_run_dir(output_dir: Path) -> Path:
    """`<output_dir>/auto_<timestamp>`, created; a second run in the same second gets a numeric suffix."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, suffix = output_dir / f"auto_{stamp}", 1
    while run_dir.exists():
        suffix += 1
        run_dir = output_dir / f"auto_{stamp}_{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def write_state(run_dir: Path, settings: AutoSettings, generation_cost_usd: float = 0.0) -> None:
    (run_dir / STATE_NAME).write_text(json.dumps({"settings": asdict(settings), "generation_cost_usd": generation_cost_usd}, indent=2), encoding="utf-8")


def read_state(run_dir: Path) -> tuple[AutoSettings, float]:
    path = run_dir / STATE_NAME
    if not path.exists():
        raise AutoError(f"{run_dir} is not an auto run directory (there is no {STATE_NAME}). Only runs started by `ragbench auto` can be resumed.")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return AutoSettings(**state["settings"]), float(state.get("generation_cost_usd", 0.0))
    except (ValueError, KeyError, TypeError) as exc:
        raise AutoError(f"{path} is unreadable ({exc}); start a new auto run.") from exc


def check_choices(preset: str, profile: str) -> None:
    """Fail early, with the usual did-you-mean hint, on a preset or selection profile that does not exist."""
    from ragbench.config.presets import preset_systems

    try:
        preset_systems(preset)
        resolve_profile(profile)
    except ValueError as exc:
        raise AutoError(str(exc)) from exc


def build_config(settings: AutoSettings, run_dir: Path, questions_path: Path, *, max_cost_usd: float | None) -> dict[str, Any]:
    """The experiment config of an auto run: the preset's systems on the dataset, the chosen selection profile, and what is left of the budget.

    `--config`, when given, supplies everything but the systems and the dataset (models, providers, pricing, evaluation, constraints).
    """
    base = load_config_dict(Path(settings.config)) if settings.config else None
    try:
        config = apply_preset(base, settings.preset, docs=settings.docs, questions=questions_path, qrels=settings.qrels)
    except ValueError as exc:
        raise AutoError(str(exc)) from exc
    config["run"] = {"name": run_dir.name, "output_dir": str(run_dir.parent)}
    config["selection"] = {**(config.get("selection") or {}), "profile": settings.profile}
    if max_cost_usd is not None:
        config["evaluation"] = {**(config.get("evaluation") or {}), "max_cost_usd": max_cost_usd}
    return expand_sweeps(config)


def remaining_budget(max_cost: float | None, generation_cost_usd: float) -> float | None:
    """What the run may spend once question generation took its share of the overall `--max-cost`."""
    if max_cost is None:
        return None
    left = max_cost - generation_cost_usd
    if left <= 0:
        raise AutoError(f"Writing the questions already cost ${generation_cost_usd:.4f}, which is all of --max-cost ${max_cost:.2f}. Raise it, or bring your own --questions.")
    return left


def winner_files(run_dir: Path) -> dict[str, Path]:
    """The deliverables of a finished run that exist: `winner.yaml`, `report.html`, `recommendation.md`."""
    names = {"winner": "winner.yaml", "report": "report.html", "recommendation": "recommendation.md"}
    return {key: run_dir / name for key, name in names.items() if (run_dir / name).exists()}
