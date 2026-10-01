"""Shared by the CLI tests: a tiny dataset, mock runs written to chosen directories, and a runner."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.evaluation.evaluator import run_benchmark

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
WORD = {"type": "word", "chunk_size": 60, "chunk_overlap": 0}


def invoke(*args: str, **kwargs):
    return CliRunner().invoke(app, list(args), **kwargs)


def demo_config(root: Path, systems: list[str] | None = None, *, name: str = "cli", **extra) -> Path:
    """A config over the demo dataset (163 questions) that runs in mock mode in about a second per system."""
    names = systems or ["bm25", "vector"]
    config = {
        "run": {"name": name, "output_dir": str(root / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": [{"type": kind, "name": kind, "chunker": WORD} for kind in names],
        "evaluation": {"max_workers": 2, "latency_probe_questions": 0},
        **extra,
    }
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def mock_run(root: Path, run_dir: Path, **kwargs) -> Path:
    """One finished mock run, written exactly to `run_dir`."""
    return run_benchmark(demo_config(root, **kwargs), force_mock=True, max_workers=2, run_dir=run_dir)
