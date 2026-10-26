"""A finished mock run with a train/dev/test split, for the evidence-tree and proposer tests: the tiny dataset with ids long enough for the
leakage guard to match (`item_q1` ... `item_q8`), two systems, scored over every question (`split: all`) so the run holds held-out rows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml
from dataset_support import write_tiny_dataset

from ragbench.datasets.loader import load_questions
from ragbench.datasets.splits import Splits, make_splits, write_splits
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.utils.jsonl import read_jsonl, write_jsonl

SYSTEMS = [
    {"type": "bm25", "name": "bm25", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"top_k": 3}},
    {"type": "hybrid_rerank", "name": "hybrid", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}},
]


def make_run(root: Path) -> tuple[Path, Splits]:
    dataset = write_tiny_dataset(root)
    questions_path = Path(dataset["questions_path"])
    write_jsonl(questions_path, [{**row, "id": f"item_{row['id']}"} for row in read_jsonl(questions_path)])
    splits = make_splits(load_questions(questions_path), train=0.5, dev=0.25, seed=3)
    write_splits(root / "splits.json", splits)
    config = {
        "run": {"name": "evidence", "output_dir": str(root / "results")},
        "dataset": {**dataset, "split": "all"},
        "systems": SYSTEMS,
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return Path(run_benchmark(path, force_mock=True)), splits


def copy_run(run_dir: Path, target: Path) -> Path:
    """A copy of a run that a test may edit (the shared run stays as it was)."""
    shutil.copytree(run_dir, target)
    return target


def edit_rows(run_dir: Path, edit) -> None:
    """Rewrite `per_question_results.jsonl` through `edit(row) -> row`."""
    path = run_dir / "per_question_results.jsonl"
    write_jsonl(path, [edit(row) for row in read_jsonl(path)])


def edit_config(run_dir: Path, edit) -> None:
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(edit(yaml.safe_load(path.read_text(encoding="utf-8")))), encoding="utf-8")


def question_texts(root: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_jsonl(root / "questions.jsonl")}


def read_manifest(evidence: Path) -> dict:
    return json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
