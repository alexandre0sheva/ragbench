"""`dataset.split`: a run scores one part of `splits.json`, records which, and cannot be resumed against a different one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from dataset_support import write_tiny_dataset
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.datasets.loader import load_questions
from ragbench.datasets.splits import SplitError, Splits, make_splits, write_splits
from ragbench.evaluation.estimate import estimate_run
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.runtime.progress import ProgressListener
from ragbench.utils.jsonl import read_jsonl


class Restored(ProgressListener):
    def __init__(self) -> None:
        self.restored: list[str] = []

    def system_restored(self, name: str, num_questions: int) -> None:
        self.restored.append(name)


def _project(root: Path, split: str = "all", *, with_splits: bool = True) -> tuple[Path, Splits | None]:
    dataset = write_tiny_dataset(root)
    splits = None
    if with_splits:
        splits = make_splits(load_questions(Path(dataset["questions_path"])), train=0.5, dev=0.25, seed=3)
        write_splits(root / "splits.json", splits)
    config = {
        "run": {"name": "split", "output_dir": str(root / "results")},
        "dataset": {**dataset, "split": split},
        "systems": [{"type": "bm25", "name": "bm25", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"top_k": 3}}],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, splits


def _question_ids(run: Path) -> set[str]:
    return {row["question_id"] for row in read_jsonl(run / "per_question_results.jsonl")}


def _summary(run: Path) -> dict:
    return json.loads((run / "run_summary.json").read_text(encoding="utf-8"))


def _manifest(run: Path) -> dict:
    return json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["train", "dev", "test"])
def test_a_run_scores_exactly_the_questions_of_its_split(tmp_path, name):
    config, splits = _project(tmp_path, name)
    assert splits is not None

    run = run_benchmark(config, force_mock=True)

    assert _question_ids(run) == splits.ids(name) and splits.ids(name)
    assert _summary(run)["dataset"]["questions"] == len(splits.ids(name))


def test_the_summary_and_manifest_record_the_split_and_its_hash(tmp_path):
    config, splits = _project(tmp_path, "test")
    assert splits is not None

    run = run_benchmark(config, force_mock=True)

    assert _summary(run)["dataset"]["split"] == {"name": "test", "hash": splits.digest}
    assert _manifest(run)["split"] == {"name": "test", "hash": splits.digest}
    assert "split" in (run / "report.html").read_text(encoding="utf-8").lower()
    assert "Split: test" in (run / "leaderboard.md").read_text(encoding="utf-8")


def test_a_run_over_everything_records_no_split_and_needs_no_splits_file(tmp_path):
    config, _ = _project(tmp_path, "all", with_splits=False)

    run = run_benchmark(config, force_mock=True)

    assert "split" not in _summary(run)["dataset"] and "split" not in _manifest(run)
    assert len(_question_ids(run)) == 8


def test_a_split_without_a_splits_file_says_how_to_make_one(tmp_path):
    config, _ = _project(tmp_path, "test", with_splits=False)

    with pytest.raises(SplitError, match="ragbench split"):
        run_benchmark(config, force_mock=True)


def test_the_splits_file_can_live_elsewhere(tmp_path):
    config, splits = _project(tmp_path, "test")
    assert splits is not None
    moved = tmp_path / "elsewhere" / "my-splits.json"
    moved.parent.mkdir()
    (tmp_path / "splits.json").rename(moved)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["dataset"]["splits_path"] = str(moved)
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert _question_ids(run_benchmark(config, force_mock=True)) == splits.ids("test")


def test_an_unknown_split_name_is_a_config_error(tmp_path):
    config, _ = _project(tmp_path, "validation")
    with pytest.raises(ValidationError, match="split"):
        load_config(config)


def test_resume_does_not_take_checkpoints_made_on_another_split(tmp_path):
    config, _ = _project(tmp_path, "train")
    run_dir = tmp_path / "results" / "the_run"
    run_benchmark(config, force_mock=True, run_dir=run_dir)
    again = Restored()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=again)
    assert again.restored == ["bm25"], "the same split resumes from its checkpoint"

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["dataset"]["split"] = "test"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    other = Restored()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=other)

    assert other.restored == [], "another split is another set of questions: the finished system is run again"


def test_resume_does_not_take_checkpoints_made_against_a_different_splits_file(tmp_path):
    config, splits = _project(tmp_path, "train")
    assert splits is not None
    run_dir = tmp_path / "results" / "the_run"
    run_benchmark(config, force_mock=True, run_dir=run_dir)
    swapped = Splits(seed=splits.seed, train=splits.train, dev=splits.test, test=splits.dev)  # same name, different questions behind it
    (tmp_path / "splits.json").write_text(json.dumps(swapped.model_dump()), encoding="utf-8")
    again = Restored()

    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=again)

    assert again.restored == []


def test_the_estimate_prices_the_questions_of_the_split(tmp_path):
    config, splits = _project(tmp_path, "test")
    assert splits is not None

    assert estimate_run(load_config(config)).n_questions == len(splits.test)


def test_comparing_runs_on_different_splits_says_they_scored_different_questions(tmp_path):
    from ragbench.evaluation.compare_runs import compare_runs

    config, _ = _project(tmp_path, "train")
    train_run = run_benchmark(config, force_mock=True)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["dataset"]["split"] = "test"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    test_run = run_benchmark(config, force_mock=True)

    notes = compare_runs(train_run, test_run).notes

    assert any("different splits (train and test)" in note for note in notes)
