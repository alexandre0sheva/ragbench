"""`ragbench split`: write splits.json for a dataset once, stratified, and say what it made."""

from __future__ import annotations

import json
from pathlib import Path

from cli_support import DEMO, invoke

from ragbench.datasets.loader import load_questions
from ragbench.datasets.splits import load_splits
from ragbench.utils.jsonl import write_jsonl


def _copy_questions(root: Path) -> Path:
    path = root / "ds" / "questions.jsonl"
    write_jsonl(path, [question.model_dump(exclude_defaults=True) for question in load_questions(DEMO / "questions.jsonl")])
    return path


def test_split_writes_splits_json_next_to_the_questions(tmp_path):
    questions = _copy_questions(tmp_path)

    result = invoke("split", str(questions), "--train", "0.5", "--dev", "0.1", "--seed", "42")

    assert result.exit_code == 0, result.output
    splits = load_splits(questions.parent / "splits.json")
    assert (splits.seed, len(splits.train) + len(splits.dev) + len(splits.test)) == (42, 163)
    assert abs(len(splits.train) - 163 * 0.5) <= 4 and abs(len(splits.dev) - 163 * 0.1) <= 4
    assert "Wrote" in result.output and all(name in result.output for name in ("train", "dev", "test"))


def test_split_takes_the_dataset_directory_too(tmp_path):
    questions = _copy_questions(tmp_path)

    assert invoke("split", str(questions.parent)).exit_code == 0
    assert (questions.parent / "splits.json").exists()


def test_split_refuses_to_overwrite_and_leaves_the_file_alone(tmp_path):
    questions = _copy_questions(tmp_path)
    assert invoke("split", str(questions), "--seed", "1").exit_code == 0
    before = (questions.parent / "splits.json").read_text(encoding="utf-8")

    again = invoke("split", str(questions), "--seed", "2")

    assert again.exit_code == 2
    assert "already exists" in again.output
    assert (questions.parent / "splits.json").read_text(encoding="utf-8") == before


def test_split_out_writes_elsewhere_and_bad_fractions_write_nothing(tmp_path):
    questions = _copy_questions(tmp_path)
    out = tmp_path / "elsewhere" / "s.json"

    assert invoke("split", str(questions), "--out", str(out)).exit_code == 0
    assert out.exists() and not (questions.parent / "splits.json").exists()

    bad = invoke("split", str(questions), "--train", "0.95", "--dev", "0.2")
    assert bad.exit_code == 2 and "test share" in bad.output
    assert not (questions.parent / "splits.json").exists()


def test_split_of_a_missing_dataset_is_a_usage_error(tmp_path):
    result = invoke("split", str(tmp_path / "nope.jsonl"))
    assert result.exit_code == 2 and "not found" in result.output.lower()


def test_split_json_prints_only_json(tmp_path):
    questions = _copy_questions(tmp_path)

    result = invoke("split", str(questions), "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload) >= {"path", "seed", "counts", "hash"}
