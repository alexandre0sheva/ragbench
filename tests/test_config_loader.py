from pathlib import Path

import pytest

from ragbench.config.loader import load_config, load_config_dict
from ragbench.config.schema import ExperimentConfig


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_parses_minimal_valid_yaml(tmp_path):
    path = _write(
        tmp_path,
        "config.yaml",
        """
run:
  name: test_run
  output_dir: results
dataset:
  documents_path: data/demo/docs
  questions_path: data/demo/questions.jsonl
systems:
  - type: bm25
    name: bm25_default
""",
    )
    config = load_config(path)
    assert isinstance(config, ExperimentConfig)
    assert config.run.name == "test_run"
    assert len(config.systems) == 1
    assert config.systems[0].resolved_name == "bm25_default"
    assert config.evaluation.judge_enabled is True


def test_load_config_resolved_name_falls_back_to_type(tmp_path):
    path = _write(
        tmp_path,
        "config.yaml",
        """
run: {name: r}
dataset:
  documents_path: data/demo/docs
  questions_path: data/demo/questions.jsonl
systems:
  - type: vector
""",
    )
    config = load_config(path)
    assert config.systems[0].resolved_name == "vector"


def test_load_config_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/path/config.yaml"))


def test_load_config_dict_returns_raw_dict(tmp_path):
    path = _write(
        tmp_path,
        "config.yaml",
        """
run: {name: r}
dataset:
  documents_path: a
  questions_path: b
systems: []
custom_key: kept
""",
    )
    raw = load_config_dict(path)
    assert raw["custom_key"] == "kept"
    assert raw["run"]["name"] == "r"


def test_load_config_empty_file_raises_validation(tmp_path):
    from pydantic import ValidationError

    path = _write(tmp_path, "empty.yaml", "")
    with pytest.raises(ValidationError):
        load_config(path)
