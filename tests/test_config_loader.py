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


def _judge_config(tmp_path: Path, evaluation: str) -> Path:
    return _write(
        tmp_path,
        "judge.yaml",
        f"""
run: {{name: r}}
dataset:
  documents_path: data/demo/docs
  questions_path: data/demo/questions.jsonl
systems:
  - type: bm25
evaluation:
{evaluation}
""",
    )


def test_judge_section_defaults_keep_the_legacy_judge_model_keys_working(tmp_path):
    legacy = load_config(_judge_config(tmp_path, "  judge_model: gpt-6-luna\n  judge_enabled: true")).evaluation
    assert legacy.judge.model == legacy.judge_model == "gpt-6-luna"
    assert (legacy.judge.samples, legacy.judge.temperature, legacy.judge.independent) == (1, 0.0, True)


def test_judge_section_sets_the_model_and_keeps_judge_model_in_sync(tmp_path):
    evaluation = load_config(_judge_config(tmp_path, "  judge:\n    model: anthropic:claude-haiku-4-5\n    samples: 3\n    temperature: 0.7\n    independent: false")).evaluation
    assert evaluation.judge_model == "anthropic:claude-haiku-4-5" == evaluation.judge.model
    assert (evaluation.judge.samples, evaluation.judge.temperature, evaluation.judge.independent) == (3, 0.7, False)


def test_judge_section_is_checked_when_the_config_loads(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        load_config(_judge_config(tmp_path, "  judge:\n    samples: 3"))  # identical samples at temperature 0 would waste money
    with pytest.raises(ValueError, match="judge_model.*judge.model|judge.model.*judge_model"):
        load_config(_judge_config(tmp_path, "  judge_model: gpt-6-luna\n  judge:\n    model: gpt-6-astra"))
    with pytest.raises(ValueError, match="sampels"):
        load_config(_judge_config(tmp_path, "  judge:\n    sampels: 3"))
    with pytest.raises(ValueError, match="evaluation.judge.model"):
        load_config(_judge_config(tmp_path, "  judge:\n    model: anthropics:claude-haiku-4-5"))
