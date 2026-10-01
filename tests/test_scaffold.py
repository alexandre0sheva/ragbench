"""`ragbench init`: a starter project (config + questions template) for your own documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.loader import load_config
from ragbench.config.presets import preset_systems
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.scaffold import ScaffoldError, scaffold_project
from ragbench.utils.jsonl import write_jsonl


def _docs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (root / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    (root / "notes.txt").write_text("Plain notes without any heading about renewals.\n")
    return root


def test_init_writes_a_config_that_loads_and_a_questions_template(tmp_path):
    docs = _docs(tmp_path / "docs")
    result = scaffold_project(tmp_path / "proj", docs)

    config = load_config(result.config_path)
    assert result.config_path == tmp_path / "proj" / "ragbench.yaml"
    assert config.dataset.documents_path == docs and config.dataset.questions_path == tmp_path / "proj" / "questions.jsonl"
    assert [system.type for system in config.systems] == [spec["type"] for spec in preset_systems("standard")]
    assert config.evaluation.judge_enabled and config.evaluation.max_cost_usd is not None
    assert result.n_documents == 3 and result.created_questions

    dataset = load_dataset(result.questions_path)
    assert 1 <= len(dataset.questions) <= 3 and dataset.label_free and all(q.is_answerable for q in dataset.questions)
    assert all(q.metadata.get("template") for q in dataset.questions)
    assert "ragbench init" in result.config_path.read_text()  # says where it came from


def test_init_uses_an_existing_questions_file_without_touching_it(tmp_path):
    docs = _docs(tmp_path / "docs")
    questions = tmp_path / "mine.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "How much?", "relevant_doc_ids": ["doc_001"]}])
    before = questions.read_text()
    result = scaffold_project(tmp_path / "proj", docs, questions)
    assert not result.created_questions and result.questions_path == questions
    assert load_config(result.config_path).dataset.questions_path == questions
    assert questions.read_text() == before and not (tmp_path / "proj" / "questions.jsonl").exists()


def test_init_refuses_to_overwrite_without_force_and_never_replaces_questions(tmp_path):
    docs = _docs(tmp_path / "docs")
    first = scaffold_project(tmp_path / "proj", docs)
    first.questions_path.write_text(json.dumps({"id": "mine", "question": "My own question here?"}) + "\n")
    with pytest.raises(ScaffoldError, match="--force"):
        scaffold_project(tmp_path / "proj", docs)
    first.config_path.write_text("# edited\n")
    scaffold_project(tmp_path / "proj", docs, preset="quick", force=True)
    assert [system["type"] for system in yaml.safe_load(first.config_path.read_text())["systems"]] == [s["type"] for s in preset_systems("quick")]
    assert "My own question" in first.questions_path.read_text()  # --force replaces the config, not your questions


def test_init_reports_bad_input_clearly(tmp_path):
    with pytest.raises(ScaffoldError, match="not found"):
        scaffold_project(tmp_path / "proj", tmp_path / "nowhere")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ScaffoldError, match="No supported documents"):
        scaffold_project(tmp_path / "proj", empty)
    docs = _docs(tmp_path / "docs")
    with pytest.raises(ScaffoldError, match="Questions file not found"):
        scaffold_project(tmp_path / "proj", docs, tmp_path / "missing.jsonl")
    with pytest.raises(ScaffoldError, match="Unknown preset"):
        scaffold_project(tmp_path / "proj", docs, preset="enormous")
    assert not (tmp_path / "proj" / "ragbench.yaml").exists()  # nothing half-written


def test_init_then_run_works_end_to_end_in_mock_mode(tmp_path):
    docs = _docs(tmp_path / "docs")
    runner = CliRunner()
    init = runner.invoke(app, ["init", str(tmp_path / "proj"), "--docs", str(docs), "--preset", "quick"])
    assert init.exit_code == 0, init.output
    assert "ragbench run --config" in init.output  # the next steps
    config = tmp_path / "proj" / "ragbench.yaml"
    run = runner.invoke(app, ["run", "--config", str(config), "--mock"])
    assert run.exit_code == 0, run.output
    assert list((tmp_path / "proj" / "results").glob("*/leaderboard.md"))
    again = runner.invoke(app, ["init", str(tmp_path / "proj"), "--docs", str(docs)])
    assert again.exit_code == 2 and "--force" in again.output
