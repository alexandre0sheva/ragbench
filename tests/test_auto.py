"""`ragbench auto`: documents in, decision out; resumable; and it never spends unasked."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.datasets.loader import load_dataset
from ragbench.evaluation.checkpoint import CHECKPOINT_DIR, CheckpointStore
from ragbench.utils.jsonl import read_jsonl, write_jsonl
from ragbench.workflows import auto as workflow

DOCS = {
    "doc_001.md": "# Pricing\n\nHarborShield costs $200 per month for the marine module. Annual billing gives two months free.",
    "doc_002.md": "# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows. The beta opens to brokers in June.",
    "doc_003.md": "# Support\n\nSupport is available around the clock for every plan. Premium plans get a named engineer.",
    "doc_004.md": "# Security\n\nAll data is encrypted at rest and in transit. Keys rotate every ninety days automatically.",
    "doc_005.md": "# Billing\n\nInvoices are issued monthly and are payable within thirty days. Late fees start at two percent.",
    "doc_006.md": "# Onboarding\n\nNew customers get a guided setup call within three business days of signing the contract.",
}


def _docs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def _auto(*args: str, **env: str):
    return CliRunner().invoke(app, ["auto", *args], env=env or None)


def _run_dir(output: Path) -> Path:
    (found,) = [path for path in output.iterdir() if path.is_dir()]
    return found


@pytest.fixture
def mock_auto(tmp_path):
    """A finished mock auto run on documents with no questions: (docs, output dir, run dir, CLI result)."""
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    result = _auto("--docs", str(docs), "--preset", "quick", "--n-questions", "10", "--mock", "--output-dir", str(output))
    assert result.exit_code == 0, result.output
    return docs, output, _run_dir(output), result


# -- end to end ------------------------------------------------------------------------------------


def test_documents_with_no_questions_become_questions_a_run_and_a_recommendation(mock_auto):
    _, _, run_dir, result = mock_auto
    for name in ("questions.jsonl", "recommendation.md", "recommendation.json", "winner.yaml", "report.html", "leaderboard.md", "config.yaml", "auto.json"):
        assert (run_dir / name).exists(), name
    rows = read_jsonl(run_dir / "questions.jsonl")
    assert len(rows) == 10 and all(row["metadata"]["needs_review"] and row["metadata"]["mock"] for row in rows)
    assert len(list((run_dir / CHECKPOINT_DIR).glob("*.json"))) == 3  # the `quick` preset: bm25, vector, hybrid_rerank
    winner = json.loads((run_dir / "recommendation.json").read_text())["winner"]
    assert winner and f"Deploy {winner}" in result.output
    assert str(run_dir / "winner.yaml") in result.output and str(run_dir / "report.html") in result.output
    assert "needs_review" in result.output  # told to read the questions
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert Path(config["dataset"]["questions_path"]) == run_dir / "questions.jsonl" and Path(config["dataset"]["documents_path"]).is_absolute()


def test_the_winner_yaml_of_an_auto_run_loads_as_a_config(mock_auto):
    from ragbench.config.loader import load_config

    _, _, run_dir, _ = mock_auto
    assert load_config(run_dir / "winner.yaml").systems


def test_your_own_questions_are_used_as_they_are_and_nothing_is_generated(tmp_path):
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    questions = tmp_path / "mine.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"]}, {"id": "q2", "question": "When does ClaimPilot ship?", "relevant_doc_ids": ["doc_002"]}])
    before = questions.read_bytes()
    result = _auto("--docs", str(docs), "--questions", str(questions), "--preset", "quick", "--mock", "--output-dir", str(output))
    assert result.exit_code == 0, result.output
    run_dir = _run_dir(output)
    assert not (run_dir / "questions.jsonl").exists() and questions.read_bytes() == before
    assert Path(yaml.safe_load((run_dir / "config.yaml").read_text())["dataset"]["questions_path"]) == questions.resolve()
    assert len(read_jsonl(run_dir / "per_question_results.jsonl")) == 6  # two questions x three systems


def test_the_profile_chooses_what_the_recommendation_optimizes(tmp_path):
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    result = _auto("--docs", str(docs), "--preset", "quick", "--profile", "max_quality", "--n-questions", "6", "--mock", "--output-dir", str(output))
    assert result.exit_code == 0, result.output
    assert json.loads((_run_dir(output) / "recommendation.json").read_text())["profile"] == "max_quality"


def test_open_shows_the_report_in_the_browser(tmp_path, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url, *a, **k: opened.append(url) or True)
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    assert _auto("--docs", str(docs), "--preset", "quick", "--n-questions", "6", "--mock", "--output-dir", str(output), "--open").exit_code == 0
    assert len(opened) == 1 and opened[0].startswith("file://") and opened[0].endswith("report.html")


# -- resume ----------------------------------------------------------------------------------------


def test_resume_after_one_system_is_lost_reruns_only_that_system(mock_auto):
    _, _, run_dir, _ = mock_auto
    store = CheckpointStore(run_dir)
    names = ["bm25", "vector", "hybrid_rerank"]
    stamps = {name: store.path_for(name).stat().st_mtime_ns for name in names}
    questions_before = (run_dir / "questions.jsonl").read_bytes()
    rows_before = {row["system"]: [] for row in read_jsonl(run_dir / "per_question_results.jsonl")}
    assert set(rows_before) == set(names)
    store.path_for("vector").unlink()

    result = _auto("--resume", str(run_dir), "--mock")
    assert result.exit_code == 0, result.output
    assert "resuming" in result.output and "Using the questions already written" in result.output
    assert store.path_for("vector").exists()  # the lost system ran again and was checkpointed
    for name in ("bm25", "hybrid_rerank"):
        assert store.path_for(name).stat().st_mtime_ns == stamps[name]  # restored, not re-run (a re-run would rewrite its checkpoint)
    assert (run_dir / "questions.jsonl").read_bytes() == questions_before  # no new questions, no new spending
    assert {row["system"] for row in read_jsonl(run_dir / "per_question_results.jsonl")} == set(names)
    assert json.loads((run_dir / "recommendation.json").read_text())["winner"]


def test_resume_of_a_run_that_never_got_to_the_benchmark_builds_it_from_the_saved_settings(mock_auto):
    import shutil

    _, _, run_dir, _ = mock_auto
    for path in run_dir.iterdir():
        if path.name not in {"auto.json", "questions.jsonl"}:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    result = _auto("--resume", str(run_dir), "--mock")
    assert result.exit_code == 0, result.output
    assert (run_dir / "winner.yaml").exists() and (run_dir / "config.yaml").exists()


def test_resume_of_a_run_that_stopped_before_writing_questions_writes_them(tmp_path):
    docs = _docs(tmp_path / "docs")
    run_dir = workflow.new_run_dir(tmp_path / "results")
    workflow.write_state(run_dir, workflow.AutoSettings(docs=str(docs), preset="quick", n_questions=6, mock=True))
    result = _auto("--resume", str(run_dir))
    assert result.exit_code == 0, result.output
    assert len(read_jsonl(run_dir / "questions.jsonl")) == 6 and (run_dir / "recommendation.md").exists()


def test_resume_rejects_settings_that_belong_to_the_original_run_and_directories_it_cannot_resume(tmp_path):
    docs = _docs(tmp_path / "docs")
    other = tmp_path / "not_auto"
    other.mkdir()
    clash = _auto("--resume", str(other), "--docs", str(docs))
    assert clash.exit_code == 2 and "--docs" in clash.output and "resumed" in clash.output
    missing = _auto("--resume", str(other))
    assert missing.exit_code == 2 and "auto.json" in missing.output
    nothing = _auto()
    assert nothing.exit_code == 2 and "--docs" in nothing.output and "--resume" in nothing.output


# -- never spends unasked --------------------------------------------------------------------------


def _expensive_config(tmp_path: Path, docs: Path) -> Path:
    config = {
        "run": {"name": "x", "output_dir": str(tmp_path / "unused")},
        "dataset": {"documents_path": str(docs), "questions_path": str(tmp_path / "unused.jsonl")},
        "systems": [{"type": "bm25", "name": "bm25"}],
        "pricing": {"gpt-6-luna": {"input": 5_000_000.0, "output": 5_000_000.0}},  # dollars per million tokens: every estimate is far above $1
    }
    path = tmp_path / "expensive.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_without_questions_an_expensive_live_run_stops_at_the_question_estimate_when_nobody_can_confirm(tmp_path):
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    result = _auto(
        "--docs", str(docs), "--config", str(_expensive_config(tmp_path, docs)), "--output-dir", str(output), OPENAI_API_KEY="sk-not-a-real-key", CI="1"
    )
    assert result.exit_code == 2 and "confirmation threshold" in result.output and "--yes" in result.output
    run_dir = _run_dir(output)
    assert not (run_dir / "questions.jsonl").exists() and not (run_dir / CHECKPOINT_DIR).exists()


def test_with_questions_an_expensive_live_run_stops_at_the_run_estimate_and_says_how_to_continue(tmp_path):
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    questions = tmp_path / "q.jsonl"
    write_jsonl(questions, [{"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"]}])
    result = _auto(
        "--docs", str(docs), "--questions", str(questions), "--config", str(_expensive_config(tmp_path, docs)), "--output-dir", str(output),
        OPENAI_API_KEY="sk-not-a-real-key", CI="1",
    )  # fmt: skip
    assert result.exit_code == 2 and "confirmation threshold" in result.output and "--yes" in result.output
    assert "ragbench auto --resume" in result.output
    run_dir = _run_dir(output)
    assert not (run_dir / "leaderboard.md").exists() and not (run_dir / CHECKPOINT_DIR).exists()


def test_the_budget_is_shared_between_writing_questions_and_the_run():
    assert workflow.remaining_budget(None, 0.5) is None
    assert workflow.remaining_budget(2.0, 0.5) == pytest.approx(1.5)
    with pytest.raises(workflow.AutoError, match="all of --max-cost"):
        workflow.remaining_budget(0.5, 0.5)


def test_the_config_carries_the_preset_the_profile_and_the_remaining_budget(tmp_path):
    settings = workflow.AutoSettings(docs=str(tmp_path / "docs"), preset="quick", profile="cheapest_acceptable")
    config = workflow.build_config(settings, tmp_path / "results" / "auto_x", tmp_path / "q.jsonl", max_cost_usd=1.25)
    assert [system["type"] for system in config["systems"]] == ["bm25", "vector", "hybrid_rerank"]
    assert config["selection"]["profile"] == "cheapest_acceptable" and config["evaluation"]["max_cost_usd"] == 1.25
    assert config["run"] == {"name": "auto_x", "output_dir": str(tmp_path / "results")}
    assert "evaluation" not in workflow.build_config(settings, tmp_path / "results" / "auto_x", tmp_path / "q.jsonl", max_cost_usd=None)


def test_a_preset_or_profile_that_does_not_exist_fails_before_anything_is_written(tmp_path):
    docs, output = _docs(tmp_path / "docs"), tmp_path / "results"
    bad_preset = _auto("--docs", str(docs), "--preset", "standrad", "--mock", "--output-dir", str(output))
    assert bad_preset.exit_code == 2 and "standard" in bad_preset.output
    bad_profile = _auto("--docs", str(docs), "--profile", "fastest", "--mock", "--output-dir", str(output))
    assert bad_profile.exit_code == 2
    assert not output.exists()


def test_state_round_trips_and_a_corrupt_state_file_is_explained(tmp_path):
    settings = workflow.AutoSettings(docs="/d", questions="/q", preset="agentic", profile="max_quality", n_questions=7, seed=3, max_cost=2.5, model="gpt-6-luna", mock=True)
    workflow.write_state(tmp_path, settings, 0.125)
    assert workflow.read_state(tmp_path) == (settings, 0.125)
    (tmp_path / workflow.STATE_NAME).write_text("{oops", encoding="utf-8")
    with pytest.raises(workflow.AutoError, match="unreadable"):
        workflow.read_state(tmp_path)


def test_the_run_directory_name_never_collides(tmp_path):
    first, second = workflow.new_run_dir(tmp_path), workflow.new_run_dir(tmp_path)
    assert first != second and first.is_dir() and second.is_dir() and os.path.dirname(first) == os.path.dirname(second)


def test_the_questions_an_auto_run_writes_load_as_a_valid_dataset(mock_auto):
    from ragbench.datasets.validation import validate_dataset
    from ragbench.documents.loaders import load_documents

    docs, _, run_dir, _ = mock_auto
    dataset = load_dataset(run_dir / "questions.jsonl")
    assert not any("missing documents" in warning for warning in validate_dataset(load_documents(docs), dataset))
