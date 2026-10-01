"""Every command through the real Typer app: it starts, reports problems as messages with the right exit code, and `--json` is clean."""

from __future__ import annotations

import json

import pytest
import typer.main
import yaml
from cli_support import demo_config, invoke, mock_run

from ragbench.cli import app
from ragbench.errors import ConfigError, ConfigIssue, config_issues


def _all_commands() -> list[list[str]]:
    root = typer.main.get_command(app)
    found: list[list[str]] = []

    def walk(command, path: list[str]) -> None:
        for name, child in getattr(command, "commands", {}).items():
            if hasattr(child, "commands"):
                walk(child, [*path, name])
            found.append([*path, name])

    walk(root, [])
    return found


@pytest.mark.parametrize("path", _all_commands(), ids=lambda p: " ".join(p))
def test_every_command_and_subcommand_answers_help(path):
    result = invoke(*path, "--help")
    assert result.exit_code == 0 and "Usage" in result.output


def test_the_commands_the_plan_adds_exist_and_evaluate_is_hidden():
    result = invoke("--help")
    assert result.exit_code == 0
    for name in ("run", "compare", "runs", "report", "compare-runs", "doctor", "completion", "estimate", "recommend", "auto"):
        assert name in result.output
    assert "evaluate" not in result.output  # hidden: it still works, but it is not advertised
    assert "--debug" in result.output


def test_evaluate_still_runs_and_says_it_is_deprecated(tmp_path):
    config = demo_config(tmp_path, ["bm25"])
    result = invoke("evaluate", "--config", str(config), "--mock")
    assert result.exit_code == 0
    assert "deprecated" in result.output and "ragbench run" in result.output and "Evaluation complete" in result.output


def test_run_and_compare_keep_their_messages(tmp_path):
    config = demo_config(tmp_path, ["bm25"])
    assert "Run complete" in invoke("run", "--config", str(config), "--mock").output
    assert "Comparison complete" in invoke("compare", "--config", str(config), "--mock").output


# -- --json ------------------------------------------------------------------------------------------------------------------------------------


def test_run_json_prints_one_document_on_stdout_and_the_rest_on_stderr(tmp_path):
    config = demo_config(tmp_path, ["bm25", "vector"])
    result = invoke("run", "--config", str(config), "--mock", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)  # nothing but the JSON on stdout
    assert payload["mode"] == "mock" and payload["status"] == "complete" and {s["system"] for s in payload["systems"]} == {"bm25", "vector"}
    assert payload["winner"] in {"bm25", "vector"} and payload["recommendation"]["winner"] == payload["winner"]
    assert payload["files"]["report"].endswith("report.html")
    assert "Leaderboard" in result.stderr and "Leaderboard" not in result.stdout


def test_recommend_json_and_exit_status(tmp_path):
    run = mock_run(tmp_path, tmp_path / "results" / "r1")
    result = invoke("recommend", "--run", str(run), "--json")
    assert result.exit_code == 0 and json.loads(result.stdout)["winner"]
    impossible = invoke("recommend", "--run", str(run), "--json", "--max-cost", "0", "--min-answer-score", "5")
    assert impossible.exit_code == 1 and json.loads(impossible.stdout)["winner"] is None


# -- --only / --skip ---------------------------------------------------------------------------------------------------------------------------


def test_only_and_skip_pick_the_systems_to_run(tmp_path):
    config = demo_config(tmp_path, ["bm25", "vector", "hybrid"])
    only = json.loads(invoke("run", "--config", str(config), "--mock", "--json", "--only", "bm25,vector").stdout)
    assert {s["system"] for s in only["systems"]} == {"bm25", "vector"}
    skipped = json.loads(invoke("run", "--config", str(config), "--mock", "--json", "--skip", "bm25", "--skip", "hybrid").stdout)
    assert {s["system"] for s in skipped["systems"]} == {"vector"}
    old_flag = json.loads(invoke("run", "--config", str(config), "--mock", "--json", "--systems", "hybrid").stdout)
    assert [s["system"] for s in old_flag["systems"]] == ["hybrid"]  # --systems keeps working as the old name of --only


def test_unknown_names_and_an_empty_selection_are_usage_errors(tmp_path):
    config = demo_config(tmp_path, ["bm25", "vector"])
    typo = invoke("run", "--config", str(config), "--mock", "--only", "bm52")
    assert typo.exit_code == 2 and "--only: no system named 'bm52'" in typo.output and "Did you mean 'bm25'" in typo.output
    nothing = invoke("run", "--config", str(config), "--mock", "--skip", "bm25,vector")
    assert nothing.exit_code == 2 and "no system to run" in nothing.output
    assert invoke("estimate", "--config", str(config), "--skip", "nope").exit_code == 2


# -- errors are messages, not tracebacks -------------------------------------------------------------------------------------------------------


def test_a_bad_config_exits_2_with_each_problem_located_and_a_suggestion(tmp_path, tiny_dataset):
    config = tmp_path / "bad.yaml"
    config.write_text(
        yaml.safe_dump({"run": {"name": "x"}, "dataset": tiny_dataset, "systems": [{"type": "bm25", "retrieval": {"top_kk": 3}}], "cache": {"enabld": True}}),
        encoding="utf-8",
    )
    result = invoke("run", "--config", str(config), "--mock")
    assert result.exit_code == 2
    assert f"Invalid config {config}" in result.output
    assert "systems[0]" in result.output and "top_kk" in result.output  # where, and what
    assert "→" in result.output and "Did you mean 'top_k'" in result.output  # what to do
    assert "cache.enabld" in result.output  # every problem, not just the first
    assert "Traceback" not in result.output


def test_a_missing_file_is_a_one_line_message(tmp_path):
    result = invoke("run", "--config", str(tmp_path / "missing.yaml"), "--mock")
    assert result.exit_code == 2 and "Config file not found" in result.output and "Traceback" not in result.output


def test_config_issues_split_the_suggestion_from_the_message():
    class Fake:
        def errors(self):
            return [
                {"loc": ("systems", 0, "retrieval", "candidate_top"), "msg": "Value error, unknown option. Did you mean 'candidate_top_k'?"},
                {"loc": (), "msg": "plain problem"},
            ]

    issues = config_issues(Fake())
    assert issues[0] == ConfigIssue("systems[0].retrieval.candidate_top", "unknown option.", "Did you mean 'candidate_top_k'?")
    assert issues[1] == ConfigIssue("", "plain problem", None)
    assert str(ConfigError("config x.yaml", issues)) == "Invalid config x.yaml" and ConfigError("c", issues).exit_code == 2


def test_an_unexpected_error_is_a_message_and_debug_shows_the_traceback(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("ragbench.cli.runs.resolve_run", boom)
    quiet = invoke("runs", "show", "x", "--results-dir", str(tmp_path))
    assert quiet.exit_code == 1 and "Unexpected RuntimeError: kaboom" in quiet.output and "--debug" in quiet.output
    assert quiet.exception is None or isinstance(quiet.exception, SystemExit)
    loud = invoke("--debug", "runs", "show", "x", "--results-dir", str(tmp_path))
    assert isinstance(loud.exception, RuntimeError)  # the original exception, for a traceback
    assert invoke("runs", "--results-dir", str(tmp_path)).exit_code == 0  # and the switch does not stick to the next command


def test_chunker_errors_name_the_option(tmp_path):
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.md").write_text("hello world")
    result = invoke("chunk-preview", "--docs", str(docs), "--chunker", "{type: markdown, chunk_sise: 3}")
    assert result.exit_code == 2 and "Invalid --chunker" in result.output and "chunk_sise" in result.output
