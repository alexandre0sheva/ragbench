"""`ragbench datasets list|pull`, driven through the CLI against a local file server."""

from __future__ import annotations

import json

import pytest
from cli_support import invoke
from test_hub_pull import _entry, server  # noqa: F401  (the fixture and the toy entry)

from ragbench.datasets.hub.registry import HUB


@pytest.fixture
def toy(server, tmp_path, monkeypatch):  # noqa: F811
    base, root, hits = server
    entry = _entry(base, root)
    monkeypatch.setitem(HUB, entry.name, entry)
    monkeypatch.setenv("RAGBENCH_DATASETS_DIR", str(tmp_path / "cache"))
    return entry, hits


def test_list_shows_every_dataset_with_its_license_and_download_size():
    result = invoke("datasets", "list")

    assert result.exit_code == 0, result.output
    assert "multihop-rag" in result.output and "hotpotqa" in result.output and "galileo-covidqa" in result.output
    assert "ODC-By" in result.output and "CC-BY-SA" in result.output and "MB" in result.output


def test_list_json_is_machine_readable():
    result = invoke("datasets", "list", "--json")

    rows = {row["name"]: row for row in json.loads(result.stdout)}
    assert rows["multihop-rag"]["license"] == "ODC-By-1.0" and rows["multihop-rag"]["download_bytes"] > 10_000_000
    assert rows["hotpotqa"]["shared_corpus"] is False


def test_pull_with_yes_downloads_converts_and_says_what_to_do_next(toy, tmp_path):
    out = tmp_path / "pulled"

    result = invoke("datasets", "pull", "toy-news", "--output", str(out), "--yes")

    assert result.exit_code == 0, result.output
    assert (out / "questions.jsonl").exists() and (out / "docs").is_dir() and (out / "SOURCE.md").exists()
    assert "6 questions" in result.output and "CC0-1.0" in result.output and "ragbench estimate" in result.output
    assert "mh_0006" in result.output, "the converter's warnings are shown"


def test_pull_without_a_terminal_or_yes_refuses_and_downloads_nothing(toy, tmp_path):
    entry, hits = toy

    result = invoke("datasets", "pull", "toy-news", "--output", str(tmp_path / "out"))

    assert result.exit_code == 1 and "--yes" in result.output and hits == []
    assert not (tmp_path / "out").exists()


def test_pull_asks_and_a_no_downloads_nothing(toy, tmp_path, monkeypatch):
    entry, hits = toy

    monkeypatch.setattr("ragbench.cli.hub.is_interactive", lambda: True)
    monkeypatch.delenv("CI", raising=False)

    declined = invoke("datasets", "pull", "toy-news", "--output", str(tmp_path / "out"), input="n\n")
    assert hits == [] and declined.exit_code == 1 and "Cancelled" in declined.output and "KB" in declined.output

    accepted = invoke("datasets", "pull", "toy-news", "--output", str(tmp_path / "out"), input="y\n")
    assert accepted.exit_code == 0 and sorted(hits) == ["/MultiHopRAG.json", "/corpus.json"]


def test_pull_subset_and_force(toy, tmp_path):
    out = tmp_path / "out"
    assert invoke("datasets", "pull", "toy-news", "--output", str(out), "--yes", "--subset", "4").exit_code == 0
    assert len((out / "questions.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    again = invoke("datasets", "pull", "toy-news", "--output", str(out), "--yes")
    assert again.exit_code == 2 and "--force" in again.output

    assert invoke("datasets", "pull", "toy-news", "--output", str(out), "--yes", "--force").exit_code == 0
    assert len((out / "questions.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_an_impossible_subset_is_a_usage_error(toy, tmp_path):
    result = invoke("datasets", "pull", "toy-news", "--output", str(tmp_path / "out"), "--yes", "--subset", "2")

    assert result.exit_code == 2 and "strata" in result.output


def test_pull_of_an_unknown_dataset_suggests_the_nearest(tmp_path):
    result = invoke("datasets", "pull", "multihop", "--output", str(tmp_path / "out"), "--yes")

    assert result.exit_code == 2 and "multihop-rag" in result.output
