from __future__ import annotations

import json

from fake_systems import question, register_fakes, write_experiment

from ragbench import __version__
from ragbench.evaluation.evaluator import run_benchmark

FAKE = {"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 3}}


def _manifest(out):
    return json.loads((out / "run_manifest.json").read_text())


def test_manifest_records_provenance(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(tmp_path, [question("q1", "What is topic 1?", ["doc_001"])], [FAKE])

    manifest = _manifest(run_benchmark(config, force_mock=True))

    assert manifest["ragbench_version"] == __version__
    assert manifest["mode"] == "mock" and "mock-llm" in manifest["models_used"]
    assert {"python", "platform", "dependencies", "git_sha", "git_dirty", "seed"} <= set(manifest)
    assert "pandas" in manifest["dependencies"]
    assert len(manifest["config_hash"]) == 64 and len(manifest["dataset_hash"]) == 64
    assert manifest["started_utc"].endswith("Z") and manifest["finished_utc"] >= manifest["started_utc"]


def test_hashes_are_stable_for_identical_inputs_and_change_with_the_dataset(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    questions = [question("q1", "What is topic 1?", ["doc_001"])]
    config = write_experiment(tmp_path, questions, [FAKE])

    first = _manifest(run_benchmark(config, force_mock=True))
    second = _manifest(run_benchmark(config, force_mock=True))
    assert first["config_hash"] == second["config_hash"] and first["dataset_hash"] == second["dataset_hash"]

    # Editing a question changes the dataset hash but not the config hash.
    (tmp_path / "questions.jsonl").write_text(json.dumps(question("q1", "What is topic 2?", ["doc_002"])) + "\n", encoding="utf-8")
    third = _manifest(run_benchmark(config, force_mock=True))
    assert third["dataset_hash"] != first["dataset_hash"] and third["config_hash"] == first["config_hash"]

    # Editing a document changes it too.
    (tmp_path / "docs" / "doc_001.md").write_text("# Topic 1\n\nChanged text.\n", encoding="utf-8")
    assert _manifest(run_benchmark(config, force_mock=True))["dataset_hash"] != third["dataset_hash"]


def test_runs_started_in_the_same_second_never_share_an_output_directory(tmp_path, monkeypatch):
    from ragbench.evaluation import evaluator as evaluator_module

    register_fakes(monkeypatch)
    config = write_experiment(tmp_path, [question("q1", "What is topic 1?", ["doc_001"])], [FAKE])

    class FrozenClock:
        @staticmethod
        def now():
            from datetime import datetime

            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(evaluator_module, "datetime", FrozenClock)
    first = run_benchmark(config, force_mock=True)
    second = run_benchmark(config, force_mock=True)
    third = run_benchmark(config, force_mock=True)

    assert len({first, second, third}) == 3
    assert [p.name for p in (first, second, third)] == ["fake_run_20260102_030405", "fake_run_20260102_030405_2", "fake_run_20260102_030405_3"]
    assert json.loads((second / "run_summary.json").read_text())["run_id"] == "fake_run_20260102_030405_2"
    assert (first / "report.html").exists() and (second / "report.html").exists()
