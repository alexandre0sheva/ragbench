"""Resuming a run in its own directory: finished systems come back from checkpoints, the rest are run, and nothing finished is paid for twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ragbench.evaluation.checkpoint import CHECKPOINT_DIR, CheckpointStore
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.runtime.progress import ProgressListener
from ragbench.utils.jsonl import read_jsonl, write_jsonl


class Recorder(ProgressListener):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.restored: list[str] = []

    def system_started(self, name: str, index: int, total: int) -> None:
        self.started.append(name)

    def system_restored(self, name: str, num_questions: int) -> None:
        self.restored.append(name)


def _dataset(root: Path) -> dict:
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    write_jsonl(
        root / "questions.jsonl",
        [
            {"id": "q1", "question": "How much does HarborShield cost?", "reference_answer": "$200 per month.", "relevant_doc_ids": ["doc_001"]},
            {"id": "q2", "question": "When does ClaimPilot ship?", "reference_answer": "Q3.", "relevant_doc_ids": ["doc_002"]},
        ],
    )
    return {"documents_path": str(docs), "questions_path": str(root / "questions.jsonl")}


def _config(root: Path, systems: list[dict] | None = None, **evaluation) -> Path:
    config = {
        "run": {"name": "resume", "output_dir": str(root / "results")},
        "dataset": _dataset(root) if not (root / "docs").exists() else {"documents_path": str(root / "docs"), "questions_path": str(root / "questions.jsonl")},
        "systems": systems
        or [
            {"type": "bm25", "name": "bm25", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"top_k": 3}},
            {"type": "vector", "name": "vector", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"top_k": 3}},
            {"type": "hybrid", "name": "hybrid", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"final_top_k": 3}},
        ],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0, **evaluation},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _rows(run: Path) -> dict[str, list[dict]]:
    by_system: dict[str, list[dict]] = {}
    for row in read_jsonl(run / "per_question_results.jsonl"):
        by_system.setdefault(row["system"], []).append(row)
    return by_system


def _first_run(tmp_path: Path, **evaluation) -> tuple[Path, Path, Recorder]:
    config = _config(tmp_path, **evaluation)
    run_dir = tmp_path / "results" / "the_run"
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    return config, run_dir, recorder


def test_a_run_in_its_own_directory_checkpoints_every_clean_system(tmp_path):
    _, run_dir, recorder = _first_run(tmp_path)
    assert recorder.started == ["bm25", "vector", "hybrid"] or set(recorder.started) == {"bm25", "vector", "hybrid"}
    assert recorder.restored == []
    assert len(list((run_dir / CHECKPOINT_DIR).glob("*.json"))) == 3
    assert (run_dir / "recommendation.md").exists() and (run_dir / "config.yaml").exists()


def test_a_normal_run_writes_no_checkpoints(tmp_path):
    out = run_benchmark(_config(tmp_path), force_mock=True)
    assert not (out / CHECKPOINT_DIR).exists()


def test_resuming_after_one_system_is_lost_reruns_only_that_system_and_changes_nothing_else(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    before = _rows(run_dir)
    summary_before = (run_dir / "metrics_summary.csv").read_text()
    CheckpointStore(run_dir).path_for("vector").unlink()

    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)

    assert recorder.started == ["vector"] and sorted(recorder.restored) == ["bm25", "hybrid"]
    after = _rows(run_dir)
    assert list(after) == ["bm25", "vector", "hybrid"]  # config order, restored or not
    for system in ("bm25", "hybrid"):  # restored systems are byte-for-byte what the first attempt produced
        assert after[system] == before[system]
    assert [r["answer"] for r in after["vector"]] == [r["answer"] for r in before["vector"]]  # the re-run is deterministic in mock mode
    assert len(list((run_dir / CHECKPOINT_DIR).glob("*.json"))) == 3
    summary_after = (run_dir / "metrics_summary.csv").read_text()
    assert set(summary_after.splitlines()[0].split(",")) == set(summary_before.splitlines()[0].split(","))
    assert len(summary_after.splitlines()) == len(summary_before.splitlines())


def test_a_complete_run_resumes_to_the_same_result_without_running_anything(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == [] and sorted(recorder.restored) == ["bm25", "hybrid", "vector"]
    assert json.loads((run_dir / "recommendation.json").read_text())["winner"]


def test_a_changed_system_or_setting_is_rerun_not_restored(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["systems"][0]["retrieval"]["top_k"] = 2  # bm25 changed
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == ["bm25"] and sorted(recorder.restored) == ["hybrid", "vector"]

    raw["evaluation"]["context_k"] = 3  # a setting that changes what every system answers
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    again = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=again)
    assert sorted(again.started) == ["bm25", "hybrid", "vector"] and again.restored == []


def test_settings_that_only_change_speed_do_not_invalidate_checkpoints(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder, max_workers=3, system_workers=2)
    assert recorder.started == [] and len(recorder.restored) == 3


def test_a_changed_dataset_invalidates_every_checkpoint(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    write_jsonl(tmp_path / "questions.jsonl", [{"id": "q1", "question": "How much does HarborShield cost?", "relevant_doc_ids": ["doc_001"]}])
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert len(recorder.started) == 3 and recorder.restored == []


def test_a_system_with_failed_questions_is_not_checkpointed_so_resuming_retries_it(tmp_path, monkeypatch):
    from ragbench.rag_systems.bm25_rag import BM25RAG

    original = BM25RAG.answer_question
    state = {"fail": True}

    def flaky(self, question, *args, **kwargs):
        if state["fail"] and "ClaimPilot" in question:
            raise RuntimeError("transient outage")
        return original(self, question, *args, **kwargs)

    monkeypatch.setattr(BM25RAG, "answer_question", flaky)
    config = _config(tmp_path, max_error_rate=1.0)
    run_dir = tmp_path / "results" / "flaky"
    run_benchmark(config, force_mock=True, run_dir=run_dir)
    assert {row["error"] is None for row in _rows(run_dir)["bm25"]} == {True, False}
    assert not CheckpointStore(run_dir).path_for("bm25").exists() and CheckpointStore(run_dir).path_for("vector").exists()

    state["fail"] = False  # the outage is over
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == ["bm25"] and all(row["error"] is None for row in _rows(run_dir)["bm25"])


def test_an_unreadable_checkpoint_is_ignored_and_its_system_runs_again(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    CheckpointStore(run_dir).path_for("hybrid").write_text("{not json", encoding="utf-8")
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == ["hybrid"]


def test_the_spending_cap_counts_what_restored_systems_already_cost(tmp_path, monkeypatch):
    """A cap is a cap on the whole run: resuming with systems restored must not start from zero."""
    from paid_fakes import PRICING
    from paid_fakes import install as install_paid_fakes

    install_paid_fakes(monkeypatch)
    systems = [{"type": "bm25", "name": f"bm25_{n}", "models": {"generator": "fake-paid-model"}, "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}} for n in range(3)]
    config = _config(tmp_path, systems=systems, judge_model="fake-judge")
    raw = yaml.safe_load(config.read_text())
    raw["pricing"], raw["cache"] = PRICING, {"enabled": False}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run_dir = tmp_path / "results" / "paid"
    run_benchmark(config, run_dir=run_dir)
    spent = json.loads((run_dir / "run_summary.json").read_text())["budget"]["spent_usd"]
    assert spent > 0

    CheckpointStore(run_dir).path_for("bm25_2").unlink()
    raw["evaluation"]["max_cost_usd"] = spent * 0.8  # the re-run system alone (a third of `spent`) fits under this; on top of the two restored ones (two thirds) it does not
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from ragbench.evaluation.budget import BudgetExceededError

    with pytest.raises(BudgetExceededError) as stopped:
        run_benchmark(config, run_dir=run_dir)
    assert stopped.value.spent_usd >= spent * 0.6 and "bm25_2" in stopped.value.incomplete
