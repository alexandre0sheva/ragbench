"""Resuming a run in its own directory: finished systems come back from checkpoints, the rest are run, and nothing finished is paid for twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from dataset_support import write_pair_dataset

from ragbench.evaluation.checkpoint import CHECKPOINT_DIR, CheckpointStore
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.runtime.progress import ProgressListener
from ragbench.utils.jsonl import read_jsonl, write_jsonl


class Recorder(ProgressListener):
    def __init__(self) -> None:
        self.started: list[str] = []
        self.restored: list[str] = []
        self.resumed: dict[str, tuple[int, int]] = {}

    def system_started(self, name: str, index: int, total: int) -> None:
        self.started.append(name)

    def system_restored(self, name: str, num_questions: int) -> None:
        self.restored.append(name)

    def system_resumed(self, name: str, answered: int, total: int) -> None:
        self.resumed[name] = (answered, total)


def _config(root: Path, systems: list[dict] | None = None, **evaluation) -> Path:
    config = {
        "run": {"name": "resume", "output_dir": str(root / "results")},
        "dataset": write_pair_dataset(root),
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


def test_a_system_with_failed_questions_keeps_its_good_answers_and_resuming_retries_only_the_failed_ones(tmp_path, monkeypatch):
    from ragbench.rag_systems.bm25_rag import BM25RAG

    original = BM25RAG.answer_question
    state = {"fail": True}
    asked: list[str] = []

    def flaky(self, question, *args, **kwargs):
        asked.append(question)
        if state["fail"] and "ClaimPilot" in question:
            raise RuntimeError("transient outage")
        return original(self, question, *args, **kwargs)

    monkeypatch.setattr(BM25RAG, "answer_question", flaky)
    config = _config(tmp_path, max_error_rate=1.0)
    run_dir = tmp_path / "results" / "flaky"
    run_benchmark(config, force_mock=True, run_dir=run_dir)
    first = _rows(run_dir)["bm25"]
    assert {row["error"] is None for row in first} == {True, False}
    store = CheckpointStore(run_dir)
    assert not store.path_for("bm25").exists() and store.path_for("vector").exists()
    partial = json.loads(store.path_for("bm25", partial=True).read_text())
    assert [row["per_question"]["question_id"] for row in partial["results"]] == ["q1"]  # the good answer only; the failed one is retried

    state["fail"], asked[:] = False, []  # the outage is over
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == ["bm25"] and recorder.resumed == {"bm25": (1, 2)}
    assert asked == ["When does ClaimPilot ship?"]  # only the failed question was asked again
    after = _rows(run_dir)["bm25"]
    assert all(row["error"] is None for row in after) and after[0] == first[0]
    assert store.path_for("bm25").exists() and not store.path_for("bm25", partial=True).exists()


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


def _paid_config(tmp_path: Path, monkeypatch, num_questions: int = 6) -> tuple[Path, dict]:
    """One bm25 system on `num_questions` questions, priced by the paid fakes (offline, but charged like a live run)."""
    from paid_fakes import PRICING
    from paid_fakes import install as install_paid_fakes

    install_paid_fakes(monkeypatch)
    systems = [{"type": "bm25", "name": "bm25", "models": {"generator": "fake-paid-model"}, "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}}]
    config = _config(tmp_path, systems=systems, judge_model="fake-judge")
    questions = [("How much does HarborShield cost?", "$200 per month.", "doc_001"), ("When does ClaimPilot ship?", "Q3.", "doc_002")]
    write_jsonl(
        tmp_path / "questions.jsonl",
        [{"id": f"q{n}", "question": questions[n % 2][0], "reference_answer": questions[n % 2][1], "relevant_doc_ids": [questions[n % 2][2]]} for n in range(num_questions)],
    )
    raw = yaml.safe_load(config.read_text())
    raw["pricing"], raw["cache"] = PRICING, {"enabled": False}
    return config, raw


def _count_questions(monkeypatch) -> list[str]:
    """The ids of the questions the run answers (and pays for) from now on."""
    from ragbench.evaluation import question_runner

    asked: list[str] = []
    original = question_runner.evaluate_question

    def counting(system, system_config, question, *args, **kwargs):
        asked.append(question.id)
        return original(system, system_config, question, *args, **kwargs)

    monkeypatch.setattr(question_runner, "evaluate_question", counting)
    return asked


def test_a_budget_stopped_system_resumes_from_the_questions_it_answered(tmp_path, monkeypatch):
    from ragbench.evaluation.budget import BudgetExceededError

    config, raw = _paid_config(tmp_path, monkeypatch)
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    full = run_benchmark(config)  # the price of the whole run, to set a cap that stops it partway
    rows = _rows(full)["bm25"]
    per_question = [float(row["cost"]["total_cost"]) for row in rows]
    assert len(per_question) == 6 and min(per_question) > 0

    raw["evaluation"]["max_cost_usd"] = sum(per_question) / 2  # about half of the answers
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run_dir = tmp_path / "results" / "capped"
    with pytest.raises(BudgetExceededError) as stopped:
        run_benchmark(config, run_dir=run_dir)
    answered = stopped.value.incomplete["bm25"]["answered"]
    assert 0 < answered < 6
    store = CheckpointStore(run_dir)
    partial = json.loads(store.path_for("bm25", partial=True).read_text())
    kept = [row["per_question"]["question_id"] for row in partial["results"]]
    assert kept == [f"q{n}" for n in range(answered)] and not store.path_for("bm25").exists()
    paid_before = stopped.value.spent_usd

    # Resumed under the same cap: what the earlier attempt charged counts, so nothing new starts and the kept answers stay kept.
    asked = _count_questions(monkeypatch)
    with pytest.raises(BudgetExceededError) as again:
        run_benchmark(config, run_dir=run_dir)
    assert asked == [] and again.value.incomplete["bm25"]["answered"] == answered
    assert again.value.spent_usd == pytest.approx(paid_before)
    assert json.loads(store.path_for("bm25", partial=True).read_text())["results"] == partial["results"]
    assert len(read_jsonl(run_dir / "per_question_partial.jsonl")) == answered

    # With the cap raised, only the missing questions are asked, and the run finishes with the earlier answers in it.
    raw["evaluation"]["max_cost_usd"] = sum(per_question) * 10
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    recorder = Recorder()
    run_benchmark(config, run_dir=run_dir, progress=recorder)
    assert asked == [f"q{n}" for n in range(answered, 6)] and recorder.resumed == {"bm25": (answered, 6)}
    resumed = _rows(run_dir)["bm25"]
    assert [row["question_id"] for row in resumed] == [f"q{n}" for n in range(6)]
    assert [row["answer"] for row in resumed] == [row["answer"] for row in rows]
    assert resumed[:answered] == [row["per_question"] for row in partial["results"]]
    assert store.path_for("bm25").exists() and not store.path_for("bm25", partial=True).exists()
    assert not (run_dir / "per_question_partial.jsonl").exists()
    spent = json.loads((run_dir / "run_summary.json").read_text())["budget"]["spent_usd"]
    assert spent >= paid_before + sum(per_question[answered:]) - 1e-9  # the earlier attempt's answers count towards the run's spend


def test_a_run_with_trials_resumes_from_its_checkpoints_and_a_changed_trial_count_reruns_everything(tmp_path):
    config, run_dir, _ = _first_run(tmp_path, trials=2)
    before = _rows(run_dir)
    assert all(len(rows) == 2 * len({row["question_id"] for row in rows}) for rows in before.values())

    again = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=again)
    assert sorted(again.restored) == ["bm25", "hybrid", "vector"] and again.started == []
    assert _rows(run_dir) == before

    changed = Recorder()
    run_benchmark(_config(tmp_path, trials=3), force_mock=True, run_dir=run_dir, progress=changed)
    assert changed.restored == [] and sorted(changed.started) == ["bm25", "hybrid", "vector"]


def test_a_changed_harness_prompt_invalidates_exactly_that_systems_checkpoint(tmp_path):
    config, run_dir, _ = _first_run(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["systems"][1]["harness"] = {"system_prompt": "You answer questions about marine insurance products."}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    recorder = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=recorder)
    assert recorder.started == ["vector"] and sorted(recorder.restored) == ["bm25", "hybrid"]

    raw["systems"][1]["harness"] = {"system_prompt": "You answer questions about marine insurance products, briefly."}
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    again = Recorder()
    run_benchmark(config, force_mock=True, run_dir=run_dir, progress=again)
    assert again.started == ["vector"]

    assert sorted(again.restored) == ["bm25", "hybrid"]


def test_ragbench_run_resumes_in_the_directory_given_by_run_dir(tmp_path):
    from typer.testing import CliRunner

    from ragbench.cli import app

    config = _config(tmp_path)
    run_dir = tmp_path / "results" / "published"
    first = CliRunner().invoke(app, ["run", "--config", str(config), "--mock", "--run-dir", str(run_dir)])
    assert first.exit_code == 0, first.output
    assert len(list((run_dir / CHECKPOINT_DIR).glob("*.json"))) == 3
    second = CliRunner().invoke(app, ["run", "--config", str(config), "--mock", "--run-dir", str(run_dir)])
    assert second.exit_code == 0, second.output
    assert "restored from the earlier attempt" in second.output
