"""`ragbench optimize`: Propose → Probe → Compose over a system's harness, picked on dev, verified on test, with an exact ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from optimize_support import ANSWER_COST, edit_config, register_tunable, scripted_reply, write_project
from scripted_llm import ScriptedLLM

from ragbench.datasets.leakage import LeakageGuard
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.splits import load_splits
from ragbench.optimize.loop import OptimizeSettings, optimize
from ragbench.optimize.proposer import Proposer


@pytest.fixture(autouse=True)
def _tunable(monkeypatch):
    register_tunable(monkeypatch)


def proposer(edits: dict[str, str]) -> tuple[Proposer, ScriptedLLM]:
    llm = ScriptedLLM(lambda prompt: scripted_reply(prompt, edits))
    return Proposer(llm), llm


def run(root: Path, edits: dict[str, str], **settings) -> tuple:
    config = root / "config.yaml" if (root / "config.yaml").exists() else write_project(root)
    chosen, llm = proposer(edits)
    result = optimize(config, OptimizeSettings(system="tuned", **settings), proposer=chosen, out_dir=root / "opt", force_mock=True)
    return result, llm


def ledger(result) -> dict:
    return json.loads((result.out_dir / "ledger.json").read_text(encoding="utf-8"))


# -- the loop and its budget identity ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
@pytest.mark.parametrize("k", [1, 2])
def test_the_ledger_counts_exactly_4k_probes_and_3k_proposer_calls(tmp_path, mode, k):
    result, llm = run(tmp_path, {"INV": "Answer briefly.", "DOM": "Name the item.", "COMPOSE": "Answer briefly and name the item."}, k=k, mode=mode)

    book = ledger(result)
    assert book["probes"] == 4 * k and book["proposer_calls"] == 3 * k
    assert book["expected"] == {"probes": 4 * k, "proposer_calls": 3 * k}
    assert llm.calls == 3 * k
    assert [entry["step"] for entry in book["entries"]].count("compose") == k
    for name in ("optimized.yaml", "optimization.md", "ledger.json"):
        assert (result.out_dir / name).is_file()


def test_parallel_mode_proposes_from_each_base_rollout(tmp_path):
    result, llm = run(tmp_path, {"INV": "Answer briefly.", "DOM": "Name the item.", "COMPOSE": "Both."}, k=2, mode="parallel")

    proposals = [entry for entry in ledger(result)["entries"] if entry["step"] == "propose"]
    assert sorted(entry["label"] for entry in proposals) == ["p0-dom", "p0-inv", "p1-dom", "p1-inv"]
    assert [entry["evidence_trial"] for entry in proposals if entry["label"].startswith("p1")] == [1, 1]
    base = next(entry for entry in ledger(result)["entries"] if entry["step"] == "base")
    assert base["rollouts"] == 2


def test_sequential_mode_carries_the_best_harness_forward(tmp_path):
    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "Name the item.", "COMPOSE": "Name the item."}, k=2, mode="sequential")

    book = ledger(result)
    second_base = next(entry for entry in book["entries"] if entry["label"] == "r1-base")
    assert second_base["harness"]["answer_instructions"] == "GOOD" and second_base["reused"]


# -- selection and verification ---------------------------------------------------------------------------------------------------------------


def test_an_edit_that_helps_on_test_too_is_adopted(tmp_path):
    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "Name the item.", "COMPOSE": "GOOD"}, k=1)

    assert result.verdict == "improved"
    optimized = yaml.safe_load((result.out_dir / "optimized.yaml").read_text(encoding="utf-8"))
    tuned = next(system for system in optimized["systems"] if system["name"] == "tuned")
    assert tuned["harness"] == {"answer_instructions": "GOOD"}
    assert result.verification is not None and result.verification["mean_diff"] > 0 and result.verification["p_holm"] < 0.05
    assert "improved" in (result.out_dir / "optimization.md").read_text(encoding="utf-8")


def test_an_edit_that_helps_on_train_but_hurts_on_test_is_not_adopted(tmp_path):
    """MAGIC wins on train and on dev, so it is the pick; the test split shows it is worse than the base, and the base is kept."""
    result, _ = run(tmp_path, {"INV": "MAGIC", "DOM": "MAGIC", "COMPOSE": "MAGIC"}, k=1)

    assert result.verdict == "no_reliable_gain"
    assert result.pick is not None and result.verification is not None and result.verification["mean_diff"] < 0
    optimized = yaml.safe_load((result.out_dir / "optimized.yaml").read_text(encoding="utf-8"))
    assert "harness" not in next(system for system in optimized["systems"] if system["name"] == "tuned")
    assert "no reliable gain" in (result.out_dir / "optimization.md").read_text(encoding="utf-8")


def test_nothing_better_than_the_base_on_train_skips_dev_and_test(tmp_path):
    result, _ = run(tmp_path, {"INV": "Answer briefly.", "DOM": "Name the item.", "COMPOSE": "Answer briefly."}, k=1)

    assert result.verdict == "no_reliable_gain" and result.pick is None and result.verification is None
    assert not any(entry["step"] in ("select", "verify") for entry in ledger(result)["entries"])


def test_without_dev_questions_the_pick_is_made_on_train(tmp_path):
    write_project(tmp_path, dev=False)
    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)

    assert result.verdict == "improved"
    assert not any(entry["step"] == "select" for entry in ledger(result)["entries"])
    assert "picked on train" in (result.out_dir / "optimization.md").read_text(encoding="utf-8")


def test_an_inherited_proposal_is_probed_by_reusing_the_base(tmp_path):
    """Both proposals are unusable twice, so they inherit the base; with no change to merge the composition inherits it too, without a call."""
    result, llm = run(tmp_path, {"INV": "", "DOM": "", "COMPOSE": "GOOD"}, k=1)

    book = ledger(result)
    assert llm.calls == 2 + 2  # INV and DOM each retried once; nothing to compose
    probes = {entry["label"]: entry for entry in book["entries"] if entry["step"] == "probe"}
    assert probes["p0-inv"]["reused"] and probes["p0-inv"]["cost_usd"] == 0
    composition = next(entry for entry in book["entries"] if entry["step"] == "compose")
    assert composition["inherited"] and "nothing to compose" in composition["error"]
    assert book["probes"] == 4 and book["proposer_calls"] == 3


# -- spending cap and resume ------------------------------------------------------------------------------------------------------------------


def test_the_cap_stops_the_loop_and_keeps_the_base(tmp_path):
    """Each train probe answers 16 questions at $0.001: a $0.04 cap allows two probes (the base's two rollouts), then stops."""
    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=2, max_cost_usd=0.04)

    book = ledger(result)
    assert result.verdict == "stopped" and book["stopped"]
    assert book["spent_usd"] <= 0.04 + 16 * ANSWER_COST
    assert book["probes"] < 8
    optimized = yaml.safe_load((result.out_dir / "optimized.yaml").read_text(encoding="utf-8"))
    assert "harness" not in optimized["systems"][0]


def test_a_second_attempt_resumes_from_the_saved_proposals_and_probes(tmp_path):
    first, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)
    again, llm = run(tmp_path, {"INV": "other", "DOM": "other", "COMPOSE": "other"}, k=1)

    assert llm.calls == 0
    assert again.verdict == first.verdict == "improved"
    book = ledger(again)
    assert all(entry["restored"] or entry["reused"] for entry in book["entries"] if entry["step"] in ("base", "probe", "propose", "compose"))
    assert book["spent_usd"] == pytest.approx(ledger(first)["spent_usd"])


def test_a_second_attempt_with_other_settings_is_refused(tmp_path):
    run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)

    with pytest.raises(ValueError, match="other settings"):
        run(tmp_path, {"INV": "GOOD"}, k=2)


# -- the method never reads the test split ----------------------------------------------------------------------------------------------------


def test_no_test_question_reaches_the_proposer_or_its_evidence(tmp_path):
    result, llm = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=2)

    dataset = load_dataset(tmp_path / "questions.jsonl")
    guard = LeakageGuard.for_splits(dataset, load_splits(tmp_path / "splits.json"), held_out=("dev", "test"))
    assert guard.scan_path(result.out_dir / "evidence") == []
    for prompt in llm.prompts:
        assert guard.scan_text(prompt) == []
        assert "omega" not in prompt


def test_the_report_has_lineage_diffs_scores_diversity_and_verdict(tmp_path):
    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "Name the item.", "COMPOSE": "GOOD"}, k=1)

    report = (result.out_dir / "optimization.md").read_text(encoding="utf-8")
    for heading in ("## Verdict", "## Candidates", "## Changes", "## Diversity", "## Ledger"):
        assert heading in report
    assert "p0-inv" in report and "c0" in report and "answer_instructions" in report


def test_an_unknown_system_or_a_dataset_without_splits_is_refused(tmp_path):
    config = write_project(tmp_path)
    chosen, _ = proposer({})
    with pytest.raises(ValueError, match="Did you mean 'tuned'"):
        optimize(config, OptimizeSettings(system="tunned"), proposer=chosen, out_dir=tmp_path / "opt", force_mock=True)
    (tmp_path / "splits.json").unlink()
    with pytest.raises(ValueError, match="ragbench split"):
        optimize(config, OptimizeSettings(system="tuned"), proposer=chosen, out_dir=tmp_path / "opt2", force_mock=True)


def test_the_configs_own_split_and_cap_do_not_leak_into_the_probes(tmp_path):
    config = write_project(tmp_path)
    edit_config(config, lambda raw: {**raw, "dataset": {**raw["dataset"], "split": "test"}})

    result, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)

    base = next(entry for entry in ledger(result)["entries"] if entry["step"] == "base")
    assert base["split"] == "train" and base["questions"] == 16


# -- the command and the estimate -------------------------------------------------------------------------------------------------------------


def tiny_project(root: Path) -> Path:
    """The tiny real corpus with a split, and `bm25` as the system: the command end to end on the mock models and the scripted proposer."""
    from dataset_support import write_tiny_dataset

    from ragbench.datasets.loader import load_questions
    from ragbench.datasets.splits import make_splits, write_splits

    dataset = write_tiny_dataset(root)
    write_splits(root / "splits.json", make_splits(load_questions(Path(dataset["questions_path"])), train=0.5, dev=0.25, seed=3))
    config = {
        "run": {"name": "tiny", "output_dir": str(root / "results")},
        "dataset": dataset,
        "systems": [{"type": "bm25", "name": "bm25", "chunker": {"type": "word", "chunk_size": 40, "chunk_overlap": 0}, "retrieval": {"top_k": 3}}],
        "evaluation": {"max_workers": 1, "latency_probe_questions": 0},
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_the_command_runs_the_whole_loop_offline_with_mock(tmp_path, mode):
    from cli_support import invoke

    config = tiny_project(tmp_path)

    result = invoke("optimize", "--config", str(config), "--system", "bm25", "--k", "2", "--mode", mode, "--mock", "--out", str(tmp_path / "opt"), "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["probes"] == 8 and payload["proposer_calls"] == 6
    assert payload["verdict"] in ("improved", "no_reliable_gain")
    book = json.loads((tmp_path / "opt" / "ledger.json").read_text(encoding="utf-8"))
    assert all(not entry.get("inherited") for entry in book["entries"] if entry["step"] == "propose")  # the scripted proposer always finds an edit


def test_a_live_optimization_needs_a_cap_and_a_reachable_proposer(tmp_path, monkeypatch):
    from cli_support import invoke

    import ragbench.cli.optimize as command

    config = tiny_project(tmp_path)
    monkeypatch.setattr(command, "resolve_run_mode", lambda loaded, mock: "live")

    no_cap = invoke("optimize", "--config", str(config), "--system", "bm25", "--out", str(tmp_path / "a"))
    assert no_cap.exit_code == 2 and "spending cap" in no_cap.output
    no_key = invoke("optimize", "--config", str(config), "--system", "bm25", "--max-cost-usd", "1", "--proposer", "anthropic:claude-sonnet-5-5", "--out", str(tmp_path / "b"))
    assert no_key.exit_code == 2 and "cannot be reached" in no_key.output
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_estimate_counts_the_loops_runs_and_proposer_requests(tmp_path):
    from ragbench.config.loader import load_config
    from ragbench.optimize.estimate import estimate_optimize

    config = write_project(tmp_path)

    planned = estimate_optimize(load_config(config), OptimizeSettings(system="tuned", k=2, finalists=2), "gpt-6-luna")

    assert planned.questions == {"train": 16, "dev": 8, "test": 16}
    per_question, ingestion = planned.per_question_usd, planned.ingestion_usd
    assert per_question > 0
    assert planned.train_usd == pytest.approx(7 * ingestion + 4 * 2 * 16 * per_question)  # 3k + 1 runs: the base's k rollouts are one run
    assert planned.dev_usd == pytest.approx(3 * (ingestion + 8 * per_question))
    assert planned.test_usd == pytest.approx(2 * (ingestion + 16 * per_question))
    assert planned.proposer_usd > 0
    assert planned.total_usd == pytest.approx(planned.train_usd + planned.dev_usd + planned.test_usd + planned.proposer_usd)


def test_ragbench_estimate_optimize_prints_the_plan(tmp_path):
    from cli_support import invoke

    config = write_project(tmp_path)

    result = invoke("estimate", "--config", str(config), "--optimize", "tuned", "--k", "1")

    assert result.exit_code == 0, result.output
    assert "Train probes" in result.output and "4 probes of 16 questions" in result.output and "Proposer" in result.output


# -- a probe that fails -----------------------------------------------------------------------------------------------------------------------


def test_a_failed_probe_ends_the_loop_cleanly_and_a_second_attempt_runs_it_again(tmp_path, monkeypatch):
    """The base run fails (every question errors, as when indexing hits a rate limit): the loop stops and keeps the base, with outputs.
    Its partial run folder must not be read back as finished: once the system works, the next attempt runs the probe again."""
    from fake_systems import BrokenIngestSystem

    from ragbench.rag_systems import SYSTEM_REGISTRY

    config = write_project(tmp_path)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_tunable", BrokenIngestSystem)
    failed, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)

    assert failed.verdict == "stopped" and "failed" in (failed.ledger.stopped or "")
    assert (failed.out_dir / "optimization.md").is_file() and (failed.out_dir / "ledger.json").is_file()

    register_tunable(monkeypatch)
    again, _ = run(tmp_path, {"INV": "GOOD", "DOM": "x", "COMPOSE": "GOOD"}, k=1)

    base = next(entry for entry in ledger(again)["entries"] if entry["step"] == "base")
    assert not base["restored"] and again.verdict == "improved"
    assert config.exists()
