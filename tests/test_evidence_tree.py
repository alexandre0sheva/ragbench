"""The evidence tree a harness proposer reads: built from a run's train questions only, in three feedback modes, behind the leakage guard."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from evidence_support import copy_run, edit_config, edit_rows, make_run, question_texts, read_manifest

from ragbench.datasets.leakage import LeakageError, LeakageGuard
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.splits import Splits
from ragbench.optimize.evidence import FORBIDDEN_WORDS, EvidenceError, assert_no_verdicts, build_evidence


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, Splits]:
    return make_run(tmp_path_factory.mktemp("evidence_run"))


@pytest.fixture
def run_dir(run) -> Path:
    return run[0]


@pytest.fixture
def splits(run) -> Splits:
    return run[1]


def held_out_guard(run_dir: Path, splits: Splits) -> LeakageGuard:
    dataset = load_dataset(run_dir.parent.parent / "questions.jsonl")
    return LeakageGuard.for_splits(dataset, splits, held_out=("dev", "test"))


def files(evidence: Path) -> set[str]:
    return {path.relative_to(evidence).as_posix() for path in evidence.rglob("*") if path.is_file()}


def test_the_tree_holds_the_mission_task_harness_rollouts_contrasts_failures_and_library(run_dir, splits, tmp_path):
    evidence = build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")

    found = files(evidence)
    assert {"MISSION.md", "task.md", "failures.md", "manifest.json", "harness/fields.md", "harness/bm25.yaml", "harness/hybrid.yaml", "library/README.md"} <= found
    assert any(name.startswith("contrasts/") for name in found)
    assert {name for name in found if name.startswith("rollouts/")} == {f"rollouts/{qid}.md" for qid in splits.train}
    mission = (evidence / "MISSION.md").read_text(encoding="utf-8")
    assert "`bm25`" in mission and "INV" in mission and "DOM" in mission
    manifest = read_manifest(evidence)
    assert manifest["system"] == "bm25" and manifest["split"] == "train" and manifest["feedback"] == "contrast"
    assert manifest["rollouts"] == len(splits.train) and manifest["split_hash"] == splits.digest


def test_the_default_location_is_inside_the_run_and_a_rebuild_replaces_it(run, tmp_path):
    run_dir = copy_run(run[0], tmp_path / "run")
    first = build_evidence(run_dir, system="bm25")
    assert first == run_dir / "evidence" / "bm25-contrast"
    (first / "stale.md").write_text("old", encoding="utf-8")

    again = build_evidence(run_dir, system="bm25")

    assert again == first and not (again / "stale.md").exists()


def test_a_folder_that_is_not_an_evidence_tree_is_never_replaced(run_dir, tmp_path):
    target = tmp_path / "mine"
    target.mkdir()
    (target / "notes.md").write_text("keep me", encoding="utf-8")

    with pytest.raises(EvidenceError, match="not an evidence folder"):
        build_evidence(run_dir, system="bm25", out_dir=target)
    assert (target / "notes.md").read_text(encoding="utf-8") == "keep me"


# -- the tripwire -----------------------------------------------------------------------------------------------------------------------------


def test_no_dev_or_test_content_reaches_the_evidence_of_a_run_over_every_question(run_dir, splits, tmp_path):
    """The run scored all eight questions; the evidence of every mode holds none of the held-out ones (id, text or reference answer)."""
    guard = held_out_guard(run_dir, splits)
    questions = question_texts(run_dir.parent.parent)
    for feedback in ("labels", "contrast", "none"):
        evidence = build_evidence(run_dir, system="hybrid", feedback=feedback, out_dir=tmp_path / feedback)

        assert guard.scan_path(evidence) == []
        text = "\n".join(path.read_text(encoding="utf-8") for path in evidence.rglob("*") if path.is_file())
        for qid in [*splits.dev, *splits.test]:
            assert qid not in text
            assert questions[qid]["question"] not in text


def test_a_train_answer_quoting_a_test_question_is_withheld_not_shown(run, tmp_path):
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    questions = question_texts(run[0].parent.parent)
    planted = questions[splits.test[0]]["question"]
    victim = splits.train[0]
    edit_rows(run_dir, lambda row: {**row, "answer": f"As asked before: {planted}"} if row["question_id"] == victim and row["system"] == "bm25" else row)

    evidence = build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")

    rollout = (evidence / "rollouts" / f"{victim}.md").read_text(encoding="utf-8")
    assert planted not in rollout and "[withheld: held-out text]" in rollout
    assert read_manifest(evidence)["withheld_passages"] >= 1
    assert held_out_guard(run[0], splits).scan_path(evidence) == []


def test_held_out_text_the_builder_cannot_withhold_stops_the_build_and_leaves_no_folder(run, tmp_path):
    """A harness prompt holding a test question (written by hand in the run's config) cannot be redacted away: the guard refuses the tree."""
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    planted = question_texts(run[0].parent.parent)[splits.test[0]]["question"]

    def plant(config: dict) -> dict:
        config["systems"][0]["harness"] = {"system_prompt": f"Remember: {planted}"}
        return config

    edit_config(run_dir, plant)
    target = tmp_path / "ev"

    with pytest.raises(LeakageError, match=re.escape(splits.test[0])) as caught:
        build_evidence(run_dir, system="bm25", out_dir=target)
    assert planted not in str(caught.value)
    assert not target.exists()


# -- feedback modes ---------------------------------------------------------------------------------------------------------------------------


def contrasted_run(run, tmp_path) -> tuple[Path, str]:
    """A copy of the run where bm25 answers one train question well and hybrid does not."""
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    question = splits.train[0]

    def score(row: dict) -> dict:
        if row["question_id"] != question:
            return row
        value = 5.0 if row["system"] == "bm25" else 1.0
        return {**row, "answer_judge": {**row["answer_judge"], "answer_score": value}, "failure_type": "no_failure" if value == 5.0 else "partial_answer"}

    edit_rows(run_dir, score)
    return run_dir, question


def test_labels_mode_shows_the_grade_reference_answer_and_who_was_right(run, tmp_path):
    run_dir, question = contrasted_run(run, tmp_path)

    evidence = build_evidence(run_dir, system="bm25", feedback="labels", out_dir=tmp_path / "ev")

    rollout = (evidence / "rollouts" / f"{question}.md").read_text(encoding="utf-8")
    assert "Judge grade: 5.00 / 5" in rollout and "Reference answer:" in rollout
    contrast = (evidence / "contrasts" / "hybrid.md").read_text(encoding="utf-8")
    assert question in contrast and "Right: `bm25`" in contrast
    assert "partial_answer" in (evidence / "contrasts" / "hybrid.md").read_text(encoding="utf-8")


def test_contrast_mode_shows_behaviour_only(run, tmp_path):
    """The Harness² setting: the contrast questions and where the traces part, but no grade, reference answer, failure type or winner."""
    run_dir, question = contrasted_run(run, tmp_path)

    evidence = build_evidence(run_dir, system="bm25", feedback="contrast", out_dir=tmp_path / "ev")

    contrast = (evidence / "contrasts" / "hybrid.md").read_text(encoding="utf-8")
    assert question in contrast and "This system (`bm25`)" in contrast
    assert_no_verdicts(evidence)
    text = "\n".join(path.read_text(encoding="utf-8") for path in evidence.rglob("*.md"))
    for marker in ("Judge grade", "Reference answer", "Right:", "partial_answer", "no_failure"):
        assert marker not in text


def test_none_mode_has_rollouts_and_no_contrasts(run, tmp_path):
    run_dir, _ = contrasted_run(run, tmp_path)

    evidence = build_evidence(run_dir, system="bm25", feedback="none", out_dir=tmp_path / "ev")

    assert files(evidence) & {"contrasts/hybrid.md"} == set()
    assert read_manifest(evidence)["contrast_questions"] == 0
    assert_no_verdicts(evidence)


def test_the_verdict_tripwire_catches_a_grade_in_a_label_free_mode(run_dir, tmp_path):
    evidence = build_evidence(run_dir, system="bm25", feedback="labels", out_dir=tmp_path / "ev")

    with pytest.raises(AssertionError, match="rollouts/"):
        assert_no_verdicts(evidence)


@pytest.mark.parametrize("feedback", ["labels", "contrast", "none"])
def test_the_framing_files_never_mention_the_measurement(run_dir, tmp_path, feedback):
    evidence = build_evidence(run_dir, system="hybrid", feedback=feedback, out_dir=tmp_path / "ev")

    for name in ("MISSION.md", "task.md", "harness/fields.md", "library/README.md", "contrasts/README.md"):
        path = evidence / name
        if path.exists():
            assert not re.search("|".join(FORBIDDEN_WORDS), path.read_text(encoding="utf-8"), flags=re.I), name


def test_task_md_describes_the_train_questions_and_the_documents(run_dir, splits, tmp_path):
    evidence = build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")

    task = (evidence / "task.md").read_text(encoding="utf-8")
    assert "6 documents" in task and f"{len(splits.train)} questions" in task and "Language: en" in task


def test_fields_md_lists_what_the_system_can_change(run_dir, tmp_path):
    evidence = build_evidence(run_dir, system="hybrid", out_dir=tmp_path / "ev")

    fields = (evidence / "harness" / "fields.md").read_text(encoding="utf-8")
    for name in ("`system_prompt`", "`answer_instructions`", "`few_shot`", "`top_k`", "`reranker`"):
        assert name in fields
    assert "`max_hops`" not in fields and "`tool_guidance`" not in fields


# -- refusals ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "all"])
def test_only_a_split_a_method_may_read_is_accepted(run_dir, tmp_path, split):
    with pytest.raises(EvidenceError, match="train or dev"):
        build_evidence(run_dir, split, system="bm25", out_dir=tmp_path / "ev")


def test_a_run_of_several_systems_needs_the_system_named(run_dir, tmp_path):
    with pytest.raises(EvidenceError, match="bm25, hybrid"):
        build_evidence(run_dir, out_dir=tmp_path / "ev")


def test_an_unknown_system_gets_a_did_you_mean(run_dir, tmp_path):
    with pytest.raises(EvidenceError, match="Did you mean 'hybrid'"):
        build_evidence(run_dir, system="hybrd", out_dir=tmp_path / "ev")


def test_a_dataset_without_splits_is_refused(run, tmp_path):
    run_dir = copy_run(run[0], tmp_path / "run")
    edit_config(run_dir, lambda config: {**config, "dataset": {**config["dataset"], "splits_path": str(tmp_path / "missing.json")}})

    with pytest.raises(EvidenceError, match="ragbench split"):
        build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")


def test_splits_changed_since_a_split_run_are_refused(run, tmp_path):
    run_dir = copy_run(run[0], tmp_path / "run")
    manifest = run_dir / "run_manifest.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text(encoding="utf-8")), "split": {"name": "train", "hash": "other"}}), encoding="utf-8")

    with pytest.raises(EvidenceError, match="changed since the run"):
        build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")


def test_a_run_without_train_questions_is_refused(run, tmp_path):
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    edit_rows(run_dir, lambda row: {**row, "question_id": "other"} if row["question_id"] in splits.train else row)

    with pytest.raises(EvidenceError, match="no train questions"):
        build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")


def test_a_folder_with_some_other_manifest_is_never_replaced(run_dir, tmp_path):
    target = tmp_path / "other"
    target.mkdir()
    (target / "manifest.json").write_text('{"name": "not evidence"}', encoding="utf-8")

    with pytest.raises(EvidenceError, match="not an evidence folder"):
        build_evidence(run_dir, system="bm25", out_dir=target)
    assert (target / "manifest.json").exists()


def test_a_system_with_no_harness_to_change_is_refused(run, tmp_path):
    run_dir = copy_run(run[0], tmp_path / "run")
    edit_config(run_dir, lambda config: {**config, "systems": [{"type": "python_callable", "name": "bm25", "retrieval": {"target": "json:dumps"}}, config["systems"][1]]})

    with pytest.raises(EvidenceError, match="no harness"):
        build_evidence(run_dir, system="bm25", out_dir=tmp_path / "ev")


def test_a_trial_keeps_the_rows_of_that_trial_only(run, tmp_path):
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    path = run_dir / "per_question_results.jsonl"
    from ragbench.utils.jsonl import read_jsonl, write_jsonl

    rows = read_jsonl(path)
    write_jsonl(path, [*({**row, "trial": 0} for row in rows), *({**row, "trial": 1, "answer": "second attempt"} for row in rows)])

    first = build_evidence(run_dir, system="bm25", out_dir=tmp_path / "t0", trial=0)
    second = build_evidence(run_dir, system="bm25", out_dir=tmp_path / "t1", trial=1)

    question = splits.train[0]
    assert "second attempt" not in (first / "rollouts" / f"{question}.md").read_text(encoding="utf-8")
    text = (second / "rollouts" / f"{question}.md").read_text(encoding="utf-8")
    assert "second attempt" in text and "Attempt 2" not in text
    assert read_manifest(second)["trial"] == 1
