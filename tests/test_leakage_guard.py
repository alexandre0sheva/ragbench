"""The tripwire: nothing an optimizer, proposer or library writes may contain the test split's questions, ids or reference answers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.datasets.leakage import LeakageError, LeakageGuard
from ragbench.datasets.schema import Dataset, Question
from ragbench.datasets.splits import Splits

TEST_QUESTION = Question(
    id="q_test_017", question="Which office hosts the claims team in Lisbon?", reference_answer="The Lisbon office hosts the claims team since 2024.", category="direct_fact"
)
TRAIN_QUESTION = Question(
    id="q_train_003", question="How much does HarborShield cost per month?", reference_answer="HarborShield costs two hundred dollars per month.", category="direct_fact"
)
DEV_QUESTION = Question(id="q_dev_001", question="When does ClaimPilot ship to the first customers?", reference_answer="ClaimPilot ships in the third quarter.", category="direct_fact")


@pytest.fixture
def guard() -> LeakageGuard:
    dataset = Dataset(questions=[TRAIN_QUESTION, DEV_QUESTION, TEST_QUESTION])
    splits = Splits(seed=1, train=[TRAIN_QUESTION.id], dev=[DEV_QUESTION.id], test=[TEST_QUESTION.id])
    return LeakageGuard.for_splits(dataset, splits)


def _kinds(leaks) -> set[tuple[str, str]]:
    return {(leak.kind, leak.question_id) for leak in leaks}


def test_a_planted_test_question_id_is_caught(guard):
    leaks = guard.scan_text("Lessons so far.\n- on q_test_017 the system gave up too early\n", "library/lesson.md")

    assert _kinds(leaks) == {("id", "q_test_017")}
    assert leaks[0].source == "library/lesson.md"


def test_a_planted_test_question_is_caught_whatever_its_case_spacing_or_punctuation(guard):
    text = "Example to avoid:\n\n  WHICH   office hosts the claims team --- in Lisbon\n"

    assert _kinds(guard.scan_text(text, "prompt")) == {("question", "q_test_017")}


def test_a_planted_reference_answer_is_caught(guard):
    assert _kinds(guard.scan_text('few_shot: "the lisbon office hosts the claims team since 2024"', "prompt")) == {("reference_answer", "q_test_017")}


def test_text_escaped_in_json_is_still_caught(guard):
    escaped = '{"note": "Which office hosts the claims team in Lisbon\\u003f"}'

    assert _kinds(guard.scan_text(escaped, "evidence.json")) == {("question", "q_test_017")}


def test_train_and_dev_content_is_not_a_leak(guard):
    text = f"{TRAIN_QUESTION.question} {TRAIN_QUESTION.reference_answer} {TRAIN_QUESTION.id} {DEV_QUESTION.question} {DEV_QUESTION.id}"

    assert guard.scan_text(text, "evidence/task.md") == []


def test_an_id_only_counts_as_a_whole_token(guard):
    assert guard.scan_text("see q_test_0178 and xq_test_017y", "notes.md") == []
    assert _kinds(guard.scan_text("see (q_test_017).", "notes.md")) == {("id", "q_test_017")}


def test_short_test_questions_and_answers_are_not_matched_as_substrings_but_their_ids_are():
    short = Question(id="item-9", question="Who is CEO?", reference_answer="Maxine", category="x")
    guard = LeakageGuard.for_questions([short])

    assert guard.scan_text("Who is the CEO? Maxine says hello", "x") == []
    assert _kinds(guard.scan_text("failed item-9 again", "x")) == {("id", "item-9")}


def test_assert_text_clean_raises_naming_where_and_what_without_repeating_the_content(guard):
    with pytest.raises(LeakageError) as caught:
        guard.assert_text_clean("The Lisbon office hosts the claims team since 2024.", "harness/system_prompt.md")

    message = str(caught.value)
    assert "harness/system_prompt.md" in message and "q_test_017" in message and "reference answer" in message
    assert "claims team" not in message


def test_a_directory_is_scanned_recursively_and_binary_files_are_skipped(tmp_path: Path, guard):
    clean = tmp_path / "evidence"
    (clean / "rollouts").mkdir(parents=True)
    (clean / "task.md").write_text("Answer questions about the company's documents.\n", encoding="utf-8")
    (clean / "rollouts" / "q_train_003.md").write_text(f"{TRAIN_QUESTION.question}\n", encoding="utf-8")
    (clean / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    guard.assert_path_clean(clean)

    (clean / "rollouts" / "q_dev_001.md").write_text(f"trace of q_test_017\n{TRAIN_QUESTION.question}\n", encoding="utf-8")
    with pytest.raises(LeakageError, match="rollouts/q_dev_001.md"):
        guard.assert_path_clean(clean)
    assert [leak.source for leak in guard.scan_path(clean)] == ["rollouts/q_dev_001.md"]


def test_a_single_file_is_scanned_by_its_name(tmp_path: Path, guard):
    path = tmp_path / "MISSION.md"
    path.write_text(TEST_QUESTION.question, encoding="utf-8")

    assert [(leak.kind, leak.source) for leak in guard.scan_path(path)] == [("question", "MISSION.md")]


def test_a_guard_over_nothing_finds_nothing():
    assert LeakageGuard.for_questions([]).scan_text("anything at all q_test_017", "x") == []


def test_the_guard_can_hold_dev_back_as_well(guard):
    dataset = Dataset(questions=[TRAIN_QUESTION, DEV_QUESTION, TEST_QUESTION])
    splits = Splits(seed=1, train=[TRAIN_QUESTION.id], dev=[DEV_QUESTION.id], test=[TEST_QUESTION.id])
    strict = LeakageGuard.for_splits(dataset, splits, held_out=("dev", "test"))

    assert _kinds(strict.scan_text(DEV_QUESTION.question, "x")) == {("question", "q_dev_001")}
    assert guard.scan_text(DEV_QUESTION.question, "x") == []
