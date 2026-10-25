from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from ragbench.datasets.schema import Dataset, Question
from ragbench.datasets.splits import SplitError, Splits, apply_split, load_splits, make_splits, write_splits


def _questions() -> list[Question]:
    """100 questions in four strata: category a (40 answerable, 20 not) and category b (30 answerable, 10 not)."""
    rows = [("a", True, 40), ("a", False, 20), ("b", True, 30), ("b", False, 10)]
    questions: list[Question] = []
    for category, answerable, count in rows:
        for i in range(count):
            questions.append(Question(id=f"{category}_{'y' if answerable else 'n'}_{i:02d}", question=f"Question {category} {answerable} {i}?", category=category, answerable=answerable))
    return questions


def test_splits_are_disjoint_and_cover_every_question():
    questions = _questions()

    splits = make_splits(questions, train=0.5, dev=0.1, seed=42)

    assigned = [*splits.train, *splits.dev, *splits.test]
    assert sorted(assigned) == sorted(q.id for q in questions)
    assert len(set(assigned)) == len(assigned)
    assert (len(splits.train), len(splits.dev), len(splits.test)) == (50, 10, 40)


def test_every_category_and_answerability_stratum_keeps_the_requested_shares():
    questions = _questions()
    splits = make_splits(questions, train=0.5, dev=0.1, seed=42)
    stratum = {q.id: (q.category, q.is_answerable) for q in questions}

    for key, size in Counter(stratum.values()).items():
        for name, share in (("train", 0.5), ("dev", 0.1), ("test", 0.4)):
            got = sum(stratum[qid] == key for qid in getattr(splits, name))
            assert abs(got - size * share) <= 1, f"{name} has {got} of the {size} questions in {key}, wanted about {size * share:g}"


def test_the_same_seed_gives_the_same_split_whatever_the_input_order_and_another_seed_a_different_one():
    questions = _questions()
    shuffled = questions[:]
    random.Random(7).shuffle(shuffled)

    first = make_splits(questions, train=0.5, dev=0.1, seed=42)

    assert make_splits(shuffled, train=0.5, dev=0.1, seed=42) == first
    assert make_splits(questions, train=0.5, dev=0.1, seed=43).test != first.test


def test_a_single_question_stratum_goes_to_one_split_and_nothing_is_lost():
    questions = [Question(id="solo", question="Only one of its kind?", category="rare", answerable=True), *_questions()]

    splits = make_splits(questions, train=0.5, dev=0.1, seed=1)

    assert sum("solo" in part for part in (splits.train, splits.dev, splits.test)) == 1


@pytest.mark.parametrize(("train", "dev"), [(0, 0.1), (-0.1, 0.1), (0.5, -0.1), (0.9, 0.1), (0.95, 0.2), (1.5, 0)])
def test_fractions_must_leave_a_train_share_and_a_test_share(train, dev):
    with pytest.raises(SplitError, match="train"):
        make_splits(_questions(), train=train, dev=dev, seed=1)


def test_duplicate_question_ids_are_refused():
    questions = [Question(id="q1", question="One?"), Question(id="q1", question="Two?")]
    with pytest.raises(SplitError, match="duplicate"):
        make_splits(questions, train=0.5, dev=0.1, seed=1)


def test_a_splits_file_round_trips_and_is_never_overwritten(tmp_path: Path):
    path = tmp_path / "splits.json"
    splits = make_splits(_questions(), train=0.5, dev=0.1, seed=42)

    write_splits(path, splits)

    assert load_splits(path) == splits
    with pytest.raises(SplitError, match="already exists"):
        write_splits(path, make_splits(_questions(), train=0.6, dev=0.1, seed=42))
    assert load_splits(path) == splits, "the refused write leaves the file as it was"


def test_the_digest_follows_the_assignment_and_nothing_else():
    questions = _questions()
    a = make_splits(questions, train=0.5, dev=0.1, seed=42)
    same = Splits.model_validate(a.model_dump())
    other = make_splits(questions, train=0.5, dev=0.1, seed=43)

    assert same.digest == a.digest
    assert other.digest != a.digest


def test_an_unreadable_splits_file_names_the_problem(tmp_path: Path):
    bad = tmp_path / "splits.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SplitError, match="splits.json"):
        load_splits(bad)
    with pytest.raises(SplitError, match="not found"):
        load_splits(tmp_path / "missing.json")


def test_apply_split_keeps_one_split_in_dataset_order_with_its_qrels():
    questions = _questions()
    dataset = Dataset(questions=questions, qrels={q.id: {"doc": 1} for q in questions})
    splits = make_splits(questions, train=0.5, dev=0.1, seed=42)

    test = apply_split(dataset, splits, "test")

    assert [q.id for q in test.questions] == [q.id for q in questions if q.id in set(splits.test)]
    assert set(test.qrels) == set(splits.test)
    assert apply_split(dataset, splits, "all") is dataset


def test_apply_split_refuses_a_dataset_with_questions_the_splits_do_not_know():
    questions = _questions()
    splits = make_splits(questions, train=0.5, dev=0.1, seed=42)
    grown = Dataset(questions=[*questions, Question(id="late", question="Added after the split?")])

    with pytest.raises(SplitError, match="late"):
        apply_split(grown, splits, "test")
