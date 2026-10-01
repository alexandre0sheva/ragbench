from __future__ import annotations

import math

import pytest

from ragbench.evaluation.answer_metrics import exact_match, is_refusal, keyword_recall, normalize_answer, refusal_metrics, token_f1


def test_normalization_ignores_case_punctuation_articles_and_citations():
    assert normalize_answer("The  Quick, Brown FOX!") == "quick brown fox"
    assert normalize_answer("Maxine Thompson [doc_005] won.") == "maxine thompson won"
    assert normalize_answer("It costs $1,200.50 in total.") == "it costs 1200.50 in total"  # digit grouping goes, the decimal point stays


def test_exact_match_uses_the_normalized_form():
    assert exact_match("The answer is 42.", "answer is 42") == 1.0
    assert exact_match("42", "43") == 0.0
    assert exact_match("", "") == 1.0


def test_token_f1_cases():
    assert token_f1("a red fox", "red fox") == 1.0  # the article is dropped
    assert token_f1("fox", "red fox") == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert token_f1("blue", "red fox") == 0.0
    assert token_f1("", "red fox") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("red red fox", "red fox") == pytest.approx(0.8)  # repeated tokens count once per occurrence in the reference


def test_keyword_recall_is_the_share_of_keywords_present():
    answer = "Maxine Thompson won the Insurance Innovator award in 2023."
    assert keyword_recall(answer, ["Maxine Thompson", "Insurance Innovator", "2023"]) == 1.0
    assert keyword_recall(answer, ["Maxine Thompson", "Antarctica"]) == 0.5
    assert keyword_recall("twenty twenty-three", ["2023"]) == 0.0
    assert keyword_recall("The 20230 budget", ["2023"]) == 0.0  # whole tokens only, no substring hits


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("I could not find the answer in the provided documents.", True),
        ("I could not find the answer from my own knowledge.", True),
        ("The documents do not contain any information about that.", True),
        ("Sorry, I cannot find the answer.", True),
        ("The Atlas plan is not available in Europe.", False),  # an answer that happens to say "not available"
        ("Maxine Thompson won in 2023.", False),
        ("", False),
    ],
)
def test_refusal_detection(answer, expected):
    assert is_refusal(answer) is expected


def test_refusal_metrics_on_a_two_by_two_table():
    rows = (
        [{"answerable": True, "refused": False}] * 6  # answered
        + [{"answerable": True, "refused": True}] * 2  # false refusals
        + [{"answerable": False, "refused": True}] * 3  # correct abstentions
        + [{"answerable": False, "refused": False}] * 1  # missed abstention
    )
    metrics = refusal_metrics(rows)
    assert metrics["abstain_precision"] == pytest.approx(3 / 5)
    assert metrics["abstain_recall"] == pytest.approx(3 / 4)
    assert metrics["false_refusal_rate"] == pytest.approx(2 / 8)


def test_refusal_metrics_are_missing_when_undefined():
    only_answerable = refusal_metrics([{"answerable": True, "refused": False}] * 3)
    assert only_answerable["false_refusal_rate"] == 0.0
    assert only_answerable["abstain_recall"] is None and only_answerable["abstain_precision"] is None  # nothing unanswerable, nothing refused
    assert all(value is None for value in refusal_metrics([]).values())
    assert not any(isinstance(value, float) and math.isnan(value) for value in only_answerable.values())  # missing is None, not NaN


def test_mock_run_reports_the_new_columns_and_flags_per_question(tmp_path):
    import json
    from pathlib import Path

    import pandas as pd
    import yaml

    from ragbench.evaluation.evaluator import run_benchmark

    demo = Path(__file__).resolve().parents[1] / "data" / "demo"
    config = {
        "run": {"name": "metrics_v2", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(demo / "docs"), "questions_path": str(demo / "questions.jsonl"), "qrels_path": str(demo / "qrels.jsonl")},
        "systems": [{"type": "bm25", "name": "bm25"}, {"type": "no_retrieval", "name": "floor"}],
        "evaluation": {"max_workers": 1},
    }
    path = tmp_path / "metrics_v2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out = run_benchmark(path, force_mock=True)

    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    for column in ("exact_match", "token_f1", "keyword_recall", "abstain_precision", "abstain_recall", "false_refusal_rate"):
        assert summary[column].dropna().between(0, 1).all(), column
    assert pd.isna(summary.loc["bm25", "abstain_precision"]) and summary.loc["bm25", "abstain_recall"] == 0.0, "mock BM25 always answers: no refusals to be precise about"
    assert summary.loc["floor", "abstain_recall"] > 0 and summary.loc["floor", "abstain_precision"] > 0
    assert summary.loc["bm25", "context_recall"] > 0 and summary.loc["bm25", "context_precision"] > 0
    assert summary.loc[["floor"], ["context_recall", "context_precision"]].isna().all().all(), "a system that retrieves nothing has no context metrics"
    assert summary["judge_fallback_rate"].isna().all(), "mock runs use the heuristic judge by design: nothing fell back"
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    unanswerable = [row for row in rows if row["system"] == "bm25" and not row["answerable"]]
    assert unanswerable and all(row["answer_judge"]["metadata"]["judge"] == "heuristic" for row in rows)
    assert all(row["answer_metrics"]["token_f1"] is None for row in unanswerable), "unanswerable questions have no F1: refusals are judged by the abstention metrics"
    answers = pd.read_csv(out / "answer_metrics.csv")
    assert {"exact_match", "token_f1", "keyword_recall", "context_recall", "refused", "answerable"} <= set(answers.columns)
