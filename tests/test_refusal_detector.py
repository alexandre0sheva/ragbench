"""`evaluation.refusal_detector`: the wording heuristic stays the default and the mock; `model` asks a decision model, and falls back to the heuristic when it gives no answer."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from ragbench.config.schema import EvaluationConfig
from ragbench.evaluation.answer_metrics import is_refusal
from ragbench.evaluation.refusal import RefusalDetector
from ragbench.models.decisions import LLMJsonDecisionModel, MockDecisionModel

ODD = "Sadly, nothing in the files you gave me speaks to that."  # a refusal the wording heuristic does not know
CLEAR = "I could not find the answer in the provided documents."
ANSWER = "HarborShield costs $200 per month."


def _model_detector(scripted_llm, probability) -> tuple[RefusalDetector, object]:
    llm = scripted_llm(lambda prompt: json.dumps({"probabilities": {"passage": probability}}), price_per_call=0.001)
    return RefusalDetector(LLMJsonDecisionModel(llm)), llm


def test_the_heuristic_is_the_default_and_misses_the_odd_phrasing():
    assert EvaluationConfig().refusal_detector == "heuristic"
    detector = RefusalDetector()
    assert not detector.uses_model
    assert detector.detect("How much?", ODD).refused is False
    assert detector.detect("How much?", CLEAR).refused is True


def test_a_decision_model_recognizes_an_oddly_phrased_refusal(scripted_llm):
    detector, llm = _model_detector(scripted_llm, 0.93)
    verdict = detector.detect("How much does HarborShield cost?", ODD)
    assert verdict.refused is True and verdict.source == "model" and verdict.probability == 0.93
    assert llm.calls == 1
    assert "HarborShield cost" in llm.prompts[0] and ODD in llm.prompts[0]  # it sees the question and the answer
    assert verdict.cost.judge_cost == pytest.approx(0.001)  # charged as judge spend


def test_a_decision_model_leaves_a_real_answer_alone(scripted_llm):
    detector, _ = _model_detector(scripted_llm, 0.04)
    assert detector.detect("How much?", ANSWER).refused is False


def test_the_model_decides_even_against_the_wording(scripted_llm):
    detector, _ = _model_detector(scripted_llm, 0.1)  # the answer quotes a refusal phrase while giving the figure
    answer = "The sheet does not provide information about discounts, but the price is $200 per month."
    assert is_refusal(answer)
    assert detector.detect("How much?", answer).refused is False


def test_an_unusable_reply_falls_back_to_the_wording(scripted_llm):
    detector = RefusalDetector(LLMJsonDecisionModel(scripted_llm(lambda prompt: "I refuse to emit JSON")))
    assert detector.detect("How much?", CLEAR).refused is True
    verdict = detector.detect("How much?", ANSWER)
    assert verdict.refused is False and verdict.source == "heuristic" and verdict.probability is None


def test_an_empty_answer_is_not_sent_to_the_model(scripted_llm):
    detector, llm = _model_detector(scripted_llm, 0.99)
    assert detector.detect("How much?", "  ").refused is False
    assert llm.calls == 0


def test_from_config_uses_the_heuristic_for_the_default_and_for_mock_runs():
    assert not RefusalDetector.from_config(EvaluationConfig(), force_mock=False).uses_model
    assert not RefusalDetector.from_config(EvaluationConfig(refusal_detector="model"), force_mock=True).uses_model
    assert RefusalDetector(MockDecisionModel()).uses_model is False  # the mock cannot read meaning


def test_the_option_is_validated():
    with pytest.raises(ValidationError):
        EvaluationConfig(refusal_detector="regex")


def test_a_mock_run_with_the_model_detector_matches_the_heuristic(tmp_path, tiny_dataset):
    from ragbench.evaluation.evaluator import run_benchmark
    from ragbench.utils.jsonl import read_jsonl

    def run(detector: str):
        config = tmp_path / f"{detector}.yaml"
        config.write_text(
            f"""
run: {{name: {detector}, output_dir: {tmp_path / "results"}}}
dataset: {json.dumps({key: str(value) for key, value in tiny_dataset.items()})}
systems: [{{type: bm25, name: bm25}}]
evaluation: {{max_workers: 1, latency_probe_questions: 0, refusal_detector: {detector}}}
""",
            encoding="utf-8",
        )
        out = run_benchmark(config, force_mock=True)
        return [(r["question_id"], r["refused"]) for r in read_jsonl(out / "per_question_results.jsonl")], pd.read_csv(
            out / "metrics_summary.csv"
        )

    heuristic_rows, heuristic_summary = run("heuristic")
    model_rows, model_summary = run("model")
    assert model_rows == heuristic_rows
    assert model_summary["false_refusal_rate"].equals(heuristic_summary["false_refusal_rate"])


def test_the_model_verdict_reaches_the_row_the_failure_type_and_the_cost(tmp_path, tiny_dataset, scripted_llm):
    from ragbench.config.schema import DatasetConfig, SystemConfig
    from ragbench.datasets.loader import load_dataset
    from ragbench.documents.loaders import load_dataset_documents
    from ragbench.evaluation.answer_checks import AnswerChecks
    from ragbench.evaluation.judge_panel import JudgePanel
    from ragbench.evaluation.question_runner import evaluate_question
    from ragbench.rag_systems import create_rag_system

    evaluation = EvaluationConfig(max_workers=1)
    system_config = SystemConfig(type="bm25", name="bm25")
    system = create_rag_system(system_config, force_mock=True)
    system.ingest(load_dataset_documents(DatasetConfig(**tiny_dataset), warnings=[]))
    dataset = load_dataset(tmp_path / "questions.jsonl")
    question = next(q for q in dataset.questions if q.id == "q1")  # answerable: the documents hold the price
    real = system.answer_question
    system.answer_question = lambda *args, **kwargs: real(*args, **kwargs).model_copy(update={"answer": ODD})  # type: ignore[method-assign]

    def score(checks: AnswerChecks) -> dict:
        panel = JudgePanel.from_config(evaluation, force_mock=True)
        return evaluate_question(system, system_config, question, {}, panel, evaluation, checks)["per_question"]

    plain = score(AnswerChecks())
    detector, _ = _model_detector(scripted_llm, 0.95)
    checked = score(AnswerChecks(detector))
    assert plain["refused"] is False and "refusal" not in plain
    assert checked["refused"] is True and checked["refusal"] == {"source": "model", "probability": 0.95}
    assert checked["failure_type"] in {"over_refusal", "insufficient_context"} and plain["failure_type"] not in {"over_refusal"}
    assert checked["cost"]["judge_cost"] == pytest.approx(plain["cost"]["judge_cost"] + 0.001)
