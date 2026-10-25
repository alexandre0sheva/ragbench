"""Claim-level faithfulness (`evaluation.claims`): an answer is split into atomic claims, each is checked against the context the generator was given."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.config.schema import EvaluationConfig
from ragbench.evaluation.checkpoint import context_hash
from ragbench.evaluation.claims import ClaimChecker
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.decisions import LLMJsonDecisionModel
from ragbench.models.refs import collect_model_refs
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.utils.jsonl import read_jsonl

PRICE_TEXT = "HarborShield costs $200 per month for the marine module."
CONTEXT = [RetrievedChunk(chunk_id="doc_001::chunk::0", doc_id="doc_001", text=PRICE_TEXT, score=1.0, rank=1)]
FREE_COFFEE = "Every subscriber also gets free coffee."
ANSWER = f"HarborShield costs $200 per month. {FREE_COFFEE} [doc_001]"


def _scripted_checker(scripted_llm, *, split_reply=None, verdicts=None, **options) -> tuple[ClaimChecker, object, object]:
    """A checker whose splitter says `split_reply` and whose support check says `verdicts[claim text]` (a probability, default 0.9)."""
    splitter = scripted_llm(
        lambda prompt: (
            split_reply if split_reply is not None else json.dumps({"claims": ["HarborShield costs $200 per month.", FREE_COFFEE]})
        ),
        price_per_call=0.002,
    )

    def judge(prompt: str) -> str:
        claim = prompt.split("Claim: ", 1)[1].split("\n\n", 1)[0]
        return json.dumps({"probabilities": {"passage": (verdicts or {}).get(claim, 0.9)}})

    checker_llm = scripted_llm(judge, price_per_call=0.001)
    return ClaimChecker(splitter, LLMJsonDecisionModel(checker_llm), **options), splitter, checker_llm


# -- the checker ----------------------------------------------------------------------------------


def test_an_unsupported_claim_is_found_and_the_rate_counts_it(scripted_llm):
    checker, splitter, _ = _scripted_checker(scripted_llm, verdicts={FREE_COFFEE: 0.05})
    result = checker.check("How much does HarborShield cost?", ANSWER, CONTEXT)
    assert result.unsupported == [FREE_COFFEE]
    assert result.support_rate == 0.5
    assert [claim.supported for claim in result.claims] == [True, False]
    assert "How much does HarborShield cost?" in splitter.prompts[0] and "free coffee" in splitter.prompts[0]


def test_the_check_sees_the_context_the_generator_saw(scripted_llm):
    checker, _, check_llm = _scripted_checker(scripted_llm)
    checker.check("q", ANSWER, CONTEXT)
    assert check_llm.calls == 2 and all(PRICE_TEXT in prompt for prompt in check_llm.prompts)


def test_spend_is_charged_as_judge_cost(scripted_llm):
    checker, _, _ = _scripted_checker(scripted_llm)
    result = checker.check("q", ANSWER, CONTEXT)
    assert result.cost.judge_cost == pytest.approx(0.002 + 2 * 0.001)
    assert result.cost.judge_prompt_tokens > 0


def test_an_answer_with_no_claims_has_no_rate_rather_than_a_perfect_one(scripted_llm):
    checker, _, check_llm = _scripted_checker(scripted_llm, split_reply=json.dumps({"claims": []}))
    result = checker.check("q", "I could not find the answer in the provided documents.", CONTEXT)
    assert result.claims == [] and result.support_rate is None and not result.split_failed
    assert check_llm.calls == 0


def test_an_unreadable_split_is_reported_and_nothing_is_checked(scripted_llm):
    checker, _, check_llm = _scripted_checker(scripted_llm, split_reply="no json here")
    result = checker.check("q", ANSWER, CONTEXT)
    assert result.split_failed and result.support_rate is None and check_llm.calls == 0


def test_a_claim_the_model_gave_no_answer_for_is_left_out_of_the_rate(scripted_llm):
    splitter = scripted_llm(lambda prompt: json.dumps({"claims": ["Claim one.", "Claim two."]}))
    replies = iter(["not json", json.dumps({"probabilities": {"passage": 0.9}})])
    checker = ClaimChecker(splitter, LLMJsonDecisionModel(scripted_llm(lambda prompt: next(replies))))
    result = checker.check("q", ANSWER, CONTEXT)
    assert [claim.supported for claim in result.claims] == [None, True]
    assert result.support_rate == 1.0 and result.unreadable == 1 and result.unsupported == []


def test_only_the_first_max_claims_are_checked(scripted_llm):
    many = json.dumps({"claims": [f"Claim number {i}." for i in range(10)]})
    checker, _, check_llm = _scripted_checker(scripted_llm, split_reply=many, max_claims=3)
    result = checker.check("q", ANSWER, CONTEXT)
    assert len(result.claims) == 3 and check_llm.calls == 3


def test_the_split_may_come_in_a_fence_and_blank_claims_are_dropped(scripted_llm):
    reply = "```json\n" + json.dumps({"claims": ["  HarborShield costs $200 per month.  ", "", 7]}) + "\n```"
    checker, _, _ = _scripted_checker(scripted_llm, split_reply=reply)
    assert [claim.text for claim in checker.check("q", ANSWER, CONTEXT).claims] == ["HarborShield costs $200 per month."]


# -- the offline stand-in -------------------------------------------------------------------------


def test_the_mock_checker_flags_a_claim_the_context_does_not_contain():
    checker = ClaimChecker.mock()
    answer = "HarborShield costs $200 per month for the marine module [doc_001]. The Zurich office serves astronauts on Mars."
    result = checker.check("How much does HarborShield cost?", answer, CONTEXT)
    assert [claim.supported for claim in result.claims] == [True, False]
    assert result.unsupported == ["The Zurich office serves astronauts on Mars."]
    assert "[doc_001]" not in result.claims[0].text  # citations are not part of a claim


def test_the_mock_checker_makes_no_claims_from_a_refusal():
    result = ClaimChecker.mock().check("q", "I could not find the answer in the provided documents.", CONTEXT)
    assert result.claims == [] and result.support_rate is None


def test_the_mock_checker_costs_nothing():
    assert ClaimChecker.mock().check("q", ANSWER, CONTEXT).cost.total_cost == 0.0


# -- config ---------------------------------------------------------------------------------------


def test_claims_are_off_by_default_and_use_the_judge_model():
    evaluation = EvaluationConfig(judge_model="anthropic:claude-haiku-5-5")
    assert evaluation.claims.enabled is False
    assert evaluation.claims_model == "anthropic:claude-haiku-5-5"
    assert EvaluationConfig(claims={"enabled": True, "model": "gpt-6-luna"}).claims_model == "gpt-6-luna"


def test_claims_options_are_validated():
    with pytest.raises(ValidationError):
        EvaluationConfig(claims={"enabled": True, "max_claim": 3})
    with pytest.raises(ValidationError):
        EvaluationConfig(claims={"max_claims": 0})


def test_from_config_builds_a_checker_only_when_enabled():
    assert ClaimChecker.from_config(EvaluationConfig(), force_mock=True) is None
    assert ClaimChecker.from_config(EvaluationConfig(claims={"enabled": True}), force_mock=True) is not None


# -- a run ----------------------------------------------------------------------------------------


def _config(tmp_path: Path, dataset: dict[str, str], evaluation: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
run: {{name: claims, output_dir: {tmp_path / "results"}}}
dataset: {json.dumps(dataset)}
systems:
  - {{type: bm25, name: bm25}}
  - {{type: no_retrieval, name: floor}}
evaluation: {{max_workers: 1, latency_probe_questions: 0, {evaluation}}}
""",
        encoding="utf-8",
    )
    return path


def test_a_mock_run_reports_claim_support_per_system_and_per_question(tmp_path, tiny_dataset):
    out = run_benchmark(_config(tmp_path, tiny_dataset, "claims: {enabled: true}"), force_mock=True)
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert 0.0 <= summary.loc["bm25", "claim_support_rate"] <= 1.0
    assert summary.loc["bm25", "unsupported_claims"] >= 0.0
    rows = {(r["system"], r["question_id"]): r for r in read_jsonl(out / "per_question_results.jsonl")}
    block = rows[("bm25", "q1")]["claims"]
    assert block["total"] == len(block["claims"]) > 0 and block["unsupported"] == [
        c["text"] for c in block["claims"] if c["supported"] is False
    ]
    refused = [row for row in rows.values() if row["refused"]]
    assert refused and all(row["claims"]["total"] == 0 for row in refused)  # a refusal asserts nothing: no claims, so no rate
    answers = pd.read_csv(out / "answer_metrics.csv")
    assert {"claim_support_rate", "unsupported_claims"} <= set(answers.columns)
    assert pd.isna(answers[answers["refused"]]["claim_support_rate"]).all()


def test_a_run_without_claims_is_unchanged(tmp_path, tiny_dataset):
    out = run_benchmark(_config(tmp_path, tiny_dataset), force_mock=True)
    assert not any("claim" in column for column in pd.read_csv(out / "metrics_summary.csv").columns)
    assert not any("claim" in column for column in pd.read_csv(out / "answer_metrics.csv").columns)
    assert all("claims" not in row for row in read_jsonl(out / "per_question_results.jsonl"))


# -- model refs, checkpoints, estimate ------------------------------------------------------------


def test_the_claims_model_is_one_of_the_models_a_run_calls(tmp_path, tiny_dataset):
    config = load_config(
        _config(tmp_path, tiny_dataset, "judge_model: gpt-6-luna, claims: {enabled: true, model: 'anthropic:claude-haiku-5-5'}")
    )
    assert "anthropic:claude-haiku-5-5" in collect_model_refs(config).llm
    off = load_config(_config(tmp_path, tiny_dataset, "judge_model: gpt-6-luna, claims: {model: 'anthropic:claude-haiku-5-5'}"))
    assert "anthropic:claude-haiku-5-5" not in collect_model_refs(off).llm


def test_checkpoint_hashes_ignore_the_defaults_and_follow_the_settings(tmp_path, tiny_dataset):
    def digest(evaluation: str) -> str:
        config = load_config(_config(tmp_path, tiny_dataset, evaluation))
        return context_hash(config.systems[0], config, "data", "mock")

    plain = digest("")
    assert digest("claims: {enabled: false}, refusal_detector: heuristic") == plain
    assert digest("claims: {enabled: true}") != plain
    assert digest("refusal_detector: model") != plain
    assert digest("claims: {enabled: true, max_claims: 3}") != digest("claims: {enabled: true}")


def test_the_estimate_charges_claim_checking(tmp_path, tiny_dataset):
    from ragbench.evaluation.estimate import estimate_run

    def judge_usd(evaluation: str) -> float:
        config = load_config(_config(tmp_path, tiny_dataset, "judge_model: gpt-6-luna" + evaluation))
        return estimate_run(config).systems[0].judge_usd_per_question

    assert judge_usd(", claims: {enabled: true}") > judge_usd("")
