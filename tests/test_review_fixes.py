"""Regression tests for the pre-Task-26 review: spend of failed questions, judge outages, out-of-credit errors, resume hashes, map scores."""

from __future__ import annotations

import httpx
import openai
import pytest

from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_checks import AnswerChecks
from ragbench.evaluation.answer_judge import AnswerJudge
from ragbench.evaluation.checkpoint import context_hash
from ragbench.evaluation.judge_panel import JudgePanel
from ragbench.evaluation.question_runner import evaluate_question
from ragbench.models.cost import CostBreakdown
from ragbench.models.errors import PermanentModelError, RateLimitError, TransientModelError, translate_openai_exception
from ragbench.models.llms import LLM, LLMResult
from ragbench.models.rerankers.llm import LLMReranker
from ragbench.models.spend import metering, note_spend
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.base import RetrievedChunk
from ragbench.rag_systems.cli_trajectory import Trajectory, Turn
from ragbench.rag_systems.graph_rag import _points

QUESTION = Question(id="q", question="Who won?", reference_answer="Alice.", expected_keywords=["Alice"], relevant_doc_ids=["doc_001"])
CONTEXT = [RetrievedChunk(chunk_id="doc_001::chunk::0", doc_id="doc_001", text="Alice won the trophy.", score=1.0, rank=1)]


class FailingLLM(LLM):
    """Raises `error` on every call."""

    model_name = "failing-judge"

    def __init__(self, error: Exception):
        self.error = error

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        raise self.error


class ScriptedLLM(LLM):
    model_name = "scripted-judge"

    def __init__(self, text: str):
        self.text = text

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        return LLMResult(text=self.text, model=self.model_name, prompt_tokens=10, completion_tokens=5, cost=CostBreakdown(llm_cost=0.01))


GOOD_VERDICT = '{"correctness": 4, "faithfulness": 4, "completeness": 4, "relevance": 4, "citation_quality": 4, "reasoning": "ok"}'


# -- spend of a question that fails partway ------------------------------------------------------------------------------------------------


def test_spend_meter_counts_only_inside_its_block_and_across_copied_contexts():
    from ragbench.runtime.parallel import ordered_parallel_map

    note_spend(CostBreakdown(llm_cost=9.0))  # outside a question: ignored
    with metering() as meter:
        note_spend(CostBreakdown(llm_cost=0.25))
        ordered_parallel_map(lambda _: note_spend(CostBreakdown(embedding_cost=0.125)), range(4), workers=4)
    assert meter.cost.total_cost == pytest.approx(0.75)


def test_a_question_that_fails_after_paying_reports_and_charges_what_it_spent(monkeypatch):
    system_config = SystemConfig(type="bm25", name="bm25")
    system = create_rag_system(system_config, force_mock=True)

    def answer_then_fail(question, top_k=None, context_k=None):
        note_spend(CostBreakdown(llm_cost=0.5))  # e.g. an agent's paid turns before its last call failed
        raise RuntimeError("provider gave up")

    monkeypatch.setattr(system, "answer_question", answer_then_fail)
    judge = JudgePanel([AnswerJudge(force_mock=True)], ["gpt-6-luna"], independent=False)
    result = evaluate_question(system, system_config, QUESTION, {}, judge, ExperimentConfig.model_validate(_config()).evaluation, AnswerChecks())

    assert result["per_question"]["error"]["message"] == "provider gave up"
    assert result["per_question"]["cost"]["total_cost"] == pytest.approx(0.5)
    assert result["cost_row"] is None  # still left out of every mean


def test_a_harness_that_fails_partway_charges_its_turns():
    from ragbench.rag_systems.cli_agent_rag import CliAgentRAG

    system = create_rag_system(SystemConfig(type="cli_agent", name="agent", retrieval={"preset": "custom", "command": ["true"], "model": "gpt-6-luna"}), force_mock=True)
    assert isinstance(system, CliAgentRAG)
    trajectory = Trajectory(items=[Turn(prompt_tokens=100_000, completion_tokens=0), Turn(prompt_tokens=100_000, completion_tokens=0)])
    with metering() as meter:
        system._charge_partial(trajectory)
    assert meter.cost.total_cost == pytest.approx(0.02)  # 200k prompt tokens at gpt-6-luna's $0.10 per 1M


# -- judge outages -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", [TransientModelError("529 overloaded"), RateLimitError("slow down")])
def test_a_judge_call_that_still_fails_after_retries_is_a_fallback_not_a_failed_question(error):
    result = AnswerJudge(llm=FailingLLM(error)).judge(QUESTION, "Alice won.", CONTEXT)
    assert result.metadata["fallback"] is True and result.metadata["judge"] == "heuristic"
    assert type(error).__name__ in result.metadata["call_error"]


def test_a_permanent_judge_error_still_fails_the_question():
    with pytest.raises(PermanentModelError):
        AnswerJudge(llm=FailingLLM(PermanentModelError("invalid api key"))).judge(QUESTION, "Alice won.", CONTEXT)


def test_one_judge_outage_leaves_the_rest_of_the_panel_scoring():
    panel = JudgePanel(
        [AnswerJudge(llm=FailingLLM(TransientModelError("down"))), AnswerJudge(llm=ScriptedLLM(GOOD_VERDICT))],
        ["anthropic:claude-haiku-5-5", "gpt-6-luna"],
        independent=False,
    )
    result = panel.judge(QUESTION, "Alice won.", CONTEXT, generator="openrouter:z-ai/glm-5.3")
    assert result.metadata["fallback"] is False and result.answer_score == pytest.approx(4.0)
    assert result.metadata["member_fallbacks"] == 1


# -- out of credit -------------------------------------------------------------------------------------------------------------------------


def _rate_limit(body: dict) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"), json=body)
    return openai.RateLimitError(body["error"]["message"], response=response, body=body["error"])


def test_out_of_credit_is_permanent_and_an_ordinary_rate_limit_is_retried():
    quota = _rate_limit({"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "code": "insufficient_quota"}})
    assert isinstance(translate_openai_exception(quota), PermanentModelError)
    limited = _rate_limit({"error": {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}})
    assert isinstance(translate_openai_exception(limited), RateLimitError)


# -- resume hashes ---------------------------------------------------------------------------------------------------------------------------


def _config(**extra) -> dict:
    return {
        "run": {"name": "t"},
        "dataset": {"documents_path": "data/demo/documents", "questions_path": "data/demo/questions.jsonl"},
        "systems": [{"type": "bm25", "name": "bm25"}],
        **extra,
    }


def test_openrouter_settings_are_in_the_checkpoint_hash_and_the_default_changes_nothing():
    def digest(**extra):
        config = ExperimentConfig.model_validate(_config(**extra))
        return context_hash(config.systems[0], config, "d", "live")

    assert digest() == digest(openrouter={})
    assert digest() != digest(openrouter={"reasoning": {"effort": "low"}})


# -- small failures that used to sink a question ---------------------------------------------------------------------------------------


def test_graph_rag_map_scores_that_are_infinite_or_nan_do_not_raise():
    reply = '{"points": [{"text": "a", "score": 1e999}, {"text": "b", "score": "Infinity"}, {"text": "c", "score": NaN}, {"text": "d", "score": 40}]}'
    assert _points(reply) == [(100, "a"), (100, "b"), (0, "c"), (40, "d")]


def test_llm_reranker_fallback_books_its_call_as_rerank_cost():
    reranker = LLMReranker(ScriptedLLM("not json"))
    result = reranker.rerank("Who won?", CONTEXT, top_k=1)
    assert result.cost.rerank_cost == pytest.approx(0.01) and result.cost.llm_cost == 0.0


# -- second round: reporting, selection, tools ------------------------------------------------------------------------------------------


def test_a_published_run_folder_without_per_question_rows_keeps_its_stage_tables(tmp_path):
    import shutil

    from cli_support import mock_run

    from ragbench.reporting.regenerate import regenerate_reports

    (tmp_path / "work").mkdir()
    run = mock_run(tmp_path / "work", tmp_path / "run")
    original = (run / "leaderboard.md").read_text(encoding="utf-8")
    assert "## Latency by stage" in original and (run / "stages.json").is_file()
    published = tmp_path / "published"
    published.mkdir()
    for name in ("config.yaml", "metrics_summary.csv", "significance.csv", "run_summary.json", "stats.json", "stages.json"):
        shutil.copy(run / name, published / name)

    _, notes = regenerate_reports(published)
    assert (published / "leaderboard.md").read_text(encoding="utf-8") == original
    (published / "stages.json").unlink()
    _, notes = regenerate_reports(published)
    assert any("stage tables left out" in note for note in notes)


def test_markdown_table_cells_keep_a_pipe_inside_their_column():
    from ragbench.reporting.markdown_report import _markdown_table

    table = _markdown_table(["Q", "N"], [{"Q": "Is A | B larger?", "N": "1"}])
    assert table.splitlines()[-1] == "| Is A \\| B larger? | 1 |"


def test_winner_yaml_keeps_the_openrouter_section(tmp_path):
    import yaml

    from ragbench.selection.recommend import winner_yaml_text

    config = {**_config(openrouter={"reasoning": {"effort": "low"}}), "systems": [{"type": "bm25", "name": "bm25"}]}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    written = yaml.safe_load(winner_yaml_text(tmp_path, "bm25"))
    assert written["openrouter"] == {"reasoning": {"effort": "low"}}


def test_an_unknown_cost_never_wins_a_tie_on_cost():
    from ragbench.selection.scoring import Weights, efficiency_penalties

    penalties = efficiency_penalties({"priced": 0.002, "unknown": None}, {"priced": 900.0, "unknown": 900.0}, Weights(quality=0.2, cost=0.8, latency=0.0))
    assert penalties["unknown"] > penalties["priced"] == 0.0


def test_lookup_table_selects_columns_ignoring_case_like_where():
    from ragbench.documents.schema import Document
    from ragbench.tools.base import CorpusView, ToolContext
    from ragbench.tools.corpus import LookupTableTool

    row = Document(doc_id="t1", path="prices.csv", title="prices", text="product: HarborShield\nprice: 200", metadata={"row": 1})
    result = LookupTableTool().run({"where": {"Product": "harborshield"}, "select": ["Price"]}, ToolContext(corpus=CorpusView([row])))
    assert result.data == [{"Price": "200"}]
