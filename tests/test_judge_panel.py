from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.config.schema import EvaluationConfig
from ragbench.datasets.schema import Question
from ragbench.evaluation.answer_judge import SCORE_AXES, AnswerJudge
from ragbench.evaluation.judge_panel import JudgePanel, panel_agreement, scoring_judges
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, LLMResult
from ragbench.models.refs import collect_model_refs
from ragbench.rag_systems.base import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
HAIKU = "anthropic:claude-haiku-5-5"
LUNA = "gpt-6-luna"
QWEN = "openrouter:qwen/qwen3.8-flash"


def _config_file(tmp_path: Path, evaluation: dict, systems: list[dict] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"name": "panel", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": systems or [{"type": "bm25", "name": "bm25"}],
        "evaluation": evaluation,
    }
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


# -- config ---------------------------------------------------------------------------------------


def test_without_a_panel_the_judge_models_are_the_one_judge():
    evaluation = EvaluationConfig(judge_model=HAIKU)
    assert evaluation.judge.panel == [] and evaluation.judge_models == [HAIKU]


def test_a_panel_lists_its_judges_and_leads_with_the_first():
    evaluation = EvaluationConfig(judge={"panel": [HAIKU, LUNA], "aggregate": "median"})
    assert evaluation.judge_models == [HAIKU, LUNA]
    assert evaluation.judge_model == evaluation.judge.model == HAIKU  # the lead judge: what needs one model (relevance grading) uses it
    assert evaluation.judge.aggregate == "median"


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        ({"judge": {"panel": [HAIKU]}}, "at least two"),
        ({"judge": {"panel": [HAIKU, HAIKU]}}, "at least two"),
        ({"judge": {"panel": [HAIKU, LUNA], "model": LUNA}}, "not both"),
        ({"judge_model": LUNA, "judge": {"panel": [HAIKU, LUNA]}}, "not both"),
    ],
)
def test_a_panel_needs_two_distinct_judges_and_no_single_judge_beside_it(evaluation, message):
    with pytest.raises(ValidationError, match=message):
        EvaluationConfig(**evaluation)


def test_every_panel_judge_is_a_model_the_run_calls_and_is_validated(tmp_path):
    config = load_config(_config_file(tmp_path, {"judge": {"panel": [HAIKU, LUNA]}}))
    refs = collect_model_refs(config).llm
    assert HAIKU in refs and LUNA in refs
    with pytest.raises(ValidationError, match=r"evaluation\.judge\.panel\[1\]"):
        load_config(_config_file(tmp_path, {"judge": {"panel": [HAIKU, "openrouter:no-vendor"]}}))


# -- independence ---------------------------------------------------------------------------------


def test_an_independent_panel_drops_the_judges_of_the_generators_family():
    assert scoring_judges([HAIKU, LUNA, QWEN], "gpt-6.1-sol", independent=True) == [HAIKU, QWEN]
    assert scoring_judges([HAIKU, LUNA, QWEN], "openrouter:anthropic/claude-haiku-5.5", independent=True) == [LUNA, QWEN]


def test_a_panel_that_is_not_independent_keeps_every_judge():
    assert scoring_judges([HAIKU, LUNA], "gpt-6.1-sol", independent=False) == [HAIKU, LUNA]


def test_when_every_judge_shares_the_generators_family_all_of_them_score():
    # Nobody else could score those answers; the self-preference notice says so instead.
    assert scoring_judges([LUNA, "openai:gpt-6.1-sol"], "gpt-6-luna", independent=True) == [LUNA, "openai:gpt-6.1-sol"]
    assert scoring_judges([LUNA], "gpt-6-luna", independent=True) == [LUNA]


# -- estimate -------------------------------------------------------------------------------------


def _estimate(tmp_path: Path, judge: dict, systems: list[dict] | None = None):
    from ragbench.evaluation.estimate import estimate_run

    # Distinct prices, so which judge was charged shows in the number.
    pricing = {
        "fake-judge": {"input": 500.0, "output": 1500.0},
        "fake-judge-2": {"input": 50.0, "output": 150.0},
        "gpt-6-luna": {"input": 1000.0, "output": 1000.0},
        "claude-haiku-5-5": {"input": 10.0, "output": 10.0},
    }
    path = _config_file(tmp_path, {"max_questions": 4, "latency_probe_questions": 0, "judge": judge}, systems)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump({**raw, "pricing": pricing}), encoding="utf-8")
    return {s.system: s.judge_usd for s in estimate_run(load_config(path)).systems}


def test_the_estimate_charges_every_judge_that_scores_a_system(tmp_path):
    one = _estimate(tmp_path / "a", {"model": "fake-judge"})["bm25"]
    two = _estimate(tmp_path / "b", {"model": "fake-judge-2"})["bm25"]
    panel = _estimate(tmp_path / "c", {"panel": ["fake-judge", "fake-judge-2"]})["bm25"]
    assert one > two > 0 and panel == pytest.approx(one + two)


def test_the_estimate_leaves_out_a_judge_that_sits_out_a_system(tmp_path):
    systems = [{"type": "bm25", "name": "luna", "models": {"generator": "gpt-6-luna"}}, {"type": "bm25", "name": "claude", "models": {"generator": HAIKU}}]
    panel = _estimate(tmp_path / "p", {"panel": [HAIKU, LUNA]}, systems)
    haiku_alone = _estimate(tmp_path / "h", {"model": HAIKU}, systems)
    luna_alone = _estimate(tmp_path / "l", {"model": LUNA}, systems)
    assert panel["luna"] == pytest.approx(haiku_alone["luna"])
    assert panel["claude"] == pytest.approx(luna_alone["claude"]) and panel["claude"] > 10 * panel["luna"]


# -- the panel ------------------------------------------------------------------------------------

CONTEXT = [RetrievedChunk(chunk_id="c1", doc_id="doc_001", text="Alice won the trophy.", score=1.0, rank=1)]


def _question(index: int) -> Question:
    return Question(id=f"q{index}", question="Who won?", reference_answer="Alice.", expected_keywords=["Alice"], relevant_doc_ids=["doc_001"])


class ScoreScript(LLM):
    """A judge model that gives the n-th answer it judges the n-th scripted score on every axis (None: an unusable reply)."""

    def __init__(self, name: str, scores: list[float | None]):
        self.model_name = name
        self.scores = scores
        self.judged = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        score = self.scores[self.judged]
        text = "not json" if score is None else json.dumps({**{axis: score for axis in SCORE_AXES}, "is_supported_by_context": score >= 3, "is_hallucinated": score < 2, "reasoning": f"{self.model_name}: {score}"})
        return LLMResult(text=text, model=self.model_name, prompt_tokens=10, completion_tokens=5, cost=CostBreakdown(llm_cost=0.01))

    def next_answer(self) -> None:
        self.judged += 1


def _panel(scripts: dict[str, list[float | None]], *, independent: bool = True, aggregate: str = "mean") -> tuple[JudgePanel, list[ScoreScript]]:
    llms = [ScoreScript(ref, scores) for ref, scores in scripts.items()]
    return JudgePanel([AnswerJudge(llm=llm) for llm in llms], list(scripts), independent=independent, aggregate=aggregate), llms


def _judge_all(panel: JudgePanel, llms: list[ScoreScript], n: int, generator: str = "openrouter:qwen/qwen3.8-flash") -> list[dict]:
    rows = []
    for index in range(n):
        result = panel.judge(_question(index), "Alice won.", CONTEXT, generator=generator)
        rows.append({"error": None, "answer_judge": {"answer_score": result.answer_score, "metadata": result.metadata}})
        for llm in llms:
            llm.next_answer()
    return rows


def test_a_panel_of_one_is_exactly_the_single_judge():
    llm = ScoreScript(LUNA, [4.0])
    alone = AnswerJudge(llm=ScoreScript(LUNA, [4.0])).judge(_question(0), "Alice won.", CONTEXT)
    panel = JudgePanel([AnswerJudge(llm=llm)], [LUNA], independent=True).judge(_question(0), "Alice won.", CONTEXT, generator=LUNA)
    assert panel.model_dump() == alone.model_dump()


def test_the_panel_verdict_is_the_mean_of_its_judges_and_records_each_one():
    panel, llms = _panel({HAIKU: [2.0], LUNA: [4.0], QWEN: [5.0]})
    result = panel.judge(_question(0), "Alice won.", CONTEXT, generator="openrouter:mistralai/mistral-small-2603")
    assert result.answer_score == pytest.approx(11 / 3) and result.correctness == pytest.approx(11 / 3)
    assert result.cost.judge_cost == pytest.approx(0.03)  # every judge's call is paid for
    meta = result.metadata
    assert meta["judge"] == "panel" and meta["fallback"] is False and meta["judges"] == [HAIKU, LUNA, QWEN] and meta["sat_out"] == []
    assert {ref: scores["answer_score"] for ref, scores in meta["panel_scores"].items()} == {HAIKU: 2.0, LUNA: 4.0, QWEN: 5.0}
    assert result.is_supported_by_context is True and result.is_hallucinated is False  # majority: 2 of 3 scored >= 3


def test_the_median_aggregate_resists_one_outlier():
    panel, _ = _panel({HAIKU: [0.0], LUNA: [4.0], QWEN: [5.0]}, aggregate="median")
    assert panel.judge(_question(0), "a", CONTEXT, generator="openrouter:mistralai/mistral-small-2603").answer_score == 4.0


def test_an_independent_panel_does_not_ask_the_generators_family():
    panel, (haiku, luna) = _panel({HAIKU: [3.0], LUNA: [5.0]})
    result = panel.judge(_question(0), "a", CONTEXT, generator="gpt-6.1-sol")
    assert result.answer_score == 3.0 and result.metadata["judges"] == [HAIKU] and result.metadata["sat_out"] == [LUNA]
    assert result.metadata["panel_scores"][LUNA] is None and result.cost.judge_cost == pytest.approx(0.01)  # never called


def test_a_judge_whose_verdict_is_unusable_is_left_out_of_the_panel_verdict():
    panel, _ = _panel({HAIKU: [None], LUNA: [4.0]})
    result = panel.judge(_question(0), "a", CONTEXT, generator=QWEN)
    assert result.answer_score == 4.0 and result.metadata["fallback"] is False and result.metadata["member_fallbacks"] == 1
    assert result.metadata["panel_scores"][HAIKU] is None and result.metadata["parse_error"] is True


def test_when_every_judge_fails_the_heuristic_scores_it_and_the_row_says_so():
    panel, _ = _panel({HAIKU: [None], LUNA: [None]})
    result = panel.judge(_question(0), "Alice won.", CONTEXT, generator=QWEN)
    assert result.metadata["judge"] == "heuristic" and result.metadata["fallback"] is True and result.metadata["member_fallbacks"] == 2


def test_injected_disagreement_yields_the_expected_alpha():
    # Two judges agree on three answers and differ by one point on the fourth: alpha = 1 - 14/222 (see test_agreement.py).
    panel, llms = _panel({HAIKU: [1.0, 2.0, 3.0, 4.0], LUNA: [1.0, 2.0, 3.0, 5.0]})
    agreement = panel_agreement(_judge_all(panel, llms, 4))
    assert set(agreement) == {*SCORE_AXES, "answer_score"}
    assert agreement["answer_score"] == pytest.approx(1 - 14 / 222) and agreement["correctness"] == pytest.approx(1 - 14 / 222)


def test_agreement_skips_judges_that_sat_out_and_is_empty_without_a_panel():
    panel, llms = _panel({HAIKU: [1.0, 5.0, 1.0], LUNA: [1.0, 5.0, 1.0], QWEN: [5.0, 1.0, 5.0]})
    rows = _judge_all(panel, llms, 3, generator="openrouter:qwen/qwen3-max")  # qwen sits out: the two left agree perfectly
    assert panel_agreement(rows)["answer_score"] == 1.0
    assert panel_agreement([{"error": None, "answer_judge": {"answer_score": 3.0, "metadata": {"judge": "gpt-6-luna"}}}]) == {}


# -- notices and the run --------------------------------------------------------------------------


def test_the_self_preference_notice_is_about_the_model_family():
    from ragbench.reporting.notices import build_notices

    [notice] = build_notices("live", judge_model="gpt-6-luna", judge_shares_family_with=["bm25", "hyde"])
    assert "Self-preference risk" in notice and "same model family" in notice and "bm25, hyde" in notice


def test_judge_notices_name_the_systems_no_other_family_could_judge():
    from ragbench.evaluation.summary import judge_notices

    single, _ = _panel({LUNA: [4.0]})
    generators = {"sol": "gpt-6.1-sol", "qwen": QWEN}
    assert judge_notices(single, generators, [], {})["judge_shares_family_with"] == ["sol"]
    panel, _ = _panel({HAIKU: [4.0], LUNA: [4.0]})
    assert judge_notices(panel, generators, [], {})["judge_shares_family_with"] == []  # haiku judges sol, both judge qwen
    lax, _ = _panel({LUNA: [4.0]}, independent=False)
    assert judge_notices(lax, generators, [], {})["judge_shares_family_with"] == []  # accepted in the config


def test_a_panel_that_disagrees_is_called_out():
    from ragbench.reporting.notices import build_notices

    assert any("disagree" in n and "0.41" in n for n in build_notices("live", judge_agreement=0.41))
    assert not build_notices("live", judge_agreement=0.9)


def test_a_mock_run_with_a_panel_records_each_judge_and_the_agreement(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    path = _config_file(tmp_path, {"max_questions": 4, "max_workers": 1, "judge": {"panel": [HAIKU, LUNA, QWEN]}})
    out = run_benchmark(path, force_mock=True)
    row = json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])
    meta = row["answer_judge"]["metadata"]
    assert meta["judge"] == "panel" and meta["judges"] == [HAIKU, QWEN] and meta["sat_out"] == [LUNA]  # bm25's generator is gpt-6-luna
    judge = json.loads((out / "run_summary.json").read_text())["judge"]
    assert judge["models"] == [HAIKU, LUNA, QWEN] and judge["aggregate"] == "mean" and judge["independent"] is True
    assert judge["agreement"]["answer_score"] == 1.0  # the mock's judges are one heuristic: they always agree
    assert "panel of 3" in (out / "report.html").read_text(encoding="utf-8")


def test_a_run_with_one_judge_has_no_panel_block(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    out = run_benchmark(_config_file(tmp_path, {"max_questions": 2, "max_workers": 1}), force_mock=True)
    assert "judge" not in json.loads((out / "run_summary.json").read_text())


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_judge_every_system_with_another_model_family(path):
    from ragbench.models.family import model_family
    from ragbench.models.refs import answering_model

    config = load_config(path)
    for system in config.systems:
        generator = answering_model(system)
        judges = scoring_judges(config.evaluation.judge_models, generator, independent=config.evaluation.judge.independent)
        assert all(model_family(judge) != model_family(generator) for judge in judges), f"{system.resolved_name}: {generator} judged by {judges}"


def test_a_disabled_judge_is_never_a_panel(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    out = run_benchmark(_config_file(tmp_path, {"max_questions": 2, "max_workers": 1, "judge_enabled": False, "judge": {"panel": [HAIKU, LUNA]}}), force_mock=True)
    row = json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])
    assert row["answer_judge"]["metadata"]["judge"] == "heuristic" and "judge" not in json.loads((out / "run_summary.json").read_text())
