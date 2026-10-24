from __future__ import annotations

import json
import re
import tempfile
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ragbench.config.loader import load_config
from ragbench.config.schema import PairwiseConfig
from ragbench.datasets.schema import Question
from ragbench.evaluation.budget import BudgetGuard
from ragbench.evaluation.pairwise import PairwiseJudge
from ragbench.evaluation.pairwise_run import run_pairwise, schedule, summarize
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLM, LLMResult

SYSTEMS = ["weak", "fair", "good", "best"]
QUALITY = {"weak": 1, "fair": 2, "good": 3, "best": 4}
QUESTIONS = [Question(id=f"q{i}", question=f"Question {i}?", answerable=True) for i in range(24)]
GENERATORS = dict.fromkeys(SYSTEMS, "openrouter:qwen/qwen3.8-flash")


class Judge(LLM):
    """Reads the hidden quality from both anonymous answers and replies with `pick(quality_1, quality_2)`."""

    def __init__(self, pick, name: str = "fake-judge"):
        self.pick = pick
        self.model_name = name
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, max_tokens=None, json_mode=False, tools=None) -> LLMResult:
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        q1, q2 = (int(re.search(r"quality=(\d+)", payload[key]).group(1)) for key in ("answer_1", "answer_2"))
        return LLMResult(text=json.dumps({"winner": self.pick(q1, q2)}), model=self.model_name, prompt_tokens=40, completion_tokens=8, cost=CostBreakdown(llm_cost=0.001))


def prefers_quality(a: int, b: int) -> str:
    return "1" if a > b else "2" if b > a else "tie"


def always_first(a: int, b: int) -> str:
    return "1"


def first_on_close_calls(a: int, b: int) -> str:
    """Knows better, but takes the first answer when the two are one level apart."""
    return "1" if abs(a - b) <= 1 else prefers_quality(a, b)


def _rows(systems=SYSTEMS, failed: set[tuple[str, str]] = frozenset()) -> list[dict]:
    return [
        {
            "system": s,
            "question_id": q.id,
            "answer": f"{s} answer, quality={QUALITY.get(s, 2)}",
            "answer_judge": {"answer_score": float(QUALITY.get(s, 2))},
            "error": {"type": "Boom"} if (s, q.id) in failed else None,
        }
        for s in systems
        for q in QUESTIONS
    ]


def _run(pick, settings: PairwiseConfig | None = None, **kwargs):
    llm = Judge(pick)
    result = run_pairwise(
        _rows(),
        QUESTIONS,
        SYSTEMS,
        judges={"fake-judge": PairwiseJudge(llm, uses_llm=True)},
        generators=GENERATORS,
        settings=settings or PairwiseConfig(enabled=True),
        independent=True,
        baseline="weak",
        seed=0,
        **kwargs,
    )
    return result, llm


def _arena(result) -> dict[str, dict]:
    return {row["system"]: row for row in summarize(result, SYSTEMS, n_boot=200, seed=0)["arena"]}


# -- schedule -------------------------------------------------------------------------------------


def _available(failed: set[tuple[str, str]] = frozenset()) -> dict[str, list[str]]:
    return {q.id: [s for s in SYSTEMS if (s, q.id) not in failed] for q in QUESTIONS}


def test_all_pairs_on_every_question_in_a_seeded_random_order():
    plan = schedule(_available(), "all", None, baseline=None, seed=0)
    assert all({frozenset(p) for p in pairs} == {frozenset(p) for p in combinations(SYSTEMS, 2)} for pairs in plan.values())
    shown_first = {pair[0] for pairs in plan.values() for pair in pairs if set(pair) == {"weak", "best"}}
    assert shown_first == {"weak", "best"}  # who is x (shown first in the first order) is shuffled, not the config order
    assert plan == schedule(_available(), "all", None, baseline=None, seed=0) != schedule(_available(), "all", None, baseline=None, seed=1)


def test_a_failed_answer_takes_no_part_and_max_pairs_caps_a_question():
    plan = schedule(_available({("best", "q0")}), "all", 2, baseline=None, seed=0)
    assert all(len(pairs) == 2 for pairs in plan.values())
    assert all("best" not in pair for pair in plan["q0"])


def test_vs_baseline_pairs_every_system_with_the_baseline():
    plan = schedule(_available({("weak", "q3")}), "vs_baseline", None, baseline="weak", seed=0)
    assert all(len(pairs) == 3 and all("weak" in pair for pair in pairs) for qid, pairs in plan.items() if qid != "q3")
    assert plan["q3"] == []  # no baseline answer to compare with


# -- run ------------------------------------------------------------------------------------------


def test_a_judge_with_a_known_preference_order_is_recovered():
    result, llm = _run(prefers_quality)
    arena = _arena(result)
    assert sorted(SYSTEMS, key=lambda s: arena[s]["arena_score"]) == SYSTEMS
    assert arena["weak"]["arena_score_ci_hi"] < arena["best"]["arena_score_ci_lo"]
    assert llm.calls == result.calls == 24 * 6 * 2  # every pair, both orders
    assert result.cost.judge_cost == pytest.approx(0.001 * llm.calls)
    assert summarize(result, SYSTEMS, n_boot=50)["summary"]["position_bias"]["flip_rate"] == 0.0


def test_a_judge_that_only_follows_position_is_detected_and_cancelled():
    result, _ = _run(always_first)
    summary = summarize(result, SYSTEMS, n_boot=50)
    assert summary["summary"]["position_bias"] == {"pairs": 144, "flip_rate": 1.0, "first_position_rate": 1.0}
    assert all(row["arena_score"] == pytest.approx(1000, abs=1e-6) for row in summary["arena"])  # every pair a draw: no ranking invented
    assert all(rate is None or rate == pytest.approx(0.5) for rates in summary["win_rates"].values() for rate in rates.values())


def test_a_partly_position_biased_judge_still_ranks_right_and_shows_its_bias():
    result, _ = _run(first_on_close_calls)
    arena = _arena(result)
    assert sorted(SYSTEMS, key=lambda s: arena[s]["arena_score"]) == SYSTEMS
    assert summarize(result, SYSTEMS, n_boot=50)["summary"]["position_bias"]["flip_rate"] == pytest.approx(0.5)  # 3 of 6 pairs are close


def test_adaptive_spends_fewer_calls_and_spends_them_on_close_pairs():
    result, llm = _run(prefers_quality, PairwiseConfig(enabled=True, pairs="adaptive", max_pairs_per_question=2))
    assert llm.calls == 24 * 2 * 2
    arena = _arena(result)
    assert sorted(SYSTEMS, key=lambda s: arena[s]["arena_score"]) == SYSTEMS
    met = {frozenset((c.x, c.y)) for c in result.comparisons}
    counts = {pair: sum(frozenset((c.x, c.y)) == pair for c in result.comparisons) for pair in met}
    assert counts.get(frozenset(("weak", "best")), 0) < counts[frozenset(("good", "best"))]


def test_the_spending_cap_stops_new_comparisons():
    result, llm = _run(prefers_quality, budget=BudgetGuard(0.0105))
    assert result.skipped > 0 and llm.calls < 24 * 6 * 2 and result.calls == llm.calls


def test_an_unusable_verdict_drops_that_comparison_only():
    result, _ = _run(lambda a, b: "maybe" if {a, b} == {1, 4} else prefers_quality(a, b))
    assert result.failed == 24 and len([c for c in result.comparisons if c.complete]) == 24 * 5


def test_an_independent_judge_never_judges_a_pair_with_its_own_family():
    haiku, luna = Judge(prefers_quality, "claude-haiku-5-5"), Judge(prefers_quality, "gpt-6-luna")
    generators = {"weak": "gpt-6-luna", "fair": "gpt-6.1-sol", "good": "anthropic:claude-haiku-5-5", "best": "openrouter:qwen/qwen3.8-flash"}
    result = run_pairwise(
        _rows(),
        QUESTIONS[:2],
        SYSTEMS,
        judges={"anthropic:claude-haiku-5-5": PairwiseJudge(haiku, uses_llm=True), "gpt-6-luna": PairwiseJudge(luna, uses_llm=True)},
        generators=generators,
        settings=PairwiseConfig(enabled=True),
        independent=True,
        baseline=None,
        seed=0,
    )
    by_pair: dict[frozenset, set[str]] = {}
    for record in result.records:
        by_pair.setdefault(frozenset(record["systems"]), set()).add(record["judge"])
    assert by_pair[frozenset(("weak", "fair"))] == {"anthropic:claude-haiku-5-5"}  # two GPT systems: Haiku
    assert by_pair[frozenset(("good", "best"))] == {"gpt-6-luna"}
    # GPT against Claude: no neutral judge, so both judge it and their leanings offset.
    assert by_pair[frozenset(("weak", "good"))] == {"anthropic:claude-haiku-5-5", "gpt-6-luna"}
    assert result.same_family == 2 * 2 * 2  # two mixed pairs x two judges x two questions


# -- the run --------------------------------------------------------------------------------------

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def _config(tmp_path: Path, systems: list[dict], pairwise: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"name": "arena", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl"), "qrels_path": str(DEMO / "qrels.jsonl")},
        "systems": systems,
        "evaluation": {"max_questions": 6, "max_workers": 2, "pairwise": pairwise, "stats": {"n_boot": 200}},
    }
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


THREE = [{"type": "bm25", "name": "bm25"}, {"type": "no_retrieval", "name": "closed_book"}, {"type": "hybrid", "name": "hybrid"}]


def test_a_mock_run_with_pairwise_writes_the_arena_everywhere(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    out = run_benchmark(_config(tmp_path, THREE, {"enabled": True}), force_mock=True)
    records = [json.loads(line) for line in (out / "pairwise.jsonl").read_text().splitlines()]
    assert len(records) == 6 * 3 and {r["judge"] for r in records} == {"gpt-6-luna"}
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert {"arena_score", "arena_score_ci_lo", "arena_score_ci_hi", "arena_win_rate"} <= set(summary.columns)
    assert summary["arena_score"].mean() == pytest.approx(1000, abs=1e-6)
    arena = pd.read_csv(out / "arena.csv")
    assert list(arena["system"]) == ["bm25", "closed_book", "hybrid"] and (arena["arena_comparisons"] == 12).all()
    block = json.loads((out / "run_summary.json").read_text())["pairwise"]
    assert block["comparisons"] == 18 and block["calls"] == 0 and block["cost_usd"] == 0 and block["pairs"] == "all"
    assert set(block["win_rates"]) == {"bm25", "closed_book", "hybrid"}
    assert "Arena" in (out / "leaderboard.md").read_text() and "Pairwise arena" in (out / "leaderboard.md").read_text()
    assert "Pairwise arena" in (out / "report.html").read_text(encoding="utf-8")


def test_the_mock_arena_follows_the_heuristic_scores(tmp_path):
    from ragbench.evaluation.evaluator import run_benchmark

    out = run_benchmark(_config(tmp_path, THREE, {"enabled": True, "pairs": "vs_baseline"}), force_mock=True)
    block = json.loads((out / "run_summary.json").read_text())["pairwise"]
    assert block["pairs"] == "vs_baseline" and block["baseline"] == "bm25"  # every mock system costs $0: the first is the cheapest
    records = [json.loads(line) for line in (out / "pairwise.jsonl").read_text().splitlines()]
    assert all("bm25" in r["systems"] for r in records)


def test_pairwise_needs_two_systems():
    with pytest.raises(ValidationError, match="two systems"):
        load_config(_config(Path(tempfile.mkdtemp()), THREE[:1], {"enabled": True}))


def _estimate(tmp_path: Path, pairwise: dict):
    from ragbench.evaluation.estimate import estimate_run

    path = _config(tmp_path, THREE, pairwise)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["evaluation"]["judge_model"] = "fake-judge"
    path.write_text(yaml.safe_dump({**raw, "pricing": {"fake-judge": {"input": 100.0, "output": 300.0}}}), encoding="utf-8")
    return estimate_run(load_config(path))


@pytest.mark.parametrize(("pairwise", "pairs_per_question"), [({"enabled": True}, 3), ({"enabled": True, "max_pairs_per_question": 1}, 1), ({"enabled": True, "pairs": "vs_baseline"}, 2), ({"enabled": True, "pairs": "adaptive", "max_pairs_per_question": 2}, 2)])
def test_the_estimate_counts_every_pairwise_call(tmp_path, pairwise, pairs_per_question):
    plain = _estimate(tmp_path / "plain", {"enabled": False})
    estimate = _estimate(tmp_path / "pw", pairwise)
    assert plain.pairwise_calls == 0 and plain.pairwise_usd == 0
    assert estimate.pairwise_calls == 6 * pairs_per_question * 2 and estimate.pairwise_usd > 0
    assert estimate.total_usd == pytest.approx(plain.total_usd + estimate.pairwise_usd)


def test_ragbench_estimate_shows_the_pairwise_line(tmp_path):
    from typer.testing import CliRunner

    from ragbench.cli import app

    path = _config(tmp_path, THREE, {"enabled": True})
    result = CliRunner().invoke(app, ["estimate", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert "Pairwise judging (36 calls)" in result.output
