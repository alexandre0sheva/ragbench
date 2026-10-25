from __future__ import annotations

import json
import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fake_systems import RankedFakeSystem, question, register_fakes, write_experiment

from ragbench.cache import CacheRuntime, DiskCache, activate_cache
from ragbench.cache.trial import current_trial, trial_key, trial_scope
from ragbench.config.schema import CacheConfig, EvaluationConfig, SystemConfig
from ragbench.datasets.schema import Question
from ragbench.evaluation.compare_runs import compare_runs
from ragbench.evaluation.evaluator import BenchmarkEvaluator, run_benchmark
from ragbench.evaluation.stats import (
    QUANTILE_MIN_N,
    bootstrap_ci,
    grouped_bootstrap_ci,
    grouped_paired_bootstrap,
    paired_bootstrap,
    pass_rates,
    quantile_ci,
)
from ragbench.evaluation.summary import build_summary
from ragbench.models.cached import CachedLLM
from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import formats
from ragbench.models.decisions.base import Predicate
from ragbench.models.decisions.cached import CachedDecisionModel
from ragbench.models.decisions.mock import MockDecisionModel
from ragbench.models.llms import LLM, LLMResult
from ragbench.rag_systems import SYSTEM_REGISTRY
from ragbench.reporting.report_data import build_report
from ragbench.runtime.executor import AUTO_PROBE_MAX, AUTO_PROBE_START, ExecutionSettings, SystemOutcome, SystemRunner
from ragbench.runtime.parallel import ordered_parallel_map
from ragbench.runtime.progress import ProgressListener
from ragbench.selection.recommend import recommend
from ragbench.tools.base import ToolContext, ToolResult
from ragbench.tools.registry import _run_with_timeout

# --- config ----------------------------------------------------------------------------------


def test_trials_default_to_one_and_must_be_positive():
    assert EvaluationConfig().trials == 1
    assert EvaluationConfig(trials=3).trials == 3
    with pytest.raises(ValueError):
        EvaluationConfig(trials=0)


def test_the_latency_probe_size_can_be_auto():
    assert EvaluationConfig(latency_probe_questions="auto").latency_probe_questions == "auto"
    for bad in ("many", -1):
        with pytest.raises(ValueError):
            EvaluationConfig(latency_probe_questions=bad)


# --- the trial index in the cache key --------------------------------------------------------


class _CountingLLM(LLM):
    model_name = "mock-llm"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, **kwargs) -> LLMResult:
        self.calls += 1
        return LLMResult(text=f"answer {self.calls}", model=self.model_name, prompt_tokens=10, completion_tokens=2, cost=CostBreakdown())


MESSAGES = [{"role": "user", "content": "what is 2+2?"}]


def test_trial_zero_keeps_the_cache_key_of_a_single_trial_run():
    assert current_trial() == 0 and trial_key() == {}
    with trial_scope(0):
        assert trial_key() == {}
    with trial_scope(2):
        assert current_trial() == 2 and trial_key() == {"trial": 2}
    assert current_trial() == 0


def test_a_later_trial_is_a_fresh_call_not_a_replay_of_the_first(tmp_path):
    inner = _CountingLLM()
    llm = CachedLLM(inner)
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "c.sqlite3"), CacheConfig())):
        first = llm.generate(MESSAGES)
        with trial_scope(1):
            second = llm.generate(MESSAGES)
            again = llm.generate(MESSAGES)
        replay = llm.generate(MESSAGES)
    assert inner.calls == 2
    assert not first.cached and not second.cached and second.text != first.text
    assert again.cached and again.text == second.text  # the same trial still hits (a resumed or repeated run is free)
    assert replay.cached and replay.text == first.text


def test_a_later_trial_is_a_fresh_decision_too(tmp_path):
    inner = MockDecisionModel()
    calls: list[str] = []
    original = inner.decide
    inner.decide = lambda text, predicates: calls.append(text) or original(text, predicates)  # type: ignore[method-assign]
    cached = CachedDecisionModel(inner)
    text, predicates = formats.single_input("q?", "a passage"), [Predicate(formats.SINGLE, "is it?")]
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "c.sqlite3"), CacheConfig())):
        cached.decide(text, predicates)
        with trial_scope(1):
            cached.decide(text, predicates)
        cached.decide(text, predicates)
    assert len(calls) == 2


def test_worker_threads_started_inside_a_trial_see_that_trial():
    with trial_scope(3):
        seen = ordered_parallel_map(lambda _: current_trial(), range(8), workers=4)
    assert seen == [3] * 8


# --- two-level statistics --------------------------------------------------------------------

RNG = np.random.default_rng(7)
SINGLE = [[float(x)] for x in RNG.uniform(0, 5, size=40)]


def test_with_one_trial_per_question_the_grouped_bootstrap_is_the_plain_one():
    flat = np.array([group[0] for group in SINGLE])
    assert grouped_bootstrap_ci(SINGLE, n_boot=500, seed=3) == bootstrap_ci(flat, n_boot=500, seed=3)
    b = [[value + 0.3 * (i % 3 - 1)] for i, (value,) in enumerate(SINGLE)]
    assert grouped_paired_bootstrap(SINGLE, b, n_boot=500, seed=3) == paired_bootstrap(flat, np.array([g[0] for g in b]), n_boot=500, seed=3)


def test_a_question_carries_all_its_trials_when_resampled():
    # Cluster bootstrap: resampling questions, each with its trials, is the bootstrap of the per-question means.
    rng = np.random.default_rng(4)
    groups = [[float(v) for v in rng.normal(m, 1.5, size=3)] for m in rng.uniform(1.5, 3.5, size=40)]
    means = np.array([np.mean(group) for group in groups])
    assert grouped_bootstrap_ci(groups, n_boot=2000, seed=0) == bootstrap_ci(means, n_boot=2000, seed=0)
    other = [[v + 0.2 for v in group[:2]] for group in groups]
    expected = paired_bootstrap(means, np.array([np.mean(group) for group in other]), n_boot=2000, seed=0)
    assert grouped_paired_bootstrap(groups, other, n_boot=2000, seed=0) == expected


def test_trial_noise_widens_the_interval_against_a_steady_system_with_the_same_expected_scores():
    rng = np.random.default_rng(5)
    expected = rng.uniform(1.5, 3.5, size=60)
    steady = [[float(m)] * 3 for m in expected]
    noisy = [[float(v) for v in rng.normal(m, 1.5, size=3)] for m in expected]
    _, steady_lo, steady_hi = grouped_bootstrap_ci(steady, n_boot=2000, seed=0)
    _, noisy_lo, noisy_hi = grouped_bootstrap_ci(noisy, n_boot=2000, seed=0)
    assert noisy_hi - noisy_lo > 1.3 * (steady_hi - steady_lo)
    paired = grouped_paired_bootstrap(noisy, steady, n_boot=2000, seed=0)
    assert paired.n == 60 and paired.ci_hi - paired.ci_lo > 0.3  # the steady side has no noise of its own: the width is the trial noise


def test_missing_trials_and_questions_are_dropped_not_zeroed():
    groups = [[1.0, math.nan, 3.0], [], [math.nan], [2.0, 2.0]]
    mean, _, _ = grouped_bootstrap_ci(groups, n_boot=200)
    assert mean == pytest.approx((2.0 + 2.0) / 2)
    assert math.isnan(grouped_bootstrap_ci([[], [math.nan]])[0])


def test_pass_at_k_counts_any_correct_trial_and_pass_hat_k_needs_all_of_them():
    correct = [[True, True, True], [True, False, True], [False, False, False], [False, True, False]]
    assert pass_rates(correct) == (pytest.approx(3 / 4), pytest.approx(1 / 4))
    assert pass_rates([]) == (None, None)


def test_the_p95_interval_is_exact_and_has_no_upper_bound_until_there_are_enough_measurements():
    assert QUANTILE_MIN_N == 72  # the fewest values whose largest is a 97.5% upper bound on the p95: 1 - 0.95**n >= 0.975
    few = list(np.linspace(100.0, 200.0, 50))
    p95, lo, hi = quantile_ci(few, 95)
    assert p95 == pytest.approx(float(np.percentile(few, 95))) and lo is not None and lo <= p95 and hi is None
    many = list(np.linspace(100.0, 200.0, 200))
    p95, lo, hi = quantile_ci(many, 95)
    assert lo is not None and hi is not None and lo <= p95 <= hi
    assert quantile_ci([], 95) == (None, None, None)
    assert quantile_ci([5.0], 95) == (5.0, None, None)


def test_the_p95_interval_covers_the_true_p95_as_often_as_it_says():
    rng = np.random.default_rng(11)
    true_p95 = float(np.exp(0.3 * 1.6448536269514722))  # lognormal(0, 0.3)
    covered = 0
    for _ in range(400):
        _, lo, hi = quantile_ci(list(rng.lognormal(0.0, 0.3, size=120)), 95)
        covered += lo is not None and hi is not None and lo <= true_p95 <= hi
    assert covered / 400 >= 0.93  # exact coverage is at least 95%; a percentile bootstrap of the p95 falls short of that at this size


# --- end to end ------------------------------------------------------------------------------


class CoinFlipSystem(RankedFakeSystem):
    """Right or wrong by a fixed scramble of (question, trial): some questions right on every trial, most on one or two."""

    def _generate_answer(self, question: str, chunks):
        index = int(question.rstrip("?").split()[-1])  # "What is topic 3?"
        right = (index * 5 + current_trial() * 3) % 7 not in (0, 1)
        text = f"Text about topic {index}. [doc_{index:03d}]" if right else "Bananas are yellow."
        return LLMResult(text=text, model="mock-llm", prompt_tokens=0, completion_tokens=0, cost=CostBreakdown())


QUESTIONS = [question(f"q{i}", f"What is topic {i}?", [f"doc_{i:03d}"]) for i in range(1, 10)]
SYSTEMS = [{"type": "fake_ranked", "name": "steady", "retrieval": {"top_k": 3}}, {"type": "fake_coinflip", "name": "flaky", "retrieval": {"top_k": 3}}]


def _run(tmp_path, monkeypatch, **evaluation):
    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_coinflip", CoinFlipSystem)
    config = write_experiment(tmp_path, QUESTIONS, SYSTEMS, {"latency_probe_questions": 0, **evaluation})
    out = run_benchmark(config, force_mock=True)
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    return out, rows, pd.read_csv(out / "metrics_summary.csv").set_index("system")


def test_one_trial_adds_no_trial_fields_or_columns(tmp_path, monkeypatch):
    out, rows, summary = _run(tmp_path, monkeypatch)
    assert len(rows) == 2 * len(QUESTIONS) and all("trial" not in row for row in rows)
    assert not {"answer_score_trial_sd", "pass_at_k", "pass_hat_k", "rank_stability"} & set(summary.columns)
    assert "trial" not in pd.read_csv(out / "answer_metrics.csv").columns
    assert "trials" not in json.loads((out / "stats.json").read_text())


def test_every_trial_is_a_fresh_answer_and_the_summary_shows_how_much_they_vary(tmp_path, monkeypatch):
    out, rows, summary = _run(tmp_path, monkeypatch, trials=3)

    assert len(rows) == 2 * len(QUESTIONS) * 3
    flaky = [row for row in rows if row["system"] == "flaky"]
    assert [(row["question_id"], row["trial"]) for row in flaky[:4]] == [("q1", 0), ("q1", 1), ("q1", 2), ("q2", 0)]
    assert {row["answer"] for row in flaky if row["question_id"] == "q1"} == {"Text about topic 1. [doc_001]", "Bananas are yellow."}
    assert pd.read_csv(out / "answer_metrics.csv")["trial"].tolist()[:3] == [0, 1, 2]

    steady, noisy = summary.loc["steady"], summary.loc["flaky"]
    right = pd.DataFrame(flaky).assign(ok=lambda f: f["answer_judge"].map(lambda j: j["answer_score"] >= 4.0)).groupby("question_id")["ok"]
    assert noisy["pass_at_k"] == pytest.approx(right.any().mean()) and noisy["pass_hat_k"] == pytest.approx(right.all().mean())
    assert noisy["pass_hat_k"] < noisy["pass_at_k"]
    assert steady["pass_at_k"] == steady["pass_hat_k"]  # a deterministic system is as reliable as it is good
    assert noisy["answer_score_trial_sd"] > 1.0 and steady["answer_score_trial_sd"] == 0.0
    assert 0.0 <= noisy["rank_stability"] <= 1.0 and steady["n_ok"] == 3 * len(QUESTIONS)

    # The interval resamples questions with all their trials: the bootstrap of the per-question means, in question order.
    means = pd.DataFrame(flaky).assign(score=lambda f: f["answer_judge"].map(lambda j: j["answer_score"])).groupby("question_id", sort=False)["score"].mean()
    _, lo, hi = bootstrap_ci(means.to_numpy(), n_boot=2000, seed=0)
    assert (noisy["answer_score_ci_lo"], noisy["answer_score_ci_hi"]) == pytest.approx((lo, hi)) and hi > lo
    assert noisy["answer_score"] == pytest.approx(means.mean())

    stats = json.loads((out / "stats.json").read_text())
    assert stats["trials"] == 3 and "trials" in stats["method"] and stats["systems"]["flaky"]["n"] == len(QUESTIONS)
    assert json.loads((out / "run_summary.json").read_text())["num_question_rows"] == len(rows)


def test_an_error_rate_counts_answers_over_all_trials(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    questions = [*QUESTIONS[:4], question("q9", "Please explode now?", [])]
    config = write_experiment(tmp_path, questions, [{"type": "fake_exploding", "name": "boom", "retrieval": {"top_k": 3}}], {"trials": 2, "latency_probe_questions": 0, "max_error_rate": 0.25})
    out = run_benchmark(config, force_mock=True)  # 2 of 10 answers fail: 20%, under the 25% allowed
    summary = pd.read_csv(out / "metrics_summary.csv").iloc[0]
    assert (summary["n_ok"], summary["n_error"]) == (8, 2)
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    right = {row["question_id"] for row in rows if row["error"] is None and row["answer_judge"]["answer_score"] >= 4.0}
    assert summary["pass_hat_k"] == summary["pass_at_k"] == pytest.approx(len(right) / 5)  # q9 always errors: never correct, still counted


def test_a_summary_score_is_the_mean_of_question_means_when_trials_are_uneven():
    answers = pd.DataFrame(
        [
            {"system": "s", "question_id": "q1", "category": "c", "trial": 0, "answer_score": 0.0, "token_f1": 0.0, "failure_type": "x"},
            {"system": "s", "question_id": "q1", "category": "c", "trial": 1, "answer_score": 4.0, "token_f1": 1.0, "failure_type": "x"},
            {"system": "s", "question_id": "q2", "category": "c", "trial": 0, "answer_score": 4.0, "token_f1": None, "failure_type": "x"},
        ]
    )
    for col in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality"]:
        answers[col] = answers["answer_score"]
    retrieval = answers[["system", "question_id", "category", "trial"]].assign(**{"recall@5": [0.0, 1.0, 1.0]})
    rows = [{"system": "s", "error": None, "latency_ms": 1.0, "answer_judge": {"metadata": {}}, "steps": [], "agent": None, "gate": None, "refused": False, "answerable": True}] * 3
    summary = build_summary([SystemConfig(type="bm25", name="s")], retrieval, answers, pd.DataFrame(), rows, None, probe_results={}, judge=None)[0]
    assert summary["answer_score"] == pytest.approx((2.0 + 4.0) / 2)  # not (0 + 4 + 4) / 3: q1's second trial must not count double
    assert summary["retrieval_recall@5"] == pytest.approx((0.5 + 1.0) / 2)
    assert summary["token_f1"] == pytest.approx(0.5)  # q2 has no reference to score against: missing, not zero


class FirstTrialWrongSystem(RankedFakeSystem):
    """Wrong on every question's first trial, right on every later one."""

    def _generate_answer(self, question: str, chunks):
        index = int(question.rstrip("?").split()[-1])
        text = "Bananas are yellow." if current_trial() == 0 else f"Text about topic {index}. [doc_{index:03d}]"
        return LLMResult(text=text, model="mock-llm", prompt_tokens=0, completion_tokens=0, cost=CostBreakdown())


def _first_wrong_run(root, monkeypatch, trials: int):
    root.mkdir()
    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_first_wrong", FirstTrialWrongSystem)
    systems = [{"type": "fake_first_wrong", "name": "late", "retrieval": {"top_k": 3}}, SYSTEMS[0]]
    return run_benchmark(write_experiment(root, QUESTIONS, systems, {"latency_probe_questions": 0, "trials": trials}), force_mock=True)


def test_the_recommendation_and_the_run_diff_use_every_trial_not_the_last(tmp_path, monkeypatch):
    once, thrice = _first_wrong_run(tmp_path / "one", monkeypatch, 1), _first_wrong_run(tmp_path / "three", monkeypatch, 3)
    expected = pd.read_csv(thrice / "metrics_summary.csv").set_index("system").loc["late", "answer_score"]
    assert expected == pytest.approx(4.75 * 2 / 3)

    ranking = {entry.system: entry for entry in recommend(thrice).ranking}
    assert ranking["late"].quality == pytest.approx(expected)

    late = next(system for system in compare_runs(once, thrice).systems if system.system == "late")
    assert late.paired_questions == len(QUESTIONS) and late.paired_diff == pytest.approx(expected)  # trial 0 scored 0 in the one-trial run


# --- latency ---------------------------------------------------------------------------------


class TimedSystem(RankedFakeSystem):
    """Reports a scripted latency: `latency(n)` milliseconds on its n-th call."""

    latency: Callable[[int], float] = staticmethod(lambda n: 100.0)
    calls = 0

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None):
        result = super().answer_question(question, top_k=top_k, context_k=context_k)
        cls = type(self)
        cls.calls += 1
        return result.model_copy(update={"latency_ms": cls.latency(cls.calls)})


def _probed(tmp_path, monkeypatch, latency: Callable[[int], float]) -> dict:
    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_timed", TimedSystem)
    TimedSystem.latency, TimedSystem.calls = staticmethod(latency), 0
    config = write_experiment(tmp_path, QUESTIONS, [{"type": "fake_timed", "name": "t", "retrieval": {"top_k": 3}}], {"latency_probe_questions": "auto"})
    evaluator = BenchmarkEvaluator(config, force_mock=True, use_cache=False)
    evaluator.mode = "live"  # models stay mock; only the probe decision reads the mode
    out = evaluator.run()
    return json.loads((out / "run_summary.json").read_text())["execution"] | {"row": pd.read_csv(out / "metrics_summary.csv").iloc[0].to_dict()}


def test_an_auto_probe_stops_once_the_p95_is_pinned_down(tmp_path, monkeypatch):
    steady = _probed(tmp_path, monkeypatch, lambda n: 100.0)
    assert steady["latency_probe_questions"] == "auto" and steady["probe_calls"] == AUTO_PROBE_START == QUANTILE_MIN_N
    assert steady["row"]["latency_ms_p95_ci_lo"] == steady["row"]["latency_ms_p95_ci_hi"] == 100.0


def test_an_auto_probe_grows_up_to_its_cap_while_the_p95_stays_uncertain(tmp_path, monkeypatch):
    noisy = _probed(tmp_path, monkeypatch, lambda n: 1.5**n)  # every call slower than the last: the top of the sample never settles
    assert noisy["probe_calls"] == AUTO_PROBE_MAX  # more than the 9 questions: questions are re-asked to get more measurements
    row = noisy["row"]
    assert row["latency_source"] == "probe" and row["latency_ms_p95_ci_lo"] < row["latency_ms_p95"] <= row["latency_ms_p95_ci_hi"]


def test_every_run_reports_an_interval_on_its_p95_open_ended_when_there_are_few_answers(tmp_path, monkeypatch):
    _, _, few = _run(tmp_path, monkeypatch)
    assert all(row["latency_ms_p95_ci_lo"] <= row["latency_ms_p95"] and pd.isna(row["latency_ms_p95_ci_hi"]) for _, row in few.iterrows())
    (tmp_path / "more").mkdir()
    _, _, many = _run(tmp_path / "more", monkeypatch, trials=8)  # 72 answers per system
    assert all(row["latency_ms_p95_ci_lo"] <= row["latency_ms_p95"] <= row["latency_ms_p95_ci_hi"] for _, row in many.iterrows())


def test_the_question_explorer_and_the_qrels_audit_list_each_answer_once(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch, trials=3)
    built = build_report(out).questions
    assert built is not None
    page = built.embedded
    assert len(page["questions"]) == len(QUESTIONS)
    assert all(len(entry["runs"]) == 2 for entry in page["questions"])  # one per system: the first trial's answer
    audit = pd.read_csv(out / "qrels_audit.csv") if (out / "qrels_audit.csv").stat().st_size > 1 else pd.DataFrame(columns=["system", "question_id"])
    assert not audit.duplicated(["system", "question_id"]).any()


def test_identical_trials_have_a_spread_of_exactly_zero():
    from ragbench.evaluation.run_stats import _trial_fields

    rows = [{"system": "s", "question_id": "q", "error": None, "answer_judge": {"answer_score": 3.7}} for _ in range(3)]
    assert _trial_fields("s", rows, {"q": [3.7] * 3})["answer_score_trial_sd"] == 0.0  # np.std gives 5e-16 here, from the mean's rounding


class _BudgetedEvaluator:
    """Stands in for the evaluator during a probe: every probe answer costs 0.1, and the cap is 0.25."""

    def __init__(self) -> None:
        self.spent = 0.0

    def budget_exhausted(self) -> bool:
        return self.spent >= 0.25

    def charge(self, usd: float) -> None:
        self.spent += usd

    def answer_for_probe(self, system, question):
        return SimpleNamespace(latency_ms=10.0, cost=CostBreakdown(llm_cost=0.1))


def test_the_probe_is_charged_to_the_spending_cap_and_stops_when_it_is_reached():
    evaluator = _BudgetedEvaluator()
    runner = SystemRunner(evaluator, ExecutionSettings(latency_probe_questions=5), ProgressListener())  # type: ignore[arg-type]
    questions = [Question(id=f"q{i}", question=f"q{i}?") for i in range(10)]
    results = [{"per_question": {"question_id": q.id, "error": None}} for q in questions]
    outcome = SystemOutcome(1, "s", "fake", object(), None, results, {})  # type: ignore[arg-type]
    probe = runner.probe([outcome], questions)["s"]
    assert probe.calls == 3 and evaluator.spent == pytest.approx(0.3)  # the third call crossed the cap; no fourth was started


def test_a_tool_runs_in_the_trial_of_the_answer_that_called_it():
    class TrialTool:
        spec = None
        side_effects = None

        def run(self, args, ctx):
            return ToolResult(text=str(current_trial()))

    with trial_scope(2):
        result = _run_with_timeout("trial_tool", TrialTool(), {}, ToolContext(corpus=None), timeout=5.0)  # type: ignore[arg-type]
    assert result.text == "2"  # a custom tool that calls a model must not replay trial 0's cache entry


def test_trials_are_counted_as_questions_where_a_table_says_questions():
    from ragbench.datasets.pooling import merge_trial_rankings
    from ragbench.evaluation.route_metrics import route_rows
    from ragbench.evaluation.tool_metrics import usage_rows

    tool = {"kind": "tool", "name": "calculator", "latency_ms": 1.0, "cost": {"total_cost": 0.0}, "metadata": {}}
    answers = [{"question_id": q, "trial": t, "steps": [tool], "route": "lexical", "answer_judge": {"answer_score": 4.0}, "cost": {"total_cost": 0.0}}
               for q in ("q1", "q2") for t in range(3)]
    (row,) = usage_rows("agent", ["calculator"], answers)
    assert (row["calls"], row["questions_using"]) == (6, 2)  # six calls, from two questions answered three times each
    (route,) = route_rows("adaptive", answers)
    assert route["questions"] == 2 and route["share"] == 1.0
    # Pooling for `ragbench label`: every trial's documents, each at its best rank, not only the last trial's.
    assert merge_trial_rankings([["d1", "d2"], ["d3", "d1"], ["d4", "d2"]]) == ["d1", "d3", "d4", "d2"]


def test_the_run_summary_and_the_report_name_the_number_of_trials(tmp_path, monkeypatch):
    out, _, _ = _run(tmp_path, monkeypatch, trials=3)
    assert json.loads((out / "run_summary.json").read_text())["trials"] == 3
    headers = {column["key"]: column["header"] for column in build_report(out).data["leaderboard"]["columns"]}
    assert headers["pass_at_k"] == "pass@3" and headers["pass_hat_k"] == "pass^3"
    (tmp_path / "one").mkdir()
    one, _, _ = _run(tmp_path / "one", monkeypatch)
    assert "trials" not in json.loads((one / "run_summary.json").read_text())  # one trial: the file is as before
