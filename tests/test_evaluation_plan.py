from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragbench.config.schema import ExperimentConfig
from ragbench.datasets.schema import Question
from ragbench.evaluation.plan import EvaluationPlan

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"
QUESTIONS = [Question(id=f"q{i}", question=f"What is topic {i}?") for i in range(1, 6)]


def _config(**evaluation: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "run": {"name": "plan", "output_dir": "results"},
            "dataset": {"documents_path": str(DEMO / "docs"), "questions_path": str(DEMO / "questions.jsonl")},
            "systems": [{"type": "bm25", "name": "b"}, {"type": "vector", "name": "v"}, {"type": "hybrid", "name": "h"}],
            "evaluation": evaluation,
        }
    )


def test_the_plan_from_a_config_is_every_system_in_config_order_on_every_question_once():
    plan = EvaluationPlan.from_config(_config(), QUESTIONS)

    assert [cfg.resolved_name for cfg in plan.systems] == ["b", "v", "h"]
    assert plan.questions == QUESTIONS and plan.num_questions == 5
    assert plan.trials == 1


@pytest.mark.parametrize(("max_questions", "expected"), [(2, ["q1", "q2"]), (None, ["q1", "q2", "q3", "q4", "q5"]), (0, ["q1", "q2", "q3", "q4", "q5"])])
def test_max_questions_keeps_the_first_questions_and_zero_or_unset_keeps_all(max_questions, expected):
    plan = EvaluationPlan.from_config(_config(max_questions=max_questions), QUESTIONS)
    assert [question.id for question in plan.questions] == expected


def test_without_systems_drops_only_the_named_systems():
    plan = EvaluationPlan.from_config(_config(), QUESTIONS)

    rest = plan.without_systems({"v"})

    assert [cfg.resolved_name for cfg in rest.systems] == ["b", "h"]
    assert rest.questions == plan.questions and rest.trials == plan.trials
    assert [cfg.resolved_name for cfg in plan.systems] == ["b", "v", "h"], "the original plan is unchanged"


def test_a_plan_needs_at_least_one_trial():
    with pytest.raises(ValueError, match="trials"):
        EvaluationPlan(systems=[], questions=[], trials=0)


def test_the_runner_runs_a_plan_and_refuses_trials_it_cannot_run_yet():
    from ragbench.runtime.executor import ExecutionSettings, SystemRunner
    from ragbench.runtime.progress import ProgressListener

    runner = SystemRunner(object(), ExecutionSettings(), ProgressListener())  # type: ignore[arg-type]  # never called: no systems
    assert runner.run_all(EvaluationPlan(systems=[], questions=QUESTIONS), documents=[]) == []
    with pytest.raises(ValueError, match="one trial"):
        runner.run_all(EvaluationPlan(systems=[], questions=QUESTIONS, trials=2), documents=[])
