"""What a run evaluates: which systems, on which questions, and how many trials of each.

The run's scope is decided here and nowhere else. A feature that narrows or widens it (repeated trials, a dataset split, an
optimizer probing one system on a subset) changes the plan instead of adding a flag to the evaluator; `SystemRunner` runs whatever
plan it is given, and the outputs describe the plan's systems and questions.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace

from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.schema import Question


@dataclass(frozen=True)
class EvaluationPlan:
    systems: list[SystemConfig]  # in config order, which is the order of every output row
    questions: list[Question]  # in dataset order
    trials: int = 1  # times each system answers each question; the runner supports one trial so far

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ValueError(f"An evaluation plan needs at least one trial, got trials={self.trials}.")

    @classmethod
    def from_config(cls, config: ExperimentConfig, questions: list[Question]) -> EvaluationPlan:
        """Every configured system on the dataset's questions, the first `evaluation.max_questions` of them when that is set (0 = all)."""
        limit = config.evaluation.max_questions
        return cls(systems=list(config.systems), questions=questions[:limit] if limit else list(questions))

    @property
    def num_questions(self) -> int:
        return len(self.questions)

    def without_systems(self, names: Collection[str]) -> EvaluationPlan:
        """This plan minus the named systems (e.g. the ones an earlier attempt of the run already finished)."""
        return replace(self, systems=[cfg for cfg in self.systems if cfg.resolved_name not in names])
