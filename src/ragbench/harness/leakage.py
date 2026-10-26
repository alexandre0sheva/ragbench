"""Few-shot examples are part of what a method may adapt, so they may come from the train split only (the leakage guard's rule)."""

from __future__ import annotations

from collections.abc import Iterator

from ragbench.config.schema import ExperimentConfig, SystemConfig, _route_configs
from ragbench.datasets.leakage import LeakageGuard
from ragbench.datasets.loader import configured_splits_path, load_dataset
from ragbench.datasets.splits import load_splits
from ragbench.harness.schema import HarnessPatch


def _few_shot(name: str, system: SystemConfig) -> Iterator[tuple[str, str]]:
    """`(where, text)` of every few-shot example in `system` and its routes."""
    for index, example in enumerate(HarnessPatch.model_validate(system.harness).few_shot or []):
        yield f"{name}.harness.few_shot[{index}]", f"{example.question}\n{example.answer}"
    for route, config in _route_configs(system):
        yield from _few_shot(f"{name}.routes.{route}", config)


def assert_few_shot_from_train(config: ExperimentConfig) -> None:
    """Raise `LeakageError` when a few-shot example holds a dev or test question (its id, text or reference answer). A dataset without a
    `split:` has no train split, so then no question of it may be an example."""
    examples = [found for system in config.systems for found in _few_shot(system.resolved_name, system)]
    if not examples:
        return
    dataset = load_dataset(config.dataset.questions_path, config.dataset.qrels_path)
    if config.dataset.split == "all":
        guard = LeakageGuard(dataset.questions)
    else:
        guard = LeakageGuard.for_splits(dataset, load_splits(configured_splits_path(config.dataset)), held_out=("dev", "test"))
    for where, text in examples:
        guard.assert_text_clean(text, where)
