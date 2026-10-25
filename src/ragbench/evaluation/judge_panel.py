"""A panel of LLM judges: each scores an answer independently, the panel's verdict aggregates them, and with `independent: true` a
judge never scores answers written by its own model family (self-preference).

A panel of one is the single judge, unchanged: its verdicts and rows are exactly what `AnswerJudge` gives.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any, Literal

from ragbench.datasets.schema import Question
from ragbench.evaluation.agreement import krippendorff_alpha
from ragbench.evaluation.answer_judge import SCORE_AXES, AnswerJudge, AnswerJudgeResult
from ragbench.evaluation.judge_prompts import JUDGE_PROMPT_VERSION
from ragbench.models.cost import CostBreakdown
from ragbench.models.family import model_family
from ragbench.rag_systems.base import RetrievedChunk

if TYPE_CHECKING:
    from ragbench.config.schema import EvaluationConfig

AGREEMENT_AXES = (*SCORE_AXES, "answer_score")


def scoring_judges(judges: list[str], generator: str | None, *, independent: bool) -> list[str]:
    """The judges that score answers from `generator`: all of them, minus its family's when `independent`. When every judge is of
    the generator's family nobody else could score those answers, so all of them do (and the report says so)."""
    if not independent:
        return list(judges)
    family = model_family(generator)
    others = [judge for judge in judges if model_family(judge) != family]
    return others or list(judges)


class JudgePanel:
    """The run's judges, one `AnswerJudge` per model ref in `refs` (in the same order)."""

    def __init__(self, judges: list[AnswerJudge], refs: list[str], *, independent: bool, aggregate: Literal["mean", "median"] = "mean"):
        self.judges = dict(zip(refs, judges, strict=True))
        self.refs = list(refs)
        self.independent = independent
        self.aggregate = aggregate

    @classmethod
    def from_config(cls, evaluation: EvaluationConfig, force_mock: bool) -> JudgePanel:
        # With the judge off the heuristic scores everything: one stand-in, not a panel of identical ones.
        refs = evaluation.judge_models if evaluation.judge_enabled else evaluation.judge_models[:1]
        judges = [
            AnswerJudge(model_name=ref, enabled=evaluation.judge_enabled, force_mock=force_mock, samples=evaluation.judge.samples, temperature=evaluation.judge.temperature)
            for ref in refs
        ]
        return cls(judges, refs, independent=evaluation.judge.independent, aggregate=evaluation.judge.aggregate)

    @property
    def lead(self) -> AnswerJudge:
        return self.judges[self.refs[0]]

    @property
    def enabled(self) -> bool:
        return self.lead.enabled

    @property
    def uses_llm(self) -> bool:
        """Whether judgments come from a model (False: the heuristic judge scores everything, by design)."""
        return any(judge.uses_llm for judge in self.judges.values())

    @property
    def is_panel(self) -> bool:
        return len(self.refs) > 1

    def model_names(self) -> list[str]:
        """The model names the judges call (the mock's when a judge is the mock)."""
        return [judge.llm.model_name for judge in self.judges.values()]

    def judge(self, question: Question, answer: str, contexts: list[RetrievedChunk], *, generator: str | None) -> AnswerJudgeResult:
        """The panel's verdict on an answer written by `generator` (a model ref), from the judges `scoring_judges` picks.

        Each judge scores independently; a judge whose verdict was unusable (it fell back to the heuristic) is left out of the
        aggregate. Only when every judge failed does the heuristic score the answer (`fallback: True`).
        """
        if not self.is_panel:
            return self.lead.judge(question, answer, contexts)
        chosen = scoring_judges(self.refs, generator, independent=self.independent)
        verdicts = {ref: self.judges[ref].judge(question, answer, contexts) for ref in chosen}
        usable = {ref: verdict for ref, verdict in verdicts.items() if not verdict.metadata.get("fallback")}
        cost = CostBreakdown()
        for verdict in verdicts.values():
            cost = cost.plus(verdict.cost)
        bookkeeping: dict[str, Any] = {
            "prompt_version": JUDGE_PROMPT_VERSION,
            "parse_error": any(v.metadata.get("parse_error") for v in verdicts.values()),
            "retries": sum(int(v.metadata.get("retries") or 0) for v in verdicts.values()),
            "judges": chosen,
            "sat_out": [ref for ref in self.refs if ref not in verdicts],
            "member_fallbacks": len(verdicts) - len(usable),
            "aggregate": self.aggregate,
            "panel_scores": {ref: _scores(usable[ref]) if ref in usable else None for ref in self.refs},
        }
        if not usable:
            result = next(iter(verdicts.values()))  # every judge fell back: the heuristic's verdict, the same for each of them
            result.cost = cost
            result.metadata = {"judge": "heuristic", "fallback": True, **bookkeeping}
            return result
        result = _aggregate(list(usable.values()), self.aggregate)
        result.cost = cost
        result.metadata = {"judge": "panel", "fallback": False, **bookkeeping}
        return result


def _scores(verdict: AnswerJudgeResult) -> dict[str, float]:
    return {**{axis: getattr(verdict, axis) for axis in SCORE_AXES}, "answer_score": verdict.answer_score}


def _aggregate(verdicts: list[AnswerJudgeResult], how: Literal["mean", "median"]) -> AnswerJudgeResult:
    """One verdict from the judges': the mean (or median) of each axis; a flag is set when most judges set it (a hallucination flag at half)."""
    combine = statistics.fmean if how == "mean" else statistics.median
    count = len(verdicts)
    return AnswerJudgeResult(
        **{axis: float(combine([getattr(verdict, axis) for verdict in verdicts])) for axis in SCORE_AXES},
        is_supported_by_context=sum(v.is_supported_by_context for v in verdicts) * 2 > count,
        is_hallucinated=sum(v.is_hallucinated for v in verdicts) * 2 >= count,
        reasoning=verdicts[0].reasoning,
        raw_output=verdicts[0].raw_output,
    )


def panel_agreement(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Krippendorff's alpha between the panel's judges, per axis, over every answer they scored (a judge that sat out an answer, or
    whose verdict was unusable, is a missing value). Empty when the rows were not judged by a panel."""
    scored = [row["answer_judge"]["metadata"]["panel_scores"] for row in rows if row.get("error") is None and (row.get("answer_judge") or {}).get("metadata", {}).get("panel_scores")]
    if not scored:
        return {}
    judges = list(scored[0])
    return {axis: krippendorff_alpha([[None if scores.get(judge) is None else scores[judge][axis] for judge in judges] for scores in scored]) for axis in AGREEMENT_AXES}
