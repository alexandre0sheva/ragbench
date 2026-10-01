"""`ragbench label`: propose relevance labels for a finished run by *pooling*.

For every question, the documents that any system in the run ranked in its top k are pooled (the classic TREC method); an LLM grades
each pooled document 0-3 against the question; the grades become proposed qrels. Only documents that were pooled *and* graded ever get a
proposed label, human labels are never overwritten, and the review file shows where the grader and the existing labels disagree.

Pooling is biased toward what the benchmarked systems can find: a relevant document that no system retrieved is never judged. See
docs/methodology.md#pooling-bias.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ragbench.config.schema import ExperimentConfig
from ragbench.datasets import prompts
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.schema import Dataset, Question
from ragbench.documents.loaders import load_dataset_documents
from ragbench.documents.schema import Document
from ragbench.evaluation.budget import BudgetGuard
from ragbench.models import cost as pricing
from ragbench.models.llms import LLM
from ragbench.utils.jsonl import read_jsonl, write_jsonl
from ragbench.utils.query_planning import loads_lenient
from ragbench.utils.text import estimate_tokens

logger = logging.getLogger(__name__)

GRADE_REPLY_TOKENS = 40
GRADES = (0, 1, 2, 3)
RETRY_REMINDER = 'That reply was not a valid grade. Reply with one JSON object only: {"grade": 0, "reason": "..."} with grade an integer from 0 to 3.'
PROPOSED_NAME = "qrels.proposed.jsonl"
REVIEW_NAME = "qrels_review.md"
MERGED_NAME = "qrels.merged.jsonl"


class LabelError(ValueError):
    """The run cannot be labeled (missing files, dataset moved, nothing to grade)."""


@dataclass
class Judgment:
    grade: int
    reason: str


@dataclass
class RunData:
    """What `label` needs from a finished run directory and the dataset its config points at."""

    run_dir: Path
    config: ExperimentConfig
    documents: dict[str, Document]
    dataset: Dataset
    rankings: dict[str, dict[str, list[str]]]  # question id -> system -> distinct documents in rank order
    flagged_gaps: dict[str, list[str]]  # question id -> systems whose failure type was `possible_qrels_gap`


def load_run(run_dir: Path) -> RunData:
    results, config_path = run_dir / "per_question_results.jsonl", run_dir / "config.yaml"
    if not results.exists() or not config_path.exists():
        raise LabelError(f"{run_dir} is not a finished run directory (it needs per_question_results.jsonl and config.yaml).")
    config = ExperimentConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    try:
        documents = load_dataset_documents(config.dataset, warnings=[])
        dataset = load_dataset(config.dataset.questions_path, config.dataset.qrels_path)
    except (FileNotFoundError, ValueError) as exc:
        raise LabelError(f"The run's dataset can no longer be loaded ({exc}). Run `ragbench label` from the directory the run used, or restore the files.") from exc
    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)
    flagged: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(results):
        if row.get("error") is not None:
            continue
        ranked: list[str] = []
        for context in row.get("retrieved_contexts", []):
            if context["doc_id"] not in ranked:
                ranked.append(context["doc_id"])
        rankings[row["question_id"]][row["system"]] = ranked
        if row.get("failure_type") == "possible_qrels_gap":
            flagged[row["question_id"]].append(row["system"])
    return RunData(run_dir, config, {document.doc_id: document for document in documents}, dataset, dict(rankings), dict(flagged))


@dataclass
class PoolPlan:
    questions: list[Question]  # the questions that will be graded, in file order
    pool: dict[str, dict[str, dict[str, int]]]  # question id -> pooled document -> {system: its rank of the document}
    skipped_unanswerable: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def pairs(self) -> int:
        return sum(len(docs) for docs in self.pool.values())


def build_pool(run: RunData, top_k: int) -> PoolPlan:
    """Per question, the union of every system's `top_k` distinct documents. Unanswerable questions are skipped (nothing should be relevant)."""
    if top_k < 1:
        raise LabelError("--top-k must be at least 1")
    plan = PoolPlan([], {})
    for question in run.dataset.questions:
        systems = run.rankings.get(question.id)
        if not systems:
            continue
        if not question.is_answerable:
            plan.skipped_unanswerable += 1
            continue
        pooled: dict[str, dict[str, int]] = {}
        for system, ranked in systems.items():
            for rank, doc_id in enumerate(ranked[:top_k], start=1):
                if doc_id in run.documents:
                    pooled.setdefault(doc_id, {})[system] = rank
                elif doc_id not in pooled:
                    plan.warnings.append(f"{doc_id} (retrieved for {question.id}) is not in the dataset's documents; not graded")
        if pooled:
            plan.questions.append(question)
            plan.pool[question.id] = dict(sorted(pooled.items(), key=lambda item: (min(item[1].values()), item[0])))
    if not plan.pool:
        raise LabelError("Nothing to grade: no answerable question of the run retrieved a document.")
    return plan


def grading_prompt(run: RunData, question: Question, doc_id: str) -> str:
    document = run.documents[doc_id]
    return prompts.grade_prompt(question.question, doc_id, document.title, document.text)


def estimate_grading_cost(run: RunData, plan: PoolPlan, model: str) -> float:
    """Dollars the grading calls are expected to cost at `model`'s price: the real prompts, a short reply each."""
    prompt_tokens = sum(estimate_tokens(grading_prompt(run, question, doc_id)) for question in plan.questions for doc_id in plan.pool[question.id])
    return pricing.estimate_model_cost(model, prompt_tokens, plan.pairs * GRADE_REPLY_TOKENS)


def parse_grade(text: str) -> Judgment | None:
    parsed = loads_lenient(text)
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("grade")
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        grade = int(float(value))
    except ValueError:
        return None
    if grade not in GRADES or float(value) != grade:
        return None
    return Judgment(grade, str(parsed.get("reason") or "").strip()[:240])


@dataclass
class GradeResult:
    grades: dict[str, dict[str, Judgment]]
    ungraded: list[tuple[str, str]] = field(default_factory=list)  # (question id, document) the grader never answered validly
    cost_usd: float = 0.0
    calls: int = 0
    stopped_by_budget: bool = False


def grade_pool(
    run: RunData,
    plan: PoolPlan,
    llm: LLM,
    *,
    workers: int = 4,
    budget: BudgetGuard | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> GradeResult:
    """Grade every pooled (question, document) pair, `workers` at a time. Results are assembled in plan order, so they do not depend on timing."""
    jobs = [(question, doc_id) for question in plan.questions for doc_id in plan.pool[question.id]]
    result = GradeResult({question.id: {} for question in plan.questions})
    finished = 0

    def one(job: tuple[Question, str]) -> tuple[Judgment | None, float, int, bool]:
        question, doc_id = job
        if budget is not None and budget.exhausted:
            return None, 0.0, 0, True
        messages: list[dict[str, Any]] = [{"role": "user", "content": grading_prompt(run, question, doc_id)}]
        cost, calls = 0.0, 0
        for _ in range(2):  # the first try and one retry with the format reminder
            reply = llm.generate(messages, json_mode=True)
            calls += 1
            cost += reply.cost.total_cost
            if budget is not None:
                budget.charge(reply.cost.total_cost)
            if (judgment := parse_grade(reply.text)) is not None:
                return judgment, cost, calls, False
            messages = [*messages, {"role": "assistant", "content": reply.text}, {"role": "user", "content": RETRY_REMINDER}]
        return None, cost, calls, False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for (question, doc_id), (judgment, cost, calls, skipped) in zip(jobs, executor.map(one, jobs), strict=True):
            result.cost_usd += cost
            result.calls += calls
            if skipped:
                result.stopped_by_budget = True
            elif judgment is None:
                result.ungraded.append((question.id, doc_id))
            else:
                result.grades[question.id][doc_id] = judgment
            finished += 1
            if progress is not None:
                progress(finished, len(jobs))
    return result


# -- proposals, merging, review ----------------------------------------------------------------


def proposed_rows(plan: PoolPlan, graded: GradeResult) -> list[dict[str, Any]]:
    """The proposed qrels: one row per graded pair (zeros included: judged and not relevant). A pair that was not graded gets no row."""
    return [
        {"query_id": question.id, "doc_id": doc_id, "relevance": judgment.grade}
        for question in plan.questions
        for doc_id, judgment in graded.grades.get(question.id, {}).items()
    ]


def existing_labels(dataset: Dataset, question: Question) -> dict[str, int]:
    """The human labels of a question: its qrels, plus every `relevant_doc_ids` entry (grade 1 where the qrels do not grade it)."""
    labels = dict(dataset.qrels.get(question.id, {}))
    for doc_id in question.relevant_doc_ids:
        labels.setdefault(doc_id, 1)
    return labels


def merge_qrels(dataset: Dataset, plan: PoolPlan, graded: GradeResult) -> list[dict[str, Any]]:
    """Existing labels plus proposed grades for documents the existing labels do not mention. Where both exist the human grade wins."""
    rows: list[dict[str, Any]] = []
    for question in dataset.questions:
        merged = existing_labels(dataset, question)
        for doc_id, judgment in graded.grades.get(question.id, {}).items():
            merged.setdefault(doc_id, judgment.grade)
        rows.extend({"query_id": question.id, "doc_id": doc_id, "relevance": grade} for doc_id, grade in sorted(merged.items(), key=lambda item: (-item[1], item[0])))
    return rows


@dataclass
class Disagreements:
    disputed: dict[str, list[str]] = field(default_factory=dict)  # question -> existing relevant documents the grader judged 0
    missing: dict[str, list[str]] = field(default_factory=dict)  # question -> documents graded 2+ that the existing labels lack (only for questions that have labels)
    unretrieved: dict[str, list[str]] = field(default_factory=dict)  # question -> existing relevant documents no system retrieved (never judged)
    unlabeled: dict[str, list[str]] = field(default_factory=dict)  # question without human labels -> documents graded 1+
    none_found: list[str] = field(default_factory=list)  # label-free questions where nothing pooled was graded above 0

    def questions_needing_attention(self) -> set[str]:
        return {*self.disputed, *self.missing, *self.unretrieved}


def compare_with_existing(run: RunData, plan: PoolPlan, graded: GradeResult) -> Disagreements:
    found = Disagreements()
    for question in plan.questions:
        labels = {doc_id: grade for doc_id, grade in existing_labels(run.dataset, question).items() if grade > 0}
        judged = graded.grades.get(question.id, {})
        if labels:
            disputed = sorted(doc_id for doc_id in labels if doc_id in judged and judged[doc_id].grade == 0)
            missing = sorted((doc_id for doc_id, j in judged.items() if j.grade >= 2 and doc_id not in labels), key=lambda d: (-judged[d].grade, d))
            unretrieved = sorted(doc_id for doc_id in labels if doc_id not in plan.pool[question.id])
            for bucket, items in ((found.disputed, disputed), (found.missing, missing), (found.unretrieved, unretrieved)):
                if items:
                    bucket[question.id] = items
        else:
            relevant = sorted((doc_id for doc_id, j in judged.items() if j.grade >= 1), key=lambda d: (-judged[d].grade, d))
            if relevant:
                found.unlabeled[question.id] = relevant
            elif judged:
                found.none_found.append(question.id)
    return found


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def review_markdown(
    run: RunData,
    plan: PoolPlan,
    graded: GradeResult,
    *,
    top_k: int,
    judge_model: str,
    mock: bool,
    applied: bool,
) -> str:
    found = compare_with_existing(run, plan, graded)
    distribution = Counter(judgment.grade for judgments in graded.grades.values() for judgment in judgments.values())
    total = sum(distribution.values())
    lines = [f"# Pooled labeling review: {run.run_dir.name}", ""]
    if mock:
        lines += ["> **Mock grader.** The grades below come from a word-overlap stand-in, not a model. They only prove the pipeline works; do not use them as labels.", ""]
    lines += [
        f"Grader: `{judge_model}` · pool depth: top {top_k} documents per system · systems pooled: {len({s for r in run.rankings.values() for s in r})}",
        "",
        "| | |",
        "| --- | --- |",
        f"| Questions graded | {len(plan.questions)} (skipped as unanswerable: {plan.skipped_unanswerable}) |",
        f"| Pooled (question, document) pairs | {plan.pairs} |",
        f"| Graded | {total} (ungraded: {len(graded.ungraded)}) |",
        "| Grade distribution | " + " · ".join(f"{grade}: {distribution.get(grade, 0)}" for grade in GRADES) + " |",
        f"| Existing labels the grader disputes (graded 0) | {sum(map(len, found.disputed.values()))} in {len(found.disputed)} questions |",
        f"| Documents graded 2+ that the labels lack | {sum(map(len, found.missing.values()))} in {len(found.missing)} questions |",
        f"| Labeled documents no system retrieved (not judged) | {sum(map(len, found.unretrieved.values()))} in {len(found.unretrieved)} questions |",
        f"| Questions with no human labels and proposed ones | {len(found.unlabeled)} |",
        "",
        "**How to read this.** The grades are an LLM's, so treat them as a first draft to check. Pooling only judges documents that some system retrieved: "
        "a relevant document that none of them found is never proposed (see docs/methodology.md#pooling-bias). Human labels are never overwritten; "
        + (f"`{MERGED_NAME}` keeps every human grade and adds proposed grades only for documents the labels do not mention." if applied else f"`--apply` would write `{MERGED_NAME}` with the same rule."),
        "",
    ]
    if graded.stopped_by_budget:
        lines += ["> **Stopped at the spending cap.** Some pairs were not graded; their questions have partial proposals.", ""]
    if graded.ungraded:
        lines += ["> **Ungraded pairs** (the grader never returned a valid grade, so nothing is proposed for them): " + ", ".join(f"{q}/{d}" for q, d in graded.ungraded[:10]) + (" …" if len(graded.ungraded) > 10 else ""), ""]

    by_id = {question.id: question for question in plan.questions}
    attention = [question.id for question in plan.questions if question.id in found.questions_needing_attention() or question.id in run.flagged_gaps]
    lines += ["## Disagreements with the existing labels", ""]
    if not attention:
        lines += ["None: where questions have labels, the grader agrees with them.", ""]
    for question_id in attention:
        question = by_id[question_id]
        labels = existing_labels(run.dataset, question)
        judged = graded.grades.get(question_id, {})
        lines += [f"### {question_id}: {question.question}", "", "Existing labels: " + (", ".join(f"{d} ({g})" for d, g in sorted(labels.items())) or "none"), ""]
        if question_id in found.disputed:
            lines.append(f"- **Disputed**: the grader says {', '.join(found.disputed[question_id])} does not answer this (graded 0). Check the label.")
        if question_id in found.missing:
            lines.append(f"- **Possible missing label**: {', '.join(f'{d} ({judged[d].grade})' for d in found.missing[question_id])} graded 2+ but not labeled relevant.")
        if question_id in found.unretrieved:
            lines.append(f"- **Not judged**: labeled documents {', '.join(found.unretrieved[question_id])} were not retrieved by any system, so nothing says whether they are relevant.")
        if question_id in run.flagged_gaps:
            lines.append(f"- The run flagged `possible_qrels_gap` for {', '.join(run.flagged_gaps[question_id])}: a well-supported answer from documents the labels do not name.")
        lines += ["", "| Document | Grade | Label | Retrieved by (rank) | Why |", "| --- | --- | --- | --- | --- |"]
        for doc_id, systems in plan.pool[question_id].items():
            judgment = judged.get(doc_id)
            retrieved = ", ".join(f"{system} ({rank})" for system, rank in sorted(systems.items(), key=lambda item: (item[1], item[0])))
            lines.append(f"| {doc_id} | {judgment.grade if judgment else '—'} | {labels.get(doc_id, '—')} | {_cell(retrieved)} | {_cell(judgment.reason) if judgment else 'not graded'} |")
        lines.append("")

    lines += ["## Proposed labels for questions without labels", ""]
    if found.unlabeled:
        lines += ["| Question | Proposed relevant documents (grade) |", "| --- | --- |"]
        for question_id, docs in found.unlabeled.items():
            judged = graded.grades[question_id]
            lines.append(f"| {question_id}: {_cell(by_id[question_id].question)} | {', '.join(f'{d} ({judged[d].grade})' for d in docs)} |")
        lines.append("")
    else:
        lines += ["None.", ""]
    if found.none_found:
        lines += [
            f"**No relevant document found** for {len(found.none_found)} question(s): {', '.join(found.none_found[:15])}{' …' if len(found.none_found) > 15 else ''}. "
            "Either no system retrieved the evidence, or the corpus does not answer them. In a dataset that is only partly labeled, a question with no relevant "
            "document is read as *unanswerable*; give the ones that do have answers `answerable: true`, and the others `answerable: false`.",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class LabelFiles:
    proposed: Path
    review: Path
    merged: Path | None


def write_label_outputs(
    out_dir: Path,
    run: RunData,
    plan: PoolPlan,
    graded: GradeResult,
    *,
    top_k: int,
    judge_model: str,
    mock: bool,
    apply: bool,
) -> LabelFiles:
    """`qrels.proposed.jsonl` and `qrels_review.md` in `out_dir`; with `apply`, also `qrels.merged.jsonl`. No existing label file is ever touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    proposed = out_dir / PROPOSED_NAME
    write_jsonl(proposed, proposed_rows(plan, graded))
    review = out_dir / REVIEW_NAME
    review.write_text(review_markdown(run, plan, graded, top_k=top_k, judge_model=judge_model, mock=mock, applied=apply), encoding="utf-8")
    merged = None
    if apply:
        merged = out_dir / MERGED_NAME
        write_jsonl(merged, merge_qrels(run.dataset, plan, graded))
    return LabelFiles(proposed, review, merged)


__all__ = [
    "GradeResult",
    "LabelError",
    "LabelFiles",
    "PoolPlan",
    "RunData",
    "build_pool",
    "estimate_grading_cost",
    "grade_pool",
    "load_run",
    "merge_qrels",
    "parse_grade",
    "proposed_rows",
    "write_label_outputs",
]
