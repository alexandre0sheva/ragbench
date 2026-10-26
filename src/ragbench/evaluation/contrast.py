"""Contrast sets: per pair of systems, the questions one answers well and the other does not, and where their traces first part.

Everything is computed from a finished run's files (`per_question_results.jsonl`, and `pairwise.jsonl` for verdicts), so `ragbench contrast`
needs no API and the evidence builder can call `build_contrasts` on any run, restricted to the questions it may show (`question_ids`).
A question is a *contrast* when exactly one of the two systems is right: by default its mean answer score is at least `threshold` for one and
below it for the other; with `source="verdicts"` the arena judge's both-order verdict decides. A failed question is left out, never scored zero.
"""

from __future__ import annotations

import difflib
import itertools
import json
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ragbench.evaluation.stats import CORRECT_THRESHOLD
from ragbench.utils.jsonl import read_jsonl, write_jsonl
from ragbench.utils.text import truncate

CONTRASTS_FILE = "contrasts.jsonl"
SUMMARY_FILE = "contrast_summary.json"
Source = Literal["score", "verdicts"]
PairMode = Literal["vs_baseline", "all"]
PREVIEW_CHARS = 160
ANSWER_CHARS = 300


class ContrastError(ValueError):
    """The run cannot be contrasted as asked (unknown system, no pairwise verdicts, no rows)."""


@dataclass
class ContrastSet:
    records: list[dict[str, Any]] = field(default_factory=list)  # one per contrast question, for `contrasts.jsonl`
    summaries: list[dict[str, Any]] = field(default_factory=list)  # one per pair, for `contrast_summary.json`


# -- aligning two traces ----------------------------------------------------------------------------------------------------------------------


def _label(step: dict[str, Any] | None) -> str | None:
    return None if step is None else f"{step.get('kind')}:{step.get('name')}"


def _documents(step: dict[str, Any]) -> list[str] | None:
    docs = (step.get("metadata") or {}).get("doc_ids")
    return [str(doc) for doc in docs] if isinstance(docs, list) else None


def _difference(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Why two steps of the same kind are not the same step, or None when they are. The names may differ (two retrievers): the work is compared."""
    if a.get("kind") == "generate":
        return "different_answer" if a.get("output_preview") != b.get("output_preview") else None
    docs_a, docs_b = _documents(a), _documents(b)
    if docs_a is not None and docs_b is not None:
        shared = min(len(docs_a), len(docs_b))  # a deeper ranking with the same head is the same evidence for the generator that reads the head
        return "different_documents" if docs_a[:shared] != docs_b[:shared] else None
    return "different_output" if a.get("output_preview") != b.get("output_preview") else None


def first_divergence(steps_a: list[dict[str, Any]], steps_b: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The first step at which two traces part, and both traces aligned step by step.

    Steps are paired by position. A pair diverges when its kinds differ (`different_stage`), or when same-kind steps retrieved other
    documents (`different_documents`, compared on the shorter ranking's depth), produced other output (`different_output`) or, for the final
    generate step, another answer (`different_answer`). `stage` is the first system's step kind (the second's where the first has no step).
    A system with no trace (an external system) has nothing to align: `(None, [])`.
    """
    if not steps_a or not steps_b:
        return None, []
    aligned: list[dict[str, Any]] = []
    divergence: dict[str, Any] | None = None
    for index in range(max(len(steps_a), len(steps_b))):
        a = steps_a[index] if index < len(steps_a) else None
        b = steps_b[index] if index < len(steps_b) else None
        if a is None or b is None or a.get("kind") != b.get("kind"):
            reason = "different_stage"
        else:
            reason = _difference(a, b) or ""
        aligned.append(
            {
                "index": index,
                "a": _label(a),
                "b": _label(b),
                "same": not reason,
                "a_output": truncate(str(a.get("output_preview") or ""), PREVIEW_CHARS) if a else None,
                "b_output": truncate(str(b.get("output_preview") or ""), PREVIEW_CHARS) if b else None,
            }
        )
        if reason and divergence is None:
            lead = a or b or {}
            divergence = {"index": index, "stage": str(lead.get("kind")), "reason": reason, "a": _label(a), "b": _label(b)}
    return divergence, aligned


# -- the run's rows ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class _Answer:
    """One system's answer to one question: the mean judge score over its trials, and the first trial's row for everything else."""

    score: float
    row: dict[str, Any]


def _score(row: dict[str, Any]) -> float | None:
    value = (row.get("answer_judge") or {}).get("answer_score")
    return float(value) if isinstance(value, int | float) else None


def _answers(rows: Iterable[dict[str, Any]], question_ids: Collection[str] | None) -> dict[str, dict[str, _Answer]]:
    """system -> question -> answer, for the questions that were answered and judged (an error or an unscored answer is absent, not zero)."""
    grouped: dict[str, dict[str, list[tuple[float, dict[str, Any]]]]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if row.get("error") is not None or (question_ids is not None and question_id not in question_ids):
            continue
        if (score := _score(row)) is not None:
            grouped.setdefault(str(row["system"]), {}).setdefault(question_id, []).append((score, row))
    return {system: {q: _Answer(sum(s for s, _ in trials) / len(trials), trials[0][1]) for q, trials in by_question.items()} for system, by_question in grouped.items()}


def _verdict_points(path: Path, a: str, b: str) -> dict[str, float]:
    """Question -> the share of points `a` won against `b` over the judges' both-order verdicts (0.5 is a draw)."""
    points: dict[str, list[float]] = {}
    for record in read_jsonl(path):
        if set(record.get("systems") or []) != {a, b} or record.get("score") is None:
            continue
        score = float(record["score"])
        points.setdefault(str(record["question_id"]), []).append(score if record["systems"][0] == a else 1.0 - score)
    return {question_id: sum(values) / len(values) for question_id, values in points.items()}


def select_pairs(systems: list[str], mode: PairMode, baseline: str | None = None) -> list[tuple[str, str]]:
    """Every pair of `systems`, or every system against `baseline` (the first system when none is named)."""
    if mode == "all":
        return list(itertools.combinations(systems, 2))
    anchor = baseline if baseline in systems else systems[0]
    return [(anchor, other) for other in systems if other != anchor]


def _known(name: str, systems: list[str]) -> str:
    if name in systems:
        return name
    close = difflib.get_close_matches(name, systems, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    raise ContrastError(f"The run has no system named '{name}'.{hint} (Systems: {', '.join(systems)})")


def _record(a: str, b: str, question_id: str, answers: dict[str, _Answer], winner: str, source: Source, points: float | None) -> dict[str, Any]:
    loser = b if winner == a else a
    row_a, row_b = answers[a].row, answers[b].row
    divergence, aligned = first_divergence(row_a.get("steps") or [], row_b.get("steps") or [])
    scores = {a: answers[a].score, b: answers[b].score}
    base = row_a
    record: dict[str, Any] = {
        "pair": [a, b],
        "question_id": question_id,
        "question": base.get("question"),
        "category": base.get("category"),
        "difficulty": base.get("difficulty"),
        "source": source,
        "winner": winner,
        "loser": loser,
        "scores": scores,
        "gap": scores[winner] - scores[loser],
        "answers": {a: truncate(str(row_a.get("answer") or ""), ANSWER_CHARS), b: truncate(str(row_b.get("answer") or ""), ANSWER_CHARS)},
        "loser_failure_type": (answers[loser].row.get("failure_type") or "unknown"),
        "diverges_at": divergence,
        "steps": aligned,
    }
    if points is not None:
        record["points"] = points
    return record


def _clusters(records: list[dict[str, Any]], pair: tuple[str, str]) -> dict[str, dict[str, dict[str, int]]]:
    """Per winner: how the questions it won split by the loser's failure type, the category, and the stage where the traces parted."""
    clusters: dict[str, dict[str, dict[str, int]]] = {}
    for winner in pair:
        won = [r for r in records if r["winner"] == winner]
        clusters[winner] = {
            "failure_type": dict(Counter(r["loser_failure_type"] for r in won)),
            "category": dict(Counter(str(r["category"]) for r in won)),
            "diverges_at": dict(Counter(r["diverges_at"]["stage"] for r in won if r["diverges_at"])),
        }
    return clusters


def build_contrasts(
    run_dir: Path,
    pairs: list[tuple[str, str]] | None = None,
    *,
    source: Source = "score",
    threshold: float = CORRECT_THRESHOLD,
    question_ids: Collection[str] | None = None,
    pair_mode: PairMode = "vs_baseline",
    baseline: str | None = None,
) -> ContrastSet:
    """The contrast set of each pair (default: `pair_mode` over the run's systems). `question_ids` restricts the questions compared."""
    rows_path = run_dir / "per_question_results.jsonl"
    if not rows_path.is_file():
        raise ContrastError(f"{run_dir} has no per_question_results.jsonl to contrast.")
    rows = read_jsonl(rows_path)
    answers = _answers(rows, question_ids)
    systems = list(dict.fromkeys(str(row["system"]) for row in rows))
    if source == "verdicts" and not (run_dir / "pairwise.jsonl").is_file():
        raise ContrastError("source 'verdicts' needs the pairwise stage: run with `evaluation.pairwise.enabled: true` (no pairwise.jsonl in this run).")
    chosen = pairs if pairs is not None else select_pairs(systems, pair_mode, baseline)
    result = ContrastSet()
    for name_a, name_b in chosen:
        a, b = _known(name_a, systems), _known(name_b, systems)
        if a == b:
            raise ContrastError("A contrast needs two different systems.")
        asked = set(question_ids) if question_ids is not None else {q for system in (a, b) for q in _all_questions(rows, system)}
        both = [q for q in sorted(asked) if q in answers.get(a, {}) and q in answers.get(b, {})]
        points = _verdict_points(run_dir / "pairwise.jsonl", a, b) if source == "verdicts" else {}
        counts = Counter[str]()
        records: list[dict[str, Any]] = []
        compared = 0
        for question_id in both:
            pair_answers = {a: answers[a][question_id], b: answers[b][question_id]}
            if source == "verdicts":
                if question_id not in points:
                    continue
                share = points[question_id]
                winner = a if share > 0.5 else b if share < 0.5 else None
                tag = "a_only" if winner == a else "b_only" if winner == b else "both_equal"
                won = share if winner == a else 1.0 - share
            else:
                right_a, right_b = pair_answers[a].score >= threshold, pair_answers[b].score >= threshold
                winner = a if right_a and not right_b else b if right_b and not right_a else None
                tag = "a_only" if winner == a else "b_only" if winner == b else "both_correct" if right_a else "both_wrong"
                won = None
            compared += 1
            counts[tag] += 1
            if winner is not None:
                records.append(_record(a, b, question_id, pair_answers, winner, source, won))
        summary: dict[str, Any] = {
            "pair": [a, b],
            "source": source,
            **({"threshold": threshold} if source == "score" else {}),
            "compared": compared,
            "skipped": len(asked) - compared,
            "a_only": counts["a_only"],
            "b_only": counts["b_only"],
            **({"both_equal": counts["both_equal"]} if source == "verdicts" else {"both_correct": counts["both_correct"], "both_wrong": counts["both_wrong"]}),
            "clusters": _clusters(records, (a, b)),
        }
        result.records.extend(records)
        result.summaries.append(summary)
    return result


def _all_questions(rows: list[dict[str, Any]], system: str) -> set[str]:
    return {str(row["question_id"]) for row in rows if row["system"] == system}


# -- files ------------------------------------------------------------------------------------------------------------------------------------


def read_summaries(run_dir: Path) -> list[dict[str, Any]]:
    """The per-pair summaries a run holds (`[]` for a run without contrasts)."""
    path = run_dir / SUMMARY_FILE
    return list(json.loads(path.read_text(encoding="utf-8")).get("pairs", [])) if path.is_file() else []


def write_contrasts(run_dir: Path, contrasts: ContrastSet, *, merge: bool = False) -> list[Path]:
    """`contrasts.jsonl` and `contrast_summary.json`. With `merge`, the pairs already in the run stay, except those this set replaces."""
    records, summaries = contrasts.records, contrasts.summaries
    if merge:
        replaced = {frozenset(summary["pair"]) for summary in summaries}
        kept_records = [r for r in (read_jsonl(run_dir / CONTRASTS_FILE) if (run_dir / CONTRASTS_FILE).is_file() else []) if frozenset(r["pair"]) not in replaced]
        kept_summaries = [s for s in read_summaries(run_dir) if frozenset(s["pair"]) not in replaced]
        records, summaries = [*kept_records, *records], [*kept_summaries, *summaries]
    write_jsonl(run_dir / CONTRASTS_FILE, records)
    (run_dir / SUMMARY_FILE).write_text(json.dumps({"version": 1, "pairs": summaries}, indent=2), encoding="utf-8")
    return [run_dir / CONTRASTS_FILE, run_dir / SUMMARY_FILE]
