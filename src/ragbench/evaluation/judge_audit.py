"""Calibrating the judge against a human: a blind, stratified sheet of a run's answers to score by hand, and the agreement between the
judge and those scores.

`write_audit_sheet` draws the sample: answers are grouped by system and by the judge's score band, and drawn from the groups in turn,
so every system and every part of the score range is represented. The sheet shows what the judge saw (question, reference, the context
the generator was given, the answer) and neither the system nor the judge's scores, so the human cannot anchor on them; the key file
maps each row back to its answer. `score_audit` compares a filled-in sheet with the judge and writes `judge_audit.md` / `.json`.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragbench.errors import UsageError
from ragbench.evaluation.agreement import krippendorff_alpha
from ragbench.evaluation.answer_judge import SCORE_AXES
from ragbench.evaluation.stats import CORRECT_THRESHOLD
from ragbench.utils.jsonl import read_jsonl

SHEET_FILE = "judge_audit_sheet.csv"
KEY_FILE = "judge_audit_key.json"
RESULT_FILE = "judge_audit.json"
REPORT_FILE = "judge_audit.md"
SHEET_COLUMNS = ["audit_id", "question", "reference_answer", "context", "answer", *SCORE_AXES, "notes"]
AXES = (*SCORE_AXES, "answer_score")


class AuditError(UsageError):
    """The run cannot be audited, or the labels cannot be read."""


@dataclass
class AuditSheet:
    sheet_path: Path
    key_path: Path
    n: int


def _band(score: float) -> str:
    return "low" if score < 2 else "high" if score >= CORRECT_THRESHOLD else "mid"


def _answers(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "per_question_results.jsonl"
    if not path.exists():
        raise AuditError(f"{run_dir} has no per_question_results.jsonl to audit.")
    return [row for row in read_jsonl(path) if row.get("error") is None and row.get("answer_judge")]


def _identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["system"]), str(row["question_id"]), int(row.get("trial") or 0)


def _context(row: dict[str, Any]) -> str:
    return "\n\n".join(f"[{c.get('doc_id')}] {c.get('text_preview', '')}" for c in row.get("retrieved_contexts") or [] if c.get("in_context", True))


def write_audit_sheet(run_dir: Path, sample: int = 50, seed: int = 0, *, overwrite: bool = False) -> AuditSheet:
    """Write `judge_audit_sheet.csv` (to fill in) and `judge_audit_key.json` (which answer each row is) into the run directory."""
    sheet_path, key_path = run_dir / SHEET_FILE, run_dir / KEY_FILE
    if sheet_path.exists() and not overwrite:
        raise AuditError(f"{sheet_path} is there already (it may hold scores). Fill that one in, or pass --force to draw a new sheet.")
    rows = _answers(run_dir)
    if not rows:
        raise AuditError(f"{run_dir} has no judged answers to audit.")
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        strata.setdefault((str(row["system"]), _band(float(row["answer_judge"]["answer_score"]))), []).append(row)
    for members in strata.values():
        rng.shuffle(members)
    picked: list[dict[str, Any]] = []
    queues = [strata[key] for key in sorted(strata)]
    while len(picked) < min(sample, len(rows)):
        for queue in queues:
            if queue and len(picked) < sample:
                picked.append(queue.pop())
    rng.shuffle(picked)
    items: dict[str, dict[str, Any]] = {}
    with sheet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(picked, start=1):
            audit_id = f"A{index:03d}"
            system, question_id, trial = _identity(row)
            items[audit_id] = {"system": system, "question_id": question_id, "trial": trial}
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "question": row.get("question", ""),
                    "reference_answer": row.get("reference_answer") or "",
                    "context": _context(row),
                    "answer": row.get("answer", ""),
                    **dict.fromkeys(SCORE_AXES, ""),
                    "notes": "",
                }
            )
    key_path.write_text(json.dumps({"seed": seed, "sample": sample, "items": items}, indent=2), encoding="utf-8")
    return AuditSheet(sheet_path, key_path, len(picked))


def _label(value: str | None, audit_id: str, axis: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        score = float(text)
    except ValueError:
        score = math.nan
    if not 0 <= score <= 5:
        raise AuditError(f"{audit_id}, {axis}: {text!r} is not a score from 0 to 5 (leave the cell blank to skip it).")
    return score


def _read_labels(labels_path: Path, items: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Human scores per audit id (blank cells left out; the answer score only when all five axes are filled)."""
    labels: dict[str, dict[str, float]] = {}
    with labels_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            audit_id = (row.get("audit_id") or "").strip()
            if audit_id not in items:
                raise AuditError(f"{labels_path}: {audit_id!r} is not an answer of the audit sheet ({KEY_FILE}).")
            scores = {axis: score for axis in SCORE_AXES if (score := _label(row.get(axis), audit_id, axis)) is not None}
            if len(scores) == len(SCORE_AXES):
                scores["answer_score"] = sum(scores.values()) / len(SCORE_AXES)
            if scores:
                labels[audit_id] = scores
    return labels


def _compare(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Agreement of (judge, human) pairs: alpha, mean absolute difference and bias (judge minus human; above 0 the judge is lenient)."""
    if not pairs:
        return {"n": 0, "alpha": None, "mean_abs_diff": None, "bias": None}
    return {
        "n": len(pairs),
        "alpha": krippendorff_alpha([list(pair) for pair in pairs]),
        "mean_abs_diff": sum(abs(j - h) for j, h in pairs) / len(pairs),
        "bias": sum(j - h for j, h in pairs) / len(pairs),
    }


def score_audit(run_dir: Path, labels_path: Path) -> dict[str, Any]:
    """Compare the human scores in `labels_path` (a filled-in audit sheet) with the judge's; write `judge_audit.json` and `.md`."""
    key_path = run_dir / KEY_FILE
    if not key_path.exists():
        raise AuditError(f"{run_dir} has no {KEY_FILE}: draw a sheet first with `ragbench judge-audit RUN --sample N`.")
    items = json.loads(key_path.read_text(encoding="utf-8"))["items"]
    labels = _read_labels(labels_path, items)
    by_identity = {_identity(row): row for row in _answers(run_dir)}
    pairs: dict[str, list[tuple[float, float]]] = {axis: [] for axis in AXES}
    by_system: dict[str, list[tuple[float, float]]] = {}
    by_judge: dict[str, list[tuple[float, float]]] = {}
    verdicts: list[bool] = []
    for audit_id, human in labels.items():
        item = items[audit_id]
        row = by_identity.get((item["system"], item["question_id"], int(item.get("trial") or 0)))
        if row is None:
            raise AuditError(f"{audit_id}: the run has no answer of {item['system']} to {item['question_id']}.")
        judged = row["answer_judge"]
        for axis, score in human.items():
            pairs[axis].append((float(judged[axis]), score))
        if "answer_score" in human:
            judge_score, human_score = float(judged["answer_score"]), human["answer_score"]
            by_system.setdefault(item["system"], []).append((judge_score, human_score))
            verdicts.append((judge_score >= CORRECT_THRESHOLD) == (human_score >= CORRECT_THRESHOLD))
            for ref, scores in (judged.get("metadata", {}).get("panel_scores") or {}).items():
                if scores is not None:
                    by_judge.setdefault(ref, []).append((float(scores["answer_score"]), human_score))
    result = {
        "run_id": run_dir.name,
        "labels": labels_path.name,
        "n_labeled": len(labels),
        "axes": {axis: _compare(values) for axis, values in pairs.items()},
        "verdict_agreement": sum(verdicts) / len(verdicts) if verdicts else None,
        "by_system": {system: {k: v for k, v in _compare(values).items() if k in {"n", "bias"}} for system, values in sorted(by_system.items())},
        "by_judge": {ref: {k: v for k, v in _compare(values).items() if k in {"n", "alpha"}} for ref, values in by_judge.items()},
    }
    (run_dir / RESULT_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / REPORT_FILE).write_text(audit_markdown(result), encoding="utf-8")
    return result


def _num(value: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if value is None else fmt.format(value)


def audit_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Judge audit: {result['run_id']}",
        "",
        f"A human scored **{result['n_labeled']}** answers (`{result['labels']}`) without seeing the system or the judge's scores. "
        "Alpha is Krippendorff's alpha between the judge and the human (1 = the same scores, 0 = chance; 0.8 and up is reliable, "
        "0.667 to 0.8 supports only tentative conclusions). Bias is the judge's score minus the human's: above 0 the judge is more lenient.",
        "",
        "| Axis | Labeled | Alpha | Mean abs. difference | Bias |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for axis, row in result["axes"].items():
        lines.append(f"| {axis} | {row['n']} | {_num(row['alpha'])} | {_num(row['mean_abs_diff'])} | {_num(row['bias'], '{:+.2f}')} |")
    if result["verdict_agreement"] is not None:
        lines += ["", f"Judge and human agree on whether an answer is correct (answer score >= {CORRECT_THRESHOLD:g}) for **{result['verdict_agreement']:.0%}** of the answers."]
    if result["by_system"]:
        lines += ["", "## Bias by system", "", "A judge more lenient with one system than the others favours it in the leaderboard.", "", "| System | Labeled | Bias |", "| --- | ---: | ---: |"]
        lines += [f"| {system} | {row['n']} | {_num(row['bias'], '{:+.2f}')} |" for system, row in result["by_system"].items()]
    if result["by_judge"]:
        lines += ["", "## Each judge of the panel against the human", "", "| Judge | Labeled | Alpha |", "| --- | ---: | ---: |"]
        lines += [f"| {ref} | {row['n']} | {_num(row['alpha'])} |" for ref, row in result["by_judge"].items()]
    return "\n".join(lines) + "\n"
