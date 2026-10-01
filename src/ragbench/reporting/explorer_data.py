"""The data behind the report's question explorer: every question, with each system's answer, judgment, retrieved contexts and trace.

`build_questions` turns the rows of `per_question_results.jsonl` into one compact entry per question (the systems side by side) and keeps the page
under a size cap. When everything fits it is embedded in `report.html`. When it does not, the complete data goes to `report_questions.json` next to the
page (the page fetches it when it is served over http) and the page keeps a reduced copy: first less detail per answer, then scores only, then only the
questions that matter most (failures and disagreements). Text is always data: the page script only ever sets `textContent`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ragbench.reporting.columns import to_float
from ragbench.utils.text import truncate

SCHEMA_VERSION = 1
QUESTIONS_FILE = "report_questions.json"
DEFAULT_MAX_EMBEDDED_MB = 4.0
DISAGREE_SPREAD = 2.0  # answer scores (0-5) this far apart, best against worst, count as "the systems disagree"

FAILURE_LABELS = {
    "no_failure": "No failure",
    "retrieval_miss": "Retrieval miss",
    "bad_reranking": "Bad reranking",
    "insufficient_context": "Insufficient context",
    "partial_answer": "Partial answer",
    "wrong_entity": "Wrong entity",
    "over_refusal": "Over-refusal",
    "wrong_date": "Wrong date",
    "answer_hallucination": "Hallucination",
    "possible_qrels_gap": "Possible label gap",
    "format_error": "Format error",
    "run_error": "Run error",
}
JUDGE_KEYS = ("correctness", "faithfulness", "completeness", "relevance", "citation_quality")
DETAILS = ("full", "reduced", "scores")

# Characters kept per field at each level of detail. Context previews and step previews are already cut to 260 / 300 characters when the run is written.
LIMITS = {
    "full": {"question": 600, "reference": 600, "answer": 1200, "reasoning": 500, "context": 260, "step": 300},
    "reduced": {"question": 300, "reference": 300, "answer": 400, "reasoning": 200, "context": 120, "step": 0},
    "scores": {"question": 200, "reference": 120, "answer": 160, "reasoning": 0, "context": 0, "step": 0},
}
MAX_META_CHARS = 100


def script_json(data: Any) -> str:
    """Compact JSON that is safe inside a `<script>` element: no `</script>`, no HTML comment opener, no line-separator characters, all still valid JSON."""
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


@dataclass
class QuestionsPayload:
    embedded: dict[str, Any]  # what goes into report.html
    full: dict[str, Any] | None  # what goes into report_questions.json; None when the page already holds everything


def _text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[: limit - 1].rstrip() + "…" if limit and len(text) > limit else text  # keeps line breaks, which `truncate` would flatten


def _round(value: Any, digits: int) -> float | None:
    number = to_float(value)
    return None if number is None else round(number, digits)


def _plain_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """The scalar entries of a step's metadata (tool name, `top_k`, `error`, `cached`, ...); nested structures are not worth the bytes."""
    return {str(k): (truncate(v, MAX_META_CHARS) if isinstance(v, str) and len(v) > MAX_META_CHARS else v) for k, v in meta.items() if isinstance(v, str | int | float | bool)}


def _step(step: dict[str, Any], detail: str) -> dict[str, Any]:
    limit = LIMITS[detail]["step"]
    out: dict[str, Any] = {"kind": step.get("kind"), "name": step.get("name"), "ms": _round(step.get("latency_ms"), 1) or 0.0, "cost": _round((step.get("cost") or {}).get("total_cost"), 7) or 0.0}
    tokens = int(step.get("prompt_tokens") or 0) + int(step.get("completion_tokens") or 0)
    if tokens:
        out["tokens"] = tokens
    if limit:
        for key, source in (("in", "input_preview"), ("out", "output_preview")):
            if step.get(source):
                out[key] = _text(step[source], limit)
        if meta := _plain_meta(step.get("metadata") or {}):
            out["meta"] = meta
    return out


def _context(chunk: dict[str, Any], relevant: set[str], good: set[str], detail: str) -> dict[str, Any]:
    doc = str(chunk.get("doc_id"))
    out: dict[str, Any] = {
        "rank": chunk.get("rank"),
        "doc": doc,
        "in": bool(chunk.get("in_context", True)),
        "score": _round(chunk.get("score"), 4),
        "mark": "relevant" if doc in relevant else "judged_good" if doc in good else None,
        "text": _text(chunk.get("text_preview"), LIMITS[detail]["context"]) or "",
    }
    for key in ("page", "heading"):
        if chunk.get(key) is not None:
            out[key] = chunk[key]
    if detail == "full" and chunk.get("chunk_id"):
        out["chunk"] = chunk["chunk_id"]
    return out


def _run(row: dict[str, Any], index: int, good: set[str], detail: str) -> dict[str, Any]:
    limits = LIMITS[detail]
    judge = row.get("answer_judge") or {}
    scored = row.get("error") is None
    relevant = {str(d) for d in row.get("relevant_doc_ids") or []}
    run: dict[str, Any] = {
        "s": index,
        "answer": _text(row.get("answer"), limits["answer"]) or "",
        "score": _round(judge.get("answer_score"), 2) if scored else None,
        "judge": {key: _round(judge.get(key), 2) for key in JUDGE_KEYS} if scored and judge else None,
        "failure": row.get("failure_type") or "no_failure",
        "error": row.get("error"),
        "refused": bool(row.get("refused")),
        "latency_ms": _round(row.get("latency_ms"), 1),
        "cost": _round((row.get("cost") or {}).get("total_cost"), 7),
    }
    if row.get("route"):
        run["route"] = row["route"]
    if detail == "scores":
        return run
    if limits["reasoning"] and judge.get("reasoning"):
        run["reasoning"] = _text(judge["reasoning"], limits["reasoning"])
    if (judge.get("metadata") or {}).get("judge"):
        run["judge_by"] = judge["metadata"]["judge"]
    if row.get("agent"):
        run["agent"] = {k: v for k, v in row["agent"].items() if isinstance(v, str | int | float | bool)}
    run["ctx"] = [_context(chunk, relevant, good if scored else set(), detail) for chunk in row.get("retrieved_contexts") or []]
    run["steps"] = [_step(step, detail) for step in row.get("steps") or []]
    return run


def _audited(rows: list[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    """(system, question) -> unlabeled documents whose retrieval went with an answer judged right and grounded (the label audit's candidates)."""
    found: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        ids = {part.strip() for part in str(row.get("unlabeled_retrieved_doc_ids") or "").split(",") if part.strip()}
        found[(str(row.get("system")), str(row.get("question_id")))] = ids
    return found


def _question(rows: list[tuple[int, dict[str, Any]]], audited: dict[tuple[str, str], set[str]], detail: str) -> dict[str, Any]:
    head = rows[0][1]
    limits = LIMITS[detail]
    scores = [s for _, row in rows if row.get("error") is None and (s := to_float((row.get("answer_judge") or {}).get("answer_score"))) is not None]
    out: dict[str, Any] = {
        "id": str(head["question_id"]),
        "q": _text(head.get("question"), limits["question"]) or "",
        "cat": str(head.get("category") or "unknown"),
        "rel": [str(d) for d in head.get("relevant_doc_ids") or []],
        "spread": round(max(scores) - min(scores), 2) if len(scores) > 1 else None,
    }
    if head.get("reference_answer"):
        out["ref"] = _text(head["reference_answer"], limits["reference"])
    if head.get("difficulty"):
        out["diff"] = head["difficulty"]
    if head.get("answerable") is False:
        out["unanswerable"] = True
    if head.get("routing_hint"):
        out["hint"] = head["routing_hint"]
    if head.get("requires_tools"):
        out["tools"] = list(head["requires_tools"])
    out["runs"] = [_run(row, index, audited.get((str(row["system"]), str(row["question_id"])), set()), detail) for index, row in rows]
    return out


def _priority(question: dict[str, Any]) -> tuple[int, float]:
    failed = sum(1 for run in question["runs"] if run["error"] or run["failure"] != "no_failure")
    return (-failed, -(question["spread"] or 0.0))


def _page(questions: list[dict[str, Any]], systems: list[str], names: dict[str, str], detail: str, *, total: int, file: str | None) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "systems": systems,
        "failure_names": names,
        "disagree_spread": DISAGREE_SPREAD,
        "detail": detail,
        "total": total,
        "shown": len(questions),
        "file": file,
        "questions": questions,
    }


def _size(data: Any) -> int:
    return len(script_json(data).encode("utf-8"))


def build_questions(rows: list[dict[str, Any]], order: list[str], audit: list[dict[str, Any]], max_bytes: int) -> QuestionsPayload:
    """One entry per question from `per_question_results.jsonl` rows, systems in `order` (systems the order does not name follow it).

    `audit` is `qrels_audit.csv` as rows; it decides which unlabeled retrieved documents are shown as probably good.
    """
    systems = [name for name in order if any(row["system"] == name for row in rows)]
    systems += [name for name in dict.fromkeys(str(row["system"]) for row in rows) if name not in systems]
    index = {name: i for i, name in enumerate(systems)}
    by_question: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        by_question[str(row["question_id"])].append((index[str(row["system"])], row))
    audited = _audited(audit)
    names = {**FAILURE_LABELS}
    for row in rows:
        kind = str(row.get("failure_type") or "no_failure")
        names.setdefault(kind, kind.replace("_", " ").capitalize())

    def at(detail: str) -> list[dict[str, Any]]:
        return [_question(sorted(items, key=lambda item: item[0]), audited, detail) for items in by_question.values()]

    full = at("full")
    total = len(full)
    whole = _page(full, systems, names, "full", total=total, file=None)
    if _size(whole) <= max_bytes:
        return QuestionsPayload(whole, None)
    stored = {**whole, "questions": full}
    for detail in DETAILS[1:]:
        shrunk = _page(at(detail), systems, names, detail, total=total, file=QUESTIONS_FILE)
        if _size(shrunk) <= max_bytes:
            return QuestionsPayload(shrunk, stored)
    # Not even the scores fit: keep the questions that matter most (failures, then disagreements) until the budget is spent, in the dataset's order.
    scores = shrunk["questions"]
    budget = max_bytes - _size(_page([], systems, names, "scores", total=total, file=QUESTIONS_FILE))
    keep: set[int] = set()
    for position in sorted(range(len(scores)), key=lambda i: (_priority(scores[i]), i)):
        cost = _size(scores[position]) + 1
        if cost > budget:
            break
        budget -= cost
        keep.add(position)
    return QuestionsPayload(_page([q for i, q in enumerate(scores) if i in keep], systems, names, "scores", total=total, file=QUESTIONS_FILE), stored)
