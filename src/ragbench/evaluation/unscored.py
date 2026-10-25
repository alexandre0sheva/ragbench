"""Blanking the retrieval scores of an external system whose passages cannot be matched to the corpus (see `external_rag`)."""

from __future__ import annotations

from typing import Any

_ROW_KEYS = ("system", "question_id", "category")


def blank_unscorable_retrieval(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One system's results, with retrieval blank on every question when, for any question, it was an external system whose passages it did
    not show or that named no corpus document. Scoring only the questions where it happened to return no passages would rank it on other
    questions than the rest, and mostly on misses."""
    def unscorable(result: dict[str, Any]) -> bool:
        external = (result.get("per_question") or {}).get("external") or {}
        return bool(external.get("no_contexts") or external.get("unmapped_doc_ids"))

    if not any(unscorable(result) for result in results):
        return results
    for result in results:
        per_question = result.get("per_question")
        if per_question is None or per_question.get("error") is not None:
            continue
        per_question["retrieval_metrics"] = {}
        context = {key: None for key in per_question.get("context_metrics") or {}}
        per_question["context_metrics"] = context
        result["retrieval_row"] = {key: value if key in _ROW_KEYS else float("nan") for key, value in result["retrieval_row"].items()}
        result["answer_row"].update(context)
    return results
