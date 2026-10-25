"""The rows one scored question becomes (`per_question_results.jsonl` and the retrieval / answer / cost tables), and the run's row lists."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragbench.config.schema import SystemConfig
from ragbench.evaluation.answer_metrics import answer_metrics, is_refusal
from ragbench.evaluation.claims import ClaimResult
from ragbench.evaluation.refusal import RefusalVerdict
from ragbench.evaluation.retrieval_metrics import unmeasured_retrieval_metrics
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import BaseRAGSystem
from ragbench.rag_systems.trace import stage_costs, step_to_dict
from ragbench.runtime import SystemOutcome
from ragbench.utils.text import truncate


@dataclass
class RunRows:
    """Every row a run produced, flattened from the systems' outcomes in config order."""

    per_question: list[dict[str, Any]] = field(default_factory=list)
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    answer: list[dict[str, Any]] = field(default_factory=list)
    cost: list[dict[str, Any]] = field(default_factory=list)
    ingestion: list[dict[str, Any]] = field(default_factory=list)
    system_runtime: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def collect(cls, outcomes: list[SystemOutcome]) -> RunRows:
        rows = cls()
        for outcome in outcomes:
            if outcome.ingestion_row is not None:
                rows.ingestion.append(outcome.ingestion_row)
            for result in outcome.results:
                rows.per_question.append(result["per_question"])
                if result["per_question"]["error"] is None:  # failed questions carry no scores and stay out of every mean
                    rows.retrieval.append(result["retrieval_row"])
                    rows.answer.append(result["answer_row"])
                    rows.cost.append(result["cost_row"])
            rows.system_runtime.append(outcome.runtime_row)
        return rows


def _location(metadata: dict[str, Any]) -> dict[str, Any]:
    """Where in its document a retrieved chunk came from (`page`, `heading`), for the report's question explorer; empty when the chunker did not record it."""
    out: dict[str, Any] = {}
    if metadata.get("page") is not None:
        out["page"] = metadata["page"] if metadata.get("page_end") in (None, metadata["page"]) else f"{metadata['page']}-{metadata['page_end']}"
    path = metadata.get("heading_path")
    if path:
        out["heading"] = " > ".join(str(part) for part in path)
    return out


def tag_trial(result: dict[str, Any], trial: int) -> dict[str, Any]:
    """Mark every row of one answer with its trial index (only done when a run has more than one trial, so one-trial outputs are unchanged)."""
    for key in ("per_question", "retrieval_row", "answer_row", "cost_row"):
        if result.get(key) is not None:
            result[key]["trial"] = trial
    return result


def error_result(system: BaseRAGSystem, system_config: SystemConfig, question, error: dict[str, str]) -> dict[str, Any]:
    zero_judge = {
        "correctness": 0.0,
        "faithfulness": 0.0,
        "completeness": 0.0,
        "relevance": 0.0,
        "citation_quality": 0.0,
        "answer_score": 0.0,
        "is_supported_by_context": False,
        "is_hallucinated": False,
        "reasoning": "",
        "metadata": {},
    }
    per_question = {
        "system": system.name,
        "system_type": system_config.type,
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "difficulty": question.difficulty,
        "answer_type": question.answer_type,
        "reference_answer": question.reference_answer,
        "relevant_doc_ids": question.relevant_doc_ids,
        "answer": "",
        "retrieved_contexts": [],
        "retrieval_metrics": {},
        "answer_judge": zero_judge,
        "cost": CostBreakdown().as_dict(),
        "latency_ms": 0.0,
        "failure_type": "run_error",
        "error": error,
        "answerable": question.is_answerable,
        "refused": False,
        "answer_metrics": {},
        "context_metrics": {},
        "steps": [],
        "requires_tools": list(question.requires_tools),
        "route": None,
        "routing_hint": question.routing_hint,
        "agent": None,
    }
    return {"per_question": per_question, "retrieval_row": None, "answer_row": None, "cost_row": None}


def build_question_rows(
    system: BaseRAGSystem,
    system_config: SystemConfig,
    question,
    answer,
    retrieval_metrics: dict[str, Any],
    judge_result,
    failure_type: str,
    in_context: set[str],
    context_metrics: dict[str, float | None],
    k_values: list[int],
    *,
    refusal: RefusalVerdict | None = None,
    claims: ClaimResult | None = None,
) -> dict[str, Any]:
    refused = refusal.refused if refusal is not None else is_refusal(answer.answer)
    deterministic = answer_metrics(answer.answer, question.reference_answer, question.expected_keywords, question.is_answerable)
    total_cost = answer.cost.plus(judge_result.cost)
    for extra in (refusal, claims):  # checks on the answer are charged as judge spend
        if extra is not None:
            total_cost = total_cost.plus(extra.cost)
    retrieved_contexts = [
        {
            "rank": chunk.rank,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "score": chunk.score,
            "in_context": chunk.chunk_id in in_context,
            "text_preview": truncate(chunk.text, 260),
            **_location(chunk.metadata),
        }
        for chunk in answer.retrieval_result.chunks
    ]
    per_question = {
        "system": system.name,
        "system_type": system_config.type,
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "difficulty": question.difficulty,
        "answer_type": question.answer_type,
        "reference_answer": question.reference_answer,
        "relevant_doc_ids": question.relevant_doc_ids,
        "answer": answer.answer,
        "retrieved_contexts": retrieved_contexts,
        "retrieval_metrics": retrieval_metrics,
        "answer_judge": {
            "correctness": judge_result.correctness,
            "faithfulness": judge_result.faithfulness,
            "completeness": judge_result.completeness,
            "relevance": judge_result.relevance,
            "citation_quality": judge_result.citation_quality,
            "answer_score": judge_result.answer_score,
            "is_supported_by_context": judge_result.is_supported_by_context,
            "is_hallucinated": judge_result.is_hallucinated,
            "reasoning": judge_result.reasoning,
            "metadata": judge_result.metadata,
        },
        "cost": total_cost.as_dict(),
        "latency_ms": answer.latency_ms,
        "failure_type": failure_type,
        "error": None,
        "answerable": question.is_answerable,
        "refused": refused,
        "answer_metrics": deterministic,
        "context_metrics": context_metrics,
        "steps": [step_to_dict(step) for step in answer.steps],
        "requires_tools": list(question.requires_tools),
        "route": answer.metadata.get("route"),  # the pipeline an `adaptive` system chose; None for every other system
        "routing_hint": question.routing_hint,
        "agent": answer.metadata.get("agent"),  # steps / hops / termination / budget_exhausted / llm_calls; None unless the system is agentic
        "gate": answer.metadata.get("gate"),  # what a `decision_gate` decided (per-chunk probabilities, sufficiency, abstain reason); None for every other system
    }
    claim_columns: dict[str, Any] = {}
    if claims is not None:
        per_question["claims"] = claims.as_dict()
        claim_columns = {"claim_support_rate": claims.support_rate, "unsupported_claims": len(claims.unsupported)}
    if refusal is not None and refusal.source == "model":
        per_question["refusal"] = {"source": refusal.source, "probability": refusal.probability}
    if answer.metadata.get("external") is not None:  # an external system: passage ids it could not map, whether its cost is known
        per_question["external"] = answer.metadata["external"]
    return {
        "per_question": per_question,
        "retrieval_row": {
            "system": system.name,
            "question_id": question.id,
            "category": question.category,
            **(retrieval_metrics or unmeasured_retrieval_metrics(k_values)),
        },
        "answer_row": {
            "system": system.name,
            "question_id": question.id,
            "category": question.category,
            "correctness": judge_result.correctness,
            "faithfulness": judge_result.faithfulness,
            "completeness": judge_result.completeness,
            "relevance": judge_result.relevance,
            "citation_quality": judge_result.citation_quality,
            "answer_score": judge_result.answer_score,
            **deterministic,
            **context_metrics,
            **claim_columns,
            "answerable": question.is_answerable,
            "refused": refused,
            "failure_type": failure_type,
        },
        "cost_row": {
            "system": system.name,
            "system_type": system_config.type,
            "stage": "question",
            "question_id": question.id,
            "latency_ms": answer.latency_ms,
            **total_cost.as_dict(),
            **{f"stage_{key}_cost": value for key, value in stage_costs(answer.steps).items()},
        },
    }

