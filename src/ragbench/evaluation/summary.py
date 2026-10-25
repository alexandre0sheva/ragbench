"""Per-system aggregates of a run (`metrics_summary.csv`), the tables derived from its rows (routes, tool usage, qrels audit) and the
facts about the judge that become report notices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ragbench.config.schema import SystemConfig
from ragbench.evaluation.answer_metrics import refusal_metrics
from ragbench.evaluation.judge_panel import JudgePanel, scoring_judges
from ragbench.evaluation.route_metrics import route_accuracy, route_rows
from ragbench.evaluation.stats import latency_percentiles, quantile_ci
from ragbench.evaluation.tool_metrics import summary_fields as tool_summary_fields
from ragbench.evaluation.tool_metrics import tool_steps
from ragbench.evaluation.tool_metrics import usage_rows as tool_usage_rows
from ragbench.models.family import model_family
from ragbench.registry import SYSTEMS
from ragbench.runtime import ProbeResult
from ragbench.utils.text import unique_preserve_order

CLAIM_COLUMNS = ("claim_support_rate", "unsupported_claims")  # `unsupported_claims` is a mean count per answer


def errors_by_system(per_question_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_question_rows:
        if row["error"] is not None:
            counts[row["system"]] = counts.get(row["system"], 0) + 1
    return counts


def failed_systems(per_question_rows: list[dict[str, Any]], num_questions: int, max_error_rate: float) -> dict[str, float]:
    """The systems whose share of failed questions is above `max_error_rate`, with that share."""
    if not num_questions:
        return {}
    rates = {name: count / num_questions for name, count in errors_by_system(per_question_rows).items()}
    return {name: rate for name, rate in rates.items() if rate > max_error_rate}


def tools_offered(systems: list[SystemConfig]) -> dict[str, list[str]]:
    """The tool names each system is configured with (empty for a system that has none)."""
    return {cfg.resolved_name: SYSTEMS.get(cfg.type).offered_tools(cfg) for cfg in systems}


def build_routes(systems: list[SystemConfig], per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-route usage of every system that routes, in the order the systems are configured."""
    rows: list[dict[str, Any]] = []
    for cfg in systems:
        ok = [r for r in per_question_rows if r["system"] == cfg.resolved_name and r["error"] is None]
        rows.extend(route_rows(cfg.resolved_name, ok))
    return rows


def build_tool_usage(systems: list[SystemConfig], per_question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_name, configured in tools_offered(systems).items():
        system_rows = [r for r in per_question_rows if r["system"] == system_name and r["error"] is None]
        if configured or any(tool_steps(r) for r in system_rows):
            rows.extend(tool_usage_rows(system_name, configured, system_rows))
    return rows


def build_qrels_audit(per_question_rows: list[dict[str, Any]], primary_k: int) -> list[dict[str, Any]]:
    """Answers judged right and grounded although retrieval missed labeled documents: candidates for a missing label. One entry per
    system and question (with repeated trials, the first trial that qualifies)."""
    audit_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in per_question_rows:
        if row["error"] is not None or (row["system"], row["question_id"]) in seen:
            continue
        metrics = row["retrieval_metrics"]
        judge = row["answer_judge"]
        relevant_doc_ids = row.get("relevant_doc_ids", [])
        if not relevant_doc_ids:
            continue
        if metrics.get(f"recall@{primary_k}", 0.0) >= 1.0:
            continue
        if judge.get("answer_score", 0.0) < 4.0 or judge.get("faithfulness", 0.0) < 4.0:
            continue
        # Only documents the generator actually saw can explain a well-supported answer.
        retrieved_doc_ids = unique_preserve_order(context["doc_id"] for context in row.get("retrieved_contexts", []) if context.get("in_context", True))
        unlabeled = [doc_id for doc_id in retrieved_doc_ids if doc_id not in set(relevant_doc_ids)]
        if not unlabeled:
            continue
        seen.add((row["system"], row["question_id"]))
        audit_rows.append(
            {
                "system": row["system"],
                "question_id": row["question_id"],
                "question": row["question"],
                "severity": "high" if metrics.get(f"hit@{primary_k}", 0.0) == 0.0 else "medium",
                "primary_k": primary_k,
                "recall": metrics.get(f"recall@{primary_k}", 0.0),
                "answer_score": judge.get("answer_score", 0.0),
                "faithfulness": judge.get("faithfulness", 0.0),
                "labeled_relevant_doc_ids": ", ".join(relevant_doc_ids),
                "unlabeled_retrieved_doc_ids": ", ".join(unlabeled),
            }
        )
    return audit_rows


def judge_fallback_rate(judge: JudgePanel | None, rows: list[dict[str, Any]]) -> float | None:
    """Share of questions the heuristic had to score because the LLM judge failed; None when no LLM judge was in use."""
    if judge is None or not judge.uses_llm or not rows:
        return None
    return sum(1 for row in rows if row["answer_judge"]["metadata"].get("fallback")) / len(rows)


def judge_notices(judge: JudgePanel | None, generators: dict[str, str | None], summary_rows: list[dict[str, Any]], agreement: dict[str, float | None]) -> dict[str, Any]:
    """Facts about the judges that `build_notices` turns into report warnings: self-preference, heuristic fallbacks, a panel that disagrees.

    `generators` maps each system to its generator's model ref. A system is at risk of self-preference when, with `independent`, every
    judge that scored it is of its generator's family (no judge of another family was there to take over).
    """
    if judge is None or not judge.uses_llm:
        return {}
    same_family = []
    if judge.independent:
        for name, generator in generators.items():
            family = model_family(generator)
            if all(model_family(ref) == family for ref in scoring_judges(judge.refs, generator, independent=True)):
                same_family.append(name)
    label = f"the judge ({judge.refs[0]})" if not judge.is_panel else f"every judge on the panel ({', '.join(judge.refs)})"
    fallbacks = {row["system"]: row["judge_fallback_rate"] for row in summary_rows if (row.get("judge_fallback_rate") or 0) > 0}
    return {"judge_model": label, "judge_shares_family_with": same_family, "judge_fallbacks": fallbacks, "judge_agreement": agreement.get("answer_score")}


def _agent_fields(agents: list[dict[str, Any]]) -> dict[str, float]:
    """Mean loop iterations, mean LLM calls and the share of questions that hit the budget; empty for a system that is not agentic."""
    if not agents:
        return {}
    return {
        "avg_steps": sum(a["steps"] for a in agents) / len(agents),
        "avg_llm_calls": sum(a["llm_calls"] for a in agents) / len(agents),
        "budget_exhausted_rate": sum(1 for a in agents if a["budget_exhausted"]) / len(agents),
    }


def _gate_fields(gates: list[dict[str, Any]]) -> dict[str, float]:
    """Share of questions the gate abstained on, mean decision requests and mean chunks kept; empty for a system without a gate."""
    if not gates:
        return {}
    return {
        "gate_abstain_rate": sum(1 for g in gates if g["abstained"]) / len(gates),
        "avg_decision_calls": sum(g["decision_calls"] for g in gates) / len(gates),
        "avg_chunks_kept": sum(g["kept"] for g in gates) / len(gates),
    }


def _latency_fields(concurrent_ms: list[float], probe: ProbeResult | None) -> dict[str, Any]:
    """Latency columns of one system.

    The headline latency comes from the one-at-a-time probe when there is one (clean: no queueing behind other
    threads, no cache); otherwise from the parallel pass, labeled as such. The parallel-pass mean is always kept.
    The p95 gets an exact order-statistic interval over the same measurements (no upper bound until there are 72 of them).
    """
    concurrent_mean = float(np.mean(concurrent_ms)) if concurrent_ms else None
    clean = probe.latencies_ms if probe is not None and probe.latencies_ms else None
    values = clean if clean is not None else concurrent_ms
    _, p95_lo, p95_hi = quantile_ci(values, 95)
    return {
        "avg_latency_ms": float(np.mean(values)) if values else None,
        **{f"latency_ms_{name}": value for name, value in latency_percentiles(values).items()},
        "latency_ms_p95_ci_lo": p95_lo,
        "latency_ms_p95_ci_hi": p95_hi,
        "latency_source": "probe" if clean is not None else "concurrent",
        "avg_latency_concurrent_ms": concurrent_mean,
    }


def _question_means(frame: pd.DataFrame) -> pd.DataFrame:
    """With repeated trials (a `trial` column), one row per question holding the mean of its trials' numeric scores, so a question
    whose trial failed is not outweighed by one answered every time; without trials, the frame as it is."""
    if frame.empty or "trial" not in frame.columns:
        return frame
    scores = [col for col in frame.columns if col not in {"system", "question_id", "category", "trial"} and col != "failure_type"]
    numeric = frame[scores].apply(pd.to_numeric, errors="coerce")
    return numeric.groupby(frame["question_id"], sort=False).mean().reset_index()


def build_summary(
    systems: list[SystemConfig],
    retrieval_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    per_question_rows: list[dict[str, Any]],
    runtime_df: pd.DataFrame | None,
    *,
    probe_results: dict[str, ProbeResult],
    judge: JudgePanel | None,
) -> list[dict[str, Any]]:
    """One row per system, in config order: mean scores, cost, refusals, tools, routing, agent / gate behaviour and latency."""
    system_types = {cfg.resolved_name: cfg.type for cfg in systems}
    offered = tools_offered(systems)
    ok_rows = [row for row in per_question_rows if row["error"] is None]
    latency = pd.DataFrame([{"system": row["system"], "latency_ms": row["latency_ms"]} for row in ok_rows], columns=["system", "latency_ms"])
    errors = errors_by_system(per_question_rows)
    rows: list[dict[str, Any]] = []
    for system_name in system_types:
        r = _question_means(retrieval_df[retrieval_df["system"] == system_name] if not retrieval_df.empty else retrieval_df)
        a = _question_means(answer_df[answer_df["system"] == system_name] if not answer_df.empty else answer_df)
        c = cost_df[cost_df["system"] == system_name] if not cost_df.empty else cost_df
        latency_rows = latency[latency["system"] == system_name]
        n_ok = sum(1 for row in ok_rows if row["system"] == system_name)
        row: dict[str, Any] = {"system": system_name, "system_type": system_types[system_name], "n_ok": n_ok, "n_error": errors.get(system_name, 0)}
        # A system with no successful question has no measurements: leave them empty (shown as an em dash), never 0.
        for col in r.columns:
            if col not in {"system", "question_id", "category"}:
                mean = r[col].mean() if not r.empty else None
                row[f"retrieval_{col}"] = None if mean is None or pd.isna(mean) else float(mean)
        for col in ["correctness", "faithfulness", "completeness", "relevance", "citation_quality", "answer_score"]:
            row[col] = float(a[col].mean()) if not a.empty else None
        # Deterministic and context metrics are missing for a question that cannot have them (no reference, no keywords, no
        # retrieval); the mean covers the questions that do, and a system with none gets an empty cell.
        for col in ["exact_match", "token_f1", "keyword_recall", "context_recall", "context_precision"]:
            mean = pd.to_numeric(a[col], errors="coerce").mean() if col in a else float("nan")
            row[col] = None if pd.isna(mean) else float(mean)
        for col in CLAIM_COLUMNS:  # present only in a run that checked claims; a system whose answers had none gets an empty cell
            if col in a:
                mean = pd.to_numeric(a[col], errors="coerce").mean()
                row[col] = None if pd.isna(mean) else float(mean)
        row["avg_cost_per_question"] = float(c["total_cost"].mean()) if not c.empty else None
        ok_system_rows = [r for r in ok_rows if r["system"] == system_name]
        row.update(refusal_metrics(ok_system_rows))
        row["judge_fallback_rate"] = judge_fallback_rate(judge, ok_system_rows)
        row.update(tool_summary_fields(ok_system_rows, tooled=bool(offered[system_name]) or any(tool_steps(r) for r in ok_system_rows)))
        if (accuracy := route_accuracy(ok_system_rows)) is not None:
            row["route_accuracy"] = accuracy
        row.update(_agent_fields([r["agent"] for r in ok_rows if r["system"] == system_name and r.get("agent")]))
        row.update(_gate_fields([r["gate"] for r in ok_system_rows if r.get("gate")]))
        row.update(_latency_fields(latency_rows["latency_ms"].tolist(), probe_results.get(system_name)))
        if runtime_df is not None and not runtime_df.empty:
            runtime = runtime_df[runtime_df["system"] == system_name]
            row["system_wall_time_ms"] = float(runtime["system_wall_time_ms"].iloc[0]) if not runtime.empty else 0.0
        rows.append(row)
    return rows


def external_facts(per_question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the report must say about external systems: whose passages named no corpus document (with examples), whose passages it did not
    show at all, and whose cost is unknown. Empty when every external system mapped and priced everything (or there is none)."""
    unmapped: dict[str, dict[str, Any]] = {}
    hidden: list[str] = []
    unknown_cost: list[str] = []
    for row in per_question_rows:
        external = row.get("external")
        if not external:
            continue
        name = row["system"]
        if external.get("unmapped_doc_ids"):
            entry = unmapped.setdefault(name, {"questions": 0, "examples": []})
            entry["questions"] += 1
            entry["examples"] = list(dict.fromkeys([*entry["examples"], *external["unmapped_doc_ids"]]))[:5]
        if external.get("no_contexts") and name not in hidden:
            hidden.append(name)
        if not external.get("cost_known", True) and name not in unknown_cost:
            unknown_cost.append(name)
    facts: dict[str, Any] = {}
    if unmapped:
        facts["unmapped_doc_ids"] = unmapped
    if hidden:
        facts["no_contexts"] = hidden
    if unknown_cost:
        facts["unknown_cost"] = unknown_cost
    return facts
