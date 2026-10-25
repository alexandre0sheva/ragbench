"""The files a finished run writes into its directory: tables, statistics, leaderboard, run summary, manifest, recommendation, report.

Everything here is computed from the run's rows; `ragbench report RUN` rebuilds the report from these files alone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from ragbench.cache import summarize_cache
from ragbench.config.schema import SelectionConfig
from ragbench.evaluation.arena import ARENA_SUMMARY_COLUMNS
from ragbench.evaluation.judge_panel import panel_agreement
from ragbench.evaluation.manifest import build_manifest, utc_now_iso
from ragbench.evaluation.pairwise_run import summarize
from ragbench.evaluation.rows import RunRows
from ragbench.evaluation.run_stats import SIGNIFICANCE_COLUMNS, RunStats, compute_run_stats
from ragbench.evaluation.summary import (
    build_qrels_audit,
    build_routes,
    build_summary,
    build_tool_usage,
    errors_by_system,
    external_facts,
    judge_notices,
)
from ragbench.models import cost as pricing
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.models.providers.openrouter import manifest_settings
from ragbench.models.refs import answering_model
from ragbench.reporting.html_report import write_report as write_html_report
from ragbench.reporting.markdown_report import stage_summary, write_failures, write_leaderboard, write_qrels_audit
from ragbench.reporting.notices import build_notices
from ragbench.selection.recommend import recommend
from ragbench.selection.recommend import write_recommendation as write_recommendation_files
from ragbench.utils.jsonl import write_jsonl

if TYPE_CHECKING:
    from ragbench.evaluation.evaluator import BenchmarkEvaluator

logger = logging.getLogger(__name__)


def write_config_copy(target: Path, config_path: Path, raw_config: dict[str, Any], config_given: bool) -> None:
    """`config.yaml` in the run directory: the file as written when nothing changed (comments and all), else the config as it ran
    (sweeps expanded, a preset's systems, a `--systems` subset), so the directory always says exactly what was benchmarked."""
    if not config_given and config_path.exists():
        text = config_path.read_text(encoding="utf-8")
        if (yaml.safe_load(text) or {}) == raw_config:
            target.write_text(text, encoding="utf-8")
            return
    target.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")


def write_report(output_dir: Path, max_embedded_mb: float) -> None:
    """`report.html` and `report_data.json`, built from the files this run just wrote (so the same call can rebuild a report later)."""
    try:
        write_html_report(output_dir, max_embedded_mb=max_embedded_mb)
    except Exception as exc:  # the results are already on disk; a page that cannot be drawn must not turn the run into an error
        logger.warning("Could not write the HTML report: %s: %s", type(exc).__name__, exc)


def write_recommendation(output_dir: Path, selection: SelectionConfig) -> None:
    """Which system to deploy (`recommendation.json` / `.md`, and `winner.yaml` when there is one), from this run's results."""
    try:
        write_recommendation_files(output_dir, recommend(output_dir, constraints=selection.constraints, weights=selection.weights, profile=selection.profile))
    except Exception as exc:  # the results are already on disk; a recommendation that cannot be made must not turn the run into an error
        logger.warning("Could not write the recommendation: %s: %s", type(exc).__name__, exc)


def _compute_stats(run: BenchmarkEvaluator, per_question_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> RunStats:
    evaluation = run.config.evaluation
    return compute_run_stats(
        per_question_rows,
        summary_rows,
        k_values=evaluation.k_values,
        primary_k=evaluation.resolved_primary_k,
        n_boot=evaluation.stats.n_boot,
        seed=evaluation.stats.seed,
        baseline=evaluation.stats.baseline,
        trials=run.plan.trials if run.plan is not None else 1,
    )


def _write_pairwise(run: BenchmarkEvaluator, out: Path, summary_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """`pairwise.jsonl` and `arena.csv`, the arena columns of `summary_rows`, and the `pairwise` block of the run summary (None without the stage)."""
    if run.pairwise is None:
        return None
    stats = run.config.evaluation.stats
    result = summarize(run.pairwise, [row["system"] for row in summary_rows], n_boot=stats.n_boot, seed=stats.seed)
    arena = {row["system"]: row for row in result["arena"]}
    for summary_row in summary_rows:
        summary_row.update({key: arena[summary_row["system"]][key] for key in ARENA_SUMMARY_COLUMNS})
    write_jsonl(out / "pairwise.jsonl", run.pairwise.records)
    pd.DataFrame(result["arena"]).to_csv(out / "arena.csv", index=False)
    settings = run.config.evaluation.pairwise
    return {**result["summary"], "max_pairs_per_question": settings.max_pairs_per_question, "win_rates": result["win_rates"]}


def write_outputs(run: BenchmarkEvaluator, rows: RunRows, run_wall_time_ms: float) -> None:
    """Every output file of a run that has at least one finished system, report last (it reads all the others)."""
    out = run.output_dir
    evaluation = run.config.evaluation
    systems = run.reported_systems
    per_question_rows = rows.per_question
    write_jsonl(out / "per_question_results.jsonl", per_question_rows)
    retrieval_df = pd.DataFrame(rows.retrieval)
    answer_df = pd.DataFrame(rows.answer)
    cost_df = pd.DataFrame([*rows.ingestion, *rows.cost])
    runtime_df = pd.DataFrame(rows.system_runtime)
    retrieval_df.to_csv(out / "retrieval_metrics.csv", index=False)
    answer_df.to_csv(out / "answer_metrics.csv", index=False)
    cost_df.to_csv(out / "cost_breakdown.csv", index=False)
    runtime_df.to_csv(out / "system_runtime.csv", index=False)

    summary_rows = build_summary(
        systems,
        retrieval_df,
        answer_df,
        pd.DataFrame(rows.cost),
        per_question_rows,
        runtime_df,
        probe_results=run.probe_results,
        judge=run.judge,
    )
    run_stats = _compute_stats(run, per_question_rows, summary_rows)
    for summary_row in summary_rows:
        summary_row.update(run_stats.summary_fields[summary_row["system"]])
    pairwise = _write_pairwise(run, out, summary_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(run_stats.significance_rows, columns=SIGNIFICANCE_COLUMNS).to_csv(out / "significance.csv", index=False)
    (out / "stats.json").write_text(json.dumps(run_stats.stats, indent=2), encoding="utf-8")
    (out / "pareto.json").write_text(json.dumps(run_stats.pareto, indent=2), encoding="utf-8")
    if routes := build_routes(systems, per_question_rows):
        pd.DataFrame(routes).to_csv(out / "routes.csv", index=False)
    if tool_usage := build_tool_usage(systems, per_question_rows):
        pd.DataFrame(tool_usage).to_csv(out / "tool_usage.csv", index=False)
    unknown_priced = pricing.unknown_priced_models()
    # Pairwise judging is charged like pointwise judging (standalone prices, cache hits included), but to the run rather than a system.
    charged = float(sum(row.get("total_cost", 0.0) for row in [*rows.ingestion, *rows.cost])) + (pairwise or {}).get("cost_usd", 0.0)
    in_process = EMBEDDING_CACHE.stats()
    probe_cost = sum(probe.cost_usd for probe in run.probe_results.values())
    probe_calls = sum(probe.calls for probe in run.probe_results.values())
    # Live runs without a probe report latency measured under concurrency: say so next to the numbers.
    concurrent_latency = run.mode == "live" and not probe_calls and run.max_workers * run.system_workers > 1
    cache_summary = summarize_cache(
        run.cache_runtime,
        charged_cost_usd=charged,
        in_process_saved_usd=float(in_process["saved_cost_usd"]),
        reason=run.cache_reason,
        extra_spend_usd=probe_cost,
    )
    panel = run.judge is not None and run.judge.is_panel
    agreement = panel_agreement(per_question_rows) if panel else {}
    generators = {cfg.resolved_name: answering_model(cfg) for cfg in systems}
    external = external_facts(per_question_rows)
    notices = build_notices(
        run.mode,
        unknown_priced,
        external=external,
        concurrent_latency=concurrent_latency,
        pairwise_flip_rate=(pairwise or {}).get("position_bias", {}).get("flip_rate") if run.mode == "live" else None,
        **judge_notices(run.judge, generators, summary_rows, agreement),
    )
    # Only a panel gets a block: a one-judge run's summary stays as it was.
    judge_block = {"judge": {"models": evaluation.judge_models, "aggregate": evaluation.judge.aggregate, "independent": evaluation.judge.independent, "agreement": agreement}} if panel else {}
    write_leaderboard(
        out / "leaderboard.md",
        summary_rows,
        notices=notices,
        primary_k=evaluation.resolved_primary_k,
        stage_rows=stage_summary(per_question_rows, [cfg.resolved_name for cfg in systems]),
        significance_rows=run_stats.significance_rows,
        stats_info=run_stats.stats,
        pairwise=pairwise,
        split=run.split_info,
    )
    write_failures(out / "failures.md", per_question_rows)
    qrels_audit_rows = build_qrels_audit(per_question_rows, evaluation.resolved_primary_k)
    pd.DataFrame(qrels_audit_rows).to_csv(out / "qrels_audit.csv", index=False)
    write_qrels_audit(out / "qrels_audit.md", qrels_audit_rows, primary_k=evaluation.resolved_primary_k)
    errors = errors_by_system(per_question_rows)
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": run.run_id,
                "mode": run.mode,
                "models_used": sorted(run.models_used),
                "unknown_priced_models": unknown_priced,
                "pricing_as_of": pricing.PRICING_AS_OF,
                "run_wall_time_ms": run_wall_time_ms,
                "max_workers": run.max_workers,
                "execution": {
                    "max_workers": run.max_workers,
                    "system_workers": run.system_workers,
                    "ingest_workers": evaluation.ingest_workers,
                    "latency_probe_questions": evaluation.latency_probe_questions,
                    "latency_source": "probe" if probe_calls else "concurrent",
                    "probe_calls": probe_calls,
                    "probe_cost_usd": probe_cost,
                },
                "num_systems": len(systems),
                "budget": run.budget_summary([cfg.resolved_name for cfg in systems], {}),
                "num_question_rows": len(per_question_rows),
                "num_errors": sum(errors.values()),
                "errors_by_system": errors,
                "k_values": evaluation.k_values,
                "retrieval_depth": evaluation.resolved_retrieval_depth,
                "context_k": evaluation.context_k,
                "primary_k": evaluation.resolved_primary_k,
                "embedding_cache": EMBEDDING_CACHE.stats(),
                "cache": cache_summary,
                "dataset": run.dataset_info,
                "notices": notices,
                **judge_block,
                **({"external": external} if external else {}),
                **({"pairwise": pairwise} if pairwise is not None else {}),
                "dataset_warnings": run.dataset_warnings,
                "document_warnings": run.document_warnings,
                "outputs": {
                    "leaderboard": "leaderboard.md",
                    "report": "report.html",
                    "qrels_audit": "qrels_audit.md",
                    "stats": "stats.json",
                    "significance": "significance.csv",
                    "pareto": "pareto.json",
                    "recommendation": "recommendation.md",
                    **({"arena": "arena.csv", "pairwise": "pairwise.jsonl"} if pairwise is not None else {}),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(
        config_path=run.config_path,
        raw_config=run.raw_config,
        dataset_digest=run.dataset_digest,
        mode=run.mode,
        models_used=sorted(run.models_used),
        started_utc=run.started_utc,
        finished_utc=utc_now_iso(),
        model_settings=manifest_settings(sorted(run.models_used), run.config.openrouter),
        split=run.split_info,
    )
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_recommendation(out, run.config.selection)
    write_report(out, run.config.report.max_embedded_mb)  # last: it reads the recommendation, the manifest and every other file above
