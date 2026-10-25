from __future__ import annotations

from typing import Any

AGREEMENT_FLOOR = 0.667  # Krippendorff's alpha below which agreement does not support even tentative conclusions
POSITION_BIAS_NOTICE = 0.25  # share of pairwise comparisons whose verdict followed the answers' position
MOCK_NOTICE = "Mock run: scores validate the pipeline only (hashing embeddings, scripted LLM, heuristic judge). They say nothing about real-world quality."


def build_notices(
    mode: str | None,
    unknown_priced_models: list[str] | None = None,
    *,
    concurrent_latency: bool = False,
    judge_model: str | None = None,
    judge_shares_family_with: list[str] | None = None,
    judge_fallbacks: dict[str, float] | None = None,
    judge_agreement: float | None = None,
    pairwise_flip_rate: float | None = None,
    external: dict[str, Any] | None = None,
) -> list[str]:
    """Run-level caveats shown prominently at the top of every report."""
    notices: list[str] = []
    if mode == "mock":
        notices.append(MOCK_NOTICE)
    if unknown_priced_models:
        models = ", ".join(sorted(unknown_priced_models))
        notices.append(f"Cost under-reported: no price is registered for {models}. Add it under `pricing:` in the config.")
    external = external or {}
    if external.get("unmapped_doc_ids"):
        systems = "; ".join(f"{name} (e.g. {', '.join(entry['examples'][:3])})" for name, entry in external["unmapped_doc_ids"].items())
        notices.append(
            f"Retrieval not scored for {systems}: the passages they returned carry document ids that match no corpus document (by id, path, file name "
            "or stem), so their retrieval metrics are blank, not zero. Map `retrieval.response.doc_id` to your document ids (see docs/benchmarking-external-systems.md)."
        )
    if external.get("no_contexts"):
        notices.append(
            f"Retrieval not scored for {', '.join(external['no_contexts'])}: they return no passages (`retrieval.response.contexts` is unset), so their retrieval "
            "metrics are blank and the judge scores faithfulness against nothing."
        )
    if external.get("unknown_cost"):
        notices.append(
            f"Cost unknown for {', '.join(external['unknown_cost'])}: they report neither a cost nor token usage with a priced `retrieval.model`, so they show "
            "$0 per question. Return `cost_usd` (or token usage and declare the model) to compare cost."
        )
    if concurrent_latency:
        notices.append(
            "Latency was measured while questions ran concurrently, so it includes queueing behind other requests. "
            "Keep `evaluation.latency_probe_questions` above 0 for a clean one-at-a-time measurement."
        )
    if judge_shares_family_with:
        systems = ", ".join(judge_shares_family_with)
        notices.append(
            f"Self-preference risk: {judge_model} scored {systems}, whose generator is from the same model family. Judges tend to rate answers from "
            "their own family higher, so those scores may be inflated. Use a judge from another family under `evaluation.judge.model`, add one to "
            "`evaluation.judge.panel`, or set `evaluation.judge.independent: false` to accept this."
        )
    if judge_fallbacks:
        systems = ", ".join(f"{name} ({rate:.0%})" for name, rate in judge_fallbacks.items())
        notices.append(
            f"The LLM judge returned an unusable verdict for some questions and the heuristic judge scored them instead: {systems}. "
            "Their answer scores mix two judges; see `judge_fallback_rate` in `metrics_summary.csv`."
        )
    if judge_agreement is not None and judge_agreement < AGREEMENT_FLOOR:
        notices.append(
            f"The judges disagree: Krippendorff's alpha on the answer score is {judge_agreement:.2f}, below {AGREEMENT_FLOOR} (the usual floor for even tentative "
            "conclusions). Small score differences between systems are not trustworthy; see `judge` in `run_summary.json` for each axis."
        )
    if pairwise_flip_rate is not None and pairwise_flip_rate >= POSITION_BIAS_NOTICE:
        notices.append(
            f"Position bias: in {pairwise_flip_rate:.0%} of the pairwise comparisons the judge picked whichever answer stood in the same place in both "
            "orders. Those comparisons count as draws, so the arena is not skewed by it, but it rests on the remaining ones; consider another pairwise judge."
        )
    return notices
