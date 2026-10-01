from __future__ import annotations

MOCK_NOTICE = "Mock run: scores validate the pipeline only (hashing embeddings, scripted LLM, heuristic judge). They say nothing about real-world quality."


def build_notices(
    mode: str | None,
    unknown_priced_models: list[str] | None = None,
    *,
    concurrent_latency: bool = False,
    judge_model: str | None = None,
    judge_shares_model_with: list[str] | None = None,
    judge_fallbacks: dict[str, float] | None = None,
) -> list[str]:
    """Run-level caveats shown prominently at the top of every report."""
    notices: list[str] = []
    if mode == "mock":
        notices.append(MOCK_NOTICE)
    if unknown_priced_models:
        models = ", ".join(sorted(unknown_priced_models))
        notices.append(f"Cost under-reported: no price is registered for {models}. Add it under `pricing:` in the config.")
    if concurrent_latency:
        notices.append(
            "Latency was measured while questions ran concurrently, so it includes queueing behind other requests. "
            "Keep `evaluation.latency_probe_questions` above 0 for a clean one-at-a-time measurement."
        )
    if judge_shares_model_with:
        systems = ", ".join(judge_shares_model_with)
        notices.append(
            f"Self-preference risk: the judge ({judge_model}) is also the generator of {systems}. Judges tend to rate answers from their own model "
            "higher, so those scores may be inflated. Use a different model under `evaluation.judge.model`, or set `evaluation.judge.independent: false` to accept this."
        )
    if judge_fallbacks:
        systems = ", ".join(f"{name} ({rate:.0%})" for name, rate in judge_fallbacks.items())
        notices.append(
            f"The LLM judge returned an unusable verdict for some questions and the heuristic judge scored them instead: {systems}. "
            "Their answer scores mix two judges; see `judge_fallback_rate` in `metrics_summary.csv`."
        )
    return notices
