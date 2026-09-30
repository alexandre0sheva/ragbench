from __future__ import annotations

MOCK_NOTICE = "Mock run: scores validate the pipeline only (hashing embeddings, scripted LLM, heuristic judge). They say nothing about real-world quality."


def build_notices(mode: str | None, unknown_priced_models: list[str] | None = None, *, concurrent_latency: bool = False) -> list[str]:
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
    return notices
