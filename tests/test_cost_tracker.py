from ragbench.models.cost import CostBreakdown, estimate_model_cost


def test_cost_tracker_computes_non_negative_totals():
    first = CostBreakdown(embedding_input_tokens=100, embedding_cost=estimate_model_cost("text-embedding-3-small", 100))
    second = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=20, llm_cost=estimate_model_cost("gpt-6-luna", 10, 20))
    total = first.plus(second)

    assert total.total_cost >= 0
    assert total.embedding_input_tokens == 100
    assert total.llm_completion_tokens == 20


def test_unknown_model_cost_is_zero_but_recorded(caplog):
    from ragbench.models import cost

    cost.reset_unknown_priced_models()

    with caplog.at_level("WARNING"):
        assert cost.estimate_model_cost("gpt-9-unknown", 1_000_000, 1_000_000) == 0.0
        cost.estimate_model_cost("gpt-9-unknown", 10, 10)

    assert cost.unknown_priced_models() == ["gpt-9-unknown"]
    assert sum("gpt-9-unknown" in record.message for record in caplog.records) == 1  # warned once
    cost.reset_unknown_priced_models()


def test_known_and_mock_models_are_not_flagged_unknown():
    from ragbench.models import cost

    cost.reset_unknown_priced_models()
    cost.estimate_model_cost("gpt-6-luna", 10, 10)
    cost.estimate_model_cost("mock-llm", 10, 10)
    cost.estimate_model_cost("hashing-embedding", 10)

    assert cost.unknown_priced_models() == []


def test_dated_snapshot_resolves_to_base_model_price():
    from ragbench.models import cost

    cost.reset_unknown_priced_models()

    assert cost.estimate_model_cost("gpt-6-luna-2026-05-18", 1_000_000, 0) == cost.estimate_model_cost("gpt-6-luna", 1_000_000, 0) == 0.10
    assert cost.unknown_priced_models() == []


def test_register_pricing_overrides_and_can_be_cleared():
    from ragbench.models import cost

    cost.reset_unknown_priced_models()
    cost.register_pricing({"my-local-model": {"input": 1.0, "output": 2.0}, "gpt-6-luna": {"input": 10.0, "output": 0.0}})
    try:
        assert cost.estimate_model_cost("my-local-model", 1_000_000, 1_000_000) == 3.0
        assert cost.estimate_model_cost("gpt-6-luna", 1_000_000, 0) == 10.0
        assert cost.unknown_priced_models() == []
    finally:
        cost.clear_pricing_overrides()
    assert cost.estimate_model_cost("gpt-6-luna", 1_000_000, 0) == 0.10
    assert isinstance(cost.PRICING_AS_OF, str) and len(cost.PRICING_AS_OF) == 10
    cost.reset_unknown_priced_models()


# --- the price table's own rules (docs/superpowers plan, Task 8 step 4) ---------------------------------------------------


def _model_refs_named_in(paths) -> set[str]:
    import re

    refs: set[str] = set()
    for path in paths:
        for match in re.finditer(r"^\s*(?:generator|embedding|judge_model):\s*([^\s#]+)", path.read_text(encoding="utf-8"), flags=re.M):
            refs.add(match.group(1).strip("\"'"))
    return refs


def test_every_model_named_in_configs_docs_and_defaults_has_a_registered_price():
    from pathlib import Path

    from ragbench.config.schema import EvaluationConfig
    from ragbench.models import cost, defaults

    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "configs").glob("*.yaml")), root / "README.md", *sorted((root / "docs").glob("*.md"))]
    refs = _model_refs_named_in(files) | {
        defaults.DEFAULT_GENERATOR_MODEL,
        defaults.DEFAULT_EMBEDDING_MODEL,
        defaults.DEFAULT_JUDGE_MODEL,
        EvaluationConfig().judge_model,
    }
    local = ("local:", "openai_compatible:")
    unpriced = sorted(ref for ref in refs if not ref.startswith(local) and cost._lookup_pricing(ref) is None)
    assert unpriced == [], f"no price row for: {unpriced}"
    assert len(refs) >= 3  # the scan itself must find something


def test_no_priced_model_is_marked_deprecated():
    from ragbench.models import cost

    assert cost.DEPRECATED_MODELS == frozenset(), "delete deprecated models from the price table instead of listing them"
    assert not cost.DEPRECATED_MODELS & set(cost.MODEL_PRICING_USD_PER_1M)


def test_price_table_is_dated_and_current_generation_models_are_present():
    from datetime import date

    from ragbench.models import cost

    date.fromisoformat(cost.PRICING_AS_OF)
    for model in ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra", "text-embedding-3-small", "claude-haiku-4-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"):
        assert model in cost.MODEL_PRICING_USD_PER_1M, model
    assert "gpt-5.4-nano" not in cost.MODEL_PRICING_USD_PER_1M and "gpt-4o" not in cost.MODEL_PRICING_USD_PER_1M
