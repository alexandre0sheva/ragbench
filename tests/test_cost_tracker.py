from ragbench.models.cost import CostBreakdown, estimate_model_cost


def test_cost_tracker_computes_non_negative_totals():
    first = CostBreakdown(embedding_input_tokens=100, embedding_cost=estimate_model_cost("text-embedding-3-small", 100))
    second = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=20, llm_cost=estimate_model_cost("gpt-5.4-nano", 10, 20))
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
    cost.estimate_model_cost("gpt-5.4-nano", 10, 10)
    cost.estimate_model_cost("mock-llm", 10, 10)
    cost.estimate_model_cost("hashing-embedding", 10)

    assert cost.unknown_priced_models() == []


def test_dated_snapshot_resolves_to_base_model_price():
    from ragbench.models import cost

    cost.reset_unknown_priced_models()

    assert cost.estimate_model_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0) == cost.estimate_model_cost("gpt-4o-mini", 1_000_000, 0)
    assert cost.unknown_priced_models() == []


def test_register_pricing_overrides_and_can_be_cleared():
    from ragbench.models import cost

    cost.reset_unknown_priced_models()
    cost.register_pricing({"my-local-model": {"input": 1.0, "output": 2.0}, "gpt-5.4-nano": {"input": 10.0, "output": 0.0}})
    try:
        assert cost.estimate_model_cost("my-local-model", 1_000_000, 1_000_000) == 3.0
        assert cost.estimate_model_cost("gpt-5.4-nano", 1_000_000, 0) == 10.0
        assert cost.unknown_priced_models() == []
    finally:
        cost.clear_pricing_overrides()
    assert cost.estimate_model_cost("gpt-5.4-nano", 1_000_000, 0) == 0.20
    assert isinstance(cost.PRICING_AS_OF, str) and len(cost.PRICING_AS_OF) == 10
    cost.reset_unknown_priced_models()
