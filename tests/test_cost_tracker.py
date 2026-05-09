from ragbench.models.cost import CostBreakdown, estimate_model_cost


def test_cost_tracker_computes_non_negative_totals():
    first = CostBreakdown(embedding_input_tokens=100, embedding_cost=estimate_model_cost("text-embedding-3-small", 100))
    second = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=20, llm_cost=estimate_model_cost("gpt-5.4-nano", 10, 20))
    total = first.plus(second)

    assert total.total_cost >= 0
    assert total.embedding_input_tokens == 100
    assert total.llm_completion_tokens == 20
