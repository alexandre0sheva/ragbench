from ragbench.utils.query_planning import generate_query_variants


def test_query_planner_splits_comparison_question():
    variants = generate_query_variants("Compare ClaimPilot and HarborShield AI by primary workflow.")

    assert "ClaimPilot primary workflow." in variants
    assert "HarborShield AI primary workflow." in variants


def test_query_planner_splits_multi_part_question():
    variants = generate_query_variants("What changed for ClaimPilot in Solstice 2.1 and what labels does ClaimPilot use?")

    assert any("What changed for ClaimPilot" in variant for variant in variants)
    assert any("what labels does ClaimPilot use" in variant for variant in variants)

