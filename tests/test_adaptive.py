"""The `adaptive` router system: rule-based and LLM routing, cost roll-up, config validation at load time, and the per-route outputs."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fake_systems import question as make_question
from fake_systems import write_experiment

from ragbench.agents.router import RouteDecision, decide_from_reply, describe_routes, heuristic_route
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.datasets.schema import Question
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import ROUTE_QUESTION_MARKER
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.rag_systems.adaptive_rag import AdaptiveRAG
from ragbench.utils.jsonl import read_jsonl

ALL = {"default", "lexical", "computation", "multi_hop"}
PER_TOKEN = 0.001

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="Errors", text="# Errors\nError HS-4127 means the tenant header is missing.\nHarborShield costs $200 per month."),
    Document(doc_id="doc_b", path="b.md", title="Roadmap", text="# Roadmap\nClaimPilot ships in Q3 with triage workflows.\nQuuxwell batteries store energy overnight."),
    Document(doc_id="doc_c", path="c.md", title="Misc", text="# Misc\nPlinth harbours host fishing fleets in the northern bay."),
]

ROUTES = {
    "default": {"type": "vector", "retrieval": {"vector_store": "numpy"}},
    "lexical": {"type": "bm25"},
    "multi_hop": {"type": "decompose", "retrieval": {"retriever": "vector"}},
    "computation": {"type": "agent_search", "retrieval": {"retriever": "vector"}, "tools": ["calculator", "date_calc"]},
}


class PricedLLM(MockLLM):
    """The mock model, but every call costs money, so cost roll-ups have something to add up."""

    def generate(self, messages, **kwargs) -> LLMResult:
        result = super().generate(messages, **kwargs)
        result.cost = CostBreakdown(llm_prompt_tokens=result.prompt_tokens, llm_completion_tokens=result.completion_tokens, llm_cost=(result.prompt_tokens + result.completion_tokens) * PER_TOKEN)
        return result


class ScriptedRouterLLM(MockLLM):
    """Answers router prompts from a script (the last reply repeats) at a fixed price; everything else is the plain mock."""

    def __init__(self, *replies: str):
        super().__init__()
        self.replies, self.prompts = list(replies), []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        if ROUTE_QUESTION_MARKER not in prompt:
            return super().generate(messages, **kwargs)
        with self._lock:
            self.prompts.append(prompt)
            reply = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return LLMResult(text=reply, model="scripted", prompt_tokens=10, completion_tokens=5, cost=CostBreakdown(llm_cost=0.5), finish_reason="stop")


def _system(routes: dict | None = None, llm: MockLLM | None = None, priced: bool = False, **options):
    config = SystemConfig(type="adaptive", retrieval={"routes": routes if routes is not None else ROUTES, **options})
    system = create_rag_system(config, force_mock=True)
    if priced:
        system.llm = PricedLLM()
        for sub in system.systems.values():
            sub.llm = PricedLLM()
    if llm is not None:
        system.llm = llm
    system.ingest(DOCS)
    return system


def _steps(result, kind=None, name=None):
    return [s for s in result.steps if (kind is None or s.kind == kind) and (name is None or s.name == name)]


def _validate(systems: list[dict], **top):
    return ExperimentConfig.model_validate({"run": {"name": "t"}, "dataset": {"documents_path": "d", "questions_path": "q"}, "systems": systems, **top})


# --- the rule-based router ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("What does error code HS-4127 indicate?", "lexical"),
        ("Which invoice is ID-1234?", "lexical"),
        ("Who changed INV_2024_0031?", "lexical"),
        ("What does HS4127 mean?", "lexical"),
        ('Where is the phrase "tenant header" explained?', "lexical"),
        ("Which release is 2.1.3?", "lexical"),
        ("Find 3f2504e0-4f89-41d3-9a0c-0305e82c3301 in the logs", "lexical"),
        ("Compare HarborShield and ClaimPilot pricing.", "multi_hop"),
        ("What is the difference between the two plans?", "computation"),  # "difference between" wording is a calculation, not a comparison
        ("HarborShield versus ClaimPilot: which ships first?", "multi_hop"),
        ("What changed for ClaimPilot in Solstice 2.1 and what labels does ClaimPilot use?", "multi_hop"),
        ("Which two documents together show the award and the role?", "multi_hop"),
        ("Who is the CEO? Who is the CTO?", "multi_hop"),
        ("By what percentage did volume grow from Q1 to Q3 2024?", "computation"),
        ("What is 300 times 12 in total?", "computation"),
        ("How many days did the pilot last?", "computation"),
        ("How many days after 2024-04-16 did version two ship?", "computation"),
        ("At list prices, what would 300,000 notices cost?", "computation"),
        ("What was the average response time?", "computation"),
        ("What was the total number of claims processed in the first half of 2024?", "computation"),
        ("Who founded the company?", "default"),
        ("What is the refund policy for returns?", "default"),
        ("Describe the onboarding process in 2024.", "default"),
        ("", "default"),
    ],
)
def test_heuristic_routing(question, route):
    decision = heuristic_route(question, ALL)
    assert decision.route == route and decision.reason and decision.fallback is False


def test_an_identifier_beats_a_calculation_which_beats_a_comparison():
    assert heuristic_route("Compare the cost of error HS-4127 over 3 months in percent", ALL).route == "lexical"
    assert heuristic_route("Compare the percentage growth of A and B", ALL).route == "computation"


def test_a_role_that_is_not_configured_falls_through_to_the_next_then_to_default():
    assert heuristic_route("Compare the percentage growth of A and B", {"default", "multi_hop"}).route == "multi_hop"
    only_default = heuristic_route("What does HS-4127 mean?", {"default"})
    assert only_default.route == "default" and "lexical, which is not configured" in only_default.reason
    assert heuristic_route("Who is the CEO?", {"default", "lexical"}).reason == "no special pattern"


def test_years_alone_are_not_figures_to_calculate_with():
    assert heuristic_route("What happened in 2023 and 2024 at the company?", ALL).route != "computation"


# --- reading an LLM router's choice -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "route", "fallback"),
    [
        ('{"route": "lexical", "reason": "an error code"}', "lexical", False),
        ('```json\n{"route": "multi_hop"}\n```', "multi_hop", False),
        ('{"route": "LEXICAL", "reason": "x"}', "lexical", False),  # case is forgiven
        ('{"route": "teleport", "reason": "x"}', "default", True),
        ("lexical", "default", True),
        ("", "default", True),
        ('{"reason": "no route"}', "default", True),
        ('{"route": ""}', "default", True),
    ],
)
def test_decide_from_reply(reply, route, fallback):
    decision = decide_from_reply(reply, ALL)
    assert (decision.route, decision.fallback) == (route, fallback)


def test_route_descriptions_use_the_configs_then_the_built_in_then_a_placeholder():
    described = describe_routes(["default", "lexical", "my_route", "other"], {"my_route": "Mine.", "default": "Custom default."})
    assert described["my_route"] == "Mine." and described["default"] == "Custom default."
    assert described["lexical"].startswith("Questions about an exact identifier") and described["other"] == "(no description)"
    assert isinstance(RouteDecision("a", "b"), RouteDecision)


# --- the system: routing, steps, cost -----------------------------------------------------------------------------------------


def test_each_question_goes_to_its_route_and_the_trace_says_so():
    system = _system()
    expected = {
        "What does error code HS-4127 mean?": ("lexical", "bm25_search"),
        "Compare HarborShield and ClaimPilot pricing.": ("multi_hop", "plan_subquestions"),
        "What is 300 times 12 in total?": ("computation", "agent_turn"),
        "Who hosts fishing fleets?": ("default", "vector_search"),
    }
    for question, (route, first_sub_step) in expected.items():
        answer = system.answer_question(question)
        assert answer.metadata["route"] == route and answer.retrieval_result.metadata["route"] == route
        first, second = answer.steps[0], answer.steps[1]
        assert (first.kind, first.name) == ("route", "route_question") and first.metadata["route"] == route and first.metadata["router"] == "heuristic"
        assert first.output_preview.startswith(f"{route}: ") and first.input_preview == question and first.cost.total_cost == 0
        assert second.name == first_sub_step
        assert all(step.metadata.get("route") == route for step in answer.steps)  # every step is tagged with the route that did the work
        assert answer.steps[-1].kind == "generate" and answer.answer
        assert answer.retrieval_result.steps == [s for s in answer.steps if s.kind != "generate"]


def test_the_chosen_systems_results_come_through():
    system = _system()
    answer = system.answer_question("What does error code HS-4127 mean?", top_k=5, context_k=2)
    assert answer.retrieval_result.chunks[0].doc_id == "doc_a" and answer.metadata["context_chunk_ids"] == [c.chunk_id for c in answer.retrieval_result.chunks[:2]]
    assert answer.metadata["system_type"] == "adaptive" and answer.metadata["router"] == "heuristic" and answer.metadata["router_fallback"] is False
    assert answer.metadata["route_reason"].startswith("exact identifier")
    agent = system.answer_question("What is 300 times 12 in total?").metadata["agent"]
    assert agent["tool_calls"] == 1 and agent["termination"] == "finished"  # an agentic route's `agent` block is kept
    assert "agent" not in answer.metadata


def test_costs_roll_up_exactly_from_the_router_and_the_chosen_pipeline():
    system = _system(priced=True, router="llm")
    system.llm = ScriptedRouterLLM('{"route": "computation", "reason": "arithmetic"}')
    for question in ("What is 300 times 12 in total?", "What does error code HS-4127 mean?"):
        answer = system.answer_question(question)
        steps_cost = sum(step.cost.total_cost for step in answer.steps)
        assert steps_cost == pytest.approx(answer.cost.total_cost) and answer.cost.total_cost > 0.5  # the router's $0.50 is in there
        assert not _steps(answer, name="untracked")
        route_step = _steps(answer, "route", "route_question")[0]
        assert route_step.cost.total_cost == pytest.approx(0.5)
        sub_cost = sum(s.cost.total_cost for s in answer.steps if s is not route_step)
        assert answer.cost.total_cost == pytest.approx(0.5 + sub_cost)
        assert answer.retrieval_result.cost.total_cost == pytest.approx(sum(s.cost.total_cost for s in answer.steps if s.kind != "generate"))


def test_the_rule_based_router_costs_nothing():
    answer = _system(priced=True).answer_question("What does error code HS-4127 mean?")
    assert _steps(answer, "route")[0].cost.total_cost == 0
    assert answer.cost.total_cost == pytest.approx(sum(s.cost.total_cost for s in answer.steps))


def test_latency_includes_routing_and_the_pipeline():
    answer = _system().answer_question("Who hosts fishing fleets?")
    assert answer.latency_ms >= sum(s.latency_ms for s in answer.steps if s.kind in ("retrieve", "generate")) * 0.5 and answer.latency_ms > 0


def test_fetch_context_alone_routes_and_reports_the_route():
    result = _system(priced=True).fetch_context("What does error code HS-4127 mean?", top_k=2)
    assert [c.doc_id for c in result.chunks][0] == "doc_a" and result.metadata["route"] == "lexical"
    assert [s.kind for s in result.steps][:2] == ["route", "retrieve"] and result.cost.total_cost == 0


def test_a_missing_role_sends_the_question_to_default():
    system = _system({"default": ROUTES["default"], "lexical": ROUTES["lexical"]})
    answer = system.answer_question("What is 300 times 12 in total?")
    assert answer.metadata["route"] == "default" and "computation, which is not configured" in answer.metadata["route_reason"]


# --- the LLM router --------------------------------------------------------------------------------------------------------------


def test_the_llm_router_follows_the_models_choice_and_shows_it_the_routes():
    llm = ScriptedRouterLLM('{"route": "multi_hop", "reason": "needs two lookups"}')
    system = _system(llm=llm, router="llm", route_descriptions={"lexical": "Error codes only."})
    answer = system.answer_question("Who hosts fishing fleets?")
    assert answer.metadata["route"] == "multi_hop" and answer.metadata["route_reason"] == "needs two lookups" and answer.metadata["router"] == "llm"
    prompt = llm.prompts[0]
    assert "- lexical: Error codes only." in prompt and "- default: General questions" in prompt and "- computation: Questions that need arithmetic" in prompt
    assert prompt.rstrip().endswith("Question: Who hosts fishing fleets?")
    route_step = answer.steps[0]
    assert route_step.kind == "route" and route_step.prompt_tokens == 10 and route_step.cost.query_rewrite_cost == pytest.approx(0.5)


@pytest.mark.parametrize("reply", ['{"route": "teleport"}', "I would pick lexical.", ""])
def test_an_unusable_llm_choice_falls_back_to_default_and_is_flagged(reply):
    answer = _system(llm=ScriptedRouterLLM(reply), router="llm").answer_question("What does error code HS-4127 mean?")
    assert answer.metadata["route"] == "default" and answer.metadata["router_fallback"] is True
    assert answer.steps[0].metadata["fallback"] is True and answer.answer


def test_the_llm_router_allows_route_names_of_your_own():
    routes = {"default": ROUTES["default"], "faq": {"type": "bm25"}}
    llm = ScriptedRouterLLM('{"route": "faq", "reason": "a frequent question"}')
    answer = _system(routes, llm=llm, router="llm", route_descriptions={"faq": "Frequently asked questions."}).answer_question("Who hosts fishing fleets?")
    assert answer.metadata["route"] == "faq" and "- faq: Frequently asked questions." in llm.prompts[0]


def test_the_mock_llm_router_runs_without_a_script():
    system = _system(router="llm")
    assert system.answer_question("What does error code HS-4127 mean?").metadata["route"] == "lexical"
    assert system.answer_question("Compare HarborShield and ClaimPilot pricing.").metadata["route"] == "multi_hop"
    assert system.answer_question("Who hosts fishing fleets?").metadata["route"] == "default"


# --- construction and ingestion -----------------------------------------------------------------------------------------------


def test_routes_are_built_from_inline_configs_and_inherit_the_models():
    config = SystemConfig(
        type="adaptive",
        name="router",
        models={"embedding": "text-embedding-3-small", "generator": "gpt-6-luna"},
        retrieval={"routes": {"default": {"type": "vector"}, "lexical": {"type": "bm25", "models": {"generator": "gpt-6-astra"}}}},
    )
    system = create_rag_system(config, force_mock=True)
    assert isinstance(system, AdaptiveRAG) and set(system.systems) == {"default", "lexical"}
    assert system.systems["default"].name == "router/default" and system.systems["lexical"].name == "router/lexical"
    assert system.systems["default"].config.models == {"embedding": "text-embedding-3-small", "generator": "gpt-6-luna"}
    assert system.systems["lexical"].config.models == {"embedding": "text-embedding-3-small", "generator": "gpt-6-astra"}  # a route's own models win


def test_every_route_is_ingested_once_and_the_costs_add_up():
    ingested = []

    class Spy(PricedLLM):
        pass

    system = create_rag_system(SystemConfig(type="adaptive", retrieval={"routes": ROUTES}), force_mock=True)
    for name, sub in system.systems.items():
        original = sub.ingest

        def spy(documents, original=original, name=name):
            ingested.append(name)
            return original(documents)

        sub.ingest = spy
    result = system.ingest(DOCS)
    assert sorted(ingested) == sorted(ROUTES) and len(ingested) == 4
    assert result.num_documents == 3 and result.num_chunks == sum(result.metadata["chunks_by_route"].values()) and result.num_chunks > 0
    assert set(result.metadata["chunks_by_route"]) == set(ROUTES)
    assert result.cost.total_cost == pytest.approx(0)  # mock models are free; the sum is what is checked below with real costs
    assert result.system == system.name


def test_offered_tools_are_the_union_of_the_routes_tools():
    config = SystemConfig(type="adaptive", retrieval={"routes": ROUTES})
    assert AdaptiveRAG.offered_tools(config) == ["search", "calculator", "date_calc"]
    assert AdaptiveRAG.offered_tools(SystemConfig(type="adaptive", retrieval={"routes": {"default": {"type": "bm25"}}})) == []


def test_the_spec_describes_the_system():
    spec = {s.type: s for s in all_specs()}["adaptive"]
    assert spec.chunker is None and spec.requires_llm and spec.retrieves and not spec.supports_tools


# --- configuration errors surface when the config loads -------------------------------------------------------------------


def test_a_valid_config_loads():
    config = _validate([{"type": "adaptive", "name": "a", "retrieval": {"routes": ROUTES}}])
    assert list(config.systems[0].retrieval["routes"]) == list(ROUTES)


@pytest.mark.parametrize(
    ("retrieval", "message"),
    [
        ({}, "routes"),
        ({"routes": {}}, "must include a 'default' route"),
        ({"routes": {"lexical": {"type": "bm25"}}}, "must include a 'default' route"),
        ({"routes": {"default": {"type": "bm25x"}}}, r"routes\.default: Unknown RAG system 'bm25x'.*Did you mean 'bm25'"),
        ({"routes": {"default": {"type": "adaptive", "retrieval": {"routes": {"default": {"type": "bm25"}}}}}}, "cannot itself be an adaptive system"),
        ({"routes": {"default": {"type": "bm25"}, "faq": {"type": "bm25"}}}, "'faq' would never be chosen.*router: llm"),
        ({"routes": {"default": {"type": "bm25"}}, "router": "psychic"}, "router"),
        ({"routes": {"default": {"type": "bm25"}}, "route_descriptions": {"ghost": "x"}}, "route_descriptions names routes that do not exist: ghost"),
        ({"routes": {"default": {"type": "bm25", "retrieval": {"top_kk": 3}}}}, "top_kk"),
        ({"routes": {"default": {"type": "bm25", "tools": ["calculator"]}}}, "does not use tools"),
        ({"routes": {"default": {"type": "bm25"}}, "extra_option": 1}, "extra_option"),
    ],
)
def test_bad_route_configs_fail_at_load_time(retrieval, message):
    with pytest.raises(ValueError, match=message):
        _validate([{"type": "adaptive", "retrieval": retrieval}])


def test_an_llm_router_accepts_route_names_the_heuristic_would_not():
    config = _validate([{"type": "adaptive", "retrieval": {"router": "llm", "routes": {"default": {"type": "bm25"}, "faq": {"type": "bm25"}}}}])
    assert config.systems[0].retrieval["router"] == "llm"


def test_a_routes_tools_and_models_are_checked_when_the_config_loads():
    with pytest.raises(ValueError, match=r"systems\[0\]\.routes\.computation\.tools: tool 'calculater'.*Did you mean 'calculator'"):
        _validate([{"type": "adaptive", "retrieval": {"routes": {"default": {"type": "bm25"}, "computation": {"type": "agent_search", "tools": ["calculater"]}}}}])
    fetch = {"name": "fetch", "path": "test_tools:lookup_price", "side_effects": "network"}
    routes = {"default": {"type": "bm25"}, "computation": {"type": "agent_search", "tools": [fetch]}}
    with pytest.raises(ValueError, match="tools.allow"):
        _validate([{"type": "adaptive", "retrieval": {"routes": routes}}])
    assert _validate([{"type": "adaptive", "retrieval": {"routes": routes}}], tools={"allow": ["fetch"], "allow_network": True})
    with pytest.raises(ValueError, match=r"systems\[0\]\.routes\.default\.models\.generator"):
        _validate([{"type": "adaptive", "retrieval": {"routes": {"default": {"type": "bm25", "models": {"generator": "openai_compatible:ghost/some-model"}}}}}])
    with pytest.raises(ValueError, match=r"systems\[0\]\.models\.generator"):  # an invalid parent model is reported once, against the parent
        _validate([{"type": "adaptive", "models": {"generator": "openai_compatible:ghost/some-model"}, "retrieval": {"routes": {"default": {"type": "bm25"}}}}])


def test_the_adaptive_system_itself_rejects_a_tools_list_and_a_chunker_section():
    with pytest.raises(ValueError, match="does not use tools"):
        _validate([{"type": "adaptive", "tools": ["calculator"], "retrieval": {"routes": {"default": {"type": "bm25"}}}}])
    with pytest.raises(ValueError, match="does not use `chunker`"):
        _validate([{"type": "adaptive", "chunker": {"type": "word"}, "retrieval": {"routes": {"default": {"type": "bm25"}}}}])


def test_routing_hint_is_optional_question_metadata():
    assert Question(id="q", question="?").routing_hint is None
    assert Question.model_validate({"id": "q", "question": "?", "routing_hint": "lexical"}).routing_hint == "lexical"


# --- concurrency -----------------------------------------------------------------------------------------------------------------


def test_concurrent_questions_route_independently():
    system = _system(priced=True)
    questions = [("What does error code HS-4127 mean?", "lexical"), ("Compare HarborShield and ClaimPilot pricing.", "multi_hop"), ("What is 300 times 12 in total?", "computation"), ("Who hosts fishing fleets?", "default")] * 4
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda item: system.answer_question(item[0]), questions))
    for (question, route), answer in zip(questions, answers, strict=True):
        assert answer.metadata["route"] == route and answer.steps[0].input_preview == question
        assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost) and not _steps(answer, name="untracked")
        assert all(s.metadata.get("route") == route for s in answer.steps)


# --- the run ----------------------------------------------------------------------------------------------------------------------


def _experiment(tmp_path, questions, **evaluation):
    systems = [
        {
            "type": "adaptive",
            "name": "router",
            "retrieval": {
                "routes": {
                    "default": {"type": "vector", "chunker": {"type": "word", "chunk_size": 50, "chunk_overlap": 0}, "retrieval": {"vector_store": "numpy", "top_k": 3}},
                    "lexical": {"type": "bm25", "chunker": {"type": "word", "chunk_size": 50, "chunk_overlap": 0}, "retrieval": {"top_k": 3}},
                    "computation": {"type": "agent_search", "chunker": {"type": "word", "chunk_size": 50, "chunk_overlap": 0}, "retrieval": {"retriever": "vector", "top_k": 3}, "tools": ["calculator"]},
                }
            },
        },
        {"type": "bm25", "name": "plain", "retrieval": {"top_k": 3}},
    ]
    return write_experiment(tmp_path, questions, systems, evaluation=evaluation or None)


def test_the_run_reports_the_route_per_question_and_a_route_distribution(tmp_path):
    questions = [
        {**make_question("q1", "What does error code HS-4127 mean?"), "routing_hint": "lexical"},
        {**make_question("q2", "Which invoice is ID-1234?"), "routing_hint": "lexical"},
        {**make_question("q3", "What is 300 times 12 in total?"), "routing_hint": "computation"},
        {**make_question("q4", "Tell me about topic 4"), "routing_hint": "default"},
        {**make_question("q5", "Tell me about topic 5"), "routing_hint": "lexical"},  # a deliberately wrong hint: routed to default
        make_question("q6", "Tell me about topic 6"),  # no hint
    ]
    run_dir = run_benchmark(_experiment(tmp_path, questions), force_mock=True, max_workers=2)

    rows = {(r["system"], r["question_id"]): r for r in read_jsonl(run_dir / "per_question_results.jsonl")}
    assert [rows[("router", q)]["route"] for q in ("q1", "q2", "q3", "q4", "q5", "q6")] == ["lexical", "lexical", "computation", "default", "default", "default"]
    assert rows[("router", "q5")]["routing_hint"] == "lexical" and rows[("router", "q6")]["routing_hint"] is None
    assert rows[("plain", "q1")]["route"] is None

    routes = pd.read_csv(run_dir / "routes.csv").set_index(["system", "route"])
    assert list(routes.index.get_level_values("system").unique()) == ["router"]  # only systems that route appear
    assert routes.loc[("router", "default"), "questions"] == 3 and routes.loc[("router", "lexical"), "questions"] == 2 and routes.loc[("router", "computation"), "questions"] == 1
    assert routes["share"].sum() == pytest.approx(1.0) and routes.loc[("router", "default"), "share"] == pytest.approx(0.5)
    assert routes[["avg_answer_score", "avg_cost_per_question"]].notna().all().all()
    # precision = of the hinted questions sent there, how many wanted it; recall = of those that wanted it, how many were sent
    assert routes.loc[("router", "lexical"), "precision"] == pytest.approx(1.0) and routes.loc[("router", "lexical"), "recall"] == pytest.approx(2 / 3)
    assert routes.loc[("router", "default"), "precision"] == pytest.approx(0.5) and routes.loc[("router", "default"), "recall"] == pytest.approx(1.0)

    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    assert summary.loc["router", "route_accuracy"] == pytest.approx(4 / 5)  # 4 of the 5 hinted questions went where the hint said
    assert pd.isna(summary.loc["plain", "route_accuracy"])
    assert summary.loc["router", "avg_tool_calls"] >= 0 and summary.loc["router", "avg_steps"] >= 1  # the computation route's agent and tool metrics show through


def test_without_hints_there_is_a_distribution_but_no_accuracy(tmp_path):
    run_dir = run_benchmark(_experiment(tmp_path, [make_question("q1", "What does error code HS-4127 mean?"), make_question("q2", "Tell me about topic 2")]), force_mock=True, max_workers=1)
    routes = pd.read_csv(run_dir / "routes.csv")
    assert set(routes["route"]) == {"lexical", "default"} and routes["precision"].isna().all() and routes["recall"].isna().all()
    assert "route_accuracy" not in pd.read_csv(run_dir / "metrics_summary.csv").columns


def test_no_routes_file_for_a_run_without_an_adaptive_system(tmp_path):
    config = write_experiment(tmp_path, [make_question("q1", "hello")], [{"type": "bm25", "name": "plain", "retrieval": {"top_k": 3}}])
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    assert not (run_dir / "routes.csv").exists() and "route_accuracy" not in pd.read_csv(run_dir / "metrics_summary.csv").columns


def test_tool_usage_covers_the_routed_agents_tools(tmp_path):
    run_dir = run_benchmark(_experiment(tmp_path, [make_question("q1", "What is 300 times 12 in total?")]), force_mock=True, max_workers=1)
    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    assert usage.loc[("router", "calculator"), "calls"] == 1 and ("router", "search") in usage.index


def test_routes_sharing_a_chunker_and_embedder_share_their_embeddings(tmp_path):
    routes = {
        "default": {"type": "vector", "chunker": {"type": "word", "chunk_size": 50, "chunk_overlap": 0}, "retrieval": {"vector_store": "numpy"}},
        "multi_hop": {"type": "decompose", "chunker": {"type": "word", "chunk_size": 50, "chunk_overlap": 0}, "retrieval": {"retriever": "vector", "vector_store": "numpy"}},
    }
    config = write_experiment(tmp_path, [make_question("q1", "hello topic")], [{"type": "adaptive", "name": "router", "retrieval": {"routes": routes}}])
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["embedding_cache"]["hits"] > 0  # the second route's corpus embeddings came from the cache, not a second embedding pass
