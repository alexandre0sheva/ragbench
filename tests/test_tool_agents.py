"""`agent_search` and `grep_agent`: native function calling and ReAct-JSON, budgets, tracing, and how the evaluator reports them."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fake_systems import question as make_question
from fake_systems import write_experiment

from ragbench.agents.react_json import describe_tools, parse_action
from ragbench.config.schema import ExperimentConfig, SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM, ToolCall
from ragbench.models.prompts import FORCE_ANSWER_MARKER
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.tools import ToolBox
from ragbench.utils.jsonl import read_jsonl

PER_CALL = 0.01

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="Pricing", text="# Pricing\nHarborShield costs $200 per month.\nThe marine module is extra.\nError code HS-4127 means a missing tenant header."),
    Document(doc_id="doc_b", path="b.md", title="Roadmap", text="# Roadmap\nClaimPilot ships in Q3.\nQuuxwell batteries store energy overnight."),
    Document(doc_id="doc_c", path="c.md", title="Misc", text="# Misc\nPlinth harbours host fishing fleets in the northern bay."),
]
QUESTION = "How much does HarborShield cost per month?"


def tool_turn(*calls: tuple[str, dict]) -> dict:
    return {"calls": list(calls)}


def say(text: str) -> dict:
    return {"text": text}


class ScriptedToolLLM(MockLLM):
    """Plays a script of agent turns. A turn is `tool_turn((name, args), ...)` (native tool calls) or `say(text)`; the last turn repeats.
    Records every request (messages and the tools offered) and charges a fixed price per call."""

    def __init__(self, *turns: dict):
        super().__init__()
        self.turns = list(turns)
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        with self._lock:
            index = len(self.requests)
            self.requests.append({"messages": json.loads(json.dumps(messages, default=str)), "tools": kwargs.get("tools"), "json_mode": kwargs.get("json_mode")})
        if FORCE_ANSWER_MARKER in str(messages[-1].get("content")):
            turn = {"text": "Best effort: " + "; ".join(str(m.get("content"))[:40] for m in messages if m.get("role") == "tool")}
        else:
            turn = self.turns[min(index, len(self.turns) - 1)]
        calls = [ToolCall(f"call_{index}_{n}", name, args) for n, (name, args) in enumerate(turn.get("calls", []))]
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
        return LLMResult(text=turn.get("text", ""), model="scripted", prompt_tokens=10, completion_tokens=5, cost=cost, tool_calls=calls, finish_reason="tool_calls" if calls else "stop")


class ScriptedReactLLM(MockLLM):
    """Replies with scripted raw strings (the last repeats)."""

    def __init__(self, *replies: str):
        super().__init__()
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def generate(self, messages, **kwargs) -> LLMResult:
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        text = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
        return LLMResult(text=text, model="scripted", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")


def _system(type_: str, llm: MockLLM | None = None, tools=None, **retrieval):
    cfg = {"type": type_, "retrieval": {"top_k": 3, **({"retriever": "vector"} if type_ == "agent_search" else {}), **retrieval}}
    if tools is not None:
        cfg["tools"] = tools
    system = create_rag_system(SystemConfig(**cfg), force_mock=True)
    if llm is not None:
        system.llm = llm
    system.ingest(DOCS)
    return system


def _steps(result, kind=None, name=None):
    return [s for s in result.steps if (kind is None or s.kind == kind) and (name is None or s.name == name)]


def _assert_costs_add_up(answer):
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)
    assert not _steps(answer, name="untracked")
    assert answer.retrieval_result.steps == [s for s in answer.steps if s.kind != "generate"]
    assert answer.steps[-1].kind == "generate"


# --- native function calling ----------------------------------------------------------------------------------------------


def test_the_agent_searches_then_answers_from_what_it_found():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "HarborShield price"})), say("HarborShield costs $200 per month [doc_a]."))
    answer = _system("agent_search", llm).answer_question(QUESTION)
    assert answer.answer == "HarborShield costs $200 per month [doc_a]."
    assert [(s.kind, s.name) for s in answer.steps if s.name != "agent_decision"] == [("llm", "agent_turn"), ("tool", "search"), ("generate", "answer")]
    agent = answer.metadata["agent"]
    assert (agent["steps"], agent["tool_calls"], agent["llm_calls"], agent["termination"], agent["best_effort"]) == (2, 1, 2, "finished", False)
    assert answer.retrieval_result.chunks and answer.retrieval_result.chunks[0].doc_id in {"doc_a", "doc_b", "doc_c"}
    assert answer.retrieval_result.metadata["retriever"] == "agent_search"
    second_request = llm.requests[1]["messages"]
    assert second_request[-1]["role"] == "tool" and "HarborShield" in second_request[-1]["content"]  # the observation went back to the model
    assert second_request[-2]["tool_calls"][0]["function"]["name"] == "search"
    assert answer.cost.total_cost >= 2 * PER_CALL
    _assert_costs_add_up(answer)


def test_the_first_request_offers_the_tools_and_forces_citations():
    llm = ScriptedToolLLM(say("done"))
    _system("agent_search", llm, tools=["calculator", "date_calc"]).answer_question(QUESTION)
    request = llm.requests[0]
    assert [t["function"]["name"] for t in request["tools"]] == ["search", "calculator", "date_calc"]
    system_prompt = request["messages"][0]["content"]
    assert "Cite the source document ID in square brackets" in system_prompt and request["messages"][1]["content"] == f"Question: {QUESTION}"


def test_the_agent_uses_the_calculator_on_a_numeric_question():
    llm = ScriptedToolLLM(tool_turn(("calculator", {"expression": "200 * 12"})), say("A year costs $2400."))
    answer = _system("agent_search", llm, tools=["calculator"]).answer_question("What does a year of HarborShield cost?")
    tool_step = _steps(answer, "tool", "calculator")[0]
    assert tool_step.output_preview == "200 * 12 = 2400" and json.loads(tool_step.input_preview) == {"expression": "200 * 12"}
    assert llm.requests[1]["messages"][-1]["content"] == "200 * 12 = 2400"
    assert answer.retrieval_result.chunks == []  # a pure calculation retrieved nothing
    _assert_costs_add_up(answer)


def test_several_tool_calls_in_one_turn_are_all_run_and_answered():
    llm = ScriptedToolLLM(tool_turn(("calculator", {"expression": "1+1"}), ("calculator", {"expression": "2+2"})), say("2 and 4"))
    answer = _system("agent_search", llm, tools=["calculator"]).answer_question("two sums")
    assert len(_steps(answer, "tool", "calculator")) == 2
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["1+1 = 2", "2+2 = 4"] and [m["tool_call_id"] for m in tool_messages] == ["call_0_0", "call_0_1"]


def test_an_unknown_tool_becomes_an_error_observation_and_the_loop_continues():
    llm = ScriptedToolLLM(tool_turn(("web_search", {"query": "x"})), tool_turn(("search", {"query": "HarborShield"})), say("HarborShield costs $200 [doc_a]."))
    answer = _system("agent_search", llm).answer_question(QUESTION)
    assert llm.requests[1]["messages"][-1]["content"].startswith("Error: unknown tool 'web_search'; available tools: search")
    assert _steps(answer, "tool", "web_search")[0].metadata["error"].startswith("unknown tool")
    assert answer.answer.startswith("HarborShield costs") and answer.metadata["agent"]["unknown_tool_calls"] == 1
    assert answer.metadata["agent"]["steps"] == 3 and answer.metadata["agent"]["termination"] == "finished"


def test_a_tool_error_is_shown_to_the_model_not_raised():
    llm = ScriptedToolLLM(tool_turn(("calculator", {"expression": "__import__('os').getcwd()"})), say("I could not compute it."))
    answer = _system("agent_search", llm, tools=["calculator"]).answer_question("compute")
    assert "not an allowed function" in llm.requests[1]["messages"][-1]["content"]
    assert answer.answer == "I could not compute it."


def test_max_steps_exhausted_gives_a_best_effort_answer_and_the_flag():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "HarborShield"})))  # never stops calling tools
    answer = _system("agent_search", llm, max_steps=3).answer_question(QUESTION)
    agent = answer.metadata["agent"]
    assert agent["termination"] == "max_steps" and agent["best_effort"] is True and agent["steps"] == 3 and agent["tool_calls"] == 3
    assert answer.answer.startswith("Best effort:")
    last = llm.requests[-1]
    assert FORCE_ANSWER_MARKER in last["messages"][-1]["content"] and last["tools"]  # the tools are still offered so providers accept the tool messages
    final = answer.steps[-1]
    assert (final.kind, final.name, final.metadata["best_effort"]) == ("generate", "answer", True)
    _assert_costs_add_up(answer)


def test_the_tool_call_cap_refuses_further_calls_and_ends_the_loop():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "a b"}), ("search", {"query": "c d"}), ("search", {"query": "e f"})))
    answer = _system("agent_search", llm, max_steps=6, max_tool_calls=2).answer_question(QUESTION)
    assert len(_steps(answer, "tool")) == 2  # the third call of the turn was refused, not run
    refused = [m for m in llm.requests[-1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "tool-call limit" in refused
    agent = answer.metadata["agent"]
    assert agent["termination"] == "budget" and agent["budget_exhausted"] is True and agent["best_effort"] is True and agent["tool_limit_hit"] is True
    assert answer.answer


def test_a_cost_cap_stops_the_agent_before_another_model_call():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "HarborShield"})))
    answer = _system("agent_search", llm, max_steps=6, max_cost_usd=PER_CALL).answer_question(QUESTION)
    # turn 1 costs $0.01 = the cap: no second turn, only the best-effort answer
    assert len(llm.requests) == 2 and answer.metadata["agent"]["termination"] == "budget"
    assert answer.answer and _assert_costs_add_up(answer) is None


def test_a_token_cap_is_enforced_too():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "HarborShield"})))
    answer = _system("agent_search", llm, max_steps=6, max_tokens=15).answer_question(QUESTION)
    assert len(llm.requests) == 2 and answer.metadata["agent"]["budget_exhausted"] is True


def test_an_empty_final_reply_still_yields_an_answer():
    llm = ScriptedToolLLM(say(""))
    answer = _system("agent_search", llm).answer_question(QUESTION)
    assert answer.metadata["agent"]["best_effort"] is True and answer.answer


def test_retrieval_is_scored_on_the_passages_the_tools_returned():
    llm = ScriptedToolLLM(
        tool_turn(("search", {"query": "HarborShield price", "top_k": 3})),
        tool_turn(("corpus_grep", {"pattern": "HS-4127"}), ("read_document", {"doc_id": "doc_b"})),
        say("done"),
    )
    answer = _system("agent_search", llm, tools=["corpus_grep", "read_document"]).answer_question(QUESTION, top_k=5)
    docs = [c.doc_id for c in answer.retrieval_result.chunks]
    assert docs.count("doc_a") >= 1 and "doc_b" in docs  # grep hit in doc_a, a window of doc_b
    assert all(c.rank == i for i, c in enumerate(answer.retrieval_result.chunks, start=1))
    assert answer.metadata["context_chunk_ids"] == [c.chunk_id for c in answer.retrieval_result.chunks[:3]]  # `top_k: 3` is how much the answer is judged on
    grep_chunk = next(c for c in answer.retrieval_result.chunks if "chunk::line" in c.chunk_id)
    assert "HS-4127" in grep_chunk.text


def test_fetch_context_alone_runs_the_agent_and_reports_the_evidence():
    llm = ScriptedToolLLM(tool_turn(("search", {"query": "HarborShield"})), say("x"))
    result = _system("agent_search", llm).fetch_context(QUESTION, top_k=2)
    assert len(result.chunks) == 2 and result.metadata["agent"]["tool_calls"] == 1
    assert result.cost.total_cost == pytest.approx(PER_CALL)  # the tool-calling turn is retrieval work; the answering turn is generation and is not counted


def test_a_custom_tool_is_available_to_the_agent():
    llm = ScriptedToolLLM(tool_turn(("price", {"product": "HarborShield", "quantity": 3})), say("$600"))
    system = _system("agent_search", llm, tools=[{"name": "price", "path": "test_tools:lookup_price"}])
    answer = system.answer_question("price of three?")
    assert llm.requests[1]["messages"][-1]["content"] == "3 x HarborShield = 600 USD"
    assert answer.metadata["agent"]["tool_calls"] == 1


class StatelessLLM(MockLLM):
    """Searches once, then answers; decides from the conversation alone, so concurrent questions cannot disturb each other."""

    def generate(self, messages, **kwargs) -> LLMResult:
        observed = any(m.get("role") == "tool" for m in messages)
        query = str(messages[1]["content"]).removeprefix("Question: ")
        calls = [] if observed else [ToolCall(f"c-{abs(hash(query))}", "search", {"query": query})]
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
        return LLMResult(text="" if calls else f"answer to: {query}", model="stateless", prompt_tokens=10, completion_tokens=5, cost=cost, tool_calls=calls)


def test_concurrent_questions_do_not_share_state():
    system = _system("agent_search", StatelessLLM())
    questions = [f"{QUESTION} variation {i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(system.answer_question, questions))
    for answer in answers:
        assert [(s.kind, s.name) for s in answer.steps if s.name != "agent_decision"] == [("llm", "agent_turn"), ("tool", "search"), ("generate", "answer")]
        _assert_costs_add_up(answer)
    for question, answer in zip(questions, answers, strict=True):  # every answer belongs to its own question
        assert answer.answer == f"answer to: {question}" and _steps(answer, "tool", "search")[0].input_preview == json.dumps({"query": question})


# --- ReAct-JSON ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "tool", "args", "answer"),
    [
        ('{"action": "tool", "tool": "calculator", "args": {"expression": "1+1"}}', "tool", "calculator", {"expression": "1+1"}, None),
        ('```json\n{"action": "tool", "tool": "search", "args": {"query": "x"}}\n```', "tool", "search", {"query": "x"}, None),
        ('I will search.\n{"action": "tool", "tool": "search", "args": {"query": "x"}}\nDone.', "tool", "search", {"query": "x"}, None),
        ('{"tool": "search", "arguments": {"query": "y"}}', "tool", "search", {"query": "y"}, None),
        ('{"action": "search", "input": {"query": "z"}}', "tool", "search", {"query": "z"}, None),
        ('{"action": "tool", "name": "search", "args": "{\\"query\\": \\"s\\"}"}', "tool", "search", {"query": "s"}, None),
        ('{"action": "tool", "tool": "list_documents"}', "tool", "list_documents", {}, None),
        ('{"action": "tool", "tool": "search", "args": {"query": "first"}}\n{"action": "answer", "answer": "ignored"}', "tool", "search", {"query": "first"}, None),
        ('{"action": "answer", "answer": "It is 42 [doc_a]."}', "answer", None, {}, "It is 42 [doc_a]."),
        ('```\n{"action": "final_answer", "answer": "Fine."}\n```', "answer", None, {}, "Fine."),
        ('{"answer": "No action key."}', "answer", None, {}, "No action key."),
        ('{"action": "finish", "final_answer": "Done."}', "answer", None, {}, "Done."),
    ],
)
def test_parse_action_handles_fenced_garbled_and_variant_json(text, kind, tool, args, answer):
    action = parse_action(text)
    assert action is not None and (action.kind, action.tool, action.args, action.answer) == (kind, tool, args, answer)


@pytest.mark.parametrize("text", ["I think the answer is 42.", "", "{not json at all", '{"thought": "hmm"}', '{"action": "tool"}', "[1, 2, 3]", "null"])
def test_parse_action_returns_none_when_there_is_no_action(text):
    assert parse_action(text) is None


def test_describe_tools_lists_each_tool_with_its_schema():
    text = describe_tools(ToolBox.from_refs(["calculator", "date_calc"]))
    lines = text.splitlines()
    assert lines[0].startswith("- calculator: ") and "expression" in lines[0] and "Required: expression" in lines[0] and lines[1].startswith("- date_calc: ")


def test_react_mode_runs_tools_from_json_actions_and_answers():
    llm = ScriptedReactLLM(
        'Let me look.\n```json\n{"action": "tool", "tool": "search", "args": {"query": "HarborShield price"}}\n```',
        '{"action": "tool", "tool": "calculator", "args": {"expression": "200 * 12"}}',
        '{"action": "answer", "answer": "A year is $2400 [doc_a]."}',
    )
    answer = _system("agent_search", llm, tools=["calculator"], agent_mode="react_json").answer_question(QUESTION)
    assert answer.answer == "A year is $2400 [doc_a]."
    assert [(s.kind, s.name) for s in answer.steps if s.name != "agent_decision"] == [
        ("llm", "agent_turn"),
        ("tool", "search"),
        ("llm", "agent_turn"),
        ("tool", "calculator"),
        ("generate", "answer"),
    ]
    first = llm.requests[0]
    assert "Reply with exactly one JSON object" in first[0]["content"] and "- calculator:" in first[0]["content"]
    assert llm.requests[1][-1]["role"] == "user" and llm.requests[1][-1]["content"].startswith("Observation from search:\n[doc_a")
    assert llm.requests[2][-1]["content"] == "Observation from calculator:\n200 * 12 = 2400"
    _assert_costs_add_up(answer)


def test_react_mode_sends_no_native_tools_and_asks_for_json():
    seen = []

    class Spy(ScriptedReactLLM):
        def generate(self, messages, **kwargs):
            seen.append((kwargs.get("tools"), kwargs.get("json_mode")))
            return super().generate(messages, **kwargs)

    _system("agent_search", Spy('{"action": "answer", "answer": "x"}'), agent_mode="react_json").answer_question(QUESTION)
    assert seen == [(None, True)]


def test_react_mode_treats_prose_as_the_answer_and_counts_it():
    llm = ScriptedReactLLM("HarborShield costs $200 a month [doc_a].")
    answer = _system("agent_search", llm, agent_mode="react_json").answer_question(QUESTION)
    assert answer.answer == "HarborShield costs $200 a month [doc_a]." and answer.metadata["agent"]["unparsed_action"] == 1
    assert answer.steps[-1].kind == "generate"


def test_react_mode_reports_an_unknown_tool_and_keeps_going():
    llm = ScriptedReactLLM('{"action": "tool", "tool": "teleport", "args": {}}', '{"action": "answer", "answer": "gave up"}')
    answer = _system("agent_search", llm, agent_mode="react_json").answer_question(QUESTION)
    assert llm.requests[1][-1]["content"].startswith("Observation from teleport:\nError: unknown tool 'teleport'")
    assert answer.answer == "gave up"


def test_react_mode_best_effort_after_max_steps():
    llm = ScriptedReactLLM('{"action": "tool", "tool": "search", "args": {"query": "HarborShield"}}')
    answer = _system("agent_search", llm, agent_mode="react_json", max_steps=2).answer_question(QUESTION)
    assert answer.metadata["agent"]["termination"] == "max_steps" and answer.metadata["agent"]["best_effort"] is True and answer.answer


# --- grep_agent -------------------------------------------------------------------------------------------------------------


def test_grep_agent_has_no_index_and_pays_no_embedding_cost():
    system = _system("grep_agent", ScriptedToolLLM(tool_turn(("corpus_grep", {"pattern": "HS-4127"})), tool_turn(("read_document", {"doc_id": "doc_a"})), say("A missing tenant header [doc_a].")))
    assert not hasattr(system, "vector_store") and not hasattr(system, "embedding_model")
    ingestion = system.ingest(DOCS)
    assert ingestion.num_chunks == 0 and ingestion.cost.total_cost == 0
    answer = system.answer_question("What does error HS-4127 mean?")
    assert answer.cost.embedding_cost == 0 and answer.cost.embedding_input_tokens == 0
    assert [s.name for s in _steps(answer, "tool")] == ["corpus_grep", "read_document"]
    assert [c.doc_id for c in answer.retrieval_result.chunks][0] == "doc_a"
    assert answer.metadata["agent"]["tool_calls"] == 2
    _assert_costs_add_up(answer)


def test_grep_agent_offers_exactly_its_three_tools():
    llm = ScriptedToolLLM(say("done"))
    _system("grep_agent", llm).answer_question("anything")
    assert [t["function"]["name"] for t in llm.requests[0]["tools"]] == ["list_documents", "corpus_grep", "read_document"]
    assert "no search index" in llm.requests[0]["messages"][0]["content"]
    with pytest.raises(ValueError, match="does not use tools"):
        ExperimentConfig.model_validate(
            {"run": {"name": "t"}, "dataset": {"documents_path": "d", "questions_path": "q"}, "systems": [{"type": "grep_agent", "tools": ["calculator"]}]}
        )


def test_grep_agent_works_end_to_end_in_mock_mode():
    answer = _system("grep_agent").answer_question("Where do Plinth harbours host fishing fleets?")
    assert answer.answer and answer.metadata["agent"]["tool_calls"] >= 1
    assert answer.retrieval_result.chunks and answer.retrieval_result.chunks[0].doc_id == "doc_c"
    _assert_costs_add_up(answer)


# --- mock mode and the two variants ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["native", "react_json"])
def test_the_mock_policy_computes_with_the_calculator_and_searches_otherwise(mode):
    system = _system("agent_search", tools=["calculator", "date_calc"], agent_mode=mode)
    calc = system.answer_question("What is 300,000 times 12 in total?")
    assert [s.name for s in _steps(calc, "tool")] == ["calculator"] and "300000 * 12 = 3600000" in calc.answer
    search = system.answer_question("Which module is extra for HarborShield?")
    assert [s.name for s in _steps(search, "tool")] == ["search"] and search.retrieval_result.chunks
    dates = system.answer_question("How many days between 2024-04-16 and 2024-05-16?")
    assert [s.name for s in _steps(dates, "tool")] == ["date_calc"] and "30 days" in dates.answer
    for answer in (calc, search, dates):
        _assert_costs_add_up(answer)


def test_without_a_calculator_the_mock_agent_just_searches():
    answer = _system("agent_search", tools=[]).answer_question("What is 300,000 times 12 in total?")
    assert [s.name for s in _steps(answer, "tool")] == ["search"]


def test_the_search_tool_is_always_there_and_never_doubled():
    from ragbench.rag_systems.agent_search_rag import AgentSearchRAG

    assert AgentSearchRAG.offered_tools(SystemConfig(type="agent_search")) == ["search"]
    assert AgentSearchRAG.offered_tools(SystemConfig(type="agent_search", tools=["calculator", {"name": "search"}])) == ["calculator", "search"]
    assert AgentSearchRAG.offered_tools(SystemConfig(type="agent_search", tools=["calculator"])) == ["search", "calculator"]


def test_specs_declare_agentic_tool_systems():
    specs = {spec.type: spec for spec in all_specs()}
    assert specs["agent_search"].supports_tools is True and specs["grep_agent"].supports_tools is False
    assert all(specs[t].agentic and specs[t].requires_llm for t in ("agent_search", "grep_agent"))
    assert specs["grep_agent"].chunker is None
    with pytest.raises(ValueError, match="agent_mode"):
        create_rag_system(SystemConfig(type="agent_search", retrieval={"agent_mode": "telepathy"}), force_mock=True)
    with pytest.raises(ValueError, match="max_stepz"):
        create_rag_system(SystemConfig(type="grep_agent", retrieval={"max_stepz": 3}), force_mock=True)


# --- the evaluator ----------------------------------------------------------------------------------------------------------


def test_the_run_reports_tool_columns_and_tool_usage_for_the_agent_variants(tmp_path):
    questions = [
        {**make_question("q1", "What is 300 times 12 in total?"), "requires_tools": ["calculator"]},
        {**make_question("q2", "How many days between 2024-01-01 and 2024-03-01?"), "requires_tools": ["date_calc"]},
        make_question("q3", "Tell me about topic 4"),
    ]
    systems = [
        {"type": "agent_search", "name": "agent_plain", "retrieval": {"top_k": 3, "retriever": "vector"}, "tools": []},
        {"type": "agent_search", "name": "agent_tools", "retrieval": {"top_k": 3, "retriever": "vector"}, "tools": ["calculator", "date_calc", "corpus_grep"]},
        {"type": "grep_agent", "name": "grepper", "retrieval": {"top_k": 3}},
        {"type": "bm25", "name": "bm25", "retrieval": {"top_k": 3}},
    ]
    config = write_experiment(tmp_path, questions, systems)
    run_dir = run_benchmark(config, force_mock=True, max_workers=2)

    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    for name in ("agent_plain", "agent_tools", "grepper"):
        assert summary.loc[name, "avg_tool_calls"] >= 1 and summary.loc[name, "avg_steps"] >= 2 and summary.loc[name, "budget_exhausted_rate"] == 0
    assert summary.loc["agent_tools", "required_tool_used_rate"] == 1.0  # the calculator and date_calc questions were answered with those tools
    assert summary.loc["agent_plain", "required_tool_used_rate"] == 0.0  # same questions, no such tools offered
    assert pd.isna(summary.loc["bm25", ["avg_tool_calls", "avg_steps", "required_tool_used_rate"]]).all()

    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    assert usage.loc[("agent_plain", "search"), "calls"] >= 3 and ("agent_plain", "calculator") not in usage.index
    assert usage.loc[("agent_tools", "calculator"), "calls"] == 1 and usage.loc[("agent_tools", "date_calc"), "calls"] == 1
    assert usage.loc[("agent_tools", "corpus_grep"), "calls"] == 0  # offered, never used: still listed
    assert {"list_documents", "corpus_grep", "read_document"} <= set(usage.loc["grepper"].index)
    assert "bm25" not in usage.index.get_level_values("system")

    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    for row in rows:
        if row["system"] != "bm25":
            assert row["agent"]["termination"] == "finished" and row["agent"]["tool_calls"] >= 1
            assert any(s["kind"] == "tool" for s in row["steps"])
    grep_rows = [r for r in rows if r["system"] == "grepper"]
    assert all(r["cost"]["embedding_cost"] == 0 for r in grep_rows)
