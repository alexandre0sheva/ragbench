"""MCP tools for agents: `tools: [{type: mcp, server: {command|url}, allow: [...]}]`.

These run against a fake MCP client (no SDK, no process, no network); `test_mcp_sdk.py` covers the real SDK and skips without the `mcp` extra.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fake_systems import RankedFakeSystem, question, write_experiment

from ragbench.config.schema import ExperimentConfig, SystemConfig, ToolsConfig
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.rag_systems import SYSTEM_REGISTRY, create_rag_system
from ragbench.rag_systems.trace import Tracer
from ragbench.tools import CorpusView, ToolBox, ToolContext, resolve_tools
from ragbench.tools import mcp as mcp_tools
from ragbench.utils.jsonl import read_jsonl

COMMAND_SERVER = {"command": ["python", "price_server.py"]}


def text(value: str) -> Any:
    return SimpleNamespace(type="text", text=value)


def tool_info(name: str, description: str = "", properties: dict | None = None) -> Any:
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": properties or {}, "additionalProperties": False},
        annotations=None,
    )


def result(*blocks: Any, structured: Any = None, is_error: bool = False) -> Any:
    return SimpleNamespace(content=list(blocks), structured_content=structured, is_error=is_error)


class FakeClient:
    """What `mcp.Client` offers the pool: `list_tools()` and `call_tool(name, arguments)`."""

    def __init__(self, tools: list[Any], handlers: dict[str, Any] | None = None):
        self.tools = tools
        self.handlers = handlers or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, cursor: str | None = None) -> Any:
        return SimpleNamespace(tools=self.tools, next_cursor=None)

    async def call_tool(self, name: str, arguments: dict | None = None, read_timeout_seconds: float | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        handler = self.handlers[name]
        value = handler(arguments or {})
        if asyncio.iscoroutine(value):
            value = await value
        return value


@pytest.fixture
def server(monkeypatch):
    """Install a fake MCP client behind every `server:`; returns it with the list of servers opened."""
    client = FakeClient(
        [tool_info("lookup_price", "Price of a product.", {"sku": {"type": "string"}}), tool_info("convert_units", "Convert units."), tool_info("delete_all", "Dangerous.")],
        {
            "lookup_price": lambda args: result(text(f"{args['sku']}: $200"), structured=[{"doc_id": "doc_a", "sku": args["sku"]}]),
            "convert_units": lambda args: result(text("12 in = 30.48 cm")),
            "delete_all": lambda args: result(text("gone")),
        },
    )
    opened: list[dict] = []

    @asynccontextmanager
    async def fake_open(config):
        opened.append(config.model_dump(exclude_none=True))
        yield client

    monkeypatch.setattr(mcp_tools, "open_client", fake_open)
    client.opened = opened  # type: ignore[attr-defined]
    yield client
    mcp_tools.close_all_servers()


def ref(allow: list[str], **extra: Any) -> dict[str, Any]:
    return {"type": "mcp", "server": COMMAND_SERVER, "allow": allow, **extra}


def _ctx() -> ToolContext:
    return ToolContext(corpus=CorpusView([]))


# --- the ref -------------------------------------------------------------------------------------------------------------------


def test_a_ref_needs_exactly_one_of_command_and_url_and_a_non_empty_allow_list():
    with pytest.raises(ValueError, match="exactly one of `command` or `url`"):
        mcp_tools.parse_mcp_ref({"type": "mcp", "server": {}, "allow": ["a"]})
    with pytest.raises(ValueError, match="exactly one of `command` or `url`"):
        mcp_tools.parse_mcp_ref({"type": "mcp", "server": {"command": "x", "url": "http://h/mcp"}, "allow": ["a"]})
    with pytest.raises(ValueError, match="allow"):
        mcp_tools.parse_mcp_ref({"type": "mcp", "server": COMMAND_SERVER, "allow": []})
    with pytest.raises(ValueError, match="allow"):
        mcp_tools.parse_mcp_ref({"type": "mcp", "server": COMMAND_SERVER})
    with pytest.raises(ValueError, match="nope"):
        mcp_tools.parse_mcp_ref({**ref(["a"]), "nope": 1})
    parsed = mcp_tools.parse_mcp_ref({"type": "mcp", "server": {"command": "python -m my.server --flag 'a b'"}, "allow": ["a"]})
    assert parsed.server.argv() == ["python", "-m", "my.server", "--flag", "a b"]  # a string command is split like a shell would


def test_a_config_with_an_mcp_ref_loads_without_contacting_the_server(monkeypatch):
    def refuse(config):
        raise AssertionError("the config load must not start or contact an MCP server")

    monkeypatch.setattr(mcp_tools, "open_client", refuse)
    config = ExperimentConfig.model_validate(
        {
            "run": {"name": "t", "output_dir": "out"},
            "dataset": {"documents_path": "docs", "questions_path": "q.jsonl"},
            "systems": [{"type": "agent_search", "tools": ["calculator", ref(["lookup_price", "convert_units"], prefix="shop_")]}],
        }
    )
    assert SYSTEM_REGISTRY["agent_search"].offered_tools(config.systems[0]) == ["search", "calculator", "shop_lookup_price", "shop_convert_units"]


def test_a_bad_mcp_ref_is_reported_against_its_system():
    with pytest.raises(ValueError, match=r"systems\[0\]\.tools.*allow"):
        ExperimentConfig.model_validate(
            {
                "run": {"name": "t", "output_dir": "out"},
                "dataset": {"documents_path": "docs", "questions_path": "q.jsonl"},
                "systems": [{"type": "agent_search", "tools": [{"type": "mcp", "server": COMMAND_SERVER}]}],
            }
        )


def test_a_system_that_does_not_use_tools_still_rejects_an_mcp_ref():
    with pytest.raises(ValueError, match="does not use tools"):
        SystemConfig(type="bm25", tools=[ref(["lookup_price"])])


def test_an_mcp_tool_clashing_with_another_tool_is_refused():
    with pytest.raises(ValueError, match="'calculator' is listed twice"):
        resolve_tools(["calculator", ref(["calculator"])])
    assert [t.spec.name for t in resolve_tools(["calculator", ref(["calculator"], prefix="remote_")])] == ["calculator", "remote_calculator"]


# --- permissions --------------------------------------------------------------------------------------------------------------


def test_a_url_server_needs_allow_network_and_a_command_server_does_not_need_tools_allow():
    remote = {"type": "mcp", "server": {"url": "https://mcp.example.com/mcp"}, "allow": ["lookup_price"]}
    with pytest.raises(ValueError, match="uses the network.*allow_network"):
        resolve_tools([remote], ToolsConfig())
    tools = resolve_tools([remote], ToolsConfig(allow_network=True))
    assert [(t.spec.name, t.side_effects) for t in tools] == [("lookup_price", "network")]
    # `allow` on the ref is the explicit list of what the agent may call, so `tools.allow` is not asked for it a second time
    assert [(t.spec.name, t.side_effects) for t in resolve_tools([ref(["lookup_price"])], ToolsConfig())] == [("lookup_price", "filesystem")]


def test_the_recommendation_sees_a_url_server_as_a_network_tool():
    from ragbench.selection.recommend import _locality

    config = ExperimentConfig.model_validate(
        {
            "run": {"name": "t", "output_dir": "out"},
            "dataset": {"documents_path": "docs", "questions_path": "q.jsonl"},
            "tools": {"allow_network": True},
            "systems": [{"type": "agent_search", "name": "a", "tools": [{"type": "mcp", "server": {"url": "https://mcp.example.com/mcp"}, "allow": ["lookup_price"]}]}],
        }
    )
    assert _locality(config, "a")[1] == ["lookup_price"]


# --- discovery ----------------------------------------------------------------------------------------------------------------


def test_the_agent_is_offered_the_servers_own_description_and_schema_for_the_allowed_tools_only(server):
    box = ToolBox.from_refs([ref(["lookup_price", "convert_units"])])
    assert box.names == ["lookup_price", "convert_units"]  # delete_all is on the server but not allowed
    schema = {s["function"]["name"]: s["function"] for s in box.schemas()}
    assert schema["lookup_price"]["description"] == "Price of a product."
    assert schema["lookup_price"]["parameters"]["properties"] == {"sku": {"type": "string"}}
    assert len(server.opened) == 1


def test_a_prefix_renames_the_tool_for_the_agent_and_not_for_the_server(server):
    box = ToolBox.from_refs([ref(["lookup_price"], prefix="shop_")])
    assert box.names == ["shop_lookup_price"]
    outcome = box.call("shop_lookup_price", {"sku": "HS-1"}, _ctx())
    assert outcome.error is None and server.calls == [("lookup_price", {"sku": "HS-1"})]


def test_an_allowed_tool_the_server_does_not_have_names_what_it_has(server):
    with pytest.raises(ValueError, match=r"does not offer 'lookup_prise'.*Did you mean 'lookup_price'.*lookup_price, convert_units, delete_all"):
        ToolBox.from_refs([ref(["lookup_prise"])])


def test_servers_with_paged_tool_lists_are_read_to_the_end(monkeypatch):
    pages = {None: (["a"], "p2"), "p2": (["b"], None)}

    class Paged(FakeClient):
        async def list_tools(self, cursor: str | None = None) -> Any:
            names, nxt = pages[cursor]
            return SimpleNamespace(tools=[tool_info(n) for n in names], next_cursor=nxt)

    client = Paged([])

    @asynccontextmanager
    async def fake_open(config):
        yield client

    monkeypatch.setattr(mcp_tools, "open_client", fake_open)
    try:
        assert ToolBox.from_refs([ref(["a", "b"])]).names == ["a", "b"]
    finally:
        mcp_tools.close_all_servers()


def test_one_server_is_started_once_for_all_the_boxes_that_use_it(server):
    ToolBox.from_refs([ref(["lookup_price"])])
    ToolBox.from_refs([ref(["convert_units"])])
    ToolBox.from_refs([{**ref(["convert_units"]), "server": {"command": ["python", "other.py"]}}])
    assert len(server.opened) == 2


# --- calls --------------------------------------------------------------------------------------------------------------------


def test_a_call_returns_the_servers_text_and_structured_result(server):
    box = ToolBox.from_refs([ref(["lookup_price"])])
    outcome = box.call("lookup_price", {"sku": "HS-1"}, _ctx())
    assert outcome.text == "HS-1: $200" and outcome.error is None
    assert outcome.data == [{"doc_id": "doc_a", "sku": "HS-1"}]


def test_text_blocks_are_joined_and_other_blocks_are_named(server):
    server.handlers["convert_units"] = lambda args: result(text("one"), SimpleNamespace(type="image", mime_type="image/png"), text("two"))
    outcome = ToolBox.from_refs([ref(["convert_units"])]).call("convert_units", {}, _ctx())
    assert outcome.text == "one\n[image image/png omitted]\ntwo"


def test_json_text_without_structured_content_still_gives_the_trace_its_doc_ids(server):
    server.handlers["convert_units"] = lambda args: result(text(json.dumps([{"doc_id": "doc_b"}, {"doc_id": "doc_a"}])))
    tracer = Tracer()
    ToolBox.from_refs([ref(["convert_units"])]).call("convert_units", {}, _ctx(), tracer)
    assert tracer.steps[0].metadata["doc_ids"] == ["doc_b", "doc_a"]


def test_a_list_wrapped_in_result_by_the_sdk_is_unwrapped(server):
    server.handlers["convert_units"] = lambda args: result(text("[...]"), structured={"result": [{"doc_id": "doc_c"}]})
    tracer = Tracer()
    outcome = ToolBox.from_refs([ref(["convert_units"])]).call("convert_units", {}, _ctx(), tracer)
    assert outcome.data == [{"doc_id": "doc_c"}] and tracer.steps[0].metadata["doc_ids"] == ["doc_c"]
    server.handlers["convert_units"] = lambda args: result(text("{}"), structured={"result": 1, "unit": "cm"})
    assert ToolBox.from_refs([ref(["convert_units"])]).call("convert_units", {}, _ctx()).data == {"result": 1, "unit": "cm"}  # a real object stays as it is


def test_a_tool_error_is_an_error_result_the_agent_can_read(server):
    server.handlers["lookup_price"] = lambda args: result(text("no such sku"), is_error=True)
    outcome = ToolBox.from_refs([ref(["lookup_price"])]).call("lookup_price", {"sku": "?"}, _ctx())
    assert outcome.error == "no such sku" and outcome.text == "no such sku"


def test_a_server_that_fails_or_vanishes_is_an_error_result_never_an_exception(server):
    def boom(args):
        raise ConnectionError("server closed the connection")

    server.handlers["lookup_price"] = boom
    outcome = ToolBox.from_refs([ref(["lookup_price"])]).call("lookup_price", {"sku": "x"}, _ctx())
    assert "ConnectionError" in (outcome.error or "") and "server closed the connection" in (outcome.error or "")


def test_a_call_that_outlasts_its_timeout_is_an_error_result(server):
    async def slow(args):
        await asyncio.sleep(5)
        return result(text("late"))

    server.handlers["lookup_price"] = slow
    box = ToolBox.from_refs([ref(["lookup_price"], timeout_s=0.2)])
    outcome = box.call("lookup_price", {"sku": "x"}, _ctx())
    assert outcome.error and "timed out" in outcome.error


def test_concurrent_calls_share_the_connection(server):
    from concurrent.futures import ThreadPoolExecutor

    box = ToolBox.from_refs([ref(["lookup_price"])])
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda n: box.call("lookup_price", {"sku": f"s{n}"}, _ctx()).text, range(16)))
    assert texts == [f"s{n}: $200" for n in range(16)]
    assert len(server.opened) == 1


# --- trace, cost ---------------------------------------------------------------------------------------------------------------


def test_a_call_is_a_traced_tool_step_with_the_configured_price(server):
    box = ToolBox.from_refs([ref(["lookup_price", "convert_units"], cost_usd={"lookup_price": 0.004}), ])
    tracer = Tracer()
    box.call("lookup_price", {"sku": "HS-1"}, _ctx(), tracer)
    box.call("convert_units", {}, _ctx(), tracer)
    priced, free = tracer.steps
    assert (priced.kind, priced.name, priced.cost.tool_cost) == ("tool", "lookup_price", pytest.approx(0.004))
    assert (free.name, free.cost.tool_cost) == ("convert_units", 0.0)
    assert priced.latency_ms > 0
    assert json.loads(priced.input_preview) == {"sku": "HS-1"}


def test_a_single_price_applies_to_every_allowed_tool(server):
    tracer = Tracer()
    ToolBox.from_refs([ref(["lookup_price"], cost_usd=0.01)]).call("lookup_price", {"sku": "x"}, _ctx(), tracer)
    assert tracer.steps[0].cost.tool_cost == pytest.approx(0.01)


# --- --mock -------------------------------------------------------------------------------------------------------------------


def test_mock_never_contacts_the_server_and_answers_deterministically(monkeypatch):
    def refuse(config):
        raise AssertionError("--mock must not start or contact an MCP server")

    monkeypatch.setattr(mcp_tools, "open_client", refuse)
    box = ToolBox.from_refs([ref(["lookup_price"], cost_usd=0.003)], mock=True)
    assert box.names == ["lookup_price"]
    first = box.call("lookup_price", {"sku": "HS-1"}, _ctx())
    assert first.text == box.call("lookup_price", {"sku": "HS-1"}, _ctx()).text and "lookup_price" in first.text and "HS-1" in first.text
    assert first.cost_usd == pytest.approx(0.003) and first.error is None


def test_a_mock_run_offers_the_mcp_tools_and_reports_them(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "open_client", lambda config: (_ for _ in ()).throw(AssertionError("contacted")))
    systems = [{"type": "agent_search", "name": "agent_mcp", "retrieval": {"top_k": 3, "retriever": "vector"}, "tools": [ref(["lookup_price", "convert_units"])]}]
    run_dir = run_benchmark(write_experiment(tmp_path, [question("q1", "What is 3 + 4?")], systems), force_mock=True, max_workers=1)
    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    assert {"lookup_price", "convert_units"} <= set(usage.loc["agent_mcp"].index)
    assert usage.loc[("agent_mcp", "lookup_price"), "calls"] == 0


# --- in a system, in a run ----------------------------------------------------------------------------------------------------


class McpUsingFake(RankedFakeSystem):
    """Calls its MCP tool on every question, the way an agent would, so the run's reports can be checked end to end."""

    def __init__(self, config, force_mock=False):
        super().__init__(config, force_mock=force_mock)
        self.box = ToolBox.from_refs(config.tools)  # live: the fake client stands in for the server

    def fetch_context(self, question, top_k=None):
        self.box.call("lookup_price", {"sku": "HS-1"}, ToolContext(corpus=CorpusView([])), self.trace)
        return super().fetch_context(question, top_k)


def test_tool_usage_csv_reports_each_mcp_tools_calls_latency_and_cost(tmp_path, monkeypatch, server):
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_mcp", McpUsingFake)
    systems = [{"type": "fake_mcp", "name": "with_mcp", "retrieval": {"top_k": 3}, "tools": [ref(["lookup_price", "convert_units"], cost_usd={"lookup_price": 0.005})]}]
    questions = [question("q1", "one"), question("q2", "two"), question("q3", "three")]
    run_dir = run_benchmark(write_experiment(tmp_path, questions, systems), force_mock=True, max_workers=2)

    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    row = usage.loc[("with_mcp", "lookup_price")]
    assert (row["calls"], row["errors"], row["questions_using"]) == (3, 0, 3)
    assert row["total_cost_usd"] == pytest.approx(0.015) and row["avg_latency_ms"] > 0
    assert usage.loc[("with_mcp", "convert_units"), "calls"] == 0  # allowed but never called: still listed
    steps = [s for r in read_jsonl(run_dir / "per_question_results.jsonl") for s in r["steps"] if s["kind"] == "tool"]
    assert len(steps) == 3 and {s["name"] for s in steps} == {"lookup_price"}
    assert len(server.opened) == 1  # three questions, two workers, one server


def test_an_agent_system_calls_the_mcp_tool_it_is_offered(server):
    from test_tool_agents import QUESTION, ScriptedToolLLM, say, tool_turn

    config = SystemConfig(type="agent_search", retrieval={"top_k": 3, "retriever": "vector"}, tools=[ref(["lookup_price"], cost_usd=0.002)])
    system = create_rag_system(config, force_mock=True)
    system.box = ToolBox.from_refs(system.tool_refs(config))  # live (a mocked system would stub the server)
    system.llm = ScriptedToolLLM(tool_turn(("lookup_price", {"sku": "HS-1"})), say("It costs $200."))
    answer = system.answer_question(QUESTION)
    assert answer.answer == "It costs $200."
    step = next(s for s in answer.steps if s.kind == "tool")
    assert (step.name, step.cost.tool_cost, step.metadata["doc_ids"]) == ("lookup_price", pytest.approx(0.002), ["doc_a"])
    assert system.llm.requests[1]["messages"][-1]["content"] == "HS-1: $200"
    assert [t["function"]["name"] for t in system.llm.requests[0]["tools"]] == ["search", "lookup_price"]


# --- the SDK ------------------------------------------------------------------------------------------------------------------


def test_a_missing_sdk_gives_the_one_line_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mcp)
    with pytest.raises(ValueError, match=r"pip install 'ragbench\[mcp\]'"):
        ToolBox.from_refs([ref(["lookup_price"])])
    mcp_tools.close_all_servers()


def test_the_example_config_loads_and_offers_the_allowed_tools():
    from pathlib import Path

    import yaml

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "mcp_example.yaml").read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(raw)
    assert SYSTEM_REGISTRY["agent_search"].offered_tools(config.systems[1]) == ["search", "convert_units", "days_between"]


def test_an_adaptive_route_can_use_mcp_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_tools, "open_client", lambda config: (_ for _ in ()).throw(AssertionError("--mock contacted a server")))
    routes = {
        "default": {"type": "vector", "retrieval": {"vector_store": "numpy"}},
        "computation": {"type": "agent_search", "retrieval": {"retriever": "vector", "vector_store": "numpy"}, "tools": ["calculator", ref(["convert_units"])]},
    }
    config = write_experiment(tmp_path, [question("q1", "What is 300 times 12 in total?")], [{"type": "adaptive", "name": "router", "retrieval": {"routes": routes}}])
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    usage = pd.read_csv(run_dir / "tool_usage.csv").set_index(["system", "tool"])
    assert ("router", "convert_units") in usage.index and usage.loc[("router", "calculator"), "calls"] == 1
