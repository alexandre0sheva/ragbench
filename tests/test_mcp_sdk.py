"""MCP tools against the real MCP SDK: an in-process server, and the example server as a real subprocess over stdio.

These skip without the `mcp` extra (CI runs them in the `mcp` job); `test_mcp_tools.py` covers the rest with a fake client.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402

from ragbench.rag_systems.trace import Tracer  # noqa: E402
from ragbench.tools import CorpusView, ToolBox, ToolContext  # noqa: E402
from ragbench.tools import mcp as mcp_tools  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mcp_server.py"


def _ctx() -> ToolContext:
    return ToolContext(corpus=CorpusView([]))


@pytest.fixture(autouse=True)
def _close_servers():
    yield
    mcp_tools.close_all_servers()


def test_the_example_server_runs_as_a_subprocess_and_its_allowed_tools_work():
    ref = {"type": "mcp", "server": {"command": [sys.executable, str(EXAMPLE)]}, "allow": ["convert_units", "days_between"], "cost_usd": {"days_between": 0.001}}
    box = ToolBox.from_refs([ref])
    assert box.names == ["convert_units", "days_between"]  # reset_counters is on the server and not allowed
    schemas = {s["function"]["name"]: s["function"] for s in box.schemas()}
    assert "Convert a length" in schemas["convert_units"]["description"]
    assert set(schemas["convert_units"]["parameters"]["properties"]) == {"value", "from_unit", "to_unit"}

    tracer = Tracer()
    converted = box.call("convert_units", {"value": 12, "from_unit": "in", "to_unit": "cm"}, _ctx(), tracer)
    assert converted.error is None and converted.text == "12 in = 30.48 cm"
    days = box.call("days_between", {"start": "2026-01-01", "end": "2026-03-01"}, _ctx(), tracer)
    assert days.error is None and days.data == {"days": 59}
    assert [step.cost.tool_cost for step in tracer.steps] == [0.0, pytest.approx(0.001)]

    failed = box.call("convert_units", {"value": 1, "from_unit": "kg", "to_unit": "km"}, _ctx())
    assert failed.error and "cannot convert kg to km" in failed.text  # the server's own error, readable by the agent


def test_a_server_that_cannot_start_is_a_clear_error():
    ref = {"type": "mcp", "server": {"command": ["definitely-not-a-real-command-xyz"]}, "allow": ["x"]}
    with pytest.raises(ValueError, match="cannot start MCP server 'definitely-not-a-real-command-xyz'"):
        ToolBox.from_refs([ref])


def test_a_tool_the_server_does_not_have_is_named_with_what_it_has():
    ref = {"type": "mcp", "server": {"command": [sys.executable, str(EXAMPLE)]}, "allow": ["convert_unit"]}
    with pytest.raises(ValueError, match=r"does not offer 'convert_unit'.*Did you mean 'convert_units'"):
        ToolBox.from_refs([ref])


def test_an_in_process_server_through_the_same_pool(monkeypatch):
    server = MCPServer("fake")

    @server.tool()
    def lookup_price(sku: str) -> list[dict[str, str]]:
        """Price of a product, with the document that states it."""
        return [{"doc_id": "doc_a", "sku": sku}]

    @asynccontextmanager
    async def open_in_process(config):
        async with Client(server) as client:
            yield client

    monkeypatch.setattr(mcp_tools, "open_client", open_in_process)
    ref = {"type": "mcp", "server": {"command": ["in-process"]}, "allow": ["lookup_price"]}
    tracer = Tracer()
    outcome = ToolBox.from_refs([ref]).call("lookup_price", {"sku": "HS-1"}, _ctx(), tracer)
    assert outcome.error is None and "HS-1" in outcome.text
    assert tracer.steps[0].metadata["doc_ids"] == ["doc_a"]
