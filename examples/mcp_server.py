"""A tiny MCP server to try `tools: [{type: mcp, ...}]` with (needs `pip install 'ragbench[mcp]'`).

    python examples/mcp_server.py        # speaks MCP on stdin/stdout; ragbench starts it for you, see configs/mcp_example.yaml

It offers two read-only, deterministic tools, and a third that the example config does not allow, to show that `allow:` is a list.
"""

from __future__ import annotations

from datetime import date

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # a deliberate error: its message reaches the agent (any other exception is a crash)

server = MCPServer("ragbench-example")

# unit -> (family, size in the family's base unit)
UNITS = {"cm": ("length", 1.0), "in": ("length", 2.54), "m": ("length", 100.0), "km": ("length", 100000.0), "mi": ("length", 160934.4), "kg": ("mass", 1.0), "lb": ("mass", 0.45359237)}


@server.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a length (cm, in, m, km, mi) or a mass (kg, lb) between units."""
    if from_unit not in UNITS or to_unit not in UNITS or UNITS[from_unit][0] != UNITS[to_unit][0]:
        raise ToolError(f"cannot convert {from_unit} to {to_unit}; units: {', '.join(UNITS)}")
    converted = value * UNITS[from_unit][1] / UNITS[to_unit][1]
    return f"{value:g} {from_unit} = {converted:.4g} {to_unit}"


@server.tool()
def days_between(start: str, end: str) -> dict[str, int]:
    """Whole days from `start` to `end`, both ISO dates (YYYY-MM-DD)."""
    return {"days": (date.fromisoformat(end) - date.fromisoformat(start)).days}


@server.tool()
def reset_counters() -> str:
    """A tool with side effects; the example config does not allow it."""
    return "reset"


if __name__ == "__main__":
    server.run()
