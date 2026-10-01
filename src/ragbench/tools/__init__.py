"""Tools an agent can call, with a registry of built-ins, a safe runner, and a loader for user-written tools (see `docs/tools.md`)."""

from ragbench.registry import TOOLS
from ragbench.tools import calculator, corpus, dates, search  # noqa: F401  (registers the built-in tools)
from ragbench.tools.base import BaseTool, CorpusView, Tool, ToolContext, ToolResult, ToolSpec
from ragbench.tools.custom import CallableTool, CustomToolError
from ragbench.tools.registry import ToolBox, resolve_tools, tool_ref_name, tool_ref_options

# Third-party packages can contribute tools through the `ragbench.tools` entry-point group (see docs/tools.md).
TOOLS.load_entry_points("ragbench.tools")

__all__ = [
    "TOOLS",
    "BaseTool",
    "CallableTool",
    "CorpusView",
    "CustomToolError",
    "Tool",
    "ToolBox",
    "ToolContext",
    "ToolResult",
    "ToolSpec",
    "resolve_tools",
    "tool_ref_name",
    "tool_ref_options",
]
