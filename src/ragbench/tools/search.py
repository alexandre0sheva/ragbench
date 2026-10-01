"""`search`: the system's own retriever as a tool, so an agent can run the same search a plain pipeline would."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ragbench.registry import TOOLS
from ragbench.tools.base import BaseTool, ToolContext, ToolResult, make_spec

PREVIEW_CHARS = 700
RETRIEVER_NAME = "search"


class SearchArgs(BaseModel):
    query: str = Field(description="What to search for; a short natural-language query or the key terms.")
    top_k: int = Field(default=5, ge=1, le=20, description="Passages to return.")


@TOOLS.register("search")
class SearchTool(BaseTool):
    Args = SearchArgs
    spec = make_spec("search", "Search the document collection and return the best matching passages with their document ids. Cite those ids in the answer.", SearchArgs)

    def _run(self, args: SearchArgs, ctx: ToolContext) -> ToolResult:
        retriever = ctx.retrievers.get(RETRIEVER_NAME)
        if retriever is None:
            return ToolResult.fail("search is not available for this system")
        result = retriever(args.query, args.top_k)
        if not result.chunks:
            return ToolResult.ok(f"No passages found for {args.query!r}.", data=[], cost_usd=result.cost.total_cost)
        lines = [f"[{c.doc_id} | {c.chunk_id.split('::chunk::')[-1]}] {' '.join(c.text.split())[:PREVIEW_CHARS]}" for c in result.chunks]
        data = [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in result.chunks]
        return ToolResult.ok("\n\n".join(lines), data=data, cost_usd=result.cost.total_cost)
