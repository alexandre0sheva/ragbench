from __future__ import annotations

from ragbench.config.schema import SystemConfig, ToolRef
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import IngestionResult, RetrievalResult
from ragbench.rag_systems.llm_query_base import LLMQueryRAG
from ragbench.rag_systems.options import AgentSearchOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.tool_agent_base import ToolAgentRAG
from ragbench.registry import SYSTEMS
from ragbench.tools import CorpusView
from ragbench.tools.base import Retriever
from ragbench.tools.registry import tool_ref_name


@SYSTEMS.register("agent_search")
class AgentSearchRAG(ToolAgentRAG, LLMQueryRAG):
    """A function-calling agent with a `search` tool (the same hybrid / vector retrieval the plain systems use) plus the tools in its `tools:` list.

    Compare one config with `tools: []` against another with `tools: [calculator, date_calc, corpus_grep]` to see what the tools are worth. The agent
    decides when to search, compute or grep, and writes the cited answer itself; retrieval is scored on the passages its tool calls returned.
    """

    spec = SystemSpec(
        type="agent_search",
        title="Tool-using agent (search)",
        summary="A function-calling agent that searches the corpus and calls its configured tools (calculator, date_calc, corpus_grep, ...), then writes the cited answer",
        best_for="Questions that need computation, exact lookups, or several searches",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=True,
        agentic=True,
        options=AgentSearchOptions,
        supports_tools=True,
    )
    options: AgentSearchOptions

    @classmethod
    def tool_refs(cls, config: SystemConfig) -> list[ToolRef]:
        refs: list[ToolRef] = list(config.tools)
        return refs if "search" in {tool_ref_name(ref) for ref in refs} else ["search", *refs]

    def ingest(self, documents: list[Document]) -> IngestionResult:
        self.corpus = CorpusView(documents)
        return super().ingest(documents)

    def _retrievers(self) -> dict[str, Retriever]:
        return {"search": self._search_tool}

    def _search_tool(self, query: str, top_k: int) -> RetrievalResult:
        """The `search` tool's retriever: untraced (the tool call is the step) and at least `per_query_top_k` deep, trimmed to what the agent asked for."""
        chunks, cost = self._rank(query, max(self.options.per_query_top_k, top_k))
        return RetrievalResult(question=query, chunks=chunks[:top_k], cost=cost)

    def _instructions(self) -> str:
        return "Use `search` for questions in natural language; use the other tools to compute exact answers or look for exact strings and identifiers."
