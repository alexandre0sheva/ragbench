from __future__ import annotations

from ragbench.config.schema import SystemConfig, ToolRef
from ragbench.documents.schema import Document
from ragbench.rag_systems.base import IngestionResult
from ragbench.rag_systems.options import GrepAgentOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.tool_agent_base import ToolAgentRAG
from ragbench.registry import SYSTEMS
from ragbench.tools import CorpusView
from ragbench.utils.timing import timer

GREP_AGENT_TOOLS = ("list_documents", "corpus_grep", "read_document")


@SYSTEMS.register("grep_agent")
class GrepAgentRAG(ToolAgentRAG):
    """Index-free agentic search: no chunking, no embeddings, no vector store. The agent lists documents, greps them, and reads what it finds.

    A deliberately different architecture from every other system: it shows whether retrieval infrastructure is needed at all for your corpus.
    Ingestion is free; the cost is paid per question, in agent turns. Retrieval is scored on the documents its greps and reads touched.
    """

    spec = SystemSpec(
        type="grep_agent",
        title="Grep agent (no index)",
        summary="An index-free agent: it lists, greps and reads the raw documents with tools. No chunking, no embeddings, no vector store",
        best_for="Small or exact-match-heavy corpora, and testing whether you need retrieval infrastructure at all",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=True,
        agentic=True,
        options=GrepAgentOptions,
        chunker=None,
    )
    options: GrepAgentOptions

    @classmethod
    def tool_refs(cls, config: SystemConfig) -> list[ToolRef]:
        return list(GREP_AGENT_TOOLS)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            self.corpus = CorpusView(documents)
        return IngestionResult(system=self.name, num_documents=len(documents), num_chunks=0, latency_ms=t.elapsed_ms, metadata={"index": "none"})

    def _instructions(self) -> str:
        return (
            "There is no search index. Use `corpus_grep` to find where names, identifiers and figures occur, `read_document` to read around a hit, "
            "and `list_documents` to see what exists."
        )
