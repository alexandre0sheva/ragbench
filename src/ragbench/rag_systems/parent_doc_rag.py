from __future__ import annotations

from ragbench.config.schema import ParentChildChunkerConfig, SystemConfig
from ragbench.documents.chunkers import WordChunker
from ragbench.documents.schema import Document, TextChunk
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.rag_systems.components import build_embedder, build_vector_index
from ragbench.rag_systems.options import ParentDocOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.utils.ids import stable_chunk_id
from ragbench.utils.timing import timer


@SYSTEMS.register("parent_doc")
class ParentDocumentRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="parent_doc",
        title="Parent document",
        summary="Retrieve small child chunks, answer from their larger parent chunks",
        best_for="Better answer context with precise retrieval",
        cost_profile="low",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=ParentDocOptions,
        chunker=ParentChildChunkerConfig,
    )
    options: ParentDocOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        chunk_cfg = ParentChildChunkerConfig.model_validate(config.chunker)
        self.parent_chunker = WordChunker(chunk_size=chunk_cfg.parent_chunk_size, chunk_overlap=chunk_cfg.parent_chunk_overlap)
        self.child_chunker = WordChunker(chunk_size=chunk_cfg.child_chunk_size, chunk_overlap=chunk_cfg.child_chunk_overlap)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.child_store = build_vector_index(self.embedding_model, self.options, self.name)
        self.parents: dict[str, TextChunk] = {}

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            parent_chunks = self.parent_chunker.chunk(documents)
            self.parents = {chunk.chunk_id: chunk for chunk in parent_chunks}
            child_chunks: list[TextChunk] = []
            child_index = 0
            for parent in parent_chunks:
                pseudo_doc = Document(
                    doc_id=parent.doc_id,
                    path=parent.metadata.get("source_path", ""),
                    title=parent.metadata.get("title", ""),
                    text=parent.text,
                    metadata=parent.metadata,
                )
                for child in self.child_chunker.chunk([pseudo_doc]):
                    metadata = dict(child.metadata)
                    metadata["parent_chunk_id"] = parent.chunk_id
                    metadata["parent_start_char"] = parent.metadata.get("start_char")
                    child_chunks.append(
                        TextChunk(
                            chunk_id=stable_chunk_id(child.doc_id, child_index, child.text, metadata["start_char"], metadata["end_char"]),
                            doc_id=child.doc_id,
                            text=child.text,
                            metadata=metadata,
                        )
                    )
                    child_index += 1
            cost = self.child_store.build(child_chunks)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(child_chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"num_parent_chunks": len(parent_chunks), "num_child_chunks": len(child_chunks)},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        top_k_parents = top_k or self.options.top_k_parents
        # Asking for a deeper parent ranking needs enough children to cover that many distinct parents.
        top_k_children = max(self.options.top_k_children, top_k_parents * 2)
        # How child scores roll up to their parent: `max` keeps the single best
        # match, `sum` rewards parents hit by several children, `mean` averages.
        aggregation = self.options.parent_score_aggregation
        with timer() as t:
            with self.trace.step("retrieve", "child_vector_search", top_k=top_k_children) as step:
                step.set_input(question)
                child_result = self.child_store.search(question, top_k=top_k_children)
                step.set_chunks(child_result.chunks, child_result.cost)
            child_scores: dict[str, list[float]] = {}
            parent_children: dict[str, list[str]] = {}
            for child in child_result.chunks:
                parent_id = child.metadata.get("parent_chunk_id")
                if not parent_id:
                    continue
                child_scores.setdefault(parent_id, []).append(child.score)
                parent_children.setdefault(parent_id, []).append(child.chunk_id)
            parent_scores = {parent_id: _aggregate(scores, aggregation) for parent_id, scores in child_scores.items()}
            ordered = sorted(parent_scores, key=lambda pid: parent_scores[pid], reverse=True)[:top_k_parents]
            chunks: list[RetrievedChunk] = []
            for rank, parent_id in enumerate(ordered, start=1):
                parent = self.parents[parent_id]
                metadata = dict(parent.metadata)
                metadata["matched_child_chunk_ids"] = parent_children.get(parent_id, [])
                chunks.append(
                    RetrievedChunk(
                        chunk_id=parent.chunk_id,
                        doc_id=parent.doc_id,
                        text=parent.text,
                        score=parent_scores[parent_id],
                        rank=rank,
                        metadata=metadata,
                    )
                )
        return RetrievalResult(
            question=question,
            chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=child_result.cost,
            metadata={"retriever": "parent_doc", "top_k_children": top_k_children, "parent_score_aggregation": aggregation},
        )


def _aggregate(scores: list[float], aggregation: str) -> float:
    if aggregation == "sum":
        return sum(scores)
    if aggregation == "mean":
        return sum(scores) / len(scores)
    return max(scores)
