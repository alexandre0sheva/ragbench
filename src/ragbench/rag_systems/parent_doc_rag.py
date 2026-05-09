from __future__ import annotations

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import TokenChunker
from ragbench.documents.schema import Document, TextChunk
from ragbench.models.embeddings import create_embedding_model
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.stores.vector_store import VectorStore
from ragbench.utils.ids import stable_chunk_id
from ragbench.utils.timing import timer


class ParentDocumentRAG(BaseRAGSystem):
    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        chunk_cfg = config.chunker
        self.parent_chunker = TokenChunker(
            chunk_size=int(chunk_cfg.get("parent_chunk_size", 1000)),
            chunk_overlap=int(chunk_cfg.get("parent_chunk_overlap", 150)),
        )
        self.child_chunker = TokenChunker(
            chunk_size=int(chunk_cfg.get("child_chunk_size", 250)),
            chunk_overlap=int(chunk_cfg.get("child_chunk_overlap", 50)),
        )
        self.embedding_model = create_embedding_model(config.models.get("embedding", "text-embedding-3-small"), force_mock=force_mock)
        self.child_store = VectorStore(
            self.embedding_model,
            backend=config.retrieval.get("vector_store", "chroma"),
            collection_name=self.name,
            persist_directory=config.retrieval.get("persist_directory"),
        )
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
        top_k_children = int(self.config.retrieval.get("top_k_children", 8))
        top_k_parents = top_k or int(self.config.retrieval.get("top_k_parents", 4))
        with timer() as t:
            child_result = self.child_store.search(question, top_k=top_k_children)
            parent_scores: dict[str, float] = {}
            parent_children: dict[str, list[str]] = {}
            for child in child_result.chunks:
                parent_id = child.metadata.get("parent_chunk_id")
                if not parent_id:
                    continue
                parent_scores[parent_id] = max(parent_scores.get(parent_id, float("-inf")), child.score)
                parent_children.setdefault(parent_id, []).append(child.chunk_id)
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
            metadata={"retriever": "parent_doc", "top_k_children": top_k_children},
        )
