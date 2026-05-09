from __future__ import annotations

import json

from ragbench.config.schema import SystemConfig
from ragbench.documents.chunkers import create_chunker
from ragbench.documents.schema import Document, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import create_embedding_model
from ragbench.models.rerankers import create_reranker
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.stores.vector_store import VectorStore
from ragbench.utils.timing import timer


class LLMHeavyRAG(BaseRAGSystem):
    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = create_chunker(config.chunker)
        self.embedding_model = create_embedding_model(config.models.get("embedding", "text-embedding-3-small"), force_mock=force_mock)
        self.store = VectorStore(
            self.embedding_model,
            backend=config.retrieval.get("vector_store", "chroma"),
            collection_name=self.name,
            persist_directory=config.retrieval.get("persist_directory"),
        )
        self.reranker = create_reranker(config.retrieval.get("reranker", "simple_keyword_overlap"), llm=self.llm)
        self.original_text_by_chunk_id: dict[str, str] = {}

    def ingest(self, documents: list[Document]) -> IngestionResult:
        features = self.config.llm_features
        enable_llm_ingestion = bool(features.get("enable_llm_ingestion", False))
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            self.original_text_by_chunk_id = {chunk.chunk_id: chunk.text for chunk in chunks}
            ingestion_cost = CostBreakdown()
            indexed_chunks: list[TextChunk] = []
            for chunk in chunks:
                metadata = dict(chunk.metadata)
                augmented_text = chunk.text
                if enable_llm_ingestion:
                    enrichment, cost = self._enrich_chunk(chunk)
                    metadata["llm_enrichment"] = enrichment
                    ingestion_cost = ingestion_cost.plus(cost)
                    augmented_text = "\n\n".join(
                        [
                            chunk.text,
                            f"Summary: {enrichment.get('summary', '')}",
                            "Key entities: " + ", ".join(enrichment.get("key_entities", [])),
                            "Hypothetical questions: " + " ".join(enrichment.get("hypothetical_questions", [])),
                        ]
                    )
                metadata["original_text"] = chunk.text
                indexed_chunks.append(TextChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=augmented_text, metadata=metadata))
            embedding_cost = self.store.build(indexed_chunks)
            total_cost = ingestion_cost.plus(embedding_cost)
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=total_cost,
            metadata={"expensive": True, "enable_llm_ingestion": enable_llm_ingestion},
        )

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        features = self.config.llm_features
        retrieval_cfg = self.config.retrieval
        final_top_k = top_k or int(retrieval_cfg.get("top_k", 5))
        per_query_top_k = int(retrieval_cfg.get("per_query_top_k", 10))
        with timer() as t:
            queries, rewrite_cost = self._rewrite_queries(question) if features.get("enable_query_rewrite", True) else ([question], CostBreakdown())
            by_id: dict[str, RetrievedChunk] = {}
            retrieval_cost = rewrite_cost
            for query in queries:
                result = self.store.search(query, top_k=per_query_top_k)
                retrieval_cost = retrieval_cost.plus(result.cost)
                for chunk in result.chunks:
                    current = by_id.get(chunk.chunk_id)
                    if current is None or chunk.score > current.score:
                        original_text = chunk.metadata.get("original_text", chunk.text)
                        by_id[chunk.chunk_id] = RetrievedChunk(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            text=original_text,
                            score=chunk.score,
                            rank=chunk.rank,
                            metadata=dict(chunk.metadata),
                        )
            candidates = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
            for rank, chunk in enumerate(candidates, start=1):
                chunk.rank = rank
            reranker_name = "llm" if features.get("enable_llm_rerank", False) else retrieval_cfg.get("reranker", "simple_keyword_overlap")
            reranker = create_reranker(reranker_name, llm=self.llm)
            reranked = reranker.rerank(question, candidates, final_top_k)
            retrieval_cost = retrieval_cost.plus(reranked.cost)
        return RetrievalResult(
            question=question,
            chunks=reranked.chunks,
            latency_ms=t.elapsed_ms,
            cost=retrieval_cost,
            metadata={"retriever": "llm_heavy", "queries": queries, "expensive": True},
        )

    def _enrich_chunk(self, chunk: TextChunk) -> tuple[dict, CostBreakdown]:
        messages = [
            {"role": "system", "content": "Create retrieval metadata. Return strict JSON."},
            {
                "role": "user",
                "content": (
                    "For this chunk, produce keys summary, key_entities, hypothetical_questions. "
                    f"Chunk:\n{chunk.text[:3500]}"
                ),
            },
        ]
        result = self.llm.generate(messages, json_mode=True, temperature=0)
        try:
            parsed = json.loads(result.text)
        except Exception:
            parsed = {"summary": chunk.text[:300], "key_entities": [], "hypothetical_questions": []}
        return parsed, result.cost

    def _rewrite_queries(self, question: str) -> tuple[list[str], CostBreakdown]:
        messages = [
            {"role": "system", "content": "Rewrite the question into 3 to 5 concise search queries. Return JSON only."},
            {"role": "user", "content": f"Question: {question}\nReturn JSON like {{\"queries\": [\"...\"]}}."},
        ]
        result = self.llm.generate(messages, json_mode=True, temperature=0)
        try:
            parsed = json.loads(result.text)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
        except Exception:
            queries = []
        queries = queries[:5] or [question]
        cost = CostBreakdown(query_rewrite_cost=result.cost.total_cost)
        return queries, cost
