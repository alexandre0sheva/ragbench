from __future__ import annotations

import json

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, TextChunk
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult, RetrievedChunk
from ragbench.rag_systems.components import build_chunker, build_embedder, build_reranker, build_vector_index
from ragbench.rag_systems.options import LLMHeavyFeatures, LLMHeavyOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.runtime.context import current_runtime
from ragbench.runtime.parallel import ordered_parallel_map
from ragbench.utils.timing import timer


@SYSTEMS.register("llm_heavy")
class LLMHeavyRAG(BaseRAGSystem):
    spec = SystemSpec(
        type="llm_heavy",
        title="LLM-heavy",
        summary="LLM-driven ingestion metadata, query rewriting, and reranking",
        best_for="Higher-cost, quality-oriented experiments",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=True,
        agentic=False,
        options=LLMHeavyOptions,
        llm_features=LLMHeavyFeatures,
    )
    options: LLMHeavyOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.features = LLMHeavyFeatures.model_validate(config.llm_features)
        self.chunker = build_chunker(config.chunker)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.store = build_vector_index(self.embedding_model, self.options, self.name)
        self.reranker = build_reranker("llm" if self.features.enable_llm_rerank else self.options.reranker, llm=self.llm)
        self.original_text_by_chunk_id: dict[str, str] = {}

    def ingest(self, documents: list[Document]) -> IngestionResult:
        enable_llm_ingestion = self.features.enable_llm_ingestion
        with timer() as t:
            chunks = self.chunker.chunk(documents)
            self.original_text_by_chunk_id = {chunk.chunk_id: chunk.text for chunk in chunks}
            ingestion_cost = CostBreakdown()
            # One independent LLM call per chunk: run `evaluation.ingest_workers` of them at a time (order preserved).
            enrichments = (
                ordered_parallel_map(self._enrich_chunk, chunks, workers=current_runtime().ingest_workers, thread_name_prefix="ragbench-enrich")
                if enable_llm_ingestion
                else [None] * len(chunks)
            )
            indexed_chunks: list[TextChunk] = []
            for chunk, enriched in zip(chunks, enrichments, strict=True):
                metadata = dict(chunk.metadata)
                augmented_text = chunk.text
                if enriched is not None:
                    enrichment, cost = enriched
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
        final_top_k = self.options.resolve_top_k(top_k)
        per_query_top_k = max(self.options.per_query_top_k, final_top_k)
        with timer() as t:
            queries, rewrite_cost = self._rewrite_queries(question) if self.features.enable_query_rewrite else ([question], CostBreakdown())
            by_id: dict[str, RetrievedChunk] = {}
            retrieval_cost = rewrite_cost
            for query in queries:
                with self.trace.step("retrieve", "vector_search", top_k=per_query_top_k) as step:
                    step.set_input(query)
                    result = self.store.search(query, top_k=per_query_top_k)
                    step.set_chunks(result.chunks, result.cost)
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
            with self.trace.step("rerank", getattr(self.reranker, "name", "rerank"), candidates=len(candidates), top_k=final_top_k) as step:
                step.set_input(question)
                reranked = self.reranker.rerank(question, candidates, final_top_k)
                step.set_chunks(reranked.chunks, reranked.cost)
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
        with self.trace.step("llm", "query_rewrite") as step:
            result = self.llm.generate(messages, json_mode=True, temperature=0)
            cost = CostBreakdown(query_rewrite_cost=result.cost.total_cost)
            step.set_llm(result, cost=cost)
            step.set_input(question)
        try:
            parsed = json.loads(result.text)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
        except Exception:
            queries = []
        queries = queries[:5] or [question]
        return queries, cost
