from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk, TextChunk
from ragbench.harness.prompts import ANSWER_PROMPTS, REDUCE_PROMPTS
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult
from ragbench.models.prompts import EXTRACT_GRAPH_MARKER, MAP_COMMUNITY_MARKER, SUMMARIZE_COMMUNITY_MARKER
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, chunk_documents
from ragbench.rag_systems.graph_index import Extraction, KnowledgeGraph, detect_communities, parse_extraction
from ragbench.rag_systems.llm_ingest import map_llm_calls
from ragbench.rag_systems.options import GraphRAGOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import sum_costs
from ragbench.registry import SYSTEMS
from ragbench.runtime.context import current_runtime
from ragbench.runtime.parallel import ordered_parallel_map
from ragbench.stores.bm25_store import BM25Store
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.query_planning import loads_lenient
from ragbench.utils.timing import timer

EXTRACT_MAX_TOKENS = 1500
REPORT_MAX_TOKENS = 400
MAP_MAX_TOKENS = 400
DESCRIPTION_CHARS = 300  # of one entity's merged descriptions, wherever they are shown
FALLBACK_REPORT_CHARS = 1500
NEIGHBOUR_WEIGHT = 0.5  # an entity one relation away from a matched one counts half as much
RANKED_CHUNKS = 50  # at least this many chunks of each ranking go into the fusion
REDUCE_MAX_POINTS = 40
MAP_WORKERS = 8  # map calls of one question in flight at once



@dataclass
class CommunityReport:
    community: int
    entities: list[str]  # keys into the graph, best connected first
    doc_ids: list[str]  # the documents its entities came from, most mentioned first
    text: str

    @property
    def chunk_id(self) -> str:
        return f"community::{self.community}"


@SYSTEMS.register("graph_rag")
class GraphRAG(BaseRAGSystem):
    """GraphRAG-lite: a knowledge graph extracted by an LLM at ingestion, its communities summarised, then local or global search.

    Every chunk goes to the LLM once to name its entities and the relations between them; the replies merge into one graph (entities
    by normalised name), Louvain splits it into communities, and each community of `min_community_size` entities gets an LLM-written
    report. `local` search matches the question to entities, follows their relations one step, and returns the chunks those entities
    came from (fused with a dense ranking); the generator also reads the entities, relations and main community report as notes.
    `global` search ranks the community reports and answers by map-reduce: one LLM call per report the generator reads draws scored
    points from it, and the answer is written from the points. Search is exact (NumPy), so there is no `vector_store` option.
    """

    spec = SystemSpec(
        type="graph_rag",
        title="GraphRAG (knowledge graph)",
        summary=(
            "An LLM extracts entities and relations from every chunk at ingestion and writes a report per graph community; `local` follows "
            "the question's entities to their chunks, `global` answers by map-reduce over the community reports"
        ),
        best_for="Questions that connect entities across documents (`local`) and corpus-wide questions about themes and groups (`global`)",
        cost_profile="high",
        latency_profile="medium",
        requires_llm=False,
        agentic=False,
        options=GraphRAGOptions,
    )
    options: GraphRAGOptions

    @classmethod
    def default_harness(cls, options: Any) -> Mapping[str, str]:
        """`global` mode answers from scored points of community reports, so its answer prompt says so."""
        return REDUCE_PROMPTS if options.mode == "global" else ANSWER_PROMPTS

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker, models=config.models, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.graph = KnowledgeGraph()
        self.titles: dict[str, str] = {}
        self.chunks: list[TextChunk] = []
        self.chunk_rows: dict[str, int] = {}
        self.chunk_vectors = np.zeros((0, 0), dtype=np.float32)
        self.entity_keys: list[str] = []
        self.entity_index = BM25Store()
        self.entity_vectors = np.zeros((0, 0), dtype=np.float32)
        self.reports: list[CommunityReport] = []
        self.report_of: dict[str, int] = {}  # entity key -> index into `reports`
        self.report_index = BM25Store()
        self.report_vectors = np.zeros((0, 0), dtype=np.float32)

    # --- ingestion ----------------------------------------------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> IngestionResult:
        workers = current_runtime().ingest_workers
        with timer() as t:
            self.chunks, cost = chunk_documents(self.chunker, documents)
            self.chunk_rows = {chunk.chunk_id: row for row, chunk in enumerate(self.chunks)}
            self.titles = {document.doc_id: document.title for document in documents}
            outcomes, extraction_failures = map_llm_calls(
                self._extract, self.chunks, workers=workers, what="a chunk's entities", thread_name_prefix="ragbench-graph"
            )
            self.graph = KnowledgeGraph()
            for chunk, outcome in zip(self.chunks, outcomes, strict=True):
                extraction, call_cost = outcome if outcome is not None else (Extraction(), CostBreakdown())
                self.graph.add(chunk, extraction)
                cost = cost.plus(call_cost)
            members = [group for group in detect_communities(list(self.graph.entities), self.graph.edges()) if len(group) >= self.options.min_community_size]
            members = [sorted(group, key=lambda key: (-self.graph.degree(key), key)) for group in members]
            written, report_failures = map_llm_calls(
                self._report,
                members,
                workers=workers,
                what="a community report",
                thread_name_prefix="ragbench-community",
                raise_when_all_fail=False,  # extraction already proved the model works, and every community has a fallback report
            )
            self.reports = []
            for number, (group, reply) in enumerate(zip(members, written, strict=True)):
                text, call_cost = reply if reply is not None else ("", CostBreakdown())
                cost = cost.plus(call_cost)
                if not text:  # the call failed or answered nothing: the community's own entities and relations stand in for it
                    text = self._fallback_report(group)
                self.reports.append(CommunityReport(community=number, entities=group, doc_ids=self.graph.doc_ids(*group), text=text))
            self.report_of = {key: number for number, report in enumerate(self.reports) for key in report.entities}
            cost = cost.plus(self._build_indexes())
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(self.chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={
                "embedding_model": self.embedding_model.model_name,
                "expensive": True,
                "entities": len(self.graph.entities),
                "relations": len(self.graph.relations),
                "communities": len(self.reports),
                "extraction_failures": extraction_failures,
                "report_failures": report_failures,
            },
        )

    def _build_indexes(self) -> CostBreakdown:
        """Embeddings and BM25 for what the mode searches: chunks and entities (`local`) or community reports (`global`)."""
        cost = CostBreakdown()
        empty = np.zeros((0, 0), dtype=np.float32)
        if self.options.mode == "global":
            cards = [TextChunk(chunk_id=report.chunk_id, doc_id=report.doc_ids[0], text=report.text) for report in self.reports]
            self.report_index.build(cards)
            embedding = self.embedding_model.embed_texts([card.text for card in cards]) if cards else None
            self.report_vectors = embedding.vectors if embedding else empty
            return cost.plus(embedding.cost) if embedding else cost
        self.entity_keys = list(self.graph.entities)
        cards = [TextChunk(chunk_id=f"entity::{key}", doc_id=self.graph.doc_ids(key)[0], text=self._entity_line(key)) for key in self.entity_keys]
        self.entity_index.build(cards)
        for texts, target in (([card.text for card in cards], "entity_vectors"), ([chunk.text for chunk in self.chunks], "chunk_vectors")):
            embedding = self.embedding_model.embed_texts(texts) if texts else None
            setattr(self, target, embedding.vectors if embedding else empty)
            if embedding is not None:
                cost = cost.plus(embedding.cost)
        return cost

    def _extract(self, chunk: TextChunk) -> tuple[Extraction, CostBreakdown]:
        messages = [
            {"role": "system", "content": "You build a knowledge graph of a document collection. Answer with JSON only."},
            {
                "role": "user",
                "content": (
                    f"{EXTRACT_GRAPH_MARKER}. Name at most {self.options.max_entities_per_chunk} entities: people, organizations, places, "
                    "products, events and other specific things, each with a short type and a one-sentence description from the text. Then "
                    "list the relations between them, each with a one-sentence description. Write names as they appear in the text.\n"
                    'Return JSON like {"entities": [{"name": "...", "type": "...", "description": "..."}], '
                    '"relations": [{"source": "...", "target": "...", "description": "..."}]}.\n'
                    f"Document title: {self.titles.get(chunk.doc_id, chunk.doc_id)}\n<text>\n{chunk.text}\n</text>"
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, json_mode=True, max_tokens=EXTRACT_MAX_TOKENS)
        extraction = parse_extraction(result.text)
        if extraction is None:  # counted as a failed call, so a model that never answers in JSON stops the run instead of yielding no graph
            raise ValueError(f"unusable extraction reply (finish_reason {result.finish_reason}): {result.text[:80]!r}")
        return extraction, result.cost

    def _report(self, group: list[str]) -> tuple[str, CostBreakdown]:
        messages = [
            {"role": "system", "content": "You write reports that help a search system answer questions about a document collection."},
            {
                "role": "user",
                "content": (
                    f"{SUMMARIZE_COMMUNITY_MARKER}. Start with a short title line, then say in a few sentences who or what the community is "
                    "about, how its members are related, and the most important facts. Use only the information below. Answer only with the report.\n"
                    f"{self._community_text(group)}"
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, max_tokens=REPORT_MAX_TOKENS)
        return result.text.strip(), result.cost

    def _community_text(self, group: list[str]) -> str:
        shown = group[: self.options.community_max_entities]
        inside = set(shown)
        keys = {(a, b) for a in shown for b, _ in self.graph.neighbours(a) if b in inside and a < b}
        relations = sorted((self.graph.relations[key] for key in keys), key=lambda relation: (-relation.weight, relation.source, relation.target))[: 2 * len(shown)]
        lines = ["Entities:", *(f"- {self._entity_line(key)}" for key in shown), "Relations:"]
        lines += [f"- {self._relation_line(relation.source, relation.target)}" for relation in relations]
        return "\n".join(lines)

    def _fallback_report(self, group: list[str]) -> str:
        return self._community_text(group)[:FALLBACK_REPORT_CHARS]

    def _entity_line(self, key: str) -> str:
        entity = self.graph.entities[key]
        return f"{entity.name} ({entity.type}): {' '.join(entity.descriptions)[:DESCRIPTION_CHARS]}"

    def _relation_line(self, a: str, b: str) -> str:
        relation = self.graph.relations[(min(a, b), max(a, b))]
        names = self.graph.entities[relation.source].name, self.graph.entities[relation.target].name
        return f"{names[0]} — {names[1]}: {' '.join(relation.descriptions[:2])[:DESCRIPTION_CHARS]}"

    # --- search ---------------------------------------------------------------------------------------------------------------

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        with timer() as t:
            query = self.embedding_model.embed_query(question)
            vector = query.vectors[0]
            if self.options.mode == "global":
                chunks, metadata = self._global_search(question, vector, k, query.cost)
            else:
                chunks, metadata = self._local_search(question, vector, k, query.cost)
        return RetrievalResult(question=question, chunks=chunks, latency_ms=t.elapsed_ms, cost=query.cost, metadata=metadata)

    def _local_search(self, question: str, vector: np.ndarray, k: int, query_cost: CostBreakdown) -> tuple[list[RetrievedChunk], dict]:
        with self.trace.step("retrieve", "entity_search", entities_k=self.options.entities_k) as step:
            step.set_input(question)
            matched = self._match_entities(question, vector)
            step.set_chunks(matched, query_cost)
        seeds = [chunk.chunk_id.removeprefix("entity::") for chunk in matched]
        with self.trace.step("retrieve", "graph_expand", entities=len(seeds)) as step:
            step.set_input(question)
            chunks = self._chunks_around(seeds, vector, k)
            step.set_chunks(chunks)
        metadata = {
            "retriever": "graph_local",
            "matched_entities": [self.graph.entities[key].name for key in seeds],
            "graph_notes": self._notes(seeds),
        }
        return chunks, metadata

    def _match_entities(self, question: str, vector: np.ndarray) -> list[RetrievedChunk]:
        """The `entities_k` entities closest to the question: BM25 and cosine over each entity's name and description, fused with RRF."""
        if not self.entity_keys:
            return []
        scores = self.entity_vectors @ vector
        order = np.argsort(-scores, kind="stable")[: max(RANKED_CHUNKS, self.options.entities_k)]
        dense = [
            RetrievedChunk(chunk_id=f"entity::{self.entity_keys[i]}", doc_id=self.graph.doc_ids(self.entity_keys[i])[0], text=self._entity_line(self.entity_keys[i]), score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]
        lexical = self.entity_index.search(question, top_k=len(self.entity_keys)).chunks
        return reciprocal_rank_fusion([dense, lexical], top_k=self.options.entities_k, rrf_k=self.options.rrf_k)

    def _chunks_around(self, seeds: list[str], vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """The chunks the matched entities and their neighbours came from (an entity weighs 1/rank, a neighbour half its entity's),
        fused by RRF with a dense ranking of every chunk, which also fills the ranking when the graph reaches too few."""
        if not self.chunks:
            return []
        weight: dict[str, float] = {}
        for rank, key in enumerate(seeds, start=1):
            weight[key] = weight.get(key, 0.0) + 1 / rank
            for neighbour, _ in self.graph.neighbours(key)[: self.options.neighbours_per_entity]:
                if neighbour not in seeds:
                    weight[neighbour] = weight.get(neighbour, 0.0) + NEIGHBOUR_WEIGHT / rank
        score: dict[str, float] = {}
        linked: dict[str, list[str]] = {}
        for key, value in weight.items():
            for chunk_id in self.graph.entities[key].chunk_ids:
                score[chunk_id] = score.get(chunk_id, 0.0) + value
                linked.setdefault(chunk_id, []).append(self.graph.entities[key].name)
        depth = max(RANKED_CHUNKS, k)
        by_graph = sorted(score, key=lambda chunk_id: (-score[chunk_id], self.chunk_rows[chunk_id]))[:depth]
        similarity = self.chunk_vectors @ vector
        by_vector = [self.chunks[row].chunk_id for row in np.argsort(-similarity, kind="stable")[:depth]]

        def ranking(chunk_ids: list[str]) -> list[RetrievedChunk]:
            return [self._as_retrieved(chunk_id, float(similarity[self.chunk_rows[chunk_id]]), rank, linked.get(chunk_id, []), score.get(chunk_id, 0.0)) for rank, chunk_id in enumerate(chunk_ids, start=1)]

        return reciprocal_rank_fusion([ranking(by_graph), ranking(by_vector)], top_k=k, rrf_k=self.options.rrf_k)

    def _as_retrieved(self, chunk_id: str, similarity: float, rank: int, entities: list[str], graph_score: float) -> RetrievedChunk:
        chunk = self.chunks[self.chunk_rows[chunk_id]]
        metadata = {**chunk.metadata, "graph_entities": entities, "graph_score": round(graph_score, 4)}
        return RetrievedChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text, score=similarity, rank=rank, metadata=metadata)

    def _notes(self, seeds: list[str]) -> str:
        """What the generator reads about the matched entities: their descriptions, the strongest relations around them, and the report
        of the community most of them belong to, cut to `notes_max_chars`."""
        if not seeds or self.options.notes_max_chars == 0:
            return ""
        chosen = set(seeds)
        touching = {(min(key, other), max(key, other)) for key in seeds for other, _ in self.graph.neighbours(key)}
        relations = sorted(touching, key=lambda key: (-(key[0] in chosen and key[1] in chosen), -self.graph.relations[key].weight, key))[: self.options.max_relations]
        lines = ["Entities:", *(f"- {self._entity_line(key)}" for key in seeds)]
        if relations:
            lines += ["Relations:", *(f"- {self._relation_line(*key)}" for key in relations)]
        communities = Counter(self.report_of[key] for key in seeds if key in self.report_of)
        if communities:
            lines += ["Community report:", self.reports[communities.most_common(1)[0][0]].text]
        return "\n".join(lines)[: self.options.notes_max_chars]

    def _global_search(self, question: str, vector: np.ndarray, k: int, query_cost: CostBreakdown) -> tuple[list[RetrievedChunk], dict]:
        with self.trace.step("retrieve", "community_search", communities=len(self.reports)) as step:
            step.set_input(question)
            reports = self._rank_reports(question, vector, k)
            step.set_chunks(reports, query_cost)
        return reports, {"retriever": "graph_global", "communities": [chunk.metadata["community"] for chunk in reports]}

    def _rank_reports(self, question: str, vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        """The `k` community reports closest to the question (BM25 and cosine, fused with RRF). A report counts as its most mentioned
        document for the retrieval metrics; all its documents are in `metadata.doc_ids`."""
        if not self.reports:
            return []
        scores = self.report_vectors @ vector
        order = np.argsort(-scores, kind="stable")
        dense = [self._report_chunk(int(i), float(scores[i]), rank) for rank, i in enumerate(order, start=1)]
        lexical = self.report_index.search(question, top_k=len(self.reports)).chunks
        fused = reciprocal_rank_fusion([dense, lexical], top_k=k, rrf_k=self.options.rrf_k)
        return fused

    def _report_chunk(self, number: int, score: float, rank: int) -> RetrievedChunk:
        report = self.reports[number]
        names = [self.graph.entities[key].name for key in report.entities]
        metadata = {"community": report.community, "doc_ids": report.doc_ids, "entities": names}
        return RetrievedChunk(chunk_id=report.chunk_id, doc_id=report.doc_ids[0], text=report.text, score=score, rank=rank, metadata=metadata)

    # --- generation -----------------------------------------------------------------------------------------------------------

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]):
        if self.options.mode == "global":
            return self._map_reduce(question, chunks)
        notes = retrieval.metadata.get("graph_notes")
        if not notes:
            return super()._generate_from_retrieval(retrieval, question, chunks)
        messages = [
            {"role": "system", "content": self.harness.answer_system()},
            {"role": "user", "content": f"Graph notes (not evidence by themselves):\n{notes}\n\nContext:\n{self._format_context(chunks)}\n\nQuestion: {question}"},
        ]
        with self.trace.step("generate", "answer", context_chunks=len(chunks), matched_entities=len(retrieval.metadata.get("matched_entities", []))) as step:
            result = self.llm.generate(messages, temperature=0)
            step.set_llm(result)
            step.set_input(question)
        return result

    def _map_reduce(self, question: str, reports: list[RetrievedChunk]) -> LLMResult:
        """Map: each report the generator reads yields scored points (the calls run at once, booked as one step). Reduce: the answer is
        written from the points that scored above 0. The returned result carries the cost and tokens of every call."""
        spent: list[LLMResult] = []
        if reports:
            with self.trace.step("generate", "map_communities", reports=len(reports)) as step:
                spent = ordered_parallel_map(
                    lambda report: self.llm.generate(self._map_messages(question, report.text), temperature=0, json_mode=True, max_tokens=MAP_MAX_TOKENS),
                    reports,
                    workers=MAP_WORKERS,
                    thread_name_prefix="ragbench-map",
                )
                step.set_cost(sum_costs(result.cost for result in spent))
                step.set_tokens(sum(result.prompt_tokens for result in spent), sum(result.completion_tokens for result in spent))
                step.set_input(question)
                if all(result.cached for result in spent):  # replayed from the cache: report the slowest original call, as they ran at once
                    step.latency_override_ms = max(result.latency_ms or 0.0 for result in spent)
        points: list[tuple[int, int, str, str]] = []  # (score, report rank, doc id, text)
        for index, (report, result) in enumerate(zip(reports, spent, strict=True)):
            points += [(score, index, report.doc_id, text) for score, text in _points(result.text) if score > 0]
        points.sort(key=lambda point: (-point[0], point[1]))
        context = "\n".join(f"[{doc_id} | community point] (score {score}) {text}" for score, _, doc_id, text in points[:REDUCE_MAX_POINTS])
        messages = [{"role": "system", "content": self.harness.answer_system()}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        with self.trace.step("generate", "reduce", reports=len(reports), points=len(points)) as step:
            answer = self.llm.generate(messages, temperature=0)
            step.set_llm(answer)
            step.set_input(question)
        return dataclasses.replace(
            answer,
            cost=answer.cost.plus(sum_costs(result.cost for result in spent)),
            prompt_tokens=answer.prompt_tokens + sum(result.prompt_tokens for result in spent),
            completion_tokens=answer.completion_tokens + sum(result.completion_tokens for result in spent),
        )

    @staticmethod
    def _map_messages(question: str, report: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "You help answer questions about a document collection. Answer with JSON only."},
            {
                "role": "user",
                "content": (
                    f"{MAP_COMMUNITY_MARKER}. Give each point a score from 0 (does not help) to 100 (answers it); if nothing in the report "
                    'helps, return no points. Return JSON like {"points": [{"text": "...", "score": 0}]}.\n'
                    f"Question: {question}\n\nReport:\n{report}"
                ),
            },
        ]


def _points(text: str) -> list[tuple[int, str]]:
    """(score, text) of each well-formed point of a map reply; the score is clamped to 0–100, a missing one counts as 0."""
    data = loads_lenient(text)
    items = data.get("points") if isinstance(data, dict) else None
    points: list[tuple[int, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            continue
        try:
            score = int(float(item.get("score") or 0))
        except (TypeError, ValueError):
            score = 0
        points.append((max(0, min(100, score)), " ".join(item["text"].split())))
    return points
