from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document, RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.prompts import SUMMARIZE_CLUSTER_MARKER
from ragbench.rag_systems.base import BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.components import build_chunker, build_embedder, chunk_documents
from ragbench.rag_systems.llm_ingest import map_llm_calls
from ragbench.rag_systems.options import RaptorOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.runtime.context import current_runtime
from ragbench.utils.text import lead_sentences, normalize_text
from ragbench.utils.timing import timer

SUMMARY_MAX_TOKENS = 400
FALLBACK_CHARS = 600
SEED = 0


@dataclass
class Node:
    """A chunk (level 0) or a cluster summary. A summary's documents are those of the chunks under it, most represented first."""

    chunk_id: str
    level: int
    text: str
    doc_counts: Counter[str]
    children: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # a chunk's own metadata (offsets, title); empty for a summary

    @property
    def doc_ids(self) -> list[str]:
        return [doc_id for doc_id, _ in self.doc_counts.most_common()]


def cluster_vectors(vectors: np.ndarray, *, max_clusters: int, pca_dims: int, min_clusters: int = 2) -> list[list[int]]:
    """Rows of `vectors` grouped by a Gaussian mixture on PCA-reduced vectors; the number of clusters (`min_clusters` to `max_clusters`, at most
    half the rows unless more are required) is the one with the lowest BIC. Deterministic (fixed seeds); empty components are dropped; clusters
    come in order of first member."""
    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    rows = len(vectors)
    if rows < 2 or np.allclose(vectors, vectors[0]):  # nothing to separate (identical chunks): one cluster
        return [list(range(rows))]
    dims = max(1, min(pca_dims, rows - 1, vectors.shape[1]))
    fitted: dict[int, tuple[float, np.ndarray]] = {}  # k -> (BIC, labels)
    with warnings.catch_warnings():  # few distinct points for many components is expected here, not a problem to report
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        reduced = PCA(n_components=dims, random_state=SEED).fit_transform(vectors.astype(np.float64))

        def bic(k: int) -> float:
            mixture = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, random_state=SEED).fit(reduced)
            fitted[k] = (mixture.bic(reduced), mixture.predict(reduced))
            return fitted[k][0]

        k = _lowest(bic, min_clusters, max(min_clusters, min(max_clusters, rows // 2)))
    labels = fitted[k][1]
    groups: dict[int, list[int]] = {}
    for row, label in enumerate(labels):
        groups.setdefault(int(label), []).append(row)
    return list(groups.values())


def _lowest(score: Callable[[int], float], lo: int, hi: int) -> int:
    """The k in `lo..hi` with the lowest `score`: a short range is scored whole; a long one on a geometric grid of 8 points, then by halving the
    gaps on both sides of the best point until none is left (BIC over k is close to unimodal, so this finds its minimum in about 15 fits)."""
    if hi - lo < 8:
        return min(range(lo, hi + 1), key=lambda k: (score(k), k))
    tried = {k: score(k) for k in sorted({lo, hi, *(round(lo * (hi / lo) ** (i / 7)) for i in range(8))})}
    while True:
        ks = sorted(tried)
        best = min(ks, key=lambda k: (tried[k], k))
        at = ks.index(best)
        probes = {(ks[at - 1] + best) // 2 if at > 0 else best, (best + ks[at + 1] + 1) // 2 if at + 1 < len(ks) else best} - set(tried)
        if not probes:
            return best
        for k in sorted(probes):
            tried[k] = score(k)


@SYSTEMS.register("raptor")
class RaptorRAG(BaseRAGSystem):
    """RAPTOR (Sarthi et al. 2024): a tree of LLM summaries over clustered chunks, searched flat (`collapsed`) or top-down (`tree`).

    Chunks are embedded and clustered (Gaussian mixture on PCA-reduced embeddings, the number of clusters by BIC); an LLM summarises each
    cluster of two or more nodes, the summaries are embedded and clustered again, until `max_levels` or fewer than `min_cluster_nodes`
    nodes are left. A node alone in its cluster is carried up unchanged. Unlike `hierarchical` (one summary per document, written once), the
    summaries here cover related passages from anywhere in the corpus, at several levels of abstraction. Search is exact (NumPy).
    """

    spec = SystemSpec(
        type="raptor",
        title="RAPTOR (recursive summary tree)",
        summary="Chunks clustered and summarised by an LLM, recursively, into a tree; the question is matched against chunks and summaries of every level",
        best_for="Questions that need the gist of several related passages, and corpora where one fact is spread over many chunks",
        cost_profile="high",
        latency_profile="fast",
        requires_llm=False,
        agentic=False,
        options=RaptorOptions,
    )
    options: RaptorOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.chunker = build_chunker(config.chunker, models=config.models, force_mock=force_mock)
        self.embedding_model = build_embedder(config.models, force_mock)
        self.nodes: list[Node] = []
        self.node_index: dict[str, int] = {}
        self.levels: list[list[int]] = []  # node ids per level; a node carried up appears in several
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        with timer() as t:
            chunks, cost = chunk_documents(self.chunker, documents)
            self.nodes = [Node(chunk_id=chunk.chunk_id, level=0, text=chunk.text, doc_counts=Counter({chunk.doc_id: 1}), metadata=chunk.metadata) for chunk in chunks]
            embedded = self.embedding_model.embed_texts([node.text for node in self.nodes]) if self.nodes else None
            vectors = [embedded.vectors] if embedded is not None else []
            if embedded is not None:
                cost = cost.plus(embedded.cost)
            self.levels = [list(range(len(self.nodes)))]
            summaries = failures = 0
            current = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)  # row i embeds node i: both are appended together
            while len(self.levels) <= self.options.max_levels and len(self.levels[-1]) >= self.options.min_cluster_nodes:
                level_ids = self.levels[-1]
                clusters = [[level_ids[row] for row in group] for group in cluster_vectors(current[level_ids], max_clusters=self.options.max_clusters, pca_dims=self.options.pca_dims)]
                clusters = [part for group in clusters for part in self._fitting(group, current)]
                groups = [group for group in clusters if len(group) > 1]
                written, failed = map_llm_calls(
                    self._summarize, groups, workers=current_runtime().ingest_workers, what="a cluster summary", thread_name_prefix="ragbench-raptor"
                )
                failures += failed
                level = len(self.levels)
                new_ids: list[int] = []
                texts: list[str] = []
                for group, outcome in zip(groups, written, strict=True):
                    text, call_cost = outcome if outcome is not None else ("", CostBreakdown())
                    cost = cost.plus(call_cost)
                    counts: Counter[str] = Counter()
                    for child in group:
                        counts.update(self.nodes[child].doc_counts)
                    node = Node(chunk_id=f"raptor::{level}::{len(new_ids)}", level=level, text=text or lead_sentences([self.nodes[c].text for c in group], FALLBACK_CHARS), doc_counts=counts, children=group)
                    new_ids.append(len(self.nodes))
                    self.nodes.append(node)
                    texts.append(node.text)
                summaries += len(groups)
                if texts:
                    embedded = self.embedding_model.embed_texts(texts)
                    cost = cost.plus(embedded.cost)
                    current = np.vstack([current, embedded.vectors])
                carried = [group[0] for group in clusters if len(group) == 1]
                next_level = sorted(new_ids + carried)
                if len(next_level) >= len(level_ids):  # clustering found no structure left to summarise
                    break
                self.levels.append(next_level)
            self.vectors = current
            self.node_index = {node.chunk_id: i for i, node in enumerate(self.nodes)}
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=len(chunks),
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={
                "embedding_model": self.embedding_model.model_name,
                "expensive": True,
                "levels": [len(level) for level in self.levels],
                "summaries": summaries,
                "summary_failures": failures,
            },
        )

    def _fitting(self, group: list[int], vectors: np.ndarray) -> list[list[int]]:
        """`group`, clustered again within itself (the paper's local clustering) until each part's members fit `summary_input_chars`
        together; a part its vectors cannot split is packed in order. A single node longer than the budget stays alone (carried up)."""
        budget = self.options.summary_input_chars
        sizes = {i: len(normalize_text(self.nodes[i].text)) for i in group}
        total = sum(sizes.values())
        if total <= budget or len(group) < 2:
            return [group]
        needed = min(len(group), math.ceil(total / budget))
        parts = cluster_vectors(vectors[group], min_clusters=max(2, needed), max_clusters=max(2, 2 * needed), pca_dims=self.options.pca_dims)
        if len(parts) < 2:  # the vectors do not separate (near duplicates): fill parts in order instead
            packed: list[list[int]] = [[]]
            for i in group:
                if packed[-1] and sum(sizes[j] for j in packed[-1]) + sizes[i] > budget:
                    packed.append([])
                packed[-1].append(i)
            return packed
        return [fit for part in parts for fit in self._fitting([group[row] for row in part], vectors)]

    def _summarize(self, group: list[int]) -> tuple[str, CostBreakdown]:
        texts = [normalize_text(self.nodes[i].text) for i in group]
        if sum(len(text) for text in texts) > self.options.summary_input_chars:  # only a part that could not be split further
            texts = [text[: self.options.summary_input_chars // len(texts)] for text in texts]
        passages = "\n".join(f"[{n}] {text}" for n, text in enumerate(texts, start=1))
        messages = [
            {"role": "system", "content": "You write summaries that help a search system answer questions about a document collection."},
            {
                "role": "user",
                "content": (
                    f"{SUMMARIZE_CLUSTER_MARKER} in one paragraph. Keep the specific facts a question might ask about (names, numbers, dates, "
                    f"conditions). Answer only with the summary.\nPassages:\n{passages}"
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, max_tokens=SUMMARY_MAX_TOKENS)
        return " ".join(result.text.split()), result.cost

    # --- search ---------------------------------------------------------------------------------------------------------------

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        k = self.options.resolve_top_k(top_k)
        if not self.nodes:
            return RetrievalResult(question=question, chunks=[], metadata={"retriever": f"raptor_{self.options.mode}"})
        with timer() as t:
            query = self.embedding_model.embed_query(question)
            scores = self.vectors @ query.vectors[0]
            metadata: dict = {"retriever": f"raptor_{self.options.mode}"}
            with self.trace.step("retrieve", f"{self.options.mode}_search", nodes=len(self.nodes)) as step:
                step.set_input(question)
                if self.options.mode == "tree":
                    chosen = self._descend(scores)
                    metadata["chosen"] = chosen
                    candidates = sorted({node_id for level in chosen for node_id in level}, key=lambda i: (-scores[i], i))
                else:
                    candidates = list(np.argsort(-scores, kind="stable"))
                chunks = [self._as_retrieved(int(i), float(scores[i]), rank) for rank, i in enumerate(candidates[:k], start=1)]
                step.set_chunks(chunks, query.cost)
        return RetrievalResult(question=question, chunks=chunks, latency_ms=t.elapsed_ms, cost=query.cost, metadata=metadata)

    def _descend(self, scores: np.ndarray) -> list[list[int]]:
        """Top-down: the `branch_k` best nodes of the top level, then the best `branch_k` of their children, down to the chunks."""
        chosen: list[list[int]] = []
        frontier = self.levels[-1]
        while frontier:
            picked = sorted(frontier, key=lambda i: (-scores[i], i))[: self.options.branch_k]
            chosen.append(picked)
            frontier = sorted({child for i in picked for child in self.nodes[i].children})
        return chosen

    def _as_retrieved(self, node_id: int, score: float, rank: int) -> RetrievedChunk:
        node = self.nodes[node_id]
        metadata = {**node.metadata, "level": node.level, "doc_ids": node.doc_ids}
        return RetrievedChunk(chunk_id=node.chunk_id, doc_id=node.doc_ids[0], text=node.text, score=score, rank=rank, metadata=metadata)

