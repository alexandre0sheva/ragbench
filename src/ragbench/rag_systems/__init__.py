"""RAG system implementations."""

from ragbench.config.schema import SystemConfig
from ragbench.rag_systems.base import BaseRAGSystem
from ragbench.rag_systems.bm25_rag import BM25RAG
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.llm_heavy_rag import LLMHeavyRAG
from ragbench.rag_systems.parent_doc_rag import ParentDocumentRAG
from ragbench.rag_systems.rerank_rag import RerankRAG
from ragbench.rag_systems.vector_rag import VectorRAG

SYSTEM_REGISTRY: dict[str, type[BaseRAGSystem]] = {
    "bm25": BM25RAG,
    "vector": VectorRAG,
    "hybrid": HybridRAG,
    "rerank": RerankRAG,
    "parent_doc": ParentDocumentRAG,
    "llm_heavy": LLMHeavyRAG,
}


def create_rag_system(config: SystemConfig, force_mock: bool = False) -> BaseRAGSystem:
    cls = SYSTEM_REGISTRY.get(config.type)
    if not cls:
        raise ValueError(f"Unknown RAG system type: {config.type}. Available: {sorted(SYSTEM_REGISTRY)}")
    return cls(config=config, force_mock=force_mock)

