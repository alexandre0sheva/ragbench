"""RAG system implementations. Importing this package registers every built-in system in `SYSTEMS`."""

from ragbench.config.schema import SystemConfig
from ragbench.rag_systems.adaptive_rag import AdaptiveRAG
from ragbench.rag_systems.agent_search_rag import AgentSearchRAG
from ragbench.rag_systems.base import BaseRAGSystem
from ragbench.rag_systems.bm25_rag import BM25RAG
from ragbench.rag_systems.contextual_rag import ContextualRAG
from ragbench.rag_systems.corrective_rag import CorrectiveRAG
from ragbench.rag_systems.decompose_rag import DecomposeRAG
from ragbench.rag_systems.full_context_rag import FullContextRAG
from ragbench.rag_systems.grep_agent_rag import GrepAgentRAG
from ragbench.rag_systems.hierarchical_rag import HierarchicalRAG
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.hybrid_rerank_rag import HybridRerankRAG
from ragbench.rag_systems.hyde_rag import HyDERAG
from ragbench.rag_systems.iterative_rag import IterativeRAG
from ragbench.rag_systems.llm_heavy_rag import LLMHeavyRAG
from ragbench.rag_systems.no_retrieval_rag import NoRetrievalRAG
from ragbench.rag_systems.parent_doc_rag import ParentDocumentRAG
from ragbench.rag_systems.rag_fusion_rag import RagFusionRAG
from ragbench.rag_systems.rerank_rag import RerankRAG
from ragbench.rag_systems.sentence_window_rag import SentenceWindowRAG
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.vector_rag import VectorRAG
from ragbench.registry import CHUNKERS, RERANKERS, SYSTEMS

# Live view of the registry, kept so `SYSTEM_REGISTRY[...]` / `monkeypatch.setitem(SYSTEM_REGISTRY, ...)` keep working.
SYSTEM_REGISTRY = SYSTEMS.mapping

# Third-party packages can contribute components via entry points (see docs/extending.md).
SYSTEMS.load_entry_points("ragbench.systems")
CHUNKERS.load_entry_points("ragbench.chunkers")
RERANKERS.load_entry_points("ragbench.rerankers")


def all_specs() -> list[SystemSpec]:
    """Specs of every registered system that declares one, ordered cheapest/fastest first, then by name."""
    cost_rank = {"free": 0, "low": 1, "medium": 2, "high": 3}
    latency_rank = {"fast": 0, "medium": 1, "slow": 2}
    specs = [spec for name in SYSTEMS.names() if (spec := SYSTEMS.get(name).spec) is not None]
    return sorted(specs, key=lambda spec: (cost_rank[spec.cost_profile], latency_rank[spec.latency_profile], spec.type))


def create_rag_system(config: SystemConfig, force_mock: bool = False) -> BaseRAGSystem:
    return SYSTEMS.get(config.type)(config=config, force_mock=force_mock)


__all__ = [
    "BM25RAG",
    "AdaptiveRAG",
    "AgentSearchRAG",
    "SYSTEMS",
    "SYSTEM_REGISTRY",
    "BaseRAGSystem",
    "ContextualRAG",
    "CorrectiveRAG",
    "DecomposeRAG",
    "FullContextRAG",
    "GrepAgentRAG",
    "HierarchicalRAG",
    "HyDERAG",
    "HybridRAG",
    "HybridRerankRAG",
    "IterativeRAG",
    "LLMHeavyRAG",
    "NoRetrievalRAG",
    "ParentDocumentRAG",
    "RagFusionRAG",
    "RerankRAG",
    "SentenceWindowRAG",
    "SystemSpec",
    "VectorRAG",
    "all_specs",
    "create_rag_system",
]
