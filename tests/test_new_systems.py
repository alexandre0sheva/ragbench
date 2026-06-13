from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.embeddings import EMBEDDING_CACHE
from ragbench.rag_systems import SYSTEM_REGISTRY, create_rag_system


def setup_function(_):
    EMBEDDING_CACHE.clear()


def _docs() -> list[Document]:
    return [
        Document(doc_id="doc_001", path="a.md", title="A", text="HarborShield AI reviews marine cargo submissions for underwriters."),
        Document(doc_id="doc_002", path="b.md", title="B", text="ClaimPilot summarizes loss notices and classifies claim severity."),
        Document(doc_id="doc_003", path="c.md", title="C", text="Aurora Risk Suite compares quote conversion across regions."),
    ]


def test_new_systems_are_registered():
    assert "hyde" in SYSTEM_REGISTRY
    assert "hybrid_rerank" in SYSTEM_REGISTRY


def test_hybrid_rerank_retrieves_relevant_doc_in_mock_mode():
    system = create_rag_system(
        SystemConfig(
            type="hybrid_rerank",
            name="hybrid_rerank_test",
            chunker={"type": "token", "chunk_size": 50, "chunk_overlap": 0},
            retrieval={"candidate_top_k": 5, "final_top_k": 2},
        ),
        force_mock=True,
    )
    system.ingest(_docs())
    result = system.fetch_context("Which product reviews marine cargo submissions?")
    assert result.chunks
    assert result.chunks[0].doc_id == "doc_001"
    assert result.metadata["retriever"] == "hybrid_rrf_rerank"


def test_hyde_generates_probe_and_retrieves_in_mock_mode():
    system = create_rag_system(
        SystemConfig(
            type="hyde",
            name="hyde_test",
            chunker={"type": "token", "chunk_size": 50, "chunk_overlap": 0},
            retrieval={"top_k": 2},
        ),
        force_mock=True,
    )
    system.ingest(_docs())
    result = system.fetch_context("Which product reviews marine cargo submissions?")
    assert result.chunks
    assert result.chunks[0].doc_id == "doc_001"
    assert result.metadata["hypothetical_document"]
    # The mock probe should stay on-topic rather than refuse.
    assert "could not find" not in result.metadata["hypothetical_document"].lower()


def test_hyde_answers_end_to_end_in_mock_mode():
    system = create_rag_system(
        SystemConfig(type="hyde", name="hyde_e2e", chunker={"type": "token", "chunk_size": 50, "chunk_overlap": 0}, retrieval={"top_k": 2}),
        force_mock=True,
    )
    system.ingest(_docs())
    answer = system.answer_question("Which product reviews marine cargo submissions?")
    assert answer.answer
    assert answer.retrieval_result.chunks
