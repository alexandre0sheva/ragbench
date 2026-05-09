from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.rag_systems.bm25_rag import BM25RAG


def test_bm25_system_can_ingest_and_retrieve_tiny_dataset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    system = BM25RAG(
        SystemConfig(
            type="bm25",
            name="bm25_test",
            chunker={"type": "token", "chunk_size": 20, "chunk_overlap": 0},
            retrieval={"top_k": 2},
            models={"generator": "gpt-5.4-nano"},
        ),
        force_mock=True,
    )
    docs = [
        Document(doc_id="doc_001", path="a.md", title="A", text="HarborShield AI handles marine cargo submissions."),
        Document(doc_id="doc_002", path="b.md", title="B", text="ClaimPilot handles claims triage."),
    ]

    ingestion = system.ingest(docs)
    result = system.fetch_context("Which product handles marine cargo?")

    assert ingestion.num_chunks == 2
    assert result.chunks
    assert result.chunks[0].doc_id == "doc_001"
