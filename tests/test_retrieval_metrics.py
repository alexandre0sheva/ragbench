from ragbench.evaluation.retrieval_metrics import compute_retrieval_metrics
from ragbench.rag_systems.base import RetrievedChunk


def test_retrieval_metrics_use_deduplicated_doc_ranking():
    chunks = [
        RetrievedChunk(chunk_id="a1", doc_id="doc_a", text="a", score=1, rank=1),
        RetrievedChunk(chunk_id="a2", doc_id="doc_a", text="a again", score=0.9, rank=2),
        RetrievedChunk(chunk_id="b1", doc_id="doc_b", text="b", score=0.8, rank=3),
    ]

    metrics = compute_retrieval_metrics(chunks, relevant_doc_ids=["doc_b"], qrels={"doc_b": 3}, k_values=[1, 2, 3])

    assert metrics["hit@1"] == 0.0
    assert metrics["hit@2"] == 1.0
    assert metrics["mrr@2"] == 0.5
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 1.0

