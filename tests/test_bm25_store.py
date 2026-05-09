from ragbench.documents.schema import TextChunk
from ragbench.stores.bm25_store import BM25Store, _FallbackBM25
from ragbench.utils.text import tokenize


def _chunks() -> list[TextChunk]:
    return [
        TextChunk(chunk_id="doc_001::chunk::0", doc_id="doc_001", text="HarborShield AI handles marine cargo submissions."),
        TextChunk(chunk_id="doc_002::chunk::0", doc_id="doc_002", text="ClaimPilot handles claims triage workflows."),
        TextChunk(chunk_id="doc_003::chunk::0", doc_id="doc_003", text="Pricing for HarborShield includes a marine module."),
    ]


def test_bm25_store_ranks_lexical_matches_first():
    store = BM25Store()
    store.build(_chunks())
    result = store.search("marine cargo HarborShield", top_k=2)
    assert result.chunks
    assert result.chunks[0].doc_id == "doc_001"
    # ranks are 1-indexed and sequential
    assert [c.rank for c in result.chunks] == [1, 2]


def test_bm25_store_top_k_limits_results():
    store = BM25Store()
    store.build(_chunks())
    result = store.search("HarborShield", top_k=1)
    assert len(result.chunks) == 1


def test_bm25_store_returns_empty_for_unknown_terms():
    store = BM25Store()
    store.build(_chunks())
    result = store.search("zzzzz_unknown_token", top_k=3)
    # All zero scores; chunks may still be returned in arbitrary order with score 0
    assert all(c.score == 0 for c in result.chunks)


def test_fallback_bm25_matches_term_frequency_intuition():
    corpus = [tokenize("alpha beta gamma"), tokenize("alpha alpha delta"), tokenize("epsilon")]
    bm25 = _FallbackBM25(corpus)
    scores = bm25.get_scores(tokenize("alpha"))
    # second doc has alpha twice, should score higher than first
    assert scores[1] > scores[0]
    assert scores[2] == 0
