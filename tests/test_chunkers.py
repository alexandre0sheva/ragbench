from ragbench.documents.chunkers import TokenChunker
from ragbench.documents.schema import Document


def test_token_chunker_creates_overlapping_chunks():
    doc = Document(doc_id="doc_test", path="doc_test.md", title="Test", text=" ".join(f"word{i}" for i in range(12)))
    chunks = TokenChunker(chunk_size=5, chunk_overlap=2).chunk([doc])

    assert len(chunks) == 4
    first = chunks[0].text.split()
    second = chunks[1].text.split()
    assert first[-2:] == second[:2]
    assert chunks[0].chunk_id != chunks[1].chunk_id

