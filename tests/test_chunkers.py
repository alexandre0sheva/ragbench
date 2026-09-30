"""Chunkers: invariants every chunker must satisfy, plus what is specific to each. Offline (a fake tokenizer stands in for tiktoken)."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from ragbench.config.schema import ChunkerConfig, SystemConfig
from ragbench.documents import tokenizer as tokenizer_module
from ragbench.documents.chunkers import (
    BaseChunker,
    FixedCharacterChunker,
    MarkdownAwareChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
    create_chunker,
)
from ragbench.documents.schema import Document
from ragbench.documents.tokenizer import ApproxTokenizer, count_tokens, get_tokenizer
from ragbench.models.embeddings import EmbeddingModel, EmbeddingResult, HashingEmbeddingModel
from ragbench.registry import CHUNKERS

PROSE = (
    "Refunds are accepted within thirty days of delivery. Damaged goods get a full refund! Do you need a receipt? "
    "Yes, a receipt is required for every return.\n\n"
    "Shipping takes five business days within the country. International orders may take up to three weeks. "
    "Tracking numbers are emailed the same day the parcel leaves the warehouse.\n\n"
    "Gift cards never expire and can be used on any order. They cannot be exchanged for cash. "
    "Lost cards are replaced only when the original receipt is available. Contact support for details.\n\n"
    "Warranty covers manufacturing defects for two years from the purchase date. Accidental damage is not covered. "
    "Repairs take about ten business days, and a loaner unit is offered for premium plans.\n"
)
MARKDOWN = """Intro text before any heading, long enough to matter for the preamble section of this document.

# Guide

The guide overview paragraph explains what this document covers in a few plain sentences so readers can orient.

## Returns

Refunds are accepted within thirty days. Damaged goods get a full refund and a prepaid label.

### Damaged goods

Photograph the packaging before opening it. Keep every piece of the original box for the carrier inspection.

## Shipping

Shipping takes five business days.

```python
# Not a heading, even though it starts with a hash
print("hello")
```

### Tracking

Tracking numbers are emailed the same day.

# Appendix

Extra notes live here.
"""

ALL = ["fixed_char", "word", "token", "recursive", "sentence", "semantic", "markdown"]
SIZES = {"fixed_char": (220, 40), "word": (30, 6), "token": (45, 8), "recursive": (45, 8), "sentence": (45, 1), "semantic": (45, 0), "markdown": (45, 8)}


class WordTokens:
    """Deterministic stand-in for tiktoken: one token per whitespace-separated word, plus one per punctuation mark."""

    name = "test_words"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch):
    get_tokenizer.cache_clear()
    monkeypatch.setattr(tokenizer_module, "_load_tokenizer", lambda: WordTokens())
    yield
    get_tokenizer.cache_clear()


def _doc(text: str = PROSE, doc_id: str = "doc_a", title: str = "Store policy") -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=title, text=text)


def _make(name: str, **overrides: Any) -> BaseChunker:
    size, overlap = SIZES[name]
    config: dict[str, Any] = {"type": name, "chunk_size": size, "chunk_overlap": overlap, **overrides}
    return create_chunker(config, embedder=HashingEmbeddingModel() if name == "semantic" else None)


def _tokens(text: str) -> int:
    return len(WordTokens().spans(text))


# --- invariants every chunker must satisfy -------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("text", [PROSE, MARKDOWN, "One short line.", "word " * 300, "  \n\n  ", ""], ids=["prose", "markdown", "short", "repetitive", "blank", "empty"])
def test_chunks_are_never_empty_and_their_spans_reproduce_their_text(name, text):
    document = _doc(text)

    chunks = _make(name).chunk([document])

    assert bool(chunks) == bool(text.strip())
    for chunk in chunks:
        start, end = chunk.metadata["start_char"], chunk.metadata["end_char"]
        assert chunk.text and chunk.text == document.text[start:end].strip(), (name, chunk.text[:40])
        assert chunk.doc_id == "doc_a" and chunk.metadata["chunker"] == name and chunk.metadata["source_path"] == "doc_a.md"


@pytest.mark.parametrize("name", ALL)
def test_chunk_indexes_are_consecutive_ids_are_unique_and_output_is_deterministic(name):
    documents = [_doc(PROSE, "doc_a"), _doc(MARKDOWN, "doc_b")]

    first, second = _make(name).chunk(documents), _make(name).chunk(documents)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second] and [c.text for c in first] == [c.text for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    for doc_id in ("doc_a", "doc_b"):
        assert [c.metadata["chunk_index"] for c in first if c.doc_id == doc_id] == list(range(len([c for c in first if c.doc_id == doc_id])))


@pytest.mark.parametrize("name", ALL)
def test_every_non_whitespace_character_of_the_document_is_in_some_chunk(name):
    for text in (PROSE, MARKDOWN):
        document = _doc(text)
        covered = np.zeros(len(text), dtype=bool)
        for chunk in _make(name).chunk([document]):
            covered[chunk.metadata["start_char"] : chunk.metadata["end_char"]] = True

        missing = [i for i, ch in enumerate(text) if not covered[i] and not ch.isspace()]
        assert missing == [], f"{name}: characters never indexed: {text[missing[0] - 20 : missing[0] + 20]!r}"


@pytest.mark.parametrize("name", ["word", "token", "recursive", "sentence", "semantic", "markdown"])
def test_no_chunk_exceeds_the_size_limit_in_its_unit(name):
    size, _ = SIZES[name]
    unit = (lambda t: len(t.split())) if name == "word" else _tokens

    for chunk in _make(name).chunk([_doc(PROSE), _doc(MARKDOWN, "doc_b")]):
        assert unit(chunk.text) <= size, (name, unit(chunk.text), chunk.text[:60])


@pytest.mark.parametrize("name", ["fixed_char", "word", "token", "recursive", "sentence", "semantic", "markdown"])
def test_zero_overlap_means_chunks_do_not_overlap_and_overlap_shows_up_when_asked_for(name):
    document = _doc(PROSE * 3)
    flat = _make(name, chunk_overlap=0).chunk([document])
    assert len(flat) > 2
    for before, after in zip(flat, flat[1:], strict=False):
        assert after.metadata["start_char"] >= before.metadata["end_char"], name

    if name in ("semantic", "markdown"):
        return  # structure decides the boundaries; their overlap applies inside long sections only (tested below)
    overlapping = _make(name, chunk_overlap=SIZES[name][1] or 1).chunk([document])
    assert any(after.metadata["start_char"] < before.metadata["end_char"] for before, after in zip(overlapping, overlapping[1:], strict=False)), name


def test_overlap_must_stay_smaller_than_the_size_for_every_chunker():
    for name in ("fixed_char", "word", "token", "recursive"):
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            ChunkerConfig(type=name, chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            _make(name, chunk_size=10, chunk_overlap=10)


def test_prefix_title_prepends_the_title_to_every_chunk_but_keeps_spans_honest():
    document = _doc(PROSE, title="Store policy")

    for name in ALL:
        for chunk in _make(name, prefix_title=True).chunk([document]):
            assert chunk.text.startswith("Store policy\n\n"), name
            body = chunk.text.removeprefix("Store policy\n\n")
            assert body == document.text[chunk.metadata["start_char"] : chunk.metadata["end_char"]].strip()


# --- fixed_char / word / token -------------------------------------------------------------------


def test_word_chunker_is_the_old_whitespace_word_behaviour():
    doc = Document(doc_id="doc_test", path="doc_test.md", title="Test", text=" ".join(f"word{i}" for i in range(12)))
    chunks = WordChunker(chunk_size=5, chunk_overlap=2).chunk([doc])

    assert len(chunks) == 4
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert all(len(c.text.split()) <= 5 for c in chunks)


def test_word_and_token_are_different_chunkers_and_the_old_alias_points_at_word():
    assert CHUNKERS.get("word") is WordChunker and CHUNKERS.get("token") is TokenChunker and CHUNKERS.get("tokenish") is WordChunker
    assert CHUNKERS.get("fixed") is FixedCharacterChunker and CHUNKERS.get("md") is MarkdownAwareChunker


def test_token_chunker_counts_punctuation_as_tokens_where_the_word_chunker_does_not():
    text = "a, b, c, d, e, f, g, h"  # 8 words, 15 tokens under the fake tokenizer

    assert len(WordChunker(chunk_size=8, chunk_overlap=0).chunk([_doc(text)])) == 1
    token_chunks = TokenChunker(chunk_size=8, chunk_overlap=0).chunk([_doc(text)])
    assert len(token_chunks) == 2 and all(_tokens(c.text) <= 8 for c in token_chunks)


def test_token_chunker_honours_the_size_in_real_tiktoken_tokens(monkeypatch):
    pytest.importorskip("tiktoken")
    real = tokenizer_module.load_tokenizer()
    if isinstance(real, ApproxTokenizer):
        pytest.skip("the o200k_base vocabulary is downloaded on first use and is unavailable offline")
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    monkeypatch.setattr(tokenizer_module, "_load_tokenizer", lambda: real)
    get_tokenizer.cache_clear()
    text = "Naïve café prices — 你好世界! " * 120 + PROSE

    chunks = TokenChunker(chunk_size=64, chunk_overlap=10).chunk([_doc(text)])

    assert get_tokenizer().name == "o200k_base" and len(chunks) > 5
    assert all(len(encoding.encode(c.text)) <= 66 for c in chunks)  # +2: a window edge can split a multi-byte character
    assert max(len(encoding.encode(c.text)) for c in chunks) >= 60


def test_the_tokenizer_falls_back_to_a_deterministic_approximation_when_the_vocabulary_is_unavailable(monkeypatch, caplog):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(tokenizer_module, "_load_tokenizer", lambda: tokenizer_module.load_tokenizer(loader=offline))
    get_tokenizer.cache_clear()
    with caplog.at_level("WARNING"):
        tokenizer = get_tokenizer()
        again = get_tokenizer()

    assert isinstance(tokenizer, ApproxTokenizer) and tokenizer is again and tokenizer.name == "approx"
    assert sum("approximate" in r.message for r in caplog.records) == 1
    assert count_tokens("Refunds are accepted.") == len(tokenizer.spans("Refunds are accepted."))
    assert tokenizer.spans("你好") == [(0, 1), (1, 2)]  # CJK: one token per character, like real BPE


# --- recursive ------------------------------------------------------------------------------------


def test_recursive_prefers_paragraph_then_line_then_sentence_then_word_boundaries():
    paragraphs = _make("recursive", chunk_size=60, chunk_overlap=0).chunk([_doc(PROSE)])
    assert len(paragraphs) >= 3 and all(c.text.endswith((".", "!", "?")) for c in paragraphs[:-1]), "split inside a sentence although a boundary fit"

    # No paragraph or sentence boundaries at all: it must fall through to word boundaries, then a hard token split.
    blob = "supercalifragilistic " * 40
    words = _make("recursive", chunk_size=10, chunk_overlap=0).chunk([_doc(blob)])
    assert all(_tokens(c.text) <= 10 for c in words) and all(c.text.endswith("supercalifragilistic") for c in words)
    assert all(_tokens(c.text) <= 10 for c in _make("recursive", chunk_size=10, chunk_overlap=0).chunk([_doc("x" * 500 + " tail")]))


def test_recursive_merge_small_chunks():
    plain = _make("recursive", chunk_size=40, chunk_overlap=0).chunk([_doc(PROSE)])
    merged = _make("recursive", chunk_size=40, chunk_overlap=0, min_chunk_size=25).chunk([_doc(PROSE)])

    assert len(merged) <= len(plain) and all(_tokens(c.text) <= 40 for c in merged)
    assert sum(_tokens(c.text) < 25 for c in merged) <= sum(_tokens(c.text) < 25 for c in plain)


# --- sentence -------------------------------------------------------------------------------------


def test_sentence_chunks_start_and_end_on_sentence_boundaries_and_overlap_counts_sentences():
    chunks = _make("sentence", chunk_size=30, chunk_overlap=0).chunk([_doc(PROSE)])
    assert all(c.text[-1] in ".!?" for c in chunks) and all(c.text[0].isupper() for c in chunks)

    overlapping = _make("sentence", chunk_size=30, chunk_overlap=1).chunk([_doc(PROSE)])
    for before, after in zip(overlapping, overlapping[1:], strict=False):
        last_sentence = re.split(r"(?<=[.!?])\s+", before.text)[-1]
        assert after.text.startswith(last_sentence), "the last sentence of a chunk opens the next one"


def test_a_sentence_longer_than_the_limit_is_split_rather_than_dropped():
    long_sentence = "word " * 100 + "end."
    chunks = _make("sentence", chunk_size=20, chunk_overlap=0).chunk([_doc(long_sentence)])

    assert len(chunks) > 4 and all(_tokens(c.text) <= 20 for c in chunks)


def test_sentence_splitting_handles_cjk_and_quotes():
    chunks = _make("sentence", chunk_size=6, chunk_overlap=0).chunk([_doc("今天天气很好。我们去公园玩！你想一起来吗？好的。")])

    assert len(chunks) >= 2 and all(c.text.endswith(("。", "！", "？")) for c in chunks)


# --- semantic -------------------------------------------------------------------------------------


class ScriptedEmbedder(EmbeddingModel):
    """Embeds a sentence as topic one-hot: sentences mentioning 'refund' vs 'shipping' vs 'cards' are orthogonal."""

    model_name = "scripted"
    TOPICS = ("refund", "shipping", "card", "warranty")

    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        from ragbench.models.cost import CostBreakdown

        self.calls += 1
        vectors = np.zeros((len(texts), len(self.TOPICS) + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            hits = [i for i, topic in enumerate(self.TOPICS) if topic in lowered]
            vectors[row, hits[0] if hits else len(self.TOPICS)] = 1.0
        return EmbeddingResult(vectors=vectors, model=self.model_name, input_tokens=len(texts) * 5, cost=CostBreakdown(embedding_input_tokens=len(texts) * 5, embedding_cost=0.01 * len(texts)))


def test_semantic_chunker_breaks_where_the_topic_changes_and_charges_its_embedding_cost():
    text = (
        "Refund rules are simple. A refund needs a receipt. Every refund takes five days. "
        "Shipping is fast. Shipping takes two days. Shipping is tracked online. "
        "Gift card balances never expire. Each card can be reloaded. A card works online. "
    )
    embedder = ScriptedEmbedder()
    chunker = create_chunker({"type": "semantic", "chunk_size": 100, "chunk_overlap": 0, "breakpoint_percentile": 75}, embedder=embedder)

    chunks = chunker.chunk([_doc(text)])

    assert [("refund" in c.text.lower(), "shipping" in c.text.lower(), "card" in c.text.lower()) for c in chunks] == [(True, False, False), (False, True, False), (False, False, True)]
    assert chunker.last_cost.embedding_cost == pytest.approx(0.09) and chunker.last_cost.embedding_input_tokens == 45
    assert chunker.embedder is embedder


def test_semantic_chunker_is_deterministic_under_the_mock_hashing_embedder_and_needs_an_embedder():
    def run() -> list[str]:
        return [c.text for c in create_chunker({"type": "semantic", "chunk_size": 60}, embedder=HashingEmbeddingModel()).chunk([_doc(PROSE)])]

    assert run() == run() and len(run()) >= 2
    with pytest.raises(ValueError, match="embedding"):
        create_chunker({"type": "semantic"})


def test_semantic_with_too_few_sentences_is_one_chunk():
    assert len(_make("semantic").chunk([_doc("Only one sentence here. And a second one.")])) == 1


# --- markdown -------------------------------------------------------------------------------------


def _by_path(chunks) -> dict[tuple[str, ...], str]:
    return {tuple(c.metadata["heading_path"]): c.text for c in chunks}


def test_markdown_heading_paths_follow_the_nesting_and_ignore_headings_inside_code_fences():
    chunks = create_chunker({"type": "markdown", "chunk_size": 200, "chunk_overlap": 0, "min_chunk_size": 1}).chunk([_doc(MARKDOWN, title="Doc")])
    paths = [tuple(c.metadata["heading_path"]) for c in chunks]

    assert paths == [(), ("Guide",), ("Guide", "Returns"), ("Guide", "Returns", "Damaged goods"), ("Guide", "Shipping"), ("Guide", "Shipping", "Tracking"), ("Appendix",)]
    by_path = _by_path(chunks)
    assert "Not a heading" in by_path[("Guide", "Shipping")] and "Photograph the packaging" in by_path[("Guide", "Returns", "Damaged goods")]
    assert by_path[()].startswith("Intro text") and all(c.metadata["chunker"] == "markdown" for c in chunks)


def test_markdown_prefix_heading_prepends_the_breadcrumb_and_small_sections_are_merged():
    kwargs = {"type": "markdown", "chunk_size": 200, "chunk_overlap": 0}
    plain = create_chunker({**kwargs, "min_chunk_size": 1}).chunk([_doc(MARKDOWN)])
    prefixed = create_chunker({**kwargs, "min_chunk_size": 1, "prefix_heading": True}).chunk([_doc(MARKDOWN)])

    damaged = next(c for c in prefixed if c.metadata["heading_path"] == ["Guide", "Returns", "Damaged goods"])
    assert damaged.text.startswith("Guide > Returns > Damaged goods\n\n### Damaged goods")
    assert prefixed[0].text == plain[0].text, "no breadcrumb for text that sits above the first heading"

    merged = create_chunker({**kwargs, "min_chunk_size": 25}).chunk([_doc(MARKDOWN)])
    assert len(merged) < len(plain) and all(_tokens(c.text) <= 200 for c in merged)
    assert sum(_tokens(c.text) < 25 for c in merged) < sum(_tokens(c.text) < 25 for c in plain)
    assert any(len(c.metadata.get("merged_headings", [])) > 1 for c in merged)


def test_a_long_markdown_section_is_split_inside_the_section_and_keeps_its_heading_path():
    body = " ".join(f"Sentence number {i} talks about refunds." for i in range(40))
    chunks = create_chunker({"type": "markdown", "chunk_size": 50, "chunk_overlap": 10, "min_chunk_size": 1}).chunk([_doc(f"# Policy\n\n## Refunds\n\n{body}\n\n## Other\n\nShort.")])

    refund_chunks = [c for c in chunks if c.metadata["heading_path"] == ["Policy", "Refunds"]]
    assert len(refund_chunks) > 3 and all(_tokens(c.text) <= 50 for c in refund_chunks)
    assert any(b.metadata["start_char"] < a.metadata["end_char"] for a, b in zip(refund_chunks, refund_chunks[1:], strict=False)), "overlap applies inside a section"
    assert chunks[-1].metadata["heading_path"] == ["Policy", "Other"]


def test_markdown_without_headings_still_chunks_the_text():
    chunks = create_chunker({"type": "markdown", "chunk_size": 40, "chunk_overlap": 0}).chunk([_doc(PROSE)])

    assert len(chunks) >= 3 and all(c.metadata["heading_path"] == [] for c in chunks)


# --- config ---------------------------------------------------------------------------------------


def test_chunker_defaults_come_from_the_chunker_class():
    assert isinstance(create_chunker(None), TokenChunker)
    token = create_chunker({"type": "token"})
    assert (token.chunk_size, token.chunk_overlap) == (500, 80)
    word = create_chunker({"type": "word"})
    assert isinstance(word, WordChunker) and (word.chunk_size, word.chunk_overlap) == (500, 80)
    fixed = create_chunker(ChunkerConfig(type="fixed"))
    assert isinstance(fixed, FixedCharacterChunker) and (fixed.chunk_size, fixed.chunk_overlap) == (1200, 150)
    markdown = create_chunker({"type": "md", "chunk_size": 100, "chunk_overlap": 0})
    assert isinstance(markdown, MarkdownAwareChunker) and (markdown.chunk_size, markdown.chunk_overlap, markdown.min_chunk_size) == (100, 0, 50)  # 0 overlap is honored
    assert isinstance(create_chunker({"type": "recursive"}), RecursiveChunker) and create_chunker({"type": "sentence"}).chunk_overlap == 1
    assert isinstance(create_chunker({"type": "semantic"}, embedder=HashingEmbeddingModel()), SemanticChunker)
    assert isinstance(create_chunker({"type": "sentence"}), SentenceChunker)


def test_options_are_validated_per_chunker():
    ChunkerConfig(type="markdown", min_chunk_size=30, prefix_heading=True, prefix_title=True)
    ChunkerConfig(type="semantic", breakpoint_percentile=95)
    ChunkerConfig(type="token", prefix_title=True)
    for bad in (
        {"type": "token", "min_chunk_size": 10},
        {"type": "word", "prefix_heading": True},
        {"type": "recursive", "breakpoint_percentile": 90},
        {"type": "markdown", "breakpoint_percentile": 90},
    ):
        with pytest.raises(ValidationError, match="does not use"):
            ChunkerConfig(**bad)
    with pytest.raises(ValidationError, match="breakpoint_percentile"):
        ChunkerConfig(type="semantic", breakpoint_percentile=120)
    with pytest.raises(ValidationError, match="Did you mean 'recursive'"):
        ChunkerConfig(type="recursiv")
    with pytest.raises(ValidationError, match=r"chunker\.min_chunk_siz"):
        SystemConfig(type="vector", chunker={"type": "markdown", "min_chunk_siz": 3})


def test_the_chunker_section_is_documented_for_every_chunker():
    docs = (__import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "configuration.md").read_text(encoding="utf-8")

    for name in CHUNKERS.names():
        assert f"`{name}`" in docs, name


def test_shipped_configs_keep_their_word_based_chunking():
    """`token` changed meaning (real tokens); the shipped configs opt into `word` so results stay comparable with 0.2.0."""
    from pathlib import Path

    for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        assert not re.search(r"type:\s*token\b", path.read_text(encoding="utf-8")), path.name


# --- cost plumbing --------------------------------------------------------------------------------


def test_systems_charge_the_chunkers_embedding_cost_to_ingestion_and_count_its_model(tmp_path):
    from ragbench.documents.loaders import load_documents
    from ragbench.rag_systems import create_rag_system

    documents = load_documents(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "demo" / "docs")
    plain = create_rag_system(SystemConfig(type="bm25", chunker={"type": "sentence", "chunk_size": 60}), force_mock=True)
    semantic = create_rag_system(SystemConfig(type="bm25", chunker={"type": "semantic", "chunk_size": 60}), force_mock=True)

    plain_result = plain.ingest(documents)
    semantic_result = semantic.ingest(documents)

    assert semantic_result.num_chunks > 0 and semantic.chunker.embedder.model_name == "hashing-embedding"
    assert semantic_result.cost.embedding_input_tokens > plain_result.cost.embedding_input_tokens == 0, "hashing embeddings are free but their tokens are counted"


def test_json_is_accepted_wherever_yaml_is_for_the_chunker_string():
    from ragbench.cli import parse_chunker_spec

    assert parse_chunker_spec('{"type": "markdown", "chunk_size": 100}') == {"type": "markdown", "chunk_size": 100}
    assert parse_chunker_spec("{type: recursive, chunk_size: 100}") == {"type": "recursive", "chunk_size": 100}
    assert parse_chunker_spec("") == {} and json.dumps(parse_chunker_spec("type: sentence")) == '{"type": "sentence"}'
    with pytest.raises(ValueError, match="mapping"):
        parse_chunker_spec("[1, 2]")
