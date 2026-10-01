"""`ragbench chunk-preview`: parsing, statistics, output shapes and errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.documents import tokenizer as tokenizer_module
from ragbench.documents.preview import chunk_stats
from ragbench.documents.schema import TextChunk
from ragbench.documents.tokenizer import get_tokenizer

DEMO_DOCS = Path(__file__).resolve().parents[1] / "data" / "demo" / "docs"
runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline_tokenizer(monkeypatch):
    class Words:
        name = "test_words"

        def spans(self, text: str) -> list[tuple[int, int]]:
            import re

            return [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]

    get_tokenizer.cache_clear()
    monkeypatch.setattr(tokenizer_module, "_load_tokenizer", lambda: Words())
    yield
    get_tokenizer.cache_clear()


def _preview(*args: str):
    return runner.invoke(app, ["chunk-preview", "--docs", str(DEMO_DOCS), *args])


def test_the_acceptance_command_works_and_shows_statistics_and_chunks():
    result = _preview("--chunker", "{type: markdown}")

    assert result.exit_code == 0, result.output
    assert "Chunking: markdown" in result.output and "Chunks" in result.output and "p95" in result.output and "First 5 chunk(s)" in result.output
    assert "under 50 tokens" in result.output


def test_json_output_has_stats_config_and_the_first_chunks():
    result = _preview("--chunker", '{"type": "recursive", "chunk_size": 40, "chunk_overlap": 5}', "--json", "-n", "2")

    payload = json.loads(result.output)
    assert result.exit_code == 0 and payload["chunker"] == {"type": "recursive", "chunk_size": 40, "chunk_overlap": 5, "prefix_title": False}
    assert len(payload["chunks"]) == 2 and payload["chunks"][0]["tokens"] <= 40 and payload["chunks"][0]["metadata"]["chunker"] == "recursive"
    stats = payload["stats"]
    n_docs = len(list(DEMO_DOCS.glob("*.md")))
    assert stats["documents"] == n_docs and stats["chunks"] >= n_docs and stats["max_tokens"] <= 40
    assert stats["tokenizer"] == "test_words" and 0 <= stats["percent_below_threshold"] <= 100 and stats["p95_tokens"] >= stats["median_tokens"]


def test_doc_filter_limits_the_preview_to_one_document_and_unknown_ids_are_reported():
    result = _preview("--chunker", "{type: sentence, chunk_size: 30}", "--doc", "doc_002", "--json", "-n", "50")

    chunks = json.loads(result.output)["chunks"]
    assert result.exit_code == 0 and chunks and {c["doc_id"] for c in chunks} == {"doc_002"}
    missing = _preview("--doc", "nope")
    assert missing.exit_code == 2 and "No document with id 'nope'" in missing.output and "doc_001" in missing.output


def test_defaults_apply_when_no_chunker_is_given_and_zero_chunks_shown_is_allowed():
    result = _preview("-n", "0")

    assert result.exit_code == 0 and "Chunking: token" in result.output and "First" not in result.output


def test_a_semantic_preview_runs_offline_with_mock_embeddings_and_reports_the_tokens_it_embedded():
    result = _preview("--chunker", "{type: semantic, chunk_size: 60}", "--mock", "--json", "-n", "1")

    assert result.exit_code == 0 and json.loads(result.output)["chunks"]


def test_bad_chunker_specs_are_reported_without_a_traceback():
    for spec, needle in [
        ("{type: markdwn}", "Did you mean 'markdown'"),
        ("{type: token, min_chunk_size: 10}", "does not use min_chunk_size"),
        ("{type: token, chunk_siz: 10}", "chunk_siz"),
        ("[1, 2]", "mapping"),
        ("{type: : :", "expected"),
    ]:
        result = _preview("--chunker", spec)
        assert result.exit_code == 2 and needle in result.output.replace("\n", " "), (spec, result.output)
        assert "Traceback" not in result.output


def test_missing_documents_path_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["chunk-preview", "--docs", str(tmp_path / "nope")])

    assert result.exit_code == 2 and "not found" in result.output


def test_chunk_stats_percentiles_and_threshold():
    chunks = [TextChunk(chunk_id=f"c{i}", doc_id="d", text=" ".join(["w"] * n), metadata={}) for i, n in enumerate([10, 20, 30, 40, 100])]

    stats = chunk_stats(chunks, documents=2, threshold_tokens=25)

    assert (stats.chunks, stats.documents, stats.min_tokens, stats.max_tokens, stats.median_tokens) == (5, 2, 10, 100, 30)
    assert stats.mean_tokens == pytest.approx(40) and stats.percent_below_threshold == pytest.approx(40) and stats.p95_tokens == pytest.approx(88)
    empty = chunk_stats([], documents=0)
    assert (empty.chunks, empty.max_tokens, empty.percent_below_threshold) == (0, 0, 0.0)
