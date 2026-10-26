"""The built-in hub entries are complete and pinned, their Parquet readers work, and each ships a config."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ragbench.config.loader import load_config
from ragbench.datasets.hub.config import render_config
from ragbench.datasets.hub.readers import READERS
from ragbench.datasets.hub.registry import HUB, HubDataset, HubFile, get_dataset
from ragbench.models.errors import MissingExtraError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "hub"


def _entries() -> list[HubDataset]:
    get_dataset("multihop-rag")  # registers the built-ins
    return sorted(HUB.values(), key=lambda entry: entry.name)


def test_the_hub_offers_multihop_rag_hotpotqa_and_every_galileo_subset():
    names = {entry.name for entry in _entries()}

    assert {"multihop-rag", "hotpotqa"} <= names
    assert {name for name in names if name.startswith("galileo-")} == {
        f"galileo-{subset}" for subset in "covidqa cuad delucionqa emanual expertqa finqa hagrid hotpotqa msmarco pubmedqa tatqa techqa".split()
    }


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: entry.name)
def test_every_entry_is_pinned_to_a_commit_and_a_checksum_and_names_its_terms(entry):
    assert re.fullmatch(r"[0-9a-f]{40}", entry.revision)
    for hub_file in entry.files:
        assert hub_file.url.startswith("https://huggingface.co/datasets/") and f"/resolve/{entry.revision}/" in hub_file.url
        assert hub_file.size > 0
        assert (hub_file.sha256 and re.fullmatch(r"[0-9a-f]{64}", hub_file.sha256)) or (hub_file.git_sha1 and re.fullmatch(r"[0-9a-f]{40}", hub_file.git_sha1))
    assert entry.license and entry.license_url.startswith("https://") and len(entry.citation) > 40 and entry.source_url.startswith("https://")
    assert entry.reader in READERS


def test_a_hub_file_needs_exactly_one_checksum():
    with pytest.raises(ValueError, match="exactly one"):
        HubFile("x", "https://x", 1)
    with pytest.raises(ValueError, match="exactly one"):
        HubFile("x", "https://x", 1, sha256="a" * 64, git_sha1="b" * 40)


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: entry.name)
def test_every_entry_ships_a_config_that_loads_and_is_what_the_generator_writes(entry):
    path = ROOT / "configs" / f"{entry.config_name}.yaml"

    assert path.exists(), f"run `python scripts/generate_docs.py` to write {path.name}"
    assert path.read_text(encoding="utf-8") == render_config(entry)
    config = load_config(path)
    assert config.dataset.documents_path == Path("data") / entry.name / "docs"
    assert config.dataset.questions_path == Path("data") / entry.name / "questions.jsonl"


# -- Parquet readers ------------------------------------------------------------------------------


def _write_parquet(path: Path, rows: list[dict]) -> Path:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as parquet

    parquet.write_table(pa.Table.from_pylist(rows), path)
    return path


def _entry(reader: str, **options: str) -> HubDataset:
    return HubDataset("x", "X", "", "MIT", "https://x", "https://x", "c", "r", (), reader, False, options)


def test_the_hotpotqa_reader_converts_a_parquet_file(tmp_path):
    rows = json.loads((FIXTURES / "hotpotqa_rows.json").read_text(encoding="utf-8"))
    path = _write_parquet(tmp_path / "validation.parquet", rows)

    converted = READERS["hotpotqa"](_entry("hotpotqa"), {"validation.parquet": path})

    assert [q["id"] for q in converted.questions] == ["hp001", "hp002", "hp003"] and len(converted.documents) == 5
    assert converted.questions[0]["relevant_doc_ids"] and converted.questions[0]["evidence"]


def test_the_galileo_reader_converts_a_parquet_file(tmp_path):
    rows = json.loads((FIXTURES / "galileo_rows.json").read_text(encoding="utf-8"))
    path = _write_parquet(tmp_path / "test.parquet", rows)

    converted = READERS["galileo"](_entry("galileo", subset="covidqa"), {"test.parquet": path})

    assert [q["answerable"] for q in converted.questions] == [True, False, True] and len(converted.documents) == 3


def test_a_missing_pyarrow_names_the_extra_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr("ragbench.utils.extras.extra_installed", lambda module: False)

    with pytest.raises(MissingExtraError, match=r"ragbench\[hub\]"):
        READERS["hotpotqa"](_entry("hotpotqa"), {"validation.parquet": tmp_path / "missing.parquet"})


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: entry.name)
def test_each_shipped_config_throttles_openai_below_the_lowest_tier_embedding_limit(entry):
    """MultiHop-RAG's corpus is ~1.4M tokens: indexing it at full concurrency hit OpenAI's 1M tokens-per-minute embedding limit (429) on
    2026-10-26 and outlasted the retries. The shipped configs throttle so a first live run gets through."""
    config = load_config(ROOT / "configs" / f"{entry.config_name}.yaml")

    assert config.limits.tokens_per_minute is not None and config.limits.tokens_per_minute <= 1_000_000
