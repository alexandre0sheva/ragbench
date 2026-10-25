"""Pulling a hub dataset: consent, download, checksum, cache, conversion and what is written. Served from a local file server; nothing leaves the machine."""

from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from ragbench.datasets.hub.download import ChecksumError, DownloadError, cached_path, fetch_files, git_blob_sha1
from ragbench.datasets.hub.pull import Declined, PullError, pull
from ragbench.datasets.hub.registry import HUB, HubDataset, HubFile, get_dataset
from ragbench.datasets.loader import load_dataset
from ragbench.datasets.validation import validate_dataset
from ragbench.documents.loaders import load_documents

FIXTURES = Path(__file__).parent / "fixtures" / "hub"


class _Handler(http.server.SimpleHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
        type(self).hits.append(self.path)
        super().do_GET()

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path, list[str]]]:
    root = tmp_path / "www"
    root.mkdir()
    _Handler.hits = []
    handler = lambda *a, **k: _Handler(*a, directory=str(root), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", root, _Handler.hits
    httpd.shutdown()
    httpd.server_close()


def _entry(base: str, root: Path, *, name: str = "toy-news", sha256: str | None = None, git_sha1: str | None = None, bad: bool = False) -> HubDataset:
    files = []
    for source, served in (("multihop_corpus.json", "corpus.json"), ("multihop_queries.json", "MultiHopRAG.json")):
        shutil.copy(FIXTURES / source, root / served)
        data = (root / served).read_bytes()
        files.append(
            HubFile(
                served,
                f"{base}/{served}",
                len(data),
                sha256=("0" * 64 if bad else sha256 or hashlib.sha256(data).hexdigest()) if git_sha1 is None else None,
                git_sha1=git_sha1 and git_blob_sha1(data),
            )
        )
    return HubDataset(
        name=name,
        title="Toy news",
        description="Five articles.",
        license="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        source_url="https://example.invalid/toy",
        citation="Nobody (2026). Toy news.",
        revision="test",
        files=tuple(files),
        reader="multihop_rag",
        shared_corpus=True,
    )


def _pull(entry: HubDataset, tmp_path: Path, **kwargs):
    kwargs.setdefault("confirm", lambda message: True)
    return pull(entry, tmp_path / "out", cache_dir=tmp_path / "cache", **kwargs)


def test_pull_writes_a_dataset_ragbench_can_load(server, tmp_path):
    base, root, _ = server
    result = _pull(_entry(base, root), tmp_path)

    out = tmp_path / "out"
    documents = load_documents(out / "docs")
    dataset = load_dataset(out / "questions.jsonl", out / "qrels.jsonl")
    assert (result.n_documents, result.n_questions) == (5, 6) and len(documents) == 5
    assert {d.doc_id for d in documents} == {f"doc_news_{n:04d}" for n in range(1, 6)}
    problems = validate_dataset(documents, dataset)
    assert len(problems) == 1 and "mh_0006" in problems[0], "only the question whose evidence is missing from the fixture corpus is unlabeled"
    assert result.warnings, "the converter's own warnings (an evidence article missing from the corpus) are passed on"


def test_pull_records_where_the_data_came_from(server, tmp_path):
    base, root, _ = server
    _pull(_entry(base, root), tmp_path)

    source = (tmp_path / "out" / "SOURCE.md").read_text(encoding="utf-8")
    assert "Toy news" in source and "CC0-1.0" in source and "Nobody (2026)" in source and "example.invalid/toy" in source
    meta = json.loads((tmp_path / "out" / "hub.json").read_text(encoding="utf-8"))
    assert meta["dataset"] == "toy-news" and meta["revision"] == "test" and meta["subset"] is None
    assert {f["name"] for f in meta["files"]} == {"corpus.json", "MultiHopRAG.json"}


def test_a_subset_keeps_every_category_and_the_unanswerable_share(server, tmp_path):
    base, root, _ = server
    result = _pull(_entry(base, root), tmp_path, subset=4, seed=1)

    questions = [json.loads(line) for line in (tmp_path / "out" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result.n_questions == len(questions) == 4
    assert {q["category"] for q in questions} == {"inference_query", "comparison_query", "temporal_query", "null_query"}
    assert any(q["answerable"] is False for q in questions)
    assert result.n_documents == 5, "a shared corpus keeps all its documents: the distractors are part of the test"
    qrels = {json.loads(line)["query_id"] for line in (tmp_path / "out" / "qrels.jsonl").read_text(encoding="utf-8").splitlines()}
    assert qrels <= {q["id"] for q in questions}
    assert json.loads((tmp_path / "out" / "hub.json").read_text(encoding="utf-8"))["subset"] == {"questions": 4, "seed": 1}


def test_files_are_downloaded_once_and_then_read_from_the_cache(server, tmp_path):
    base, root, hits = server
    entry = _entry(base, root)

    _pull(entry, tmp_path)
    assert sorted(hits) == ["/MultiHopRAG.json", "/corpus.json"]
    asked: list[str] = []
    pull(entry, tmp_path / "again", cache_dir=tmp_path / "cache", confirm=lambda message: asked.append(message) or True)

    assert sorted(hits) == ["/MultiHopRAG.json", "/corpus.json"], "the second pull makes no request"
    assert asked == [], "and asks for no consent: nothing is downloaded"


def test_consent_names_the_files_the_size_and_the_host_and_a_no_downloads_nothing(server, tmp_path):
    base, root, hits = server
    asked: list[str] = []

    with pytest.raises(Declined):
        _pull(_entry(base, root), tmp_path, confirm=lambda message: asked.append(message) or False)

    assert hits == [] and not (tmp_path / "out").exists()
    assert len(asked) == 1 and "corpus.json" in asked[0] and "127.0.0.1" in asked[0] and "KB" in asked[0]


def test_a_wrong_checksum_is_refused_and_nothing_is_kept(server, tmp_path):
    base, root, _ = server
    entry = _entry(base, root, bad=True)

    with pytest.raises(ChecksumError, match="corpus.json"):
        _pull(entry, tmp_path)

    assert not list((tmp_path / "cache").rglob("corpus.json*")) and not (tmp_path / "out").exists()


def test_git_blob_checksums_are_verified_too(server, tmp_path):
    base, root, _ = server

    assert _pull(_entry(base, root, git_sha1=True), tmp_path).n_questions == 6


def test_a_missing_file_is_a_download_error_naming_the_url(server, tmp_path):
    base, root, _ = server
    entry = _entry(base, root)
    (root / "MultiHopRAG.json").unlink()

    with pytest.raises(DownloadError, match="MultiHopRAG.json"):
        _pull(entry, tmp_path)


def test_pull_refuses_to_replace_an_earlier_pull_without_force(server, tmp_path):
    base, root, _ = server
    entry = _entry(base, root)
    _pull(entry, tmp_path)
    stale = tmp_path / "out" / "docs" / "doc_stale.md"
    stale.write_text("old", encoding="utf-8")

    with pytest.raises(PullError, match="--force"):
        _pull(entry, tmp_path)
    assert stale.exists()

    _pull(entry, tmp_path, force=True)
    assert not stale.exists(), "a forced pull removes the earlier documents, so a smaller subset cannot keep stale ones"


def test_unknown_names_get_a_did_you_mean():
    with pytest.raises(PullError, match="multihop-rag"):
        get_dataset("multihop_rag_")


def test_the_cache_path_names_dataset_revision_and_file(tmp_path):
    entry = _entry("http://x", tmp_path)

    assert cached_path(entry, entry.files[0], tmp_path) == tmp_path / "toy-news" / "test" / "corpus.json"


def test_fetch_files_reports_progress_per_file(server, tmp_path):
    base, root, _ = server
    entry = _entry(base, root)
    seen: list[str] = []

    paths = fetch_files(entry, tmp_path / "cache", on_file=lambda hub_file: seen.append(hub_file.name))

    assert seen == ["corpus.json", "MultiHopRAG.json"] and set(paths) == {"corpus.json", "MultiHopRAG.json"} and all(p.exists() for p in paths.values())


def test_registered_entries_are_listed_by_name():
    assert "toy-news" not in HUB
