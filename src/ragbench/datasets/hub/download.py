"""Downloading a hub dataset's files into the cache, checked against their pinned checksum."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ragbench.datasets.hub.registry import HubDataset, HubFile
from ragbench.errors import UsageError

CACHE_ENV = "RAGBENCH_DATASETS_DIR"
CHUNK = 1 << 20


class DownloadError(UsageError):
    """A file could not be fetched."""


class ChecksumError(DownloadError):
    """A file is not the one the entry pins."""


def cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "ragbench" / "datasets")


def cached_path(entry: HubDataset, hub_file: HubFile, cache_dir: Path) -> Path:
    return cache_dir / entry.name / entry.revision / hub_file.name


def git_blob_sha1(data: bytes) -> str:
    """The id git gives a file with this content (what Hugging Face lists for files that are not stored in LFS)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _digest_of(path: Path, hub_file: HubFile) -> str:
    if hub_file.sha256 is not None:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(b"blob %d\0" % hub_file.size, usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, hub_file: HubFile) -> bool:
    """Whether the file at `path` is exactly the pinned one (size, then checksum)."""
    if not path.is_file() or path.stat().st_size != hub_file.size:
        return False
    return _digest_of(path, hub_file) == (hub_file.sha256 or hub_file.git_sha1)


def missing_files(entry: HubDataset, cache_dir: Path) -> list[HubFile]:
    """The files that still have to be downloaded: not cached, or cached but not what the entry pins."""
    return [item for item in entry.files if not verified(cached_path(entry, item, cache_dir), item)]


def _download(hub_file: HubFile, target: Path) -> None:
    request = urllib.request.Request(hub_file.url, headers={"User-Agent": "ragbench-datasets"})
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:  # noqa: S310 (the url is the entry's own)
            while chunk := response.read(CHUNK):
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {hub_file.name} from {hub_file.url}: {exc}") from exc
    if not verified(partial, hub_file):
        found = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise ChecksumError(
            f"{hub_file.name} from {hub_file.url} is not the file this RAGBench pins (expected {hub_file.size} bytes and "
            f"{'sha256' if hub_file.sha256 else 'git sha1'} {hub_file.sha256 or hub_file.git_sha1}; got {found} bytes). Nothing was kept."
        )
    os.replace(partial, target)


def fetch_files(entry: HubDataset, cache_dir: Path, on_file: Callable[[HubFile], None] | None = None) -> dict[str, Path]:
    """Every file of the entry, downloading (and verifying) those the cache lacks. Returns their paths by file name."""
    paths: dict[str, Path] = {}
    for hub_file in entry.files:
        path = cached_path(entry, hub_file, cache_dir)
        if not verified(path, hub_file):
            if on_file is not None:
                on_file(hub_file)
            _download(hub_file, path)
        paths[hub_file.name] = path
    return paths
