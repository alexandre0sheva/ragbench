"""Run manifest: enough provenance to tell what produced a result directory and to key caches/resume later."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ragbench import __version__
from ragbench.documents.schema import Document

TRACKED_DEPENDENCIES = ["openai", "pandas", "numpy", "scikit-learn", "pydantic", "chromadb", "faiss-cpu", "qdrant-client", "rank-bm25", "tiktoken", "typer", "jinja2"]


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(raw_config: dict[str, Any]) -> str:
    """Hash of the experiment config as written (independent of key order)."""
    return hashlib.sha256(json.dumps(raw_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def dataset_hash(documents: Iterable[Document], questions_path: Path, qrels_path: Path | None) -> str:
    """Hash of the corpus text plus the question and qrels files; any edit to the dataset changes it."""
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda doc: doc.doc_id):
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(hashlib.sha256(document.text.encode("utf-8")).digest())
    for path in (questions_path, qrels_path):
        digest.update(path.read_bytes() if path is not None and path.exists() else b"<none>")
    return digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_info(start: Path) -> dict[str, Any]:
    cwd = start if start.is_dir() else start.parent
    sha = _git(["rev-parse", "HEAD"], cwd)
    dirty = None if sha is None else bool(_git(["status", "--porcelain"], cwd))
    return {"git_sha": sha, "git_dirty": dirty}


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def build_manifest(
    *,
    config_path: Path,
    raw_config: dict[str, Any],
    dataset_digest: str,
    mode: str,
    models_used: list[str],
    started_utc: str,
    finished_utc: str,
) -> dict[str, Any]:
    return {
        "ragbench_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": dependency_versions(),
        **git_info(config_path.resolve()),
        "config_hash": config_hash(raw_config),
        "dataset_hash": dataset_digest,
        "mode": mode,
        "models_used": models_used,
        "seed": None,  # Placeholder: no component is randomized yet; Task 22 (bootstrap) is the first seeded step.
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }
