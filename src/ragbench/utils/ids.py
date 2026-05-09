from __future__ import annotations

import re
from pathlib import Path

from ragbench.utils.hashing import stable_hash

DOC_ID_RE = re.compile(r"^doc_[A-Za-z0-9_-]+$")


def stable_doc_id(path: Path, root: Path | None = None) -> str:
    """Generate a stable document id.

    Demo files named doc_001.md keep their stem as the ID so qrels are readable.
    Other files use a hash of their path relative to the document root.
    """
    stem = path.stem
    if DOC_ID_RE.match(stem):
        return stem
    relative = path.relative_to(root) if root and path.is_relative_to(root) else path
    slug = re.sub(r"[^A-Za-z0-9]+", "_", relative.as_posix()).strip("_").lower()
    return f"doc_{stable_hash(slug, 10)}"


def stable_chunk_id(doc_id: str, chunk_index: int, text: str, start_char: int, end_char: int) -> str:
    digest = stable_hash(f"{start_char}:{end_char}:{text}", 10)
    return f"{doc_id}::chunk::{chunk_index:04d}::{digest}"

