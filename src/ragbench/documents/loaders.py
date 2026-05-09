from __future__ import annotations

import re
from pathlib import Path

from ragbench.documents.schema import Document
from ragbench.utils.ids import stable_doc_id

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm"}


def _extract_title(path: Path, text: str) -> str:
    if path.suffix.lower() in {".md", ".markdown"}:
        for line in text.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*$", line)
            if match:
                return match.group(1).strip()
    html_title = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    if html_title:
        return re.sub(r"\s+", " ", html_title.group(1)).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _html_to_text(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Documents path not found: {path}")
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        root = path
    documents: list[Document] = []
    for file_path in files:
        raw = file_path.read_text(encoding="utf-8")
        text = _html_to_text(raw) if file_path.suffix.lower() in {".html", ".htm"} else raw
        documents.append(
            Document(
                doc_id=stable_doc_id(file_path, root),
                path=str(file_path),
                title=_extract_title(file_path, raw),
                text=text,
                metadata={
                    "source_path": str(file_path),
                    "extension": file_path.suffix.lower(),
                },
            )
        )
    if not documents:
        raise ValueError(f"No supported documents found under {path}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return documents

