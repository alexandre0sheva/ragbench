"""Plain-text formats: `.txt`, `.md`, `.markdown`, `.rst`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ragbench.documents.loaders.registry import LoadContext, decode_bytes
from ragbench.documents.schema import Document
from ragbench.registry import LOADERS
from ragbench.utils.ids import stable_doc_id


def title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


def extract_title(path: Path, text: str) -> str:
    """The first `# Heading` of a Markdown file, else an HTML `<title>`, else the file name."""
    if path.suffix.lower() in {".md", ".markdown"}:
        for line in text.splitlines():
            match = re.match(r"^\s*#\s+(.+?)\s*$", line)
            if match:
                return match.group(1).strip()
    html_title = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    if html_title:
        return re.sub(r"\s+", " ", html_title.group(1)).strip()
    return title_from_filename(path)


def make_document(path: Path, context: LoadContext, title: str, text: str, extra: dict[str, Any] | None = None, doc_id: str | None = None) -> Document:
    return Document(
        doc_id=doc_id or stable_doc_id(path, context.root),
        path=str(path),
        title=title,
        text=text,
        metadata={"source_path": str(path), "extension": path.suffix.lower(), **(extra or {})},
    )


def read_text_file(path: Path, context: LoadContext) -> str:
    return decode_bytes(path.read_bytes(), path.name, context.warn)


@LOADERS.register(".txt", aliases=(".md", ".markdown", ".rst"))
def load_text(path: Path, context: LoadContext) -> list[Document]:
    text = read_text_file(path, context)
    return [make_document(path, context, extract_title(path, text), text)]
