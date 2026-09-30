"""PDF (optional extra: `pip install 'ragbench[pdf]'`). Pages become `page_spans` so chunks know their page."""

from __future__ import annotations

from pathlib import Path

from ragbench.documents.loaders.registry import LoadContext
from ragbench.documents.loaders.text import make_document, title_from_filename
from ragbench.documents.schema import Document
from ragbench.models.errors import MissingExtraError
from ragbench.registry import LOADERS


@LOADERS.register(".pdf")
def load_pdf(path: Path, context: LoadContext) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingExtraError(
            f"PDF support requires the optional dependency pypdf. Install it with `pip install 'ragbench[pdf]'` to load {path.name}."
        ) from exc
    reader = PdfReader(str(path))
    parts: list[str] = []
    spans: list[list[int]] = []
    offset = 0
    for number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue  # blank pages are skipped but later pages keep their real numbers
        if parts:
            offset += 2  # the "\n\n" between pages
        spans.append([offset, offset + len(page_text), number])
        parts.append(page_text)
        offset += len(page_text)
    title = str((reader.metadata.title if reader.metadata else "") or "").strip() or title_from_filename(path)
    return [make_document(path, context, title, "\n\n".join(parts), {"page_spans": spans})]
