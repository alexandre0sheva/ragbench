"""Word documents (optional extra: `pip install 'ragbench[docx]'`). Headings become Markdown headings so the `markdown` chunker works on them."""

from __future__ import annotations

import re
from pathlib import Path

from ragbench.documents.loaders.registry import LoadContext
from ragbench.documents.loaders.text import make_document, title_from_filename
from ragbench.documents.schema import Document
from ragbench.models.errors import MissingExtraError
from ragbench.registry import LOADERS


@LOADERS.register(".docx")
def load_docx(path: Path, context: LoadContext) -> list[Document]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise MissingExtraError(
            f"Word support requires the optional dependency python-docx. Install it with `pip install 'ragbench[docx]'` to load {path.name}."
        ) from exc
    document = docx.Document(str(path))
    blocks: list[str] = []
    first_heading = ""
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = (paragraph.style.name if paragraph.style is not None else "") or ""
            level = _heading_level(style)
            if level:
                first_heading = first_heading or text
                text = f"{'#' * level} {text}"
            blocks.append(text)
        elif tag == "tbl":
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in Table(element, document).rows]
            if any(row.strip(" |") for row in rows):
                blocks.append("\n".join(rows))
    title = (document.core_properties.title or "").strip() or first_heading or title_from_filename(path)
    return [make_document(path, context, title, "\n\n".join(blocks))]


def _heading_level(style: str) -> int:
    if style == "Title":
        return 1
    match = re.fullmatch(r"Heading (\d)", style)
    return min(int(match.group(1)), 6) if match else 0
