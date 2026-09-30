"""HTML: visible text with paragraph structure, using only the standard library (no extra dependency)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ragbench.documents.loaders.registry import LoadContext
from ragbench.documents.loaders.text import make_document, read_text_file, title_from_filename
from ragbench.documents.schema import Document
from ragbench.registry import LOADERS

_SKIPPED = {"script", "style", "noscript", "template"}
_PARAGRAPH = {"p", "div", "section", "article", "header", "footer", "main", "aside", "nav", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "ul", "ol", "table", "form", "figure", "dl", "details"}
_LINE = {"br", "li", "tr", "dt", "dd", "hr", "figcaption", "summary"}
_SPACE = re.compile(r"[ \t\r\n\f\v]+")  # not \xa0: a non-breaking space is content, not layout


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self.first_h1 = ""
        self._skip = 0
        self._in_title = self._in_h1 = self._in_pre = False
        self._cells = 0
        self._pending = 0  # newlines owed before the next text

    def _break(self, newlines: int) -> None:
        self._pending = max(self._pending, newlines)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "h1" and not self.first_h1:
            self._in_h1 = True
        if tag == "pre":
            self._in_pre = True
        if tag in _PARAGRAPH:
            self._break(2)
        elif tag in _LINE:
            self._break(1)
        if tag == "tr":
            self._cells = 0
        elif tag in ("td", "th"):
            if self._cells:
                self.parts.append(" | ")
            self._cells += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "h1":
            self._in_h1 = False
        elif tag == "pre":
            self._in_pre = False
        if tag in _PARAGRAPH:
            self._break(2)

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data
            return
        if self._in_h1:
            self.first_h1 += data
        text = data if self._in_pre else _SPACE.sub(" ", data)
        if not text.strip() and not self._in_pre:
            if text and self.parts and not self._pending:
                self.parts.append(" ")
            return
        if self._pending:
            self.parts.append("\n" * self._pending)
            self._pending = 0
        self.parts.append(text)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self.parts).split("\n")]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(raw: str) -> tuple[str, str]:
    """`(title, text)` of an HTML page. The title is `<title>`, else the first `<h1>`, else empty."""
    extractor = _TextExtractor()
    extractor.feed(raw)
    extractor.close()
    title = _SPACE.sub(" ", extractor.title).strip() or _SPACE.sub(" ", extractor.first_h1).strip()
    return title, extractor.text()


@LOADERS.register(".html", aliases=(".htm",))
def load_html(path: Path, context: LoadContext) -> list[Document]:
    title, text = html_to_text(read_text_file(path, context))
    return [make_document(path, context, title or title_from_filename(path), text)]
