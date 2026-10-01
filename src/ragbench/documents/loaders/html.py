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
    """Collects the visible text. Its state uses `__name` attributes (mangled per class) so it can never clash with the parser's own."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.__parts: list[str] = []
        self.__title = ""
        self.__first_h1 = ""
        self.__skip = 0
        self.__in_title = self.__in_h1 = self.__in_pre = False
        self.__cells = 0
        self.__pending = 0  # newlines owed before the next text

    @property
    def title(self) -> str:
        return self.__title

    @property
    def first_h1(self) -> str:
        return self.__first_h1

    def __break(self, newlines: int) -> None:
        self.__pending = max(self.__pending, newlines)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED:
            self.__skip += 1
        elif tag == "title":
            self.__in_title = True
        elif tag == "h1" and not self.__first_h1:
            self.__in_h1 = True
        if tag == "pre":
            self.__in_pre = True
        if tag in _PARAGRAPH:
            self.__break(2)
        elif tag in _LINE:
            self.__break(1)
        if tag == "tr":
            self.__cells = 0
        elif tag in ("td", "th"):
            if self.__cells:
                self.__parts.append(" | ")
            self.__cells += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self.__skip = max(0, self.__skip - 1)
        elif tag == "title":
            self.__in_title = False
        elif tag == "h1":
            self.__in_h1 = False
        elif tag == "pre":
            self.__in_pre = False
        if tag in _PARAGRAPH:
            self.__break(2)

    def handle_data(self, data: str) -> None:
        if self.__skip:
            return
        if self.__in_title:
            self.__title += data
            return
        if self.__in_h1:
            self.__first_h1 += data
        text = data if self.__in_pre else _SPACE.sub(" ", data)
        if not text.strip() and not self.__in_pre:
            if text and self.__parts and not self.__pending:
                self.__parts.append(" ")
            return
        if self.__pending:
            self.__parts.append("\n" * self.__pending)
            self.__pending = 0
        self.__parts.append(text)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self.__parts).split("\n")]
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
