"""Tools that read the corpus: `list_documents`, `read_document`, `corpus_grep`, `lookup_table`."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragbench.documents.schema import Document
from ragbench.registry import TOOLS
from ragbench.tools.base import BaseTool, ToolContext, ToolResult, make_spec

MAX_READ_CHARS = 8000
MAX_PATTERN_CHARS = 200
MAX_REGEX_LINE_CHARS = 2000  # longer lines are only searched up to this point, which bounds what a regex can chew on
MAX_GREP_RESULTS = 50


# --- list_documents -------------------------------------------------------------------------------------------------------


class ListDocumentsArgs(BaseModel):
    contains: str | None = Field(default=None, description="Only documents whose title or id contains this text (case-insensitive).")
    limit: int = Field(default=50, ge=1, le=200, description="Most documents to list.")
    offset: int = Field(default=0, ge=0, description="Documents to skip (to page through a long list).")


@TOOLS.register("list_documents")
class ListDocumentsTool(BaseTool):
    Args = ListDocumentsArgs
    spec = make_spec("list_documents", "List the documents in the collection: id, title and length, in order. Use it to see what exists before reading.", ListDocumentsArgs)

    def _run(self, args: ListDocumentsArgs, ctx: ToolContext) -> ToolResult:
        needle = (args.contains or "").lower()
        matching = [d for d in ctx.corpus.documents if not needle or needle in d.title.lower() or needle in d.doc_id.lower()]
        page = matching[args.offset : args.offset + args.limit]
        if not page:
            return ToolResult.ok(f"No documents{' match ' + repr(args.contains) if needle else ''}{' beyond the first ' + str(args.offset) if args.offset else ''}.", data=[])
        lines = [f"{d.doc_id} | {d.title} | {len(d.text)} chars" for d in page]
        more = len(matching) - args.offset - len(page)
        footer = f"\n({more} more; use offset={args.offset + len(page)})" if more > 0 else ""
        return ToolResult.ok("\n".join(lines) + footer, data=[{"doc_id": d.doc_id, "title": d.title, "chars": len(d.text)} for d in page])


# --- read_document --------------------------------------------------------------------------------------------------------


class ReadDocumentArgs(BaseModel):
    doc_id: str = Field(description="Document id, as listed by `list_documents` or cited in search results.")
    start: int = Field(default=0, ge=0, description="Character offset to start reading at.")
    length: int = Field(default=2000, ge=1, le=MAX_READ_CHARS, description=f"Characters to read (at most {MAX_READ_CHARS}).")


def _missing_document(ctx: ToolContext, doc_id: str) -> ToolResult:
    close = ctx.corpus.suggest(doc_id)
    return ToolResult.fail(f"no document with id {doc_id!r}" + (f"; did you mean {', '.join(close)}?" if close else "; use list_documents to see the ids"))


@TOOLS.register("read_document")
class ReadDocumentTool(BaseTool):
    Args = ReadDocumentArgs
    spec = make_spec("read_document", "Read a window of a document's text by id. Long documents are read in several calls by moving `start`.", ReadDocumentArgs)

    def _run(self, args: ReadDocumentArgs, ctx: ToolContext) -> ToolResult:
        document = ctx.corpus.get(args.doc_id)
        if document is None:
            return _missing_document(ctx, args.doc_id)
        total = len(document.text)
        if args.start >= total:
            return ToolResult.fail(f"start={args.start} is past the end of {document.doc_id} ({total} characters)")
        end = min(args.start + args.length, total)
        header = f"[{document.doc_id} | {document.title} | characters {args.start}-{end} of {total}]"
        footer = f"\n[continues: use start={end}]" if end < total else "\n[end of document]"
        return ToolResult.ok(f"{header}\n{document.text[args.start : end]}{footer}", data={"doc_id": document.doc_id, "start": args.start, "end": end, "total": total})


# --- corpus_grep ----------------------------------------------------------------------------------------------------------


def _is_counted_repeat(text: str) -> bool:
    """Whether `text` starts with a `{n}`, `{n,}`, `{,m}` or `{n,m}` quantifier that can repeat more than once."""
    match = re.match(r"\{(\d*)(,?)(\d*)\}", text)
    return match is not None and (match.group(1), match.group(2), match.group(3)) not in {("1", "", ""), ("0", "", ""), ("", "", "")}


def regex_problem(pattern: str) -> str | None:
    """Why `pattern` is refused as a regex, or None if it is acceptable.

    A backtracking regex can take exponential time and Python's `re` never yields the interpreter while it runs, so no timeout can
    interrupt it. Instead of trying, refuse the shapes that cause it: very long patterns, back-references, and a repeated group
    that itself contains a repeat or an alternation (`(a+)+`, `(a|aa)*`, `(.*)*`).
    """
    if len(pattern) > MAX_PATTERN_CHARS:
        return f"the pattern is too long (limit {MAX_PATTERN_CHARS} characters)"
    if re.search(r"\\[1-9]|\(\?P=|\\k<", pattern):
        return "back-references are not supported"
    stack: list[dict[str, bool]] = []
    index, in_class = 0, False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "(":
            stack.append({"repeat": False, "alt": False})
        elif char == ")" and stack:
            group = stack.pop()
            follower = pattern[index + 1 : index + 2]
            repeated = follower in ("*", "+") or _is_counted_repeat(pattern[index + 1 :])
            if repeated and (group["repeat"] or group["alt"]):
                return "a repeated group that contains a repeat or an alternation can take exponential time; simplify the pattern"
            if stack and (repeated or group["repeat"]):
                stack[-1]["repeat"] = True
            if stack and group["alt"]:
                stack[-1]["alt"] = True
        elif stack and char in "*+":
            stack[-1]["repeat"] = True
        elif stack and _is_counted_repeat(pattern[index:]):
            stack[-1]["repeat"] = True
        elif stack and char == "|":
            stack[-1]["alt"] = True
        index += 1
    return None


class GrepArgs(BaseModel):
    pattern: str = Field(description="What to look for. By default every word must appear on the same line (case-insensitive); with `regex` it is a regular expression.")
    regex: bool = Field(default=False, description="Treat `pattern` as a regular expression (simple patterns only).")
    case_sensitive: bool = Field(default=False)
    doc_id: str | None = Field(default=None, description="Search only this document.")
    context_lines: int = Field(default=1, ge=0, le=5, description="Lines of context around each match.")
    max_results: int = Field(default=20, ge=1, le=MAX_GREP_RESULTS, description="Most matching lines to return.")


class GrepOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_results: int = Field(default=MAX_GREP_RESULTS, ge=1, le=MAX_GREP_RESULTS, description="Hard cap on matching lines per call, whatever the model asks for.")


@TOOLS.register("corpus_grep")
class CorpusGrepTool(BaseTool):
    Args = GrepArgs
    Options = GrepOptions
    spec = make_spec(
        "corpus_grep",
        "Search every document's lines for words or a pattern, like grep. Returns `doc_id:line: text` with a little context. Good for identifiers, names and exact figures.",
        GrepArgs,
    )
    options: GrepOptions

    def _run(self, args: GrepArgs, ctx: ToolContext) -> ToolResult:
        matcher, problem = self._matcher(args)
        if matcher is None:
            return ToolResult.fail(problem or "bad pattern")
        documents: list[Document]
        if args.doc_id:
            document = ctx.corpus.get(args.doc_id)
            if document is None:
                return _missing_document(ctx, args.doc_id)
            documents = [document]
        else:
            documents = ctx.corpus.documents
        cap = min(args.max_results, self.options.max_results)
        blocks: list[str] = []
        hits: list[dict[str, Any]] = []
        total_docs = 0
        more = False
        for document in documents:
            lines = document.text.splitlines()
            matched = [number for number, line in enumerate(lines) if matcher(line[:MAX_REGEX_LINE_CHARS])]
            if not matched:
                continue
            total_docs += 1
            for number in matched:
                if len(hits) >= cap:
                    more = True
                    break
                first, last = max(0, number - args.context_lines), min(len(lines), number + args.context_lines + 1)
                blocks.append("\n".join(f"{document.doc_id}:{n + 1}{':' if n == number else '-'} {lines[n]}" for n in range(first, last)))
                hits.append({"doc_id": document.doc_id, "line": number + 1, "text": lines[number]})
            if more:
                break
        if not hits:
            return ToolResult.ok(f"No matches for {args.pattern!r}.", data=[])
        footer = f"\n(stopped at {cap} matching lines; narrow the pattern or pass doc_id)" if more else f"\n({len(hits)} matching lines in {total_docs} documents)"
        return ToolResult.ok("\n--\n".join(blocks) + footer, data=hits)

    @staticmethod
    def _matcher(args: GrepArgs):
        flags = 0 if args.case_sensitive else re.IGNORECASE
        if args.regex:
            if problem := regex_problem(args.pattern):
                return None, problem
            try:
                compiled = re.compile(args.pattern, flags)
            except re.error as exc:
                return None, f"invalid regular expression: {exc}"
            return (lambda line: compiled.search(line) is not None), None
        words = args.pattern.split() if args.case_sensitive else args.pattern.lower().split()
        if not words:
            return None, "the pattern is empty"
        if args.case_sensitive:
            return (lambda line: all(word in line for word in words)), None
        return (lambda line: all(word in line.lower() for word in words)), None


# --- lookup_table ---------------------------------------------------------------------------------------------------------


def parse_row(document: Document) -> dict[str, str]:
    """A table-row document (one `column: value` line per column, as the tabular loader writes it) as a dict."""
    row: dict[str, str] = {}
    for line in document.text.splitlines():
        column, separator, value = line.partition(": ")
        if separator:
            row[column.strip()] = value.strip()
    return row


class LookupTableArgs(BaseModel):
    where: dict[str, str] = Field(description='Column values the row must have, e.g. {"product": "HarborShield"}. Compared ignoring case and surrounding spaces.')
    select: list[str] | None = Field(default=None, description="Columns to return (default: all of them).")
    source: str | None = Field(default=None, description="Only rows of the table whose file path or title contains this text.")
    limit: int = Field(default=5, ge=1, le=20, description="Most rows to return.")


@TOOLS.register("lookup_table")
class LookupTableTool(BaseTool):
    Args = LookupTableArgs
    spec = make_spec(
        "lookup_table",
        "Look up rows of the CSV / JSON tables in the collection by column value and return chosen columns. Exact cell values instead of searching prose.",
        LookupTableArgs,
    )

    def _run(self, args: LookupTableArgs, ctx: ToolContext) -> ToolResult:
        rows = [(d, parse_row(d)) for d in ctx.corpus.documents if "row" in d.metadata]
        if not rows:
            return ToolResult.fail("the collection has no table rows (CSV, TSV, JSON or JSONL files)")
        if args.source:
            needle = args.source.lower()
            rows = [(d, r) for d, r in rows if needle in d.path.lower() or needle in d.title.lower()]
        wanted = {column.strip().lower(): value.strip().lower() for column, value in args.where.items()}
        found = []
        for document, row in rows:
            folded = {column.lower(): value.lower() for column, value in row.items()}
            if all(folded.get(column) == value for column, value in wanted.items()):
                found.append((document, row))
        if not found:
            columns = sorted({column for _, row in rows for column in row})[:20]
            return ToolResult.ok(f"No row matches {json.dumps(args.where)}. Columns: {', '.join(columns)}.", data=[])
        selected = [{c: r.get(c, "") for c in (args.select or list(r))} for _, r in found[: args.limit]]
        lines = [f"{d.doc_id}: " + "; ".join(f"{c}={v}" for c, v in s.items()) for (d, _), s in zip(found, selected, strict=False)]
        footer = f"\n({len(found) - args.limit} more rows match)" if len(found) > args.limit else ""
        return ToolResult.ok("\n".join(lines) + footer, data=selected)
