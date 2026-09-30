"""Tables and records: `.csv`, `.tsv`, `.json`, `.jsonl`. Each row / record becomes one document with its column names as context."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from ragbench.documents.loaders.registry import DocumentLoadError, LoadContext
from ragbench.documents.loaders.text import make_document, read_text_file, title_from_filename
from ragbench.documents.schema import Document
from ragbench.registry import LOADERS
from ragbench.utils.ids import stable_doc_id


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _records_to_documents(path: Path, context: LoadContext, columns: list[str] | None, records: list[dict[str, Any]]) -> list[Document]:
    options = context.tabular
    available = columns if columns is not None else list(dict.fromkeys(key for record in records for key in record))  # first-seen order
    for wanted in [*(options.text_columns or []), *([options.id_column] if options.id_column else [])]:
        if wanted not in available:
            raise DocumentLoadError(f"{path.name}: no column {wanted!r} (columns: {', '.join(available)})")
    text_columns = options.text_columns or available
    base = stable_doc_id(path, context.root)
    title_base = title_from_filename(path)
    documents: list[Document] = []
    for row, record in enumerate(records, start=1):
        lines = [f"{column}: {_cell(record.get(column))}" for column in text_columns if _cell(record.get(column)).strip()]
        if not lines:
            continue
        identifier = _cell(record.get(options.id_column)).strip() if options.id_column else ""
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier) if identifier else str(row)
        extra: dict[str, Any] = {"row": row}
        if identifier:
            extra["id"] = identifier
        documents.append(make_document(path, context, f"{title_base} row {row}", "\n".join(lines), extra, doc_id=f"{base}#{suffix}"))
    if not documents:
        context.warn(f"{path.name}: no data rows with content; skipped.")
    return documents


def _load_delimited(path: Path, context: LoadContext, delimiter: str) -> list[Document]:
    text = read_text_file(path, context)
    rows = [row for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter) if any(cell.strip() for cell in row)]
    if not rows:
        context.warn(f"{path.name}: empty file; skipped.")
        return []
    header = [name.strip() for name in rows[0]]
    records = [dict(zip(header, row, strict=False)) for row in rows[1:]]
    return _records_to_documents(path, context, header, records)


@LOADERS.register(".csv")
def load_csv(path: Path, context: LoadContext) -> list[Document]:
    return _load_delimited(path, context, ",")


@LOADERS.register(".tsv")
def load_tsv(path: Path, context: LoadContext) -> list[Document]:
    return _load_delimited(path, context, "\t")


def _as_record(item: Any) -> dict[str, Any]:
    return item if isinstance(item, dict) else {"value": item}


@LOADERS.register(".json")
def load_json(path: Path, context: LoadContext) -> list[Document]:
    try:
        data = json.loads(read_text_file(path, context))
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(f"{path.name}: invalid JSON ({exc})") from exc
    if isinstance(data, list):
        return _records_to_documents(path, context, None, [_as_record(item) for item in data])
    return [make_document(path, context, title_from_filename(path), json.dumps(data, ensure_ascii=False, indent=2))]  # one object: the whole file is the text


@LOADERS.register(".jsonl")
def load_jsonl(path: Path, context: LoadContext) -> list[Document]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(read_text_file(path, context).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(_as_record(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(f"{path.name}: invalid JSON on line {number} ({exc})") from exc
    return _records_to_documents(path, context, None, records)
