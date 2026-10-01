"""`ragbench import`: turn a dataset you already have (a CSV of questions, a BEIR download, Markdown Q/A pairs) into RAGBench files.

Every importer writes `questions.jsonl` (plus `qrels.jsonl` and a `docs/` folder where the source has them). A question that comes
without relevance labels is written with an explicit `answerable: true`: "no relevant documents" must never be read as "unanswerable"
(see docs/dataset-format.md#label-free-mode).
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragbench.utils.hashing import stable_hash
from ragbench.utils.jsonl import read_jsonl, write_jsonl

FORMATS = ("csv", "beir", "qa-md")
_LIST_SPLIT = re.compile(r"[;|,\n]")
_TRUE, _FALSE = {"true", "yes", "y", "1"}, {"false", "no", "n", "0"}


class ImportDatasetError(ValueError):
    """The source cannot be imported (missing file, unknown format, nothing usable in it)."""


@dataclass
class ImportResult:
    output: Path
    questions_path: Path
    qrels_path: Path | None
    docs_dir: Path | None
    n_questions: int
    n_documents: int = 0
    n_qrels: int = 0
    warnings: list[str] = field(default_factory=list)


def import_dataset(fmt: str, source: Path, output: Path, *, docs: Path | None = None, split: str = "test", force: bool = False) -> ImportResult:
    """Import `source` (a file or folder, depending on `fmt`) into `output`. `docs` is an existing documents folder that csv / qa-md questions refer to:
    it is only used to check their `doc_ids`. `split` picks the BEIR qrels file. Refuses to replace an earlier import unless `force`."""
    if fmt not in FORMATS:
        raise ImportDatasetError(f"Unknown format {fmt!r}. Available: {', '.join(FORMATS)}")
    if not source.exists():
        raise ImportDatasetError(f"Input not found: {source}")
    writers: dict[str, Callable[..., ImportResult]] = {"csv": _import_csv, "beir": _import_beir, "qa-md": _import_qa_markdown}
    existing = [name for name in ("questions.jsonl", "qrels.jsonl", "id_map.json", *(("docs",) if fmt == "beir" else ())) if (output / name).exists()]
    if existing and not force:
        raise ImportDatasetError(f"{output} already holds {', '.join(existing)}. Use --force to replace the files of an earlier import, or pick another --output.")
    result = writers[fmt](source, output, split=split, force=force)
    if docs is not None:
        result.warnings.extend(_check_against_documents(result, docs))
    return result


def _check_against_documents(result: ImportResult, docs: Path) -> list[str]:
    from ragbench.datasets.loader import load_dataset
    from ragbench.datasets.validation import validate_dataset
    from ragbench.documents.loaders import load_documents

    return validate_dataset(load_documents(docs), load_dataset(result.questions_path, result.qrels_path))


def _question_row(question_id: str, text: str, answer: str | None, doc_ids: list[str], answerable: bool | None, extra: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"id": question_id, "question": text}
    if answer:
        row["reference_answer"] = answer
    if doc_ids:
        row["relevant_doc_ids"] = doc_ids
    if answerable is not None:
        row["answerable"] = answerable
    elif not doc_ids:
        row["answerable"] = True  # nothing is labeled: say explicitly that the question has an answer
    return {**row, **extra}


def _write_questions(output: Path, rows: list[dict[str, Any]]) -> Path:
    path = output / "questions.jsonl"
    write_jsonl(path, rows)
    return path


def _ids(count: int) -> list[str]:
    return [f"q_{number:03d}" for number in range(1, count + 1)]


def _split_list(cell: str) -> list[str]:
    return [part.strip() for part in _LIST_SPLIT.split(cell) if part.strip()]


# -- csv -----------------------------------------------------------------------------------------

_CSV_ALIASES = {
    "question": ("question", "query", "q"),
    "answer": ("answer", "reference_answer", "reference"),
    "doc_ids": ("doc_ids", "relevant_doc_ids", "docs"),
    "id": ("id",),
    "category": ("category",),
    "difficulty": ("difficulty",),
    "answer_type": ("answer_type",),
    "keywords": ("keywords", "expected_keywords"),
    "answerable": ("answerable",),
}


def _import_csv(source: Path, output: Path, **_: Any) -> ImportResult:
    if source.is_dir():
        raise ImportDatasetError(f"--input for csv must be a file, not a folder: {source}")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
        columns = {key: next((headers[alias] for alias in aliases if alias in headers), None) for key, aliases in _CSV_ALIASES.items()}
        if columns["question"] is None:
            raise ImportDatasetError(f"{source.name} needs a `question` column (or `query`); it has: {', '.join(reader.fieldnames or []) or 'no header'}")
        known = {name for name in columns.values() if name}
        records = list(reader)

    def cell(record: dict[str, str | None], key: str) -> str:
        column = columns[key]
        return (record.get(column) or "").strip() if column else ""

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for number, record in enumerate(records, start=2):  # row 1 is the header
        text = cell(record, "question")
        if not text:
            continue
        flag = cell(record, "answerable").lower()
        if flag and flag not in _TRUE | _FALSE:
            warnings.append(f"Row {number}: answerable={flag!r} is not yes/no/true/false; ignored")
        question_id = cell(record, "id") or f"q_{len(rows) + 1:03d}"
        if question_id in seen:
            raise ImportDatasetError(f"Row {number}: duplicate question id {question_id!r}")
        seen.add(question_id)
        extra: dict[str, Any] = {}
        for key in ("category", "difficulty", "answer_type"):
            if cell(record, key):
                extra[key] = cell(record, key)
        if cell(record, "keywords"):
            extra["expected_keywords"] = _split_list(cell(record, "keywords"))
        metadata = {name: value.strip() for name, value in record.items() if name and name not in known and value and value.strip()}
        if metadata:
            extra["metadata"] = metadata
        answerable = True if flag in _TRUE else False if flag in _FALSE else None
        rows.append(_question_row(question_id, text, cell(record, "answer") or None, _split_list(cell(record, "doc_ids")), answerable, extra))
    if not rows:
        raise ImportDatasetError(f"{source.name} has no question rows")
    return ImportResult(output, _write_questions(output, rows), None, None, len(rows), warnings=warnings)


# -- beir ----------------------------------------------------------------------------------------


def _doc_id_map(original_ids: list[str]) -> dict[str, str]:
    """Loader-safe document ids (`doc_<id>`, only letters, digits, `_` and `-`), unique even on a case-insensitive file system."""
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for original in original_ids:
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", original).strip("_") or "x"
        new = base if base.startswith("doc_") else f"doc_{base}"
        if new.lower() in taken:
            new = f"{new}_{stable_hash(original, 6)}"
        taken.add(new.lower())
        mapping[original] = new
    return mapping


def _read_beir_qrels(path: Path, warnings: list[str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for number, parts in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not any(part.strip() for part in parts):
                continue
            try:
                if len(parts) < 3:
                    raise ValueError
                grade = int(float(parts[2]))
            except ValueError:
                if number > 1:  # the first row is usually a `query-id corpus-id score` header
                    warnings.append(f"{path.name} line {number}: not `query-id<TAB>corpus-id<TAB>score`; skipped")
                continue
            qrels.setdefault(parts[0].strip(), {})[parts[1].strip()] = grade
    return qrels


def _import_beir(source: Path, output: Path, *, split: str, force: bool, **_: Any) -> ImportResult:
    corpus_path, queries_path, qrels_dir = source / "corpus.jsonl", source / "queries.jsonl", source / "qrels"
    missing = [path.name for path in (corpus_path, queries_path) if not path.is_file()] + ([] if qrels_dir.is_dir() else ["qrels/"])
    if not source.is_dir() or missing:
        raise ImportDatasetError(f"{source} is not a BEIR dataset folder (missing {', '.join(missing) or 'the folder'}): expected corpus.jsonl, queries.jsonl and qrels/<split>.tsv")
    qrels_file = qrels_dir / f"{split}.tsv"
    if not qrels_file.is_file():
        available = sorted(path.stem for path in qrels_dir.glob("*.tsv"))
        raise ImportDatasetError(f"No qrels/{split}.tsv in {source}. Splits available: {', '.join(available) or 'none'} (choose one with --split)")
    warnings: list[str] = []
    corpus = read_jsonl(corpus_path)
    id_map = _doc_id_map([str(row["_id"]) for row in corpus])
    docs_dir = output / "docs"
    if force and docs_dir.exists():
        shutil.rmtree(docs_dir)  # the folder of the earlier import: stale documents would corrupt the corpus
    docs_dir.mkdir(parents=True, exist_ok=True)
    for row in corpus:
        title, text = str(row.get("title") or "").strip(), str(row.get("text") or "")
        (docs_dir / f"{id_map[str(row['_id'])]}.md").write_text(f"# {title}\n\n{text}\n" if title else f"{text}\n", encoding="utf-8")

    raw_qrels = _read_beir_qrels(qrels_file, warnings)
    queries = {str(row["_id"]): str(row.get("text") or "").strip() for row in read_jsonl(queries_path)}
    rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    for query_id, text in queries.items():
        if query_id not in raw_qrels:
            continue  # a query of another split
        graded = {}
        for corpus_id, grade in raw_qrels[query_id].items():
            if corpus_id in id_map:
                graded[id_map[corpus_id]] = grade
            else:
                warnings.append(f"Qrels for query {query_id} name the document {corpus_id}, which is not in corpus.jsonl; left out")
        relevant = [doc_id for doc_id, grade in sorted(graded.items(), key=lambda item: -item[1]) if grade > 0]
        if not text or not relevant:
            warnings.append(f"Query {query_id} has {'no text' if not text else 'no document graded above 0'}; left out (nothing to score)")
            continue
        rows.append(_question_row(query_id, text, None, relevant, None, {}))
        qrel_rows.extend({"query_id": query_id, "doc_id": doc_id, "relevance": grade} for doc_id, grade in graded.items())
    unknown = sorted(set(raw_qrels) - set(queries))
    if unknown:
        warnings.append(f"Qrels name {len(unknown)} queries that are not in queries.jsonl (first: {', '.join(unknown[:3])}); left out")
    if not rows:
        raise ImportDatasetError(f"The {split} split leaves no questions with a relevant document; see the warnings: {'; '.join(warnings[:3])}")
    qrels_path = output / "qrels.jsonl"
    write_jsonl(qrels_path, qrel_rows)
    (output / "id_map.json").write_text(json.dumps(id_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return ImportResult(output, _write_questions(output, rows), qrels_path, docs_dir, len(rows), n_documents=len(corpus), n_qrels=len(qrel_rows), warnings=warnings)


# -- Markdown Q/A pairs ------------------------------------------------------------------------

_MARK = r"(?:\*\*|__)?"
_LEAD = rf"^\s*(?:#{{1,6}}\s+)?(?:[-*+]\s+)?{_MARK}"
_QUESTION = re.compile(rf"{_LEAD}(?:Q|Question){_MARK}\s*[:.]\s*{_MARK}\s*(.*)$", re.IGNORECASE)
_ANSWER = re.compile(rf"{_LEAD}(?:A|Answer){_MARK}\s*[:.]\s*{_MARK}\s*(.*)$", re.IGNORECASE)
_DOCS = re.compile(rf"{_LEAD}(?:Docs|Documents|Sources?){_MARK}\s*:\s*{_MARK}\s*(.*)$", re.IGNORECASE)


def _parse_qa_markdown(text: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    target: str | None = None  # the field a following plain line continues; a blank line ends it
    for line in text.splitlines():
        if match := _QUESTION.match(line):
            current = {"question": [match.group(1).strip()], "answer": [], "docs": []}
            pairs.append(current)
            target = "question"
        elif current is not None and (match := _ANSWER.match(line)):
            current["answer"].append(match.group(1).strip())
            target = "answer"
        elif current is not None and (match := _DOCS.match(line)):
            current["docs"].extend(_split_list(match.group(1)))
            target = None
        elif not line.strip():
            target = None
        elif current is not None and target is not None:
            current[target].append(line.strip())
    return [
        {"question": " ".join(part for part in pair["question"] if part), "answer": " ".join(part for part in pair["answer"] if part) or None, "docs": pair["docs"]}
        for pair in pairs
        if any(pair["question"])
    ]


def _import_qa_markdown(source: Path, output: Path, **_: Any) -> ImportResult:
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt"}) if source.is_dir() else [source]
    pairs = [pair for path in files for pair in _parse_qa_markdown(path.read_text(encoding="utf-8-sig", errors="replace"))]
    if not pairs:
        raise ImportDatasetError(f"No question/answer pairs found in {source}. Write each as `Q: question` followed by `A: answer` (optionally `Docs: doc_001, doc_002`).")
    rows = [_question_row(question_id, pair["question"], pair["answer"], pair["docs"], None, {}) for question_id, pair in zip(_ids(len(pairs)), pairs, strict=True)]
    return ImportResult(output, _write_questions(output, rows), None, None, len(rows))
