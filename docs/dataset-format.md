# Dataset Format

RAGBench evaluates retrieval at the document level. Do not add static `relevant_chunk_ids`; chunks are generated dynamically by each RAG system and may differ by chunker or strategy.

## Documents

Put your files in a directory and point `dataset.documents_path` at it (subfolders are searched). Supported formats:

| Extension | Becomes | Needs |
| --- | --- | --- |
| `.txt`, `.md`, `.markdown`, `.rst` | one document; a Markdown file's title is its first `# Heading` | nothing |
| `.html`, `.htm` | one document of the visible text: `<title>` (else the first `<h1>`) is the title; scripts, styles, comments and `<noscript>` are dropped; entities are decoded; paragraphs, list items and table rows keep their line breaks | nothing (standard library parser) |
| `.pdf` | one document; `metadata["page_spans"]` records the character range of every page, and each chunk gets `page` (and `page_end` when it crosses pages) in its metadata | `pip install 'ragbench[pdf]'` |
| `.docx` | one document; Word headings become `#` Markdown headings (so the `markdown` chunker can use them), tables become ` | `-separated lines | `pip install 'ragbench[docx]'` |
| `.csv`, `.tsv`, `.json` (an array of records), `.jsonl` | **one document per row / record**, written as `column: value` lines so the column names travel with the text | nothing |
| a single `.json` object | one document of the pretty-printed JSON | nothing |

```text
docs/
  doc_001.md
  handbook.docx
  contract.pdf
  faq.csv
  .ragbenchignore
```

**Document ids** are stable. A file named like `doc_001.md` uses its stem as the id; other files get a deterministic hash-based id. The documents of a table file are named `<file id>#<row>` (`doc_3fa9c1d2e0#7`), where the row counts data rows from 1 and is also stored as `metadata["row"]`. Qrels and `relevant_doc_ids` refer to these ids.

**Tables and records.** By default every column goes into the text. Pick columns and an id in the config:

```yaml
dataset:
  documents_path: data/faq
  tabular:
    text_columns: [question, answer]   # the document text (default: all columns)
    id_column: id                      # documents are named <file id>#<value of id> instead of #<row>
```

**Choosing files.** `dataset.include` and `dataset.exclude` take globs matched against each file's path relative to `documents_path` (`*` within a folder, `**` across folders; a pattern without `/` matches the file name anywhere). A `.ragbenchignore` file in the documents folder adds gitignore-style rules: `#` comments, blank lines, `dir/`, a leading `/` to anchor at the folder, `*`, `**`, `?`, and `!` to re-include (a file cannot be re-included once its folder is excluded, as in git):

```text
# .ragbenchignore
drafts/
*.tmp.md
!keep.tmp.md
```

**Encodings.** Files are read as UTF-8 (a UTF-8/16/32 byte-order mark is honoured). A file that is not valid UTF-8 is decoded as Windows-1252, which covers legacy Western-European text, and a warning says so; bytes that fit no encoding are replaced with `U+FFFD` with a warning. Re-save such files as UTF-8 when characters look wrong.

**Errors.** `dataset.on_error: raise` (default) stops at the first file that cannot be loaded (corrupt PDF, invalid JSON, a file above `dataset.max_file_mb`, default 25) and names it. `on_error: skip` skips such files and records a warning instead. The warnings (skips and encoding fallbacks) are listed by `ragbench inspect-dataset` and stored as `document_warnings` in each run's `run_summary.json`. A missing optional package (`pypdf`, `python-docx`) always stops the run with the `pip install` to run, because no file is at fault. `inspect-dataset` accepts `--include`, `--exclude` and `--on-error` to try these out.

Loaders are pluggable: register a function for an extension in `LOADERS` (see [extending.md](extending.md)).

## Validation

`ragbench inspect-dataset` checks the dataset before you spend money on a run. It flags duplicate question ids, questions or qrels referencing documents that are not on disk, qrels for unknown question ids, `relevant_doc_ids` missing from the qrels file, and documents with empty text. The same warnings are embedded in each run's `run_summary.json`, next to the `document_warnings` described above.

## Questions

Questions are stored as JSONL:

```json
{
  "id": "q_001",
  "question": "Who won the prestigious IIOTY award in 2023?",
  "reference_answer": "Maxine Thompson won the prestigious Insurance Innovator of the Year award in 2023.",
  "expected_keywords": ["Maxine Thompson", "Insurance Innovator of the Year", "2023"],
  "relevant_doc_ids": ["doc_005", "doc_014"],
  "category": "direct_fact",
  "difficulty": "easy",
  "answer_type": "single_fact"
}
```

For unanswerable questions, use an empty `relevant_doc_ids` list and make the reference answer explicit that the documents do not contain the answer.

## Optional Qrels

`qrels.jsonl` supports graded relevance:

```json
{"query_id": "q_001", "doc_id": "doc_005", "relevance": 3}
```

Recommended relevance levels:

| Value | Meaning |
| --- | --- |
| 0 | Not relevant |
| 1 | Partially relevant |
| 2 | Relevant |
| 3 | Contains exact answer |

If qrels are missing, RAGBench derives binary relevance from `relevant_doc_ids`.

## Qrels Audit

Runs generate `qrels_audit.md` and `qrels_audit.csv`. These files flag cases where an answer was judged highly supported even though retrieval metrics were low because retrieved documents were not labeled relevant. Treat those rows as candidates for human review.

