# Dataset Format

RAGBench evaluates retrieval at the document level. Do not add static `relevant_chunk_ids`; chunks are generated dynamically by each RAG system and may differ by chunker or strategy.

## Documents

Put `.md`, `.txt`, `.rst`, `.html`, or `.pdf` files in a directory (PDF requires the optional extra: `pip install 'ragbench[pdf]'`):

```text
docs/
  doc_001.md
  doc_002.md
  contract.pdf
```

Document IDs are stable. Files named like `doc_001.md` use the stem as the document ID. Other files receive deterministic hash-based IDs.

## Validation

`ragbench inspect-dataset` checks the dataset before you spend money on a run. It flags duplicate question ids, questions or qrels referencing documents that are not on disk, qrels for unknown question ids, `relevant_doc_ids` missing from the qrels file, and documents with empty text. The same warnings are embedded in each run's `run_summary.json`.

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

