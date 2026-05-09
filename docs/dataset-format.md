# Dataset Format

RAGBench evaluates retrieval at the document level. Do not add static `relevant_chunk_ids`; chunks are generated dynamically by each RAG system and may differ by chunker or strategy.

## Documents

Put `.md` or `.txt` files in a directory:

```text
docs/
  doc_001.md
  doc_002.md
```

Document IDs are stable. Files named like `doc_001.md` use the stem as the document ID. Other files receive deterministic hash-based IDs.

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

