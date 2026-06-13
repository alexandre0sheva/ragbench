# Changelog

All notable changes to RAGBench will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning once it reaches public releases.

## [0.2.0] - 2026-06-12

### Added

- **Two new RAG systems**, both dataset-agnostic and mock-safe:
  - `hybrid_rerank`: hybrid BM25 + vector RRF retrieval followed by a reranking pass.
  - `hyde`: Hypothetical Document Embeddings — the LLM writes a hypothetical answer passage used as the vector-search probe, fused with the raw-question ranking via RRF.
- Shared **embedding cache** across systems in a run (`evaluation.embedding_cache`, on by default). Cache hits are still charged to each system at standalone prices so comparisons stay fair; actual API savings are reported in `run_summary.json` and the console.
- **Live progress display**: per-system ingestion status and question-by-question progress bars during `run` / `compare` / `evaluate`.
- **Console leaderboard** printed after each run with the best value per column highlighted.
- **Dataset validation**: `inspect-dataset` now flags duplicate question ids, references to missing documents, qrels for unknown questions, `relevant_doc_ids` missing from qrels, and empty documents. Warnings are also embedded in `run_summary.json`.
- **PDF and reStructuredText document support** (`.pdf` requires `pip install 'ragbench[pdf]'`).
- Weighted RRF for `hybrid` and `hybrid_rerank` (`bm25_weight`, `vector_weight`).
- `parent_doc` child-score aggregation option (`parent_score_aggregation: max | sum | mean`).
- Five `paraphrase` questions in the demo dataset that avoid document vocabulary, so semantic retrieval can differentiate from lexical matching (now 50 questions across 11 categories).
- Configs `configs/hyde.yaml` and `configs/hybrid_rerank.yaml`; both new systems included in `configs/all.yaml`.

### Changed

- `report.html` redesigned: self-contained (no CDN), winner summary cards, sortable leaderboard, comparison bar charts, per-category quality, aggregated cost breakdown, light/dark mode.
- BM25 retrieval no longer pads results with chunks that share no term with the query.
- In-memory vector search uses partial sorting (`argpartition`) for top-k selection.

### Fixed

- `llm_heavy` rebuilt its reranker on every question; it is now constructed once at system init.
- `ragbench --version` previously failed with "Missing command"; the option is now eager and works standalone.

## [0.1.0] - 2026-05-09

### Added

- Evaluation-first RAG benchmark harness.
- BM25, vector, hybrid, rerank, parent-document, and LLM-heavy RAG systems.
- Chroma-backed vector search with in-memory fallback.
- Mock mode for embeddings, LLM answers, and answer judging.
- Retrieval metrics, answer judge, failure analysis, qrels audit, cost tracking, and reports.
- Demo fictional insurance technology dataset.
- Typer CLI with `demo`, `run`, `compare`, `evaluate`, `inspect-dataset`, and `list-systems`.

