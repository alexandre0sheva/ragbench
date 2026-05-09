# Changelog

All notable changes to RAGBench will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning once it reaches public releases.

## [0.1.0] - 2026-05-09

### Added

- Evaluation-first RAG benchmark harness.
- BM25, vector, hybrid, rerank, parent-document, and LLM-heavy RAG systems.
- Chroma-backed vector search with in-memory fallback.
- Mock mode for embeddings, LLM answers, and answer judging.
- Retrieval metrics, answer judge, failure analysis, qrels audit, cost tracking, and reports.
- Demo fictional insurance technology dataset.
- Typer CLI with `demo`, `run`, `compare`, `evaluate`, `inspect-dataset`, and `list-systems`.

