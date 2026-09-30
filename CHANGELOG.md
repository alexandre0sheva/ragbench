# Changelog

All notable changes to RAGBench will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning once it reaches public releases.

## [Unreleased]

### Added

- `pricing:` config section to add or override per-model prices; dated model snapshots now resolve to their base model's price.
- Retry with exponential backoff, jitter, `Retry-After` support and a 120 s timeout for OpenAI chat and embedding calls, with an error taxonomy (`RateLimitError`, `TransientModelError`, `PermanentModelError`, `ModelInitError`).
- `run_summary.json` now records `mode` (`mock`/`live`), `models_used`, `unknown_priced_models` and `pricing_as_of`; reports show a banner for mock runs and for models with no registered price.
- `evaluation.context_k`, `retrieval_depth`, `primary_k` and `max_error_rate` options; `run_manifest.json` (versions, git commit, config and dataset hashes, models, mode, timings); `n_ok` / `n_error` columns in `metrics_summary.csv`; an `in_context` flag on every retrieved chunk in `per_question_results.jsonl`.
- A question that raises no longer aborts the run: it is recorded as a `run_error` row and excluded from all means. A system whose ingestion fails is marked failed while the other systems finish. The run exits non-zero (after writing all results) when a system exceeds `max_error_rate`.
- **Component registries** (`ragbench.registry`): systems, chunkers and rerankers register by name, unknown names fail with a did-you-mean suggestion, and third-party packages can add components through the `ragbench.systems` / `ragbench.chunkers` / `ragbench.rerankers` entry-point groups.
- **Typed system configs**: every `retrieval:`, `chunker:` and `llm_features:` section is validated against a Pydantic option model, so a typo such as `candidate_top` fails at load time with a suggestion instead of being ignored. Systems also declare a `SystemSpec` (title, summary, best-for, cost/latency profile).
- `docs/systems.md`, `docs/cli.md` and the README system table are generated from the code by `scripts/generate_docs.py`; CI and a test fail when they are stale.
- A golden mock-run snapshot (`tests/golden/mock_metrics_v3.json`, `python scripts/update_golden.py`) that guards refactors.
- **Step tracing**: every answer carries an ordered trace (`AnswerResult.steps`, `RetrievalResult.steps`) of what the system did: retrieve / rerank / LLM / generate steps with latency, tokens, cost, and short input/output previews. Steps are persisted as `steps` in `per_question_results.jsonl`, each question's stage costs appear as `stage_<kind>_cost` columns in `cost_breakdown.csv` (plus a new `tool_cost` field), and `leaderboard.md` gains "Cost by stage" and "Latency by stage" tables. Step costs always add up to the answer's cost; anything a system fails to trace shows up as an explicit `untracked` step.

- **Persistent disk cache** (`cache:` config section, `--no-cache`, `ragbench cache stats|clear`): live runs cache temperature-0 LLM responses (generation, rewriting, reranking, judging) and corpus embeddings in `.ragbench_cache/cache.sqlite3` (SQLite, WAL), so re-running an unchanged benchmark or resuming after a crash costs almost nothing. Cache hits are still charged to each system at the current standalone price, cached LLM calls replay their original latency, and `run_summary.json` gains a `cache` block with hits, avoided spend (`saved_cost_usd`) and `real_spend_usd`. Mock runs never use the disk cache. A corrupted cache file is ignored with a warning.
- **Parallel execution engine** (`ragbench.runtime`): `evaluation.system_workers` runs systems concurrently (each still ingests once), `evaluation.ingest_workers` parallelizes `llm_heavy`'s per-chunk LLM enrichment and embedding batches, and `--system-workers` sets the former from the CLI. Output rows, scores and their order are identical to a sequential run. On a simulated I/O-bound workload (8 systems, 24 questions, ~50 ms per call) wall time drops from 13.9 s to 0.74 s with `system_workers=8, max_workers=4`; mock runs are CPU-bound and gain little.
- **Rate limiting**: a `limits:` section (`max_concurrent_requests`, `requests_per_minute`, `tokens_per_minute`) throttles OpenAI requests across all threads; every retry attempt counts, cache hits do not.
- **Latency probe**: live runs re-ask `evaluation.latency_probe_questions` (default 5) questions per system one at a time with the cache off, after the parallel pass. The Latency column shows the clean mean, a `p95` column is added, and `metrics_summary.csv` gains `latency_ms_p50`, `latency_ms_p95`, `latency_source` and `avg_latency_concurrent_ms`. Probe spend is reported as `probe_cost_usd` (in `run_summary.json`'s new `execution` block) and included in `real_spend_usd`. Mock runs skip it.
- Progress callbacks are serialized, so listeners never need their own locking.

### Changed

- Embedding cost is now charged per text from one token estimate whether the text was a cache hit or had to be embedded. Before, the first system to embed a text was charged the provider's token count and later systems a tokenizer estimate, so a system's ingestion charge depended on run order (and, with parallel systems, on thread timing).
- Chroma index builds take a lock (systems may now be built on several threads).
- `evaluation.max_workers` must be at least 1 (0 used to be silently treated as 1).
- The in-process embedding cache is now the first tier in front of the disk cache (its semantics and `evaluation.embedding_cache` are unchanged); the HTML report header counts reuse from both tiers.
- **Invalid configs now fail at load time**: unknown options, unknown `type`/chunker/reranker names, unknown top-level system keys, `vector_store` values other than `chroma`/`in_memory`, and duplicate system names used to be silently ignored or merged. The CLI reports them as a short list instead of a traceback.
- Systems share chunker/embedder/vector-store construction through `rag_systems/components.py`; `ragbench list-systems` and the leaderboard's "Best For" column read each system's `SystemSpec` (no behavior change in retrieval or scores: verified against the golden snapshot).
- `rerank` now also accepts `top_k` as an alias for `final_top_k`, like the hybrid systems.
- **Retrieval metrics are now measured on a deeper ranking than the LLM context (breaking for comparisons with 0.2.0).** Systems used to return only `top_k` (5) chunks, so every `@10` metric (Recall@10, MRR@10, nDCG@10) silently equalled its `@5` value. Systems now return `max(k_values)` ranked chunks, and the generator still receives only its configured `top_k` (or `evaluation.context_k`). Numbers labelled `@10` are not comparable with 0.2.0 results; `@5` values are unchanged. `parent_doc` searches enough child chunks to fill the deeper parent ranking.
- Leaderboard columns (console, `leaderboard.md`, `report.html`) are generated from one shared helper and follow `k_values`/`primary_k` instead of hard-coding `@5`/`@10`; missing values show `—` instead of `0.000`. `leaderboard.md` headers are now the short names used elsewhere (`Answer`, `Faithful`, `$/Q`, `Latency`) and it gains an `Errors` column. Failure classification and the qrels audit use `primary_k` and only count chunks the generator actually saw.
- **Python 3.11+ is now required** (3.10 reaches end-of-life in October 2026 and blocks current numpy/pandas). CI covers 3.11-3.13 and runs 3.14 as best-effort.
- Dependency floors raised to current majors: openai 3, pandas 3, numpy 2.3, scikit-learn 1.9, pydantic 2.13, typer 0.27, rich 15, tiktoken 0.14, chromadb 1.5, pytest 9, ruff 0.16, mypy 2.3.
- Dependabot groups routine pip minor/patch bumps into a single PR; pre-commit hooks updated.
- `mypy` no longer pins `python_version` (numpy 2.5 type stubs need a 3.12+ parser); CI runs it on each supported interpreter.
- **A present `OPENAI_API_KEY` with a failing OpenAI client now raises `ModelInitError`** instead of silently producing mock scores (`strict=False` on `create_llm`/`create_embedding_model` restores the old fallback).
- The Responses API fallback is used only when the model cannot be called via Chat Completions, not after any error (which previously doubled billable requests). Chat requests send `max_completion_tokens`, and models that reject `temperature` are called without it.
- Models with no registered price log a warning and are flagged in reports rather than silently costing $0.

### Fixed

- Two runs started within the same second used the same output directory and overwrote each other (plausible now that cached re-runs are fast); run ids get a numeric suffix when the directory exists.
- Token estimation resolved `tiktoken.encoding_for_model` on every call and always failed for new model names, silently using a word-count heuristic; the encoding is now resolved once per model with an `o200k_base` fallback.
- Leftover "RAGLab" naming in `LICENSE` and `.env.example`; stale version and release date in `CITATION.cff`.

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

