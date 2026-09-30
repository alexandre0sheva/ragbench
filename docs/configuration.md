# Configuration Guide

RAGBench experiments are YAML files. Four sections describe the experiment:

- `run`
- `dataset`
- `systems`
- `evaluation`

Three optional sections tune how it executes: [`pricing`](#pricing), [`cache`](#caching) and [`limits`](#rate-limits).

## Minimal Example

```yaml
run:
  name: my_rag_eval
  output_dir: results

dataset:
  documents_path: data/demo/docs
  questions_path: data/demo/questions.jsonl
  qrels_path: data/demo/qrels.jsonl

systems:
  - type: hybrid
    name: hybrid_default
    chunker:
      type: token
      chunk_size: 500
      chunk_overlap: 80
    retrieval:
      vector_store: chroma
      bm25_top_k: 20
      vector_top_k: 20
      final_top_k: 5
      rrf_k: 60
      multi_query: true
      max_query_variants: 4
    models:
      embedding: text-embedding-3-small
      generator: gpt-5.4-nano

evaluation:
  k_values: [1, 3, 5, 10]
  judge_enabled: true
  judge_model: gpt-5.4-nano
  max_questions: null
  max_workers: 4
  embedding_cache: true
  # system_workers: 1, ingest_workers: 4, latency_probe_questions: 5  (see Concurrency)
```

## Retrieval depth, context size, and failures

Retrieval is scored on a deep ranking, but the generator only reads the top of it:

| Key | Default | Meaning |
| --- | --- | --- |
| `evaluation.k_values` | `[1, 3, 5, 10]` | Cut-offs for Recall / Precision / Hit / MRR / nDCG |
| `evaluation.retrieval_depth` | `max(k_values)` | Ranked chunks every system returns and metrics are computed on. May not be smaller than `max(k_values)` |
| `evaluation.context_k` | `5` | Chunks handed to the LLM, for systems that do not set their own `top_k` / `final_top_k` (`top_k_parents` for `parent_doc`) — a system's own setting wins |
| `evaluation.primary_k` | `5` (or the median `k`) | Recall cut-off for headline columns, failure classification and the qrels audit; must be one of `k_values` |
| `evaluation.max_error_rate` | `0.2` | Share of a system's questions that may fail before the run is reported as failed |

Leaderboard columns follow your `k_values`: recall at `primary_k`, MRR and nDCG at the largest `k`. A value that could not be measured shows as `—`, never as `0`.

A question that raises (provider outage, bad response, bug) no longer aborts the run. It is recorded in `per_question_results.jsonl` with an `error` field and `failure_type: run_error`, excluded from every mean, and counted in `n_error` / `run_summary.json` (`num_errors`, `errors_by_system`). A system whose ingestion fails has all of its questions marked this way while the other systems finish. All results are written first; then, if any system's failure share exceeds `max_error_rate`, `ragbench` prints the error and exits with status 1.

Every run also writes `run_manifest.json`: RAGBench and dependency versions, Python/platform, git commit (and whether the tree was dirty), a hash of the config and of the dataset (documents + questions + qrels), models used, mode, and start/finish times.

## System options

Each system's `retrieval:`, `chunker:` and (for `llm_heavy`) `llm_features:` options, with types, defaults and descriptions, are listed in [systems.md](systems.md). Unknown options, unknown system types, unknown chunker or reranker names, and duplicate system names are rejected when the config is loaded, with a suggestion when a name is close to a valid one.

## Caching

Two layers avoid paying twice for the same call:

- **In process** (`evaluation.embedding_cache`, default `true`): the systems of one run embed the same corpus, so they share the embeddings instead of paying for them once each. Set `false` to disable.
- **On disk** (`cache:` section): LLM responses and corpus embeddings are kept in `.ragbench_cache/cache.sqlite3` between runs, so repeating an unchanged benchmark, fixing one system, or resuming after a crash costs almost nothing. Only **live** runs use it; mock runs have nothing paid to save.

```yaml
cache:
  enabled: true                 # `--no-cache` switches it off for one run
  dir: .ragbench_cache          # or set RAGBENCH_CACHE_DIR
  llm: true                     # generation, query rewriting, reranking and judging, at temperature 0
  embeddings: true              # corpus embeddings
  cache_query_embeddings: false # see below
  llm_nonzero_temperature: false
  ttl_days: null                # ignore entries older than this
```

An entry is keyed by everything that determines the answer (provider, model, messages, parameters, text), so any change to a prompt, model, or document is a miss. Requests with a temperature above 0 are not cached unless `llm_nonzero_temperature` is on, and errors are never cached.

Caching does not change what a system appears to cost or how fast it appears to be:

- **Cost stays standalone.** A hit is charged to the requesting system at the *current* price, as if the call had been made, so per-system `$/Q` and ingestion cost reflect what that system would cost deployed alone. The spend actually avoided is reported separately: `run_summary.json` has a `cache` block (hits, misses, hit rate, `saved_cost_usd`, `charged_cost_usd`, `real_spend_usd`) and the console prints it.
- **Latency stays honest.** A cached LLM response replays the latency of the original call (the step is marked `cached: true`). Query embeddings are not cached by default; with `cache_query_embeddings: true` their measured latency becomes unrealistically low.

```bash
ragbench cache stats                      # what is stored, per namespace
ragbench cache clear --namespace llm      # or everything; asks first unless --yes
ragbench run --config my.yaml --no-cache  # one run without reading or writing it
```

A cache file that is corrupted or cannot be opened is ignored with a warning rather than failing the run.

## Mock Mode

RAGBench automatically uses mock mode when `OPENAI_API_KEY` is not available. You can force mock mode even when a key is configured:

```bash
ragbench compare --config configs/all.yaml --mock
```

The mode is recorded as `mode: "mock" | "live"` in `run_summary.json` (next to `models_used`), and mock runs carry a banner at the top of `leaderboard.md` and `report.html`. If a key **is** set but the OpenAI client cannot be created, RAGBench raises an error instead of quietly switching to mock scores.

## Pricing

Costs come from the price table in `src/ragbench/models/cost.py` (USD per 1M tokens, reviewed as of `PRICING_AS_OF`). Dated snapshots such as `gpt-4o-mini-2024-07-18` use the price of their base model. Override or extend the table per experiment:

```yaml
pricing:
  my-finetuned-model: {input: 0.30, output: 1.20}
  gpt-5.4-nano: {input: 0.25, output: 1.50}   # overrides the built-in price
```

A model with no registered price is billed at $0 — RAGBench logs a warning, lists the model under `unknown_priced_models` in `run_summary.json`, and shows a "cost under-reported" banner in the reports, so a zero is never silent.

## Retries and timeouts

OpenAI calls (chat and embeddings) time out after 120 s and are retried up to 6 attempts with exponential backoff (1 s doubling to a 30 s cap, plus jitter, and honoring `Retry-After`) on rate limits, 5xx responses, timeouts, and connection errors. Auth and bad-request errors fail immediately. The Responses API is used only for models that cannot be called through Chat Completions, never as a generic fallback. Models that reject `temperature` (for example the `o`-series) are called without it; other models that reject it are detected from the API error and remembered for the rest of the run.

## Concurrency

Four independent knobs spread a run over threads (work is I/O-bound, so threads help a lot on live runs):

| Key | Default | Runs at the same time |
| --- | --- | --- |
| `evaluation.system_workers` | `1` | Systems (each still ingests exactly once) |
| `evaluation.max_workers` | `4` | Questions within a system |
| `evaluation.ingest_workers` | `4` | LLM enrichment calls (`llm_heavy`) and embedding batches while a system ingests |
| `limits:` | none | Caps on provider requests, applied across all of the above (see below) |

Peak concurrency is roughly `system_workers × max_workers` requests, plus `system_workers × ingest_workers` during ingestion. Results do not depend on any of these: rows, scores and order are identical to a sequential run (only timings differ). Override from the CLI with `--system-workers` and `--max-workers`:

```bash
ragbench compare --config configs/all.yaml --system-workers 4 --max-workers 4
```

Mock runs are CPU-bound and gain little from threads; live runs, which wait on the network, speed up roughly in proportion to the workers until the provider's limits bind.

### Rate limits

A `limits:` section throttles requests to the provider (OpenAI for now) across every thread. Each retry attempt counts as a request, cache hits do not, and unset limits are not enforced:

```yaml
limits:
  max_concurrent_requests: 8   # requests in flight at once
  requests_per_minute: 500     # started per rolling minute
  tokens_per_minute: 200000    # estimated prompt tokens per rolling minute
```

Waiting for a slot never holds up other threads, and a single request larger than the whole token budget is let through on its own rather than waiting forever. Rate-limit responses (HTTP 429) are still retried with backoff (see [Retries and timeouts](#retries-and-timeouts)), so limits are an optimization for staying under quota, not a requirement.

### Latency measurement

Latency measured while many questions run at once includes time spent queueing behind other requests. So **live** runs finish with a *latency probe*: `evaluation.latency_probe_questions` (default `5`; `0` disables it) questions per system, spread evenly across the dataset, are re-asked one at a time after everything else is done, with the disk cache bypassed. The leaderboard's Latency column then shows the probe's mean, `p95` is added, and `metrics_summary.csv` gains `latency_ms_p50`, `latency_ms_p95`, `latency_source` (`probe` or `concurrent`) and `avg_latency_concurrent_ms`. Probe answers are not scored and are not charged to any system; their cost appears as `probe_cost_usd` in `run_summary.json` and is included in `real_spend_usd`. Mock runs skip the probe, and if you disable it in a live run the reports say latency was measured under concurrency.

## Vector Store

Chroma is the default vector backend for vector-capable systems:

```yaml
retrieval:
  vector_store: chroma
```

If Chroma is unavailable, RAGBench falls back to local in-memory NumPy similarity.

