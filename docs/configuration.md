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
      type: word
      chunk_size: 500
      chunk_overlap: 80
    retrieval:
      vector_store: numpy
      bm25_top_k: 20
      vector_top_k: 20
      final_top_k: 5
      rrf_k: 60
      multi_query: true
      max_query_variants: 4
    models:
      embedding: text-embedding-3-small
      generator: gpt-6-luna

evaluation:
  k_values: [1, 3, 5, 10]
  judge_enabled: true
  judge_model: gpt-6-luna      # shorthand for judge.model (see Judge)
  max_questions: null
  max_workers: 4
  embedding_cache: true
  # system_workers: 1, ingest_workers: 4, latency_probe_questions: 5  (see Concurrency)
  # judge: {...} (see Judge), stats: {n_boot: 2000, seed: 0, baseline: null} (see Statistics)
# selection: {profile: balanced, constraints: {...}} (see Selection)
# max_cost_usd: 25, cost_confirm_threshold_usd: 1.0  (see Estimating cost and capping it)
```

## Chunkers

How documents are cut into chunks is a comparison axis of its own. Every system except `parent_doc` has a `chunker:` section; `type` picks the chunker and the common fields tune it (full option table: [systems.md](systems.md)).

| `type` | Splits on | `chunk_size` / `chunk_overlap` unit | Notes |
| --- | --- | --- | --- |
| `token` (default) | windows of real model tokens (tiktoken `o200k_base`) | tokens | Accurate for code, CJK and punctuation-heavy text |
| `word` | windows of whitespace-separated words | words | What `token` meant before 0.3.0; all shipped configs use it so results stay comparable with 0.2.0 |
| `fixed_char` | character windows | characters | Default 1200 / 150 |
| `recursive` | paragraphs, then lines, sentences, words, then a hard split | tokens | Keeps paragraphs and sentences whole whenever they fit |
| `sentence` | sentences, packed up to the size | size in tokens, overlap in **sentences** (default 1) | A sentence longer than the size is split by tokens |
| `semantic` | topic shifts between consecutive sentences | size in tokens, overlap in sentences (default 0) | Embeds every sentence with the system's embedding model (billed as ingestion cost; free under `--mock`) and breaks where the distance between neighbours exceeds `breakpoint_percentile` (default 90) |
| `markdown` | headings (`#` to `######`, not inside code fences) | tokens | One chunk per section with `metadata["heading_path"]`; sections under `min_chunk_size` (default 50) are merged into the next one while they fit; long sections are split recursively |

Common fields: `chunk_size`, `chunk_overlap`, `prefix_title` (prepend the document title to every chunk's text, so it is embedded and searched) and, for the chunkers that use them, `min_chunk_size` (`recursive`, `sentence`, `semantic`, `markdown`), `prefix_heading` (`markdown`: prepend `Guide > Returns` to the text) and `breakpoint_percentile` (`semantic`). A field the chosen chunker does not use is rejected when the config loads.

```yaml
chunker:
  type: markdown
  chunk_size: 400
  chunk_overlap: 40
  min_chunk_size: 60
  prefix_heading: true
```

**Preview a chunker before paying for a benchmark:**

```bash
ragbench chunk-preview --docs data/demo/docs --chunker '{type: markdown, chunk_size: 300}'
```

It prints the number of chunks, mean/median/p95/max tokens, the share of chunks below `min_chunk_size` (50 tokens when unset) and the first chunks (`--limit`, `--doc DOC_ID`, `--json`; `--mock` uses hashing embeddings for `semantic`). If the real tokenizer's vocabulary cannot be downloaded (no network), token-based chunkers fall back to a deterministic approximation and log a warning; chunk metadata records the `tokenizer` used.

Chunk sizes change what is retrieved, so results are comparable only between runs that used the same chunker settings.

## Retrieval depth, context size, and failures

Retrieval is scored on a deep ranking, but the generator only reads the top of it:

| Key | Default | Meaning |
| --- | --- | --- |
| `evaluation.k_values` | `[1, 3, 5, 10]` | Cut-offs for Recall / Precision / Hit / MRR / nDCG |
| `evaluation.retrieval_depth` | `max(k_values)` | Ranked chunks every system returns and metrics are computed on. May not be smaller than `max(k_values)` |
| `evaluation.context_k` | `5` | Chunks handed to the LLM, for systems that do not set their own `top_k` / `final_top_k` (`top_k_parents` for `parent_doc`) — a system's own setting wins |
| `evaluation.primary_k` | `5` (or the median `k`) | Recall cut-off for headline columns, failure classification and the qrels audit; must be one of `k_values` |
| `evaluation.max_error_rate` | `0.2` | Share of a system's questions that may fail before the run is reported as failed |

Leaderboard columns follow your `k_values`: recall at `primary_k`, MRR and nDCG at the largest `k`. A value that could not be measured shows as `—`, never as `0`. The `Ctx recall` column (`context_recall`) scores only the first `context_k` chunks, the ones the generator actually read; see [methodology.md](methodology.md#context-metrics).

A question that raises (provider outage, bad response, bug) no longer aborts the run. It is recorded in `per_question_results.jsonl` with an `error` field and `failure_type: run_error`, excluded from every mean, and counted in `n_error` / `run_summary.json` (`num_errors`, `errors_by_system`). A system whose ingestion fails has all of its questions marked this way while the other systems finish. All results are written first; then, if any system's failure share exceeds `max_error_rate`, `ragbench` prints the error and exits with status 1.

Every run also writes `run_manifest.json`: RAGBench and dependency versions, Python/platform, git commit (and whether the tree was dirty), a hash of the config and of the dataset (documents + questions + qrels), models used, mode, and start/finish times.

## Judge

`judge_enabled` turns the LLM judge on or off (a heuristic judge scores mock runs and `judge_enabled: false`). The `judge:` section tunes it; how answers are scored, and where the judge can mislead you, is in [methodology.md](methodology.md#the-llm-judge).

```yaml
evaluation:
  judge_enabled: true
  judge:
    model: anthropic:claude-haiku-4-5   # a model ref; default: evaluation.judge_model
    samples: 3                          # independent judgments per question, averaged (default 1)
    temperature: 0.5                    # judge sampling temperature (default 0); must be above 0 when samples > 1
    independent: true                   # warn in the reports when the judge model is also a generator (default true)
```

`evaluation.judge_model` keeps working as a shorthand for `judge.model`; set only one of them (two different models are an error). The two always agree after loading. Choose a judge from a different model family than your generators: when it is the same model as a system's generator, the leaderboard and `report.html` warn about self-preference, and `independent: false` silences that on purpose. Every sample is a paid call, and temperature-0 judgments are cached on disk like any other LLM call; sampled ones are not unless `cache.llm_nonzero_temperature` is on.

## Statistics

`evaluation.stats` controls the bootstrap behind the leaderboard's confidence intervals and `significance.csv`; what they mean is in [methodology.md](methodology.md#statistics).

```yaml
evaluation:
  stats:
    n_boot: 2000     # bootstrap resamples (at least 100)
    seed: 0          # same results + seed = same intervals and p-values
    baseline: bm25   # system the others are compared with; default: the cheapest ($/Q). Must be a system name from this config
```

## Selection

`selection:` decides what each run recommends deploying (`recommendation.md`, `recommendation.json` and `winner.yaml` in the run directory; how it works is in [methodology.md](methodology.md#selection)). Everything is optional; by default the `balanced` profile and no constraints.

```yaml
selection:
  profile: balanced          # balanced | max_quality | cheapest_acceptable | lowest_latency
  constraints:               # hard limits: a system that breaks one is never recommended
    max_cost_per_question: 0.002     # dollars, mean $/Q
    max_latency_ms_p95: 2500
    min_faithfulness: 4.5            # 0-5
    min_answer_score: 3.5            # 0-5
    max_ingestion_cost: 5.0          # dollars to index the corpus once
    require_local_models: false      # every model runs on this machine (`local:` or a localhost endpoint)
    require_no_network: false        # ... and no tool uses the network
  weights: {quality: 0.6, cost: 0.2, latency: 0.2}   # replaces the profile's weights
```

`ragbench recommend --run results/<run>` re-asks the question of a finished run, starting from its own `selection:` settings; `--profile`, `--max-cost`, `--max-latency`, `--min-faithfulness`, `--min-answer-score`, `--max-ingestion-cost`, `--local-models` and `--no-network` override them, and `--export winner.yaml` writes the winner's runnable config (see [cli.md](cli.md)). It exits with status 1 when no system qualifies. An unknown profile or constraint name is an error when the config loads.

## Sweeps

To explore an axis (chunk size × reranker × tool set) without writing a block per combination, give one system a `sweep:`. It becomes one system per combination, named `base[axis=value,...]`:

```yaml
systems:
  - type: hybrid_rerank
    name: hr                              # the base name
    sweep:
      chunker.chunk_size: [300, 500]
      retrieval.reranker: [local_relevance, cross_encoder]
      tools: [[], [calculator]]           # a tool set is a value like any other (agent_search only)
```

- **Axes** are `chunker.<option>`, `retrieval.<option>`, `llm_features.<option>`, `models.generator`, `models.embedding` and `tools`: the options listed for the system in [systems.md](systems.md). Anything else is an error that lists the valid axes and suggests the closest. Nested paths into an option's value (`retrieval.routes.default...`) are not supported.
- **Names** are fixed by the config: the first axis varies slowest, values are written compactly (`chunk_size=300`, `tools=calculator+corpus_grep`, `tools=none`), and an axis is labelled by its last path segment (the whole path when two axes would end alike). The example above yields `hr[chunk_size=300,reranker=local_relevance]`, `hr[chunk_size=300,reranker=cross_encoder]`, and so on, four systems in all.
- Every variant is validated like a hand-written system, a duplicate name is an error, and one sweep may not expand to more than 100 systems.
- The run directory's `config.yaml` holds the **expanded** systems (a config without sweeps is copied verbatim, comments included), so `ragbench recommend` and `winner.yaml` work on variant names.

`--systems hr,bm25` (or repeated `--systems`) runs only some of the systems of a config, after sweeps are expanded; a sweep's base name (`hr`) selects all of its variants, and a full variant name selects one.

## Presets

`--preset` replaces a config's systems with a ready-made list, so a first comparison needs no YAML at all:

```bash
ragbench compare --preset quick --docs my_dataset/docs --questions my_dataset/questions.jsonl
ragbench compare --preset thorough --config my.yaml      # keep my dataset, models, judge, ... and swap the systems
```

| Preset | Systems |
| --- | --- |
| `quick` | `bm25`, `vector`, `hybrid_rerank` |
| `standard` | `quick` + `hybrid`, `rerank`, `parent_doc`, `hyde`, `contextual` |
| `thorough` | `standard` + `hierarchical`, `sentence_window`, `rag_fusion`, `decompose`, `llm_heavy` and a chunk-size sweep of `hybrid_rerank` (250 and 1000 words) |
| `agentic` | `hybrid_rerank` as the reference, `corrective`, `iterative`, `agent_search` with and without tools, `grep_agent`, `adaptive` |

The dataset comes from `--docs` / `--questions` / `--qrels` or from the `--config` you pass (flags win); without either the command stops and says so. A preset uses default models and options, so pair it with a config when you need other models. The assembled config is saved as the run's `config.yaml`. `--preset` works with `run`, `compare`, `evaluate` and `estimate`.

## Estimating cost and capping it

```bash
ragbench estimate --config configs/all.yaml          # a table: ingestion, query and judge cost per system, and the total
ragbench estimate --preset standard --docs docs/ --questions q.jsonl
```

`estimate` runs every system on the offline mock models (indexing the whole corpus and answering a sample of your questions) and counts what they send: prompt sizes, retrieved context, number of model calls, embedding volume. Those tokens are then priced with the configured models' rates ([Pricing](#pricing)), and judging is added from the real judge prompt (`evaluation.judge.samples` times). It never calls a paid model, and it takes a few seconds. It is **not** a guarantee:

- Prompts, call counts and embedding volume are measured; against a mock run priced as if live it lands within a few percent (the test holds it to ±35%). What it cannot know is how long a real model's answers are (the mock's are short, so output cost is probably low) and how many steps a real agent takes (agentic systems are marked `~`; their `max_steps` and `max_cost_usd` bound the real spend).
- It assumes no cache hits, so a re-run with a warm [cache](#caching) costs less. Time is a rough projection from typical API latencies and your concurrency settings.
- It prints the date of the price table and warns when it is more than 90 days old, and when a configured model has no price (its cost would be counted as $0).

Two settings turn the estimate into guard rails:

```yaml
evaluation:
  cost_confirm_threshold_usd: 1.0   # a live run estimated above this asks first (default 1.00)
  max_cost_usd: 25                  # hard cap on what the run may be charged (default: none)
```

- **Confirmation.** Before a live run (never a mock one) `run` / `compare` print the estimate. Above the threshold they ask `Run it for about $X?`, unless you pass `--yes`; with no terminal to ask (a pipe, a cron job) or with `CI` set they refuse and tell you to pass `--yes`. If the estimate itself fails, the run goes ahead with a warning.
- **Budget cap.** `max_cost_usd` counts what the run is charged (ingestion, answers and judging, at standalone prices, so it is an upper bound of real spend when the cache is warm). Once it is reached no new question or system starts; questions already running finish, so the total can end up over the cap by about one question's cost per worker. The run then writes the results of every system that **finished**, saves the answers of unfinished systems in `per_question_partial.jsonl`, leaves them out of the leaderboard and the recommendation (a mean over some questions cannot be compared with a mean over all of them), records the outcome under `budget` in `run_summary.json`, prints which systems were and were not finished, and exits with status 1. Raise the cap, run fewer systems (`--systems`) or questions, or estimate first.

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

RAGBench runs in mock mode (deterministic hashing embeddings, mock LLM, heuristic judge) when none of the models a config uses can be reached: no `OPENAI_API_KEY` for OpenAI refs, no `ANTHROPIC_API_KEY` for Claude refs, and no `providers:` endpoint or `local:` model in the config. You can force mock mode even when keys are set:

```bash
ragbench compare --config configs/all.yaml --mock
```

The mode is recorded as `mode: "mock" | "live"` in `run_summary.json` (next to `models_used`), and mock runs carry a banner at the top of `leaderboard.md` and `report.html`. A live run never quietly mixes real and mock models: if some model it uses has no key (say a Claude generator with the default OpenAI judge and no `OPENAI_API_KEY`), it stops before spending anything and lists what is missing. If a key **is** set but a client cannot be created, RAGBench raises an error instead of switching to mock scores.

## Providers & model refs

Every model setting (`models.generator`, `models.embedding`, `evaluation.judge_model`) is a **model ref**: `provider:model`. A bare name is an OpenAI model, so `gpt-6-luna` and `openai:gpt-6-luna` are the same.

| Ref | Serves | Needs |
| --- | --- | --- |
| `gpt-6-luna`, `openai:gpt-6.1-sol`, `openai:text-embedding-3-small` | chat, embeddings | `OPENAI_API_KEY` |
| `anthropic:claude-haiku-4-5`, `anthropic:claude-sonnet-5-5` | chat | `ANTHROPIC_API_KEY` and `pip install 'ragbench[anthropic]'` |
| `openai_compatible:<endpoint>/<model>` | chat, embeddings | an entry under `providers:` (below) |
| `local:BAAI/bge-small-en-v1.5` | embeddings | `pip install 'ragbench[local]'`; downloads the Hugging Face model on first use |

```yaml
providers:                        # named OpenAI-compatible endpoints: Ollama, vLLM, LM Studio, OpenRouter, Together, ...
  ollama:
    base_url: http://localhost:11434/v1
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY     # omit for servers that need no key
    limits: {max_concurrent_requests: 4, requests_per_minute: 120}   # this endpoint only

systems:
  - type: vector
    models:
      generator: anthropic:claude-haiku-4-5
      embedding: local:BAAI/bge-small-en-v1.5
  - type: hybrid
    models:
      generator: openai_compatible:ollama/llama3.1:8b          # the model name may contain `/` and `:`
      embedding: openai_compatible:ollama/nomic-embed-text
evaluation:
  judge_model: openai_compatible:openrouter/anthropic/claude-sonnet-5-5
```

- Refs are checked when the config loads: an unknown endpoint, a chat model used as an embedder (or the reverse), or a mistyped provider (`anthropics:`) fails with the list of valid choices.
- **Endpoints use `max_tokens`**, the parameter nearly every OpenAI-compatible server understands; OpenAI itself gets `max_completion_tokens`.
- **Claude models** are called through the native Messages API. They take no sampling parameters, so `temperature` is not sent (older models that still honour it, such as Haiku 4.5, get it); Claude has no JSON mode, so judge calls ask for JSON in the system prompt and RAGBench strips any code fence from the reply. Models that think by default are called at low effort with extra `max_tokens` headroom, because thinking tokens count toward the limit.
- **GPT-6 models** are sent `reasoning_effort: none` (their default is to reason, which would spend short calls' token budgets on hidden thinking and disables function calling on Chat Completions); an endpoint that rejects the parameter is detected and it is left out.
- **Local embeddings** use [sentence-transformers](https://www.sbert.net) (chosen over fastembed because it runs any Hugging Face embedding model and ships wheels for Python 3.11–3.14 on macOS and Linux). Vectors are normalized and cost $0; results are cached on disk like any other embedding.
- **Cost:** OpenAI and Claude prices come from the built-in table. `local:` and `openai_compatible:` models cost **$0 unless you add a price** under `pricing:`, keyed by the full ref, `<endpoint>/<model>`, or the bare model name — without one, a paid proxy such as OpenRouter is reported as free.
- **Tool calling** (used by the agentic systems) is part of the common interface: `LLM.generate(messages, tools=[...])` takes OpenAI function schemas and returns `LLMResult.tool_calls`, and each provider translates to its own format.
- Keys are read from the environment or the project's `.env`; see `.env.example`.

## Pricing

Costs come from the price table in `src/ragbench/models/cost.py` (USD per 1M tokens, reviewed as of `PRICING_AS_OF`, with the source pages listed above the table). Dated snapshots such as `claude-haiku-4-5-20251001` use the price of their base model; `local:` and `openai_compatible:` models are $0 unless priced here (see [Providers & model refs](#providers--model-refs)). Override or extend the table per experiment:

```yaml
pricing:
  my-finetuned-model: {input: 0.30, output: 1.20}
  gpt-6-luna: {input: 0.25, output: 1.50}   # overrides the built-in price
```

A model with no registered price is billed at $0 — RAGBench logs a warning, lists the model under `unknown_priced_models` in `run_summary.json`, and shows a "cost under-reported" banner in the reports, so a zero is never silent.

## Retries and timeouts

Calls to every provider (chat and embeddings; OpenAI, Claude and `providers:` endpoints) time out after 120 s and are retried up to 6 attempts with exponential backoff (1 s doubling to a 30 s cap, plus jitter, and honoring `Retry-After`) on rate limits, 5xx responses, timeouts, and connection errors. Auth and bad-request errors fail immediately. The Responses API is used only for models that cannot be called through Chat Completions, never as a generic fallback. Models that reject `temperature` (for example the `o`-series) are called without it; other models that reject it are detected from the API error and remembered for the rest of the run.

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

A `limits:` section throttles requests across every thread. It applies to each hosted API in use (OpenAI and Anthropic) with a separate limiter apiece, since their quotas are independent; every `providers:` endpoint sets its own `limits:` instead. Each retry attempt counts as a request, cache hits do not, and unset limits are not enforced:

```yaml
limits:
  max_concurrent_requests: 8   # requests in flight at once
  requests_per_minute: 500     # started per rolling minute
  tokens_per_minute: 200000    # estimated prompt tokens per rolling minute
```

Waiting for a slot never holds up other threads, and a single request larger than the whole token budget is let through on its own rather than waiting forever. Rate-limit responses (HTTP 429) are still retried with backoff (see [Retries and timeouts](#retries-and-timeouts)), so limits are an optimization for staying under quota, not a requirement.

### Latency measurement

Latency measured while many questions run at once includes time spent queueing behind other requests. So **live** runs finish with a *latency probe*: `evaluation.latency_probe_questions` (default `5`; `0` disables it) questions per system, spread evenly across the dataset, are re-asked one at a time after everything else is done, with the disk cache bypassed. The leaderboard's Latency column then shows the probe's mean, `p95` is added, and `metrics_summary.csv` gains `latency_ms_p50`, `latency_ms_p95`, `latency_source` (`probe` or `concurrent`) and `avg_latency_concurrent_ms`. Probe answers are not scored and are not charged to any system; their cost appears as `probe_cost_usd` in `run_summary.json` and is included in `real_spend_usd`. Mock runs skip the probe, and if you disable it in a live run the reports say latency was measured under concurrency.

## Tools

Systems that call tools take a `tools:` list in their own block (built-in names, `{name, ...options}` mappings, or `{name, path: "pkg.module:function"}` for your own functions). A top-level `tools:` section sets the rules every call runs under: `allow` and `allow_network` (permissions for tools with side effects), `timeout_s` and `max_output_chars`. `evaluation.tools_now` fixes what "today" means for `date_calc`. The keys, the safety model and the tool metrics are described in [tools.md](tools.md).

## Adaptive routing

The `adaptive` system sends each question to one of several pipelines you define inline under `retrieval.routes`; every route is a complete system config (`type`, `chunker`, `retrieval`, `tools`, ...) and inherits the adaptive system's `models` unless it sets its own:

```yaml
- type: adaptive
  name: router
  models: {embedding: text-embedding-3-small, generator: gpt-6-luna}
  retrieval:
    router: heuristic            # or `llm`: the generator model picks the route, and you can name routes freely
    routes:
      default:     {type: hybrid_rerank}
      lexical:     {type: bm25}                                   # exact identifiers, error codes, quoted phrases
      multi_hop:   {type: decompose}                              # comparisons and questions with several linked parts
      computation: {type: agent_search, tools: [calculator, date_calc]}   # arithmetic, percentages, date calculations
```

`default` is required; with `router: heuristic` the other names must be `lexical`, `computation` or `multi_hop`, and a role you leave out falls back to `default`. Routes, their types, options, tools and model refs are all checked when the config loads, not when question 37 arrives. Every route is ingested once, and routes with the same chunker and embedding model share corpus embeddings through the run's embedding cache (each route is still charged at standalone prices). The router's choice is a `route` step in the trace, the chosen pipeline's steps follow it tagged with the route, and costs add up exactly. Options: [systems.md](systems.md#adaptive); output: `routes.csv` and, with `routing_hint` on questions ([dataset-format.md](dataset-format.md)), `route_accuracy`.

## Vector Store

`retrieval.vector_store` selects the backend of every system that searches embeddings (`vector`, `hybrid`, `hybrid_rerank`, `rerank`, `parent_doc`, `hyde`, `llm_heavy`). The backend is a comparison axis like the chunker or the reranker:

| `vector_store` | Search | Install | Notes |
| --- | --- | --- | --- |
| `numpy` (default) | exact | nothing | Brute-force cosine; plenty for up to ~100k chunks, deterministic across runs and processes |
| `faiss` | exact | `pip install 'ragbench[faiss]'` | Flat inner-product index |
| `faiss_hnsw` | approximate | `ragbench[faiss]` | HNSW graph (M=32, efConstruction=200, efSearch=128) |
| `chroma` | approximate (HNSW) | `ragbench[chroma]` | Can persist with `persist_directory`; its index is not deterministic across processes (ties in the tail of a ranking can shuffle) |
| `qdrant` | exact | `ragbench[qdrant]` | Qdrant's embedded local mode, in memory or on disk with `persist_directory`; local mode locks its directory, so give each system its own |

```yaml
retrieval:
  vector_store: faiss_hnsw
```

- **There is no silent fallback.** Asking for a backend whose library is not installed fails when the system is built, with the `pip install` to run. `RetrievalResult.metadata["vector_backend"]` (and so `per_question_results.jsonl`) always names the backend that actually ran.
- `in_memory` is a deprecated alias of `numpy` (it logs a warning once). Chroma used to be the default and a mandatory dependency; configs that say `vector_store: chroma` keep working once `ragbench[chroma]` is installed.
- `persist_directory` only applies to `chroma` and `qdrant`; setting it with another backend is a config error.
- FAISS publishes no macOS wheel for Python 3.14 yet (Linux and Python 3.11-3.13 are fine).
- Backends are pluggable: register a class with `@VECTOR_BACKENDS.register("name")` or an entry point in the `ragbench.vector_backends` group (see [extending.md](extending.md)).
