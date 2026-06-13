# Configuration Guide

RAGBench experiments are YAML files with four top-level sections:

- `run`
- `dataset`
- `systems`
- `evaluation`

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
```

## System-Specific Retrieval Options

Every system accepts `top_k` (or `final_top_k`); these are the notable extras:

| System | Option | Default | Meaning |
| --- | --- | --- | --- |
| `hybrid`, `hybrid_rerank` | `bm25_weight` / `vector_weight` | 1.0 / 1.0 | Weighted RRF: raise one side to favor lexical or semantic evidence |
| `hybrid`, `hybrid_rerank` | `rrf_k` | 60 | RRF smoothing constant |
| `hybrid`, `hybrid_rerank`, `rerank` | `multi_query` + `max_query_variants` | false / 4 | Local (non-LLM) query variants for multi-hop questions |
| `hybrid_rerank`, `rerank` | `candidate_top_k` | 30 | Candidate pool size handed to the reranker |
| `hybrid_rerank`, `rerank`, `llm_heavy` | `reranker` | varies | `simple_keyword_overlap`, `local_relevance` (TF-IDF), or `llm` |
| `parent_doc` | `parent_score_aggregation` | `max` | How child scores roll up to parents: `max`, `sum` (rewards parents hit by several children), or `mean` |
| `parent_doc` | `top_k_children` / `top_k_parents` | 8 / 4 | Children searched / parents returned |
| `hyde` | `probe_top_k` | 2 × top_k | Results fetched for the hypothetical-document probe |
| `hyde` | `fuse_with_question` | true | Fuse the probe ranking with the raw-question ranking via RRF |
| `llm_heavy` | `llm_features.*` | — | `enable_llm_ingestion`, `enable_query_rewrite`, `enable_llm_rerank` |

## Embedding Cache

`evaluation.embedding_cache` (default `true`) shares corpus embeddings across systems within one run, so six systems do not pay the embedding bill six times. Cache hits are still charged to the requesting system at standalone prices — per-system cost stays comparable — and the real API savings appear in `run_summary.json` and the console summary. Query embeddings are never cached, so per-question latency is honest for every system. Set to `false` to disable entirely.

## Mock Mode

RAGBench automatically uses mock mode when `OPENAI_API_KEY` is not available. You can force mock mode even when a key is configured:

```bash
ragbench compare --config configs/all.yaml --mock
```

## Concurrency

`evaluation.max_workers` controls question-level parallelism within each system. Higher values can improve wall-clock time for live LLM runs but may hit provider rate limits.

```yaml
evaluation:
  max_workers: 4
```

Override it from the CLI:

```bash
ragbench compare --config configs/all.yaml --max-workers 8
```

## Vector Store

Chroma is the default vector backend for vector-capable systems:

```yaml
retrieval:
  vector_store: chroma
```

If Chroma is unavailable, RAGBench falls back to local in-memory NumPy similarity.

