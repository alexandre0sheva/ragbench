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
```

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

