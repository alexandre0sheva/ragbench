"""The config each hub dataset ships (`configs/<name>.yaml`): a small, honest first comparison of the data you just pulled."""

from __future__ import annotations

from ragbench.datasets.hub.registry import HubDataset

TEMPLATE = """# {title}: pull the data first with `ragbench datasets pull {name}` (add `--subset 300` for a cheaper run).
# Check the cost before a live run: `ragbench estimate --config configs/{config_name}.yaml`.
run:
  name: {config_name}
  output_dir: results

dataset:
  documents_path: data/{name}/docs
  questions_path: data/{name}/questions.jsonl
  qrels_path: data/{name}/qrels.jsonl

systems:
  - type: bm25
    name: bm25
    chunker:
      type: word
      chunk_size: 450
      chunk_overlap: 70
    retrieval:
      top_k: 5
    models:
      generator: gpt-6-luna
  - type: hybrid
    name: hybrid
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
    models:
      embedding: text-embedding-3-small
      generator: gpt-6-luna
  - type: hybrid_rerank
    name: hybrid_rerank
    chunker:
      type: word
      chunk_size: 500
      chunk_overlap: 80
    retrieval:
      vector_store: numpy
      bm25_top_k: 20
      vector_top_k: 20
      candidate_top_k: 30
      final_top_k: 5
      rrf_k: 60
      reranker: local_relevance
    models:
      embedding: text-embedding-3-small
      generator: gpt-6-luna

evaluation:
  k_values: [1, 3, 5, 10]
  judge_enabled: true
  judge_model: anthropic:claude-haiku-5-5   # another model family than the generators (self-preference); live: ANTHROPIC_API_KEY + ragbench[anthropic]
  max_questions: null
  max_workers: 4
"""


def render_config(entry: HubDataset) -> str:
    return TEMPLATE.format(title=entry.title, name=entry.name, config_name=entry.config_name)
