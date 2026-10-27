# Recommendation

**Deploy `hybrid_rerank_default`** (profile `balanced`, run `all_solutions_multihop`). Its config is in `winner.yaml`.

## Why

- `hybrid_rerank_default` is recommended under the `balanced` profile (quality first, then the best mix of cost and speed among systems that are equally good): answer score 3.73, $0.00109 per question, p95 latency 3826 ms.
- `hybrid_rerank_default` ties with `compression_llm_extract` and `llm_heavy_default` on answer score: they are statistically indistinguishable from the best (`llm_heavy_default`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ 0.05). Quality does not separate them, so it wins on cost and latency ($0.00109 per question against up to $0.00983; p95 latency 3826 ms against up to 8092 ms).
- hybrid_rerank_default vs llm_heavy_default (the best answer score): -0.11 answer score (95% CI -0.22 to -0.00, significant), 0.1× cost
- hybrid_rerank_default vs no_retrieval_floor (the cheapest feasible system): +1.82 answer score (95% CI +1.67 to +1.97, significant), 5.2× cost

## Ranking

The winner first, then the systems tied with it (best choice first), then the rest by weighted score. *Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; *tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.

| # | System | Score | Answer score | $/Q | p95 | Ingestion | Calls/Q | Tie | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `hybrid_rerank_default` | 0.91 | 3.73 | $0.00109 | 3826 ms | $0.03301 | 1.0 | tie | ✓ |
| 2 | `compression_llm_extract` | 0.84 | 3.72 | $0.00112 | 7614 ms | $0.03341 | 2.0 | tie |  |
| 3 | `llm_heavy_default` | 0.69 | 3.83 | $0.00983 | 8092 ms | $0.03301 | 2.0 | tie | ✓ |
| 4 | `bm25_default` | 0.91 | 3.62 | $0.00104 | 2298 ms | $0.00000 | 1.0 |  | ✓ |
| 5 | `rerank_default` | 0.91 | 3.65 | $0.00108 | 2784 ms | $0.03301 | 1.0 |  | ✓ |
| 6 | `contextual_default` | 0.89 | 3.60 | $0.00104 | 2771 ms | $0.97876 | 1.0 |  | ✓ |
| 7 | `vector_default` | 0.86 | 3.48 | $0.00106 | 2105 ms | $0.03301 | 1.0 |  | ✓ |
| 8 | `adaptive_default` | 0.86 | 3.57 | $0.00088 | 4143 ms | $0.10022 | 1.5 |  | ✓ |
| 9 | `hierarchical_default` | 0.86 | 3.45 | $0.00076 | 2309 ms | $0.14719 | 1.0 |  | ✓ |
| 10 | `vector_mmr` | 0.85 | 3.44 | $0.00102 | 2519 ms | $0.03301 | 1.0 |  | ✓ |
| 11 | `decision_gate_default` | 0.85 | 3.55 | $0.00181 | 3611 ms | $0.03301 | 2.6 |  | ✓ |
| 12 | `sentence_window_default` | 0.84 | 3.44 | $0.00051 | 3254 ms | $0.02827 | 1.0 |  | ✓ |
| 13 | `late_interaction_rerank` | 0.84 | 3.41 | $0.00076 | 2546 ms | $0.03341 | 1.0 |  |  |
| 14 | `decompose_default` | 0.84 | 3.61 | $0.00087 | 6084 ms | $0.03341 | 2.0 |  | ✓ |
| 15 | `raptor_collapsed` | 0.83 | 3.36 | $0.00076 | 2245 ms | $0.48246 | 1.0 |  | ✓ |
| 16 | `rag_fusion_default` | 0.83 | 3.58 | $0.00085 | 6168 ms | $0.03341 | 2.0 |  | ✓ |
| 17 | `parent_doc_default` | 0.82 | 3.40 | $0.00123 | 2905 ms | $0.03802 | 1.0 |  |  |
| 18 | `hyde_default` | 0.82 | 3.44 | $0.00111 | 4126 ms | $0.03301 | 2.0 |  |  |
| 19 | `graph_rag_local` | 0.79 | 3.26 | $0.00092 | 2431 ms | $2.01740 | 1.0 |  |  |
| 20 | `corrective_default` | 0.79 | 3.68 | $0.00100 | 9877 ms | $0.03341 | 2.8 |  | ✓ |
| 21 | `hybrid_default` | 0.77 | 3.25 | $0.00106 | 3255 ms | $0.03301 | 1.0 |  |  |
| 22 | `full_context_50k` | 0.72 | 3.41 | $0.00581 | 3329 ms | $0.00000 | 1.0 |  |  |
| 23 | `iterative_default` | 0.72 | 3.52 | $0.00123 | 10908 ms | $0.03341 | 3.9 |  |  |
| 24 | `agent_search_tools` | 0.66 | 3.53 | $0.00259 | 13071 ms | $0.03341 | 5.5 |  |  |
| 25 | `agent_search_plain` | 0.65 | 3.35 | $0.00226 | 10607 ms | $0.03341 | 5.3 |  |  |
| 26 | `grep_agent_default` | 0.65 | 3.30 | $0.00309 | 8792 ms | $0.00000 | 6.8 |  |  |
| 27 | `decision_gate_strict` | 0.58 | 2.60 | $0.00154 | 2374 ms | $0.03301 | 2.1 |  | ✓ |
| 28 | `langchain_default` | 0.57 | 2.85 | $0.00088 | 8353 ms | $0.00000 | 1.0 |  |  |
| 29 | `llamaindex_default` | 0.51 | 2.60 | $0.00176 | 5798 ms | $0.00000 | 3.6 |  |  |
| 30 | `haystack_default` | 0.47 | 2.71 | $0.00115 | 10978 ms | $0.00000 | 1.0 |  |  |
| 31 | `graph_rag_global` | 0.41 | 2.04 | $0.00061 | 3087 ms | $1.96643 | 2.0 |  |  |
| 32 | `no_retrieval_floor` | 0.40 | 1.90 | $0.00021 | 1754 ms | $0.00000 | 1.0 |  | ✓ |

## Winner by category

The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.

| Category | Winner | `hybrid_rerank_default` | `compression_llm_extract` | `llm_heavy_default` | `bm25_default` | `rerank_default` | `contextual_default` | `vector_default` | `adaptive_default` | `hierarchical_default` | `vector_mmr` | `decision_gate_default` | `sentence_window_default` | `late_interaction_rerank` | `decompose_default` | `raptor_collapsed` | `rag_fusion_default` | `parent_doc_default` | `hyde_default` | `graph_rag_local` | `corrective_default` | `hybrid_default` | `full_context_50k` | `iterative_default` | `agent_search_tools` | `agent_search_plain` | `grep_agent_default` | `decision_gate_strict` | `langchain_default` | `llamaindex_default` | `haystack_default` | `graph_rag_global` | `no_retrieval_floor` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comparison_query | `compression_llm_extract` | 3.57 | 3.69 | 3.64 | 3.50 | 3.45 | 3.55 | 3.36 | 3.48 | 3.23 | 3.25 | 3.10 | 3.26 | 3.34 | 3.52 | 3.31 | 3.48 | 3.41 | 3.39 | 3.16 | 3.37 | 3.19 | 3.10 | 3.28 | 3.29 | 3.05 | 2.92 | 1.83 | 3.01 | 2.28 | 2.90 | 1.51 | 1.12 |
| inference_query | `compression_llm_extract` | 3.97 | 4.15 | 4.08 | 3.89 | 3.95 | 3.84 | 3.71 | 3.96 | 3.87 | 3.64 | 3.98 | 3.80 | 3.55 | 3.91 | 3.63 | 3.88 | 3.60 | 3.70 | 3.50 | 4.08 | 3.46 | 3.77 | 3.76 | 3.78 | 3.74 | 3.52 | 3.05 | 3.29 | 3.42 | 3.42 | 2.86 | 2.68 |
| null_query | `decision_gate_strict` | 4.38 | 3.91 | 4.51 | 4.36 | 4.31 | 4.35 | 4.32 | 4.08 | 4.34 | 4.57 | 4.70 | 4.50 | 4.40 | 3.76 | 4.28 | 4.36 | 4.39 | 4.29 | 4.41 | 4.28 | 4.37 | 4.37 | 4.42 | 4.38 | 4.31 | 4.67 | 4.84 | 1.73 | 2.22 | 0.59 | 2.76 | 3.23 |
| temporal_query | `llm_heavy_default` | 3.28 | 3.06 | 3.42 | 3.05 | 3.18 | 2.92 | 2.88 | 2.88 | 2.72 | 2.85 | 3.01 | 2.64 | 2.79 | 3.23 | 2.60 | 2.92 | 2.58 | 2.69 | 2.48 | 3.27 | 2.45 | 2.87 | 3.06 | 3.08 | 2.76 | 2.82 | 1.92 | 2.62 | 2.11 | 2.56 | 1.29 | 1.27 |

## How this was chosen

- Quality is the mean **answer score** over 300 questions. Weights: quality 0.6, cost 0.2, latency 0.2.
- Constraints: none.
- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. See docs/methodology.md#selection.
