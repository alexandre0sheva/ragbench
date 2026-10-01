# Recommendation

**Deploy `bm25_default`** (profile `balanced`, run `all_systems_demo_20261001_221453`). Its config is in `winner.yaml`.

## Why

- `bm25_default` is recommended under the `balanced` profile (quality first, then the best mix of cost and speed among systems that are equally good): answer score 4.69, $0.00049 per question, p95 latency 1378 ms.
- `bm25_default` ties with `parent_doc_default`, `vector_default`, `contextual_default`, `hybrid_rerank_default`, `rerank_default`, `decompose_default`, `rag_fusion_default`, `adaptive_default`, `hyde_default`, `iterative_default`, `full_context_ceiling`, `agent_search_tools`, `corrective_default`, `grep_agent_default` and `llm_heavy_default` on answer score: they are statistically indistinguishable from the best (`iterative_default`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ 0.05). Quality does not separate them, so it wins on cost and latency ($0.00049 per question against up to $0.00512; p95 latency 1378 ms against up to 9437 ms).
- bm25_default vs iterative_default (the best answer score): -0.12 answer score (95% CI -0.27 to +0.01, not significant), 0.9× cost
- bm25_default vs no_retrieval_floor (the cheapest feasible system): +3.15 answer score (95% CI +2.95 to +3.35, significant), 3.9× cost

## Ranking

The winner first, then the systems tied with it (best choice first), then the rest by weighted score. *Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; *tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.

| # | System | Score | Answer score | $/Q | p95 | Ingestion | Calls/Q | Tie | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `bm25_default` | 0.96 | 4.69 | $0.00049 | 1378 ms | $0.00000 | 1.0 | tie | ✓ |
| 2 | `parent_doc_default` | 0.96 | 4.71 | $0.00049 | 1514 ms | $0.00058 | 1.0 | tie | ✓ |
| 3 | `vector_default` | 0.95 | 4.67 | $0.00044 | 1637 ms | $0.00052 | 1.0 | tie | ✓ |
| 4 | `contextual_default` | 0.95 | 4.69 | $0.00046 | 1605 ms | $0.01456 | 1.0 | tie | ✓ |
| 5 | `hybrid_rerank_default` | 0.94 | 4.72 | $0.00053 | 2140 ms | $0.00052 | 1.0 | tie | ✓ |
| 6 | `rerank_default` | 0.95 | 4.74 | $0.00052 | 2201 ms | $0.00052 | 1.0 | tie | ✓ |
| 7 | `decompose_default` | 0.92 | 4.72 | $0.00041 | 3368 ms | $0.00053 | 2.0 | tie | ✓ |
| 8 | `rag_fusion_default` | 0.92 | 4.75 | $0.00042 | 3488 ms | $0.00053 | 2.0 | tie | ✓ |
| 9 | `adaptive_default` | 0.91 | 4.70 | $0.00044 | 3492 ms | $0.00158 | 1.3 | tie | ✓ |
| 10 | `hyde_default` | 0.91 | 4.68 | $0.00046 | 3479 ms | $0.00052 | 2.0 | tie |  |
| 11 | `iterative_default` | 0.91 | 4.81 | $0.00057 | 3952 ms | $0.00053 | 2.8 | tie | ✓ |
| 12 | `full_context_ceiling` | 0.86 | 4.78 | $0.00280 | 2257 ms | $0.00000 | 1.0 | tie | ✓ |
| 13 | `agent_search_tools` | 0.85 | 4.75 | $0.00082 | 5783 ms | $0.00053 | 3.0 | tie |  |
| 14 | `corrective_default` | 0.79 | 4.77 | $0.00060 | 8715 ms | $0.00053 | 2.9 | tie |  |
| 15 | `grep_agent_default` | 0.76 | 4.78 | $0.00109 | 9437 ms | $0.00000 | 4.4 | tie |  |
| 16 | `llm_heavy_default` | 0.62 | 4.72 | $0.00512 | 8055 ms | $0.00052 | 2.0 | tie |  |
| 17 | `hierarchical_default` | 0.94 | 4.56 | $0.00033 | 1148 ms | $0.00489 | 1.0 |  | ✓ |
| 18 | `hybrid_default` | 0.93 | 4.61 | $0.00048 | 1666 ms | $0.00052 | 1.0 |  |  |
| 19 | `vector_mmr` | 0.93 | 4.54 | $0.00040 | 1533 ms | $0.00052 | 1.0 |  | ✓ |
| 20 | `sentence_window_default` | 0.91 | 4.42 | $0.00025 | 1562 ms | $0.00047 | 1.0 |  | ✓ |
| 21 | `agent_search_plain` | 0.83 | 4.58 | $0.00072 | 5403 ms | $0.00053 | 3.0 |  |  |
| 22 | `no_retrieval_floor` | 0.40 | 1.54 | $0.00012 | 1093 ms | $0.00000 | 1.0 |  | ✓ |

## Winner by category

The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.

| Category | Winner | `bm25_default` | `parent_doc_default` | `vector_default` | `contextual_default` | `hybrid_rerank_default` | `rerank_default` | `decompose_default` | `rag_fusion_default` | `adaptive_default` | `hyde_default` | `iterative_default` | `full_context_ceiling` | `agent_search_tools` | `corrective_default` | `grep_agent_default` | `llm_heavy_default` | `hierarchical_default` | `hybrid_default` | `vector_mmr` | `sentence_window_default` | `agent_search_plain` | `no_retrieval_floor` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregation | `agent_search_plain` | 3.73 | 4.00 | 3.84 | 3.84 | 3.76 | 3.71 | 4.29 | 4.04 | 4.11 | 3.84 | 4.27 | 4.53 | 4.78 | 4.31 | 4.64 | 3.84 | 3.69 | 3.56 | 3.69 | 3.40 | 4.87 | 1.09 |
| comparison | `agent_search_tools` | 4.55 | 4.72 | 4.72 | 4.80 | 4.72 | 4.90 | 4.70 | 4.67 | 4.70 | 4.65 | 4.97 | 4.75 | 5.00 | 4.90 | 5.00 | 4.57 | 4.90 | 4.80 | 4.17 | 3.90 | 4.97 | 1.05 |
| date_arithmetic | `hierarchical_default` | 5.00 | 5.00 | 5.00 | 4.96 | 5.00 | 4.78 | 4.98 | 4.96 | 4.96 | 5.00 | 4.73 | 4.93 | 4.98 | 4.80 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.36 | 4.69 | 1.18 |
| direct_fact | `rag_fusion_default` | 4.88 | 4.95 | 5.00 | 4.98 | 4.99 | 4.97 | 4.98 | 5.00 | 5.00 | 4.99 | 4.87 | 4.95 | 4.85 | 4.99 | 4.92 | 5.00 | 4.98 | 4.85 | 4.96 | 4.74 | 4.97 | 1.15 |
| distractor | `rag_fusion_default` | 4.96 | 4.96 | 4.95 | 5.00 | 4.96 | 4.96 | 4.98 | 5.00 | 4.98 | 4.96 | 5.00 | 4.89 | 4.98 | 4.96 | 4.71 | 4.95 | 4.98 | 4.98 | 4.95 | 3.64 | 4.98 | 1.24 |
| exact_identifier | `hierarchical_default` | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.78 | 4.67 | 4.96 | 5.00 | 4.96 | 5.00 | 4.80 | 4.98 | 5.00 | 4.98 | 5.00 | 4.55 | 4.47 | 1.24 |
| long_context | `sentence_window_default` | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.72 | 5.00 | 5.00 | 5.00 | 5.00 | 4.33 | 5.00 | 5.00 | 3.65 | 5.00 | 5.00 | 5.00 | 3.72 | 1.17 |
| multi_hop | `agent_search_tools` | 4.28 | 4.20 | 4.17 | 4.13 | 4.27 | 4.55 | 3.68 | 4.13 | 4.02 | 4.10 | 4.45 | 4.75 | 4.92 | 4.58 | 4.72 | 4.47 | 3.78 | 4.02 | 3.90 | 4.53 | 4.72 | 1.17 |
| numeric_reasoning | `sentence_window_default` | 5.00 | 4.74 | 4.44 | 5.00 | 5.00 | 5.00 | 4.80 | 5.00 | 5.00 | 4.66 | 5.00 | 4.84 | 5.00 | 5.00 | 4.62 | 5.00 | 4.96 | 4.76 | 3.84 | 5.00 | 4.76 | 1.14 |
| paraphrase | `decompose_default` | 4.17 | 4.90 | 4.98 | 4.45 | 4.45 | 4.62 | 5.00 | 4.97 | 4.50 | 4.97 | 4.98 | 4.38 | 4.48 | 4.98 | 4.97 | 4.77 | 4.08 | 4.40 | 4.28 | 4.90 | 4.32 | 1.18 |
| temporal_conflict | `decompose_default` | 4.98 | 4.45 | 4.72 | 4.82 | 4.74 | 4.74 | 4.98 | 4.70 | 4.98 | 4.72 | 4.98 | 4.91 | 4.62 | 4.98 | 4.78 | 4.74 | 4.77 | 4.75 | 4.51 | 4.23 | 4.69 | 1.15 |
| unanswerable | `iterative_default` | 4.48 | 4.48 | 4.18 | 4.30 | 4.52 | 4.48 | 4.30 | 4.44 | 4.46 | 4.34 | 4.59 | 4.45 | 4.18 | 4.41 | 4.44 | 4.24 | 4.46 | 4.21 | 4.42 | 4.15 | 4.04 | 3.61 |

## How this was chosen

- Quality is the mean **answer score** over 163 questions. Weights: quality 0.6, cost 0.2, latency 0.2.
- Constraints: none.
- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. See docs/methodology.md#selection.
