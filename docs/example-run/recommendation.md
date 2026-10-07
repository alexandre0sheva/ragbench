# Recommendation

**Deploy `bm25_default`** (profile `balanced`, run `all_systems_plus_decision_gate_20261007_184150`). Its config is in `winner.yaml`.

## Why

- `bm25_default` is recommended under the `balanced` profile (quality first, then the best mix of cost and speed among systems that are equally good): answer score 4.69, $0.00049 per question, p95 latency 1371 ms.
- `bm25_default` ties with `contextual_default`, `vector_default`, `decompose_default`, `hyde_default`, `rerank_default`, `hybrid_rerank_default`, `decision_gate_strict`, `rag_fusion_default`, `adaptive_default`, `decision_gate_default`, `parent_doc_default`, `full_context_ceiling`, `agent_search_tools`, `iterative_default`, `grep_agent_default`, `corrective_default` and `llm_heavy_default` on answer score: they are statistically indistinguishable from the best (`iterative_default`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ 0.05). Quality does not separate them, so it wins on cost and latency ($0.00049 per question against up to $0.00512; p95 latency 1371 ms against up to 9595 ms).
- bm25_default vs iterative_default (the best answer score): -0.12 answer score (95% CI -0.27 to +0.01, not significant), 0.9× cost
- bm25_default vs no_retrieval_floor (the cheapest feasible system): +3.15 answer score (95% CI +2.95 to +3.35, significant), 3.9× cost

## Ranking

The winner first, then the systems tied with it (best choice first), then the rest by weighted score. *Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; *tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.

| # | System | Score | Answer score | $/Q | p95 | Ingestion | Calls/Q | Tie | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `bm25_default` | 0.96 | 4.69 | $0.00049 | 1371 ms | $0.00000 | 1.0 | tie | ✓ |
| 2 | `contextual_default` | 0.96 | 4.69 | $0.00046 | 1690 ms | $0.01456 | 1.0 | tie | ✓ |
| 3 | `vector_default` | 0.94 | 4.67 | $0.00044 | 2374 ms | $0.00052 | 1.0 | tie | ✓ |
| 4 | `decompose_default` | 0.94 | 4.72 | $0.00041 | 2888 ms | $0.00053 | 2.0 | tie | ✓ |
| 5 | `hyde_default` | 0.92 | 4.68 | $0.00046 | 3110 ms | $0.00052 | 2.0 | tie |  |
| 6 | `rerank_default` | 0.93 | 4.74 | $0.00052 | 3054 ms | $0.00052 | 1.0 | tie | ✓ |
| 7 | `hybrid_rerank_default` | 0.92 | 4.72 | $0.00053 | 3463 ms | $0.00052 | 1.0 | tie |  |
| 8 | `decision_gate_strict` | 0.90 | 4.65 | $0.00091 | 3086 ms | $0.00052 | 2.6 | tie |  |
| 9 | `rag_fusion_default` | 0.91 | 4.75 | $0.00042 | 4093 ms | $0.00053 | 2.0 | tie | ✓ |
| 10 | `adaptive_default` | 0.90 | 4.70 | $0.00044 | 4119 ms | $0.00158 | 1.3 | tie |  |
| 11 | `decision_gate_default` | 0.88 | 4.80 | $0.00094 | 4801 ms | $0.00052 | 2.8 | tie | ✓ |
| 12 | `parent_doc_default` | 0.86 | 4.71 | $0.00049 | 5704 ms | $0.00058 | 1.0 | tie |  |
| 13 | `full_context_ceiling` | 0.87 | 4.78 | $0.00280 | 2038 ms | $0.00000 | 1.0 | tie | ✓ |
| 14 | `agent_search_tools` | 0.85 | 4.75 | $0.00082 | 5753 ms | $0.00053 | 3.0 | tie |  |
| 15 | `iterative_default` | 0.86 | 4.81 | $0.00057 | 6423 ms | $0.00053 | 2.8 | tie | ✓ |
| 16 | `grep_agent_default` | 0.79 | 4.78 | $0.00109 | 8197 ms | $0.00000 | 4.4 | tie |  |
| 17 | `corrective_default` | 0.77 | 4.77 | $0.00060 | 9595 ms | $0.00053 | 2.9 | tie |  |
| 18 | `llm_heavy_default` | 0.63 | 4.72 | $0.00512 | 7579 ms | $0.00052 | 2.0 | tie |  |
| 19 | `hierarchical_default` | 0.94 | 4.56 | $0.00033 | 1604 ms | $0.00489 | 1.0 |  | ✓ |
| 20 | `vector_mmr` | 0.94 | 4.54 | $0.00040 | 1541 ms | $0.00052 | 1.0 |  |  |
| 21 | `hybrid_default` | 0.94 | 4.61 | $0.00048 | 1895 ms | $0.00052 | 1.0 |  |  |
| 22 | `sentence_window_default` | 0.92 | 4.42 | $0.00025 | 1518 ms | $0.00047 | 1.0 |  | ✓ |
| 23 | `agent_search_plain` | 0.84 | 4.58 | $0.00072 | 5345 ms | $0.00053 | 3.0 |  |  |
| 24 | `no_retrieval_floor` | 0.40 | 1.54 | $0.00012 | 1430 ms | $0.00000 | 1.0 |  | ✓ |

## Winner by category

The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.

| Category | Winner | `bm25_default` | `contextual_default` | `vector_default` | `decompose_default` | `hyde_default` | `rerank_default` | `hybrid_rerank_default` | `decision_gate_strict` | `rag_fusion_default` | `adaptive_default` | `decision_gate_default` | `parent_doc_default` | `full_context_ceiling` | `agent_search_tools` | `iterative_default` | `grep_agent_default` | `corrective_default` | `llm_heavy_default` | `hierarchical_default` | `vector_mmr` | `hybrid_default` | `sentence_window_default` | `agent_search_plain` | `no_retrieval_floor` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggregation | `agent_search_plain` | 3.73 | 3.84 | 3.84 | 4.29 | 3.84 | 3.73 | 3.76 | 3.42 | 4.04 | 4.11 | 3.80 | 4.00 | 4.53 | 4.78 | 4.27 | 4.64 | 4.31 | 3.84 | 3.69 | 3.69 | 3.56 | 3.40 | 4.87 | 1.09 |
| comparison | `agent_search_tools` | 4.55 | 4.80 | 4.72 | 4.70 | 4.65 | 4.90 | 4.72 | 4.50 | 4.67 | 4.70 | 4.83 | 4.72 | 4.75 | 5.00 | 4.97 | 5.00 | 4.90 | 4.57 | 4.90 | 4.17 | 4.80 | 3.90 | 4.97 | 1.05 |
| date_arithmetic | `hierarchical_default` | 5.00 | 4.96 | 5.00 | 4.98 | 5.00 | 4.78 | 5.00 | 4.62 | 4.96 | 4.96 | 4.96 | 5.00 | 4.93 | 4.98 | 4.73 | 5.00 | 4.80 | 5.00 | 5.00 | 5.00 | 5.00 | 4.36 | 4.69 | 1.18 |
| direct_fact | `rag_fusion_default` | 4.88 | 4.98 | 5.00 | 4.98 | 4.99 | 4.97 | 4.99 | 4.99 | 5.00 | 5.00 | 4.99 | 4.95 | 4.95 | 4.85 | 4.87 | 4.92 | 4.99 | 5.00 | 4.98 | 4.96 | 4.85 | 4.74 | 4.97 | 1.15 |
| distractor | `rag_fusion_default` | 4.96 | 5.00 | 4.95 | 4.98 | 4.96 | 4.96 | 4.96 | 4.96 | 5.00 | 4.98 | 4.95 | 4.96 | 4.89 | 4.98 | 5.00 | 4.71 | 4.96 | 4.95 | 4.98 | 4.95 | 4.98 | 3.64 | 4.98 | 1.24 |
| exact_identifier | `hierarchical_default` | 5.00 | 5.00 | 5.00 | 5.00 | 4.67 | 5.00 | 5.00 | 4.98 | 5.00 | 4.78 | 4.96 | 5.00 | 5.00 | 4.96 | 4.96 | 4.80 | 5.00 | 4.98 | 5.00 | 5.00 | 4.98 | 4.55 | 4.47 | 1.24 |
| long_context | `sentence_window_default` | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.72 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.33 | 5.00 | 3.65 | 5.00 | 5.00 | 5.00 | 3.72 | 1.17 |
| multi_hop | `agent_search_tools` | 4.28 | 4.13 | 4.17 | 3.68 | 4.10 | 4.55 | 4.27 | 3.43 | 4.13 | 4.02 | 4.40 | 4.20 | 4.75 | 4.92 | 4.45 | 4.72 | 4.58 | 4.47 | 3.78 | 3.90 | 4.02 | 4.53 | 4.72 | 1.17 |
| numeric_reasoning | `sentence_window_default` | 5.00 | 5.00 | 4.44 | 4.80 | 4.66 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 4.74 | 4.84 | 5.00 | 5.00 | 4.62 | 5.00 | 5.00 | 4.96 | 3.84 | 4.76 | 5.00 | 4.76 | 1.14 |
| paraphrase | `decompose_default` | 4.17 | 4.45 | 4.98 | 5.00 | 4.97 | 4.62 | 4.45 | 4.67 | 4.97 | 4.50 | 4.70 | 4.90 | 4.38 | 4.48 | 4.98 | 4.97 | 4.98 | 4.77 | 4.08 | 4.28 | 4.40 | 4.90 | 4.32 | 1.18 |
| temporal_conflict | `decision_gate_default` | 4.98 | 4.82 | 4.72 | 4.98 | 4.72 | 4.74 | 4.74 | 4.69 | 4.72 | 4.98 | 5.00 | 4.45 | 4.91 | 4.62 | 4.98 | 4.78 | 4.98 | 4.74 | 4.77 | 4.51 | 4.75 | 4.23 | 4.69 | 1.15 |
| unanswerable | `decision_gate_strict` | 4.48 | 4.30 | 4.18 | 4.30 | 4.34 | 4.48 | 4.52 | 4.72 | 4.44 | 4.46 | 4.72 | 4.48 | 4.45 | 4.18 | 4.59 | 4.44 | 4.41 | 4.24 | 4.46 | 4.42 | 4.21 | 4.15 | 4.04 | 3.61 |

## How this was chosen

- Quality is the mean **answer score** over 163 questions. Weights: quality 0.6, cost 0.2, latency 0.2.
- Constraints: none.
- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. See docs/methodology.md#selection.
