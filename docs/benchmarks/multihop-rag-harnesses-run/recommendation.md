# Recommendation

**Deploy `hybrid_rerank_default`** (profile `balanced`, run `all_solutions_harnesses_multihop`). Its config is in `winner.yaml`.

## Why

- `hybrid_rerank_default` is recommended under the `balanced` profile (quality first, then the best mix of cost and speed among systems that are equally good): answer score 3.65, $0.00109 per question, p95 latency 3305 ms.
- `hybrid_rerank_default` ties with `agent_search_tools`, `opencode_glm` and `claude_code` on answer score: they are statistically indistinguishable from the best (`hybrid_rerank_default`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ 0.05). Quality does not separate them, so it wins on cost and latency ($0.00109 per question against up to $0.01353; p95 latency 3305 ms against up to 31459 ms).
- hybrid_rerank_default vs no_retrieval_floor (the cheapest feasible system): +1.71 answer score (95% CI +1.49 to +1.91, significant), 5.2× cost

## Ranking

The winner first, then the systems tied with it (best choice first), then the rest by weighted score. *Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; *tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.

| # | System | Score | Answer score | $/Q | p95 | Ingestion | Calls/Q | Tie | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `hybrid_rerank_default` | 0.98 | 3.65 | $0.00109 | 3305 ms | $0.03301 | 1.0 | tie | ✓ |
| 2 | `agent_search_tools` | 0.82 | 3.44 | $0.00264 | 16177 ms | $0.03341 | 5.6 | tie |  |
| 3 | `opencode_glm` | 0.76 | 3.64 | $0.01023 | 22541 ms | $0.00000 | 5.4 | tie |  |
| 4 | `claude_code` | 0.64 | 3.55 | $0.01353 | 31459 ms | $0.00000 | 6.5 | tie |  |
| 5 | `no_retrieval_floor` | 0.40 | 1.95 | $0.00021 | 2150 ms | $0.00000 | 1.0 |  | ✓ |
| 6 | `codex` | 0.34 | 2.92 | $0.01598 | 40058 ms | $0.00000 | 1.0 |  |  |

## Winner by category

The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.

| Category | Winner | `hybrid_rerank_default` | `agent_search_tools` | `opencode_glm` | `claude_code` | `no_retrieval_floor` | `codex` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| comparison_query | `hybrid_rerank_default` | 3.53 | 3.11 | 3.50 | 2.91 | 1.15 | 2.56 |
| inference_query | `claude_code` | 3.94 | 3.72 | 3.84 | 4.31 | 2.64 | 3.43 |
| null_query | `hybrid_rerank_default` | 4.36 | 4.33 | 3.75 | 3.91 | 3.61 | 3.99 |
| temporal_query | `opencode_glm` | 3.06 | 3.05 | 3.50 | 3.24 | 1.27 | 2.18 |

## How this was chosen

- Quality is the mean **answer score** over 150 questions. Weights: quality 0.6, cost 0.2, latency 0.2.
- Constraints: none.
- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. See docs/methodology.md#selection.
