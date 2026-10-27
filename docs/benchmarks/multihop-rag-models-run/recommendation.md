# Recommendation

**Deploy `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]`** (profile `balanced`, run `all_solutions_models_multihop`). Its config is in `winner.yaml`.

## Why

- `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` is recommended under the `balanced` profile (quality first, then the best mix of cost and speed among systems that are equally good): answer score 3.97, $0.00123 per question, p95 latency 6168 ms.
- `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` ties with `iterative_default[generator=anthropic:claude-haiku-5-5]` and `agent_search_tools[generator=anthropic:claude-haiku-5-5]` on answer score: they are statistically indistinguishable from the best (`hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]`) by a paired bootstrap on the same questions (Holm-adjusted p ≥ 0.05). Quality does not separate them, so it wins on cost and latency ($0.00123 per question against up to $0.00358; p95 latency 6168 ms against up to 18606 ms).
- hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] vs hybrid_rerank_default[generator=gpt-6-luna] (the cheapest feasible system): +0.32 answer score (95% CI +0.14 to +0.50, significant), 1.1× cost

## Ranking

The winner first, then the systems tied with it (best choice first), then the rest by weighted score. *Score* is the weighted, min-max normalized mix of quality, cost and latency across the feasible systems; *tie* marks quality that is statistically indistinguishable from the best; *Pareto* marks systems no other feasible system beats on quality, cost and latency at once.

| # | System | Score | Answer score | $/Q | p95 | Ingestion | Calls/Q | Tie | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` | 0.98 | 3.97 | $0.00123 | 6168 ms | $0.03301 | 1.0 | tie | ✓ |
| 2 | `iterative_default[generator=anthropic:claude-haiku-5-5]` | 0.86 | 3.84 | $0.00181 | 14812 ms | $0.03341 | 3.6 | tie |  |
| 3 | `agent_search_tools[generator=anthropic:claude-haiku-5-5]` | 0.89 | 3.97 | $0.00358 | 18606 ms | $0.03341 | 4.3 | tie |  |
| 4 | `hybrid_rerank_default[generator=gpt-6-luna]` | 0.83 | 3.65 | $0.00108 | 3491 ms | $0.03301 | 1.0 |  | ✓ |
| 5 | `hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3]` | 0.76 | 3.61 | $0.00294 | 7831 ms | $0.03301 | 1.0 |  |  |
| 6 | `agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash]` | 0.73 | 3.60 | $0.00250 | 15294 ms | $0.03341 | 4.6 |  |  |
| 7 | `hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash]` | 0.72 | 3.50 | $0.00131 | 9896 ms | $0.03301 | 1.0 |  |  |
| 8 | `iterative_default[generator=openrouter:minimax/minimax-m3]` | 0.72 | 3.60 | $0.00292 | 15878 ms | $0.03341 | 3.8 |  |  |
| 9 | `agent_search_tools[generator=openrouter:z-ai/glm-5.3]` | 0.71 | 3.63 | $0.00474 | 13354 ms | $0.03341 | 3.8 |  |  |
| 10 | `hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro]` | 0.71 | 3.58 | $0.00246 | 17186 ms | $0.03301 | 1.0 |  |  |
| 11 | `hybrid_rerank_default[generator=openrouter:minimax/minimax-m3]` | 0.68 | 3.43 | $0.00206 | 6729 ms | $0.03301 | 1.0 |  |  |
| 12 | `iterative_default[generator=gpt-6-luna]` | 0.68 | 3.44 | $0.00124 | 11458 ms | $0.03341 | 3.9 |  |  |
| 13 | `iterative_default[generator=openrouter:z-ai/glm-5.3]` | 0.67 | 3.54 | $0.00380 | 17586 ms | $0.03341 | 3.5 |  |  |
| 14 | `hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3]` | 0.66 | 3.45 | $0.00408 | 5228 ms | $0.03301 | 1.0 |  |  |
| 15 | `agent_search_tools[generator=openrouter:qwen/qwen3.8-flash]` | 0.65 | 3.51 | $0.00269 | 21213 ms | $0.03341 | 5.7 |  |  |
| 16 | `agent_search_tools[generator=gpt-6-luna]` | 0.63 | 3.42 | $0.00259 | 15081 ms | $0.03341 | 5.6 |  |  |
| 17 | `iterative_default[generator=openrouter:qwen/qwen3.8-flash]` | 0.60 | 3.41 | $0.00157 | 24813 ms | $0.03341 | 3.4 |  |  |
| 18 | `iterative_default[generator=openrouter:moonshotai/kimi-k3]` | 0.57 | 3.56 | $0.01098 | 13899 ms | $0.03341 | 3.6 |  |  |
| 19 | `iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro]` | 0.57 | 3.50 | $0.00338 | 36879 ms | $0.03341 | 3.8 |  |  |
| 20 | `agent_search_tools[generator=openrouter:minimax/minimax-m3]` | 0.57 | 3.48 | $0.00491 | 28783 ms | $0.03341 | 4.8 |  |  |
| 21 | `iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash]` | 0.53 | 3.12 | $0.00146 | 7212 ms | $0.03341 | 3.5 |  |  |
| 22 | `agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro]` | 0.52 | 3.56 | $0.00513 | 48893 ms | $0.03341 | 5.6 |  |  |
| 23 | `hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash]` | 0.40 | 2.82 | $0.00121 | 2486 ms | $0.03301 | 1.0 |  | ✓ |
| 24 | `agent_search_tools[generator=openrouter:moonshotai/kimi-k3]` | 0.31 | 3.16 | $0.01315 | 17170 ms | $0.03341 | 3.6 |  |  |

## Winner by category

The best mean quality in each question category among the feasible systems. This is descriptive: categories hold few questions, so a different winner here is a lead worth checking, not a significant result.

| Category | Winner | `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` | `iterative_default[generator=anthropic:claude-haiku-5-5]` | `agent_search_tools[generator=anthropic:claude-haiku-5-5]` | `hybrid_rerank_default[generator=gpt-6-luna]` | `hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3]` | `agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash]` | `hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash]` | `iterative_default[generator=openrouter:minimax/minimax-m3]` | `agent_search_tools[generator=openrouter:z-ai/glm-5.3]` | `hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro]` | `hybrid_rerank_default[generator=openrouter:minimax/minimax-m3]` | `iterative_default[generator=gpt-6-luna]` | `iterative_default[generator=openrouter:z-ai/glm-5.3]` | `hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3]` | `agent_search_tools[generator=openrouter:qwen/qwen3.8-flash]` | `agent_search_tools[generator=gpt-6-luna]` | `iterative_default[generator=openrouter:qwen/qwen3.8-flash]` | `iterative_default[generator=openrouter:moonshotai/kimi-k3]` | `iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro]` | `agent_search_tools[generator=openrouter:minimax/minimax-m3]` | `iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash]` | `agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro]` | `hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash]` | `agent_search_tools[generator=openrouter:moonshotai/kimi-k3]` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| comparison_query | `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` | 3.60 | 3.29 | 3.42 | 3.28 | 3.26 | 3.31 | 3.02 | 3.18 | 3.12 | 3.28 | 2.95 | 3.00 | 3.22 | 2.85 | 3.12 | 2.97 | 2.94 | 3.04 | 3.09 | 3.08 | 2.63 | 3.18 | 1.97 | 2.67 |
| inference_query | `hybrid_rerank_default[generator=anthropic:claude-haiku-5-5]` | 4.47 | 4.44 | 4.36 | 3.98 | 3.88 | 3.72 | 3.85 | 3.68 | 3.75 | 3.78 | 3.80 | 3.74 | 3.80 | 3.91 | 3.80 | 3.60 | 3.66 | 3.83 | 3.78 | 3.71 | 3.50 | 3.73 | 3.23 | 2.83 |
| null_query | `iterative_default[generator=anthropic:claude-haiku-5-5]` | 4.25 | 4.73 | 4.50 | 4.57 | 4.42 | 4.19 | 4.65 | 4.40 | 4.10 | 4.38 | 4.35 | 4.62 | 4.40 | 4.65 | 3.87 | 4.47 | 4.50 | 4.48 | 4.35 | 4.16 | 4.48 | 4.17 | 4.65 | 4.58 |
| temporal_query | `agent_search_tools[generator=openrouter:z-ai/glm-5.3]` | 3.67 | 3.30 | 3.92 | 3.25 | 3.30 | 3.53 | 3.13 | 3.65 | 3.93 | 3.31 | 3.10 | 3.02 | 3.21 | 3.02 | 3.50 | 3.27 | 3.15 | 3.47 | 3.26 | 3.38 | 2.57 | 3.55 | 2.54 | 3.58 |

## How this was chosen

- Quality is the mean **answer score** over 100 questions. Weights: quality 0.6, cost 0.2, latency 0.2.
- Constraints: none.
- Systems whose quality is not significantly worse than the best (paired bootstrap, Holm-adjusted) are tied; the tie is broken on cost and latency by the weights, then on the fewest model calls. See docs/methodology.md#selection.
