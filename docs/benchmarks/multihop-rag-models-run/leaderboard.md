# RAGBench Leaderboard

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 2.82 [2.53, 3.11] | 4.00 [3.78, 4.21] | 0.02 | 0.65 | $0.00121 | $0.00303 | 1595 ms | 2486 ms [2390 ms, 5241 ms] | ✓ | 0 | 96272 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.61 [3.40, 3.80] | 3.94 [3.76, 4.12] | 0.03 | 0.65 | $0.00294 | $0.00576 | 4267 ms | 7831 ms [7056 ms, 47184 ms] |  | 0 | 173181 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.45 [3.19, 3.69] | 4.00 [3.80, 4.20] | 0.05 | 0.65 | $0.00408 | $0.00801 | 2799 ms | 5228 ms [4501 ms, 13956 ms] |  | 0 | 128339 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.58 [3.38, 3.77] | 3.94 [3.75, 4.11] | 0.05 | 0.65 | $0.00246 | $0.00513 | 8740 ms | 17186 ms [13829 ms, 36240 ms] |  | 0 | 287327 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.50 [3.27, 3.73] | 3.96 [3.77, 4.13] | 0.07 | 0.65 | $0.00131 | $0.00257 | 4936 ms | 9896 ms [8566 ms, 13843 ms] |  | 0 | 179956 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=openrouter:minimax/minimax-m3] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.43 [3.20, 3.64] | 3.96 [3.77, 4.14] | 0.05 | 0.65 | $0.00206 | $0.00468 | 3167 ms | 6729 ms [5274 ms, 17031 ms] |  | 0 | 144561 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=gpt-6-luna] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.65 [3.47, 3.83] | 4.08 [3.92, 4.23] | 0.07 | 0.65 | $0.00108 | $0.00213 | 2410 ms | 3491 ms [3079 ms, 7931 ms] | ✓ | 0 | 18502 ms | Recall of hybrid plus rerank precision |
| hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] | 0.627 [0.562, 0.690] | 0.742 [0.668, 0.810] | 0.627 [0.569, 0.686] | 3.97 [3.76, 4.17] | 4.11 [3.91, 4.30] | 0.02 | 0.65 | $0.00123 | $0.00175 | 3860 ms | 6168 ms [5451 ms, 7804 ms] | ✓ | 0 | 155393 ms | Recall of hybrid plus rerank precision |
| iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | 0.618 [0.543, 0.688] | 0.679 [0.601, 0.753] | 0.587 [0.522, 0.652] | 3.12 [2.87, 3.37] | 4.08 [3.90, 4.24] | 0.05 | 0.61 | $0.00146 | $0.00340 | 5171 ms | 7212 ms [6905 ms, 8726 ms] |  | 0 | 192360 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=openrouter:z-ai/glm-5.3] | 0.593 [0.520, 0.662] | 0.673 [0.595, 0.747] | 0.569 [0.502, 0.635] | 3.54 [3.35, 3.72] | 3.84 [3.65, 4.02] | 0.04 | 0.61 | $0.00380 | $0.00809 | 9067 ms | 17586 ms [12330 ms, 37223 ms] |  | 0 | 290147 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=openrouter:moonshotai/kimi-k3] | 0.607 [0.536, 0.676] | 0.656 [0.579, 0.729] | 0.580 [0.517, 0.641] | 3.56 [3.34, 3.78] | 3.69 [3.45, 3.92] | 0.02 | 0.59 | $0.01098 | $0.02112 | 8753 ms | 13899 ms [12287 ms, 20335 ms] |  | 0 | 287428 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | 0.613 [0.542, 0.682] | 0.634 [0.559, 0.707] | 0.566 [0.500, 0.631] | 3.50 [3.34, 3.67] | 3.98 [3.81, 4.15] | 0.02 | 0.57 | $0.00338 | $0.00804 | 26402 ms | 36879 ms [34324 ms, 45886 ms] |  | 0 | 737268 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=openrouter:qwen/qwen3.8-flash] | 0.601 [0.529, 0.668] | 0.665 [0.592, 0.737] | 0.579 [0.513, 0.638] | 3.41 [3.18, 3.62] | 3.74 [3.53, 3.95] | 0.04 | 0.60 | $0.00157 | $0.00364 | 16258 ms | 24813 ms [22988 ms, 30659 ms] |  | 0 | 472412 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=openrouter:minimax/minimax-m3] | 0.610 [0.538, 0.680] | 0.641 [0.565, 0.716] | 0.577 [0.511, 0.642] | 3.60 [3.40, 3.79] | 3.90 [3.75, 4.06] | 0.03 | 0.61 | $0.00292 | $0.00596 | 8972 ms | 15878 ms [13524 ms, 26931 ms] |  | 0 | 290630 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=gpt-6-luna] | 0.583 [0.510, 0.651] | 0.679 [0.602, 0.754] | 0.577 [0.510, 0.639] | 3.44 [3.21, 3.65] | 4.13 [3.96, 4.29] | 0.05 | 0.60 | $0.00124 | $0.00258 | 8676 ms | 11458 ms [10802 ms, 14796 ms] |  | 0 | 32097 ms | Multi-hop questions whose second search depends on what the first one found |
| iterative_default[generator=anthropic:claude-haiku-5-5] | 0.615 [0.542, 0.685] | 0.672 [0.595, 0.745] | 0.583 [0.519, 0.646] | 3.84 [3.61, 4.04] | 4.10 [3.93, 4.26] | 0.02 | 0.57 | $0.00181 | $0.00318 | 10307 ms | 14812 ms [13868 ms, 16502 ms] |  | 0 | 323422 ms | Multi-hop questions whose second search depends on what the first one found |
| agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash] | 0.688 [0.612, 0.757] | 0.696 [0.620, 0.774] | 0.657 [0.589, 0.724] | 3.60 [3.41, 3.78] | 3.21 [2.98, 3.43] | 0.02 | 0.69 | $0.00250 | $0.00544 | 7614 ms | 15294 ms [12188 ms, 23177 ms] |  | 0 | 272471 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=openrouter:z-ai/glm-5.3] | 0.676 [0.603, 0.743] | 0.702 [0.626, 0.772] | 0.640 [0.572, 0.705] | 3.63 [3.43, 3.81] | 3.33 [3.11, 3.53] | 0.02 | 0.70 | $0.00474 | $0.00832 | 7330 ms | 13354 ms [11927 ms, 23533 ms] |  | 0 | 74566 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=openrouter:moonshotai/kimi-k3] | 0.427 [0.339, 0.517] | 0.435 [0.344, 0.527] | 0.411 [0.326, 0.498] | 3.16 [2.92, 3.41] | 3.38 [3.08, 3.69] | 0.02 | 0.46 | $0.01315 | $0.03554 | 8199 ms | 17170 ms [15604 ms, 46547 ms] |  | 0 | 225809 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro] | 0.670 [0.593, 0.740] | 0.747 [0.668, 0.820] | 0.667 [0.596, 0.734] | 3.56 [3.35, 3.76] | 3.17 [2.93, 3.40] | 0.01 | 0.65 | $0.00513 | $0.01007 | 26621 ms | 48893 ms [45211 ms, 61439 ms] |  | 0 | 181182 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=openrouter:qwen/qwen3.8-flash] | 0.667 [0.597, 0.733] | 0.728 [0.653, 0.798] | 0.640 [0.574, 0.705] | 3.51 [3.32, 3.69] | 3.10 [2.90, 3.32] | 0.01 | 0.66 | $0.00269 | $0.00641 | 17102 ms | 21213 ms |  | 0 | 212719 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=openrouter:minimax/minimax-m3] | 0.649 [0.573, 0.724] | 0.691 [0.611, 0.766] | 0.628 [0.556, 0.698] | 3.48 [3.27, 3.69] | 3.33 [3.12, 3.53] | 0.02 | 0.64 | $0.00491 | $0.01143 | 12371 ms | 28783 ms [23779 ms, 42972 ms] |  | 0 | 117712 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=gpt-6-luna] | 0.672 [0.598, 0.738] | 0.687 [0.613, 0.757] | 0.630 [0.565, 0.691] | 3.42 [3.21, 3.64] | 3.57 [3.35, 3.78] | 0.05 | 0.67 | $0.00259 | $0.00631 | 11133 ms | 15081 ms [13408 ms, 46149 ms] |  | 0 | 33985 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools[generator=anthropic:claude-haiku-5-5] | 0.648 [0.574, 0.720] | 0.701 [0.626, 0.771] | 0.645 [0.577, 0.708] | 3.97 [3.78, 4.15] | 3.89 [3.70, 4.07] | 0.01 | 0.68 | $0.00358 | $0.00534 | 10134 ms | 18606 ms [15241 ms, 20112 ms] |  | 0 | 136516 ms | Questions that need computation, exact lookups, or several searches |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs hybrid_rerank_default[generator=gpt-6-luna]

Each system minus the baseline `hybrid_rerank_default[generator=gpt-6-luna]` (cheapest system), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | Answer | -0.830 | [-1.082, -0.586] | 0.023 | worse | 23 / 18 / 59 |
| hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3] | Answer | -0.047 | [-0.193, +0.100] | 1.000 | no clear difference | 42 / 13 / 45 |
| hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3] | Answer | -0.207 | [-0.396, -0.018] | 0.416 | no clear difference | 33 / 22 / 45 |
| hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | Answer | -0.074 | [-0.239, +0.092] | 1.000 | no clear difference | 41 / 11 / 48 |
| hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash] | Answer | -0.148 | [-0.319, +0.017] | 0.924 | no clear difference | 34 / 23 / 43 |
| hybrid_rerank_default[generator=openrouter:minimax/minimax-m3] | Answer | -0.226 | [-0.400, -0.067] | 0.126 | no clear difference | 37 / 13 / 50 |
| hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] | Answer | +0.319 | [+0.142, +0.498] | 0.023 | better | 60 / 8 / 32 |
| iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | Answer | -0.533 | [-0.769, -0.297] | 0.023 | worse | 30 / 7 / 63 |
| iterative_default[generator=openrouter:z-ai/glm-5.3] | Answer | -0.108 | [-0.290, +0.070] | 1.000 | no clear difference | 38 / 13 / 49 |
| iterative_default[generator=openrouter:moonshotai/kimi-k3] | Answer | -0.089 | [-0.277, +0.098] | 1.000 | no clear difference | 37 / 14 / 49 |
| iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | Answer | -0.149 | [-0.305, +0.005] | 0.742 | no clear difference | 37 / 12 / 51 |
| iterative_default[generator=openrouter:qwen/qwen3.8-flash] | Answer | -0.247 | [-0.450, -0.039] | 0.357 | no clear difference | 36 / 9 / 55 |
| iterative_default[generator=openrouter:minimax/minimax-m3] | Answer | -0.056 | [-0.256, +0.143] | 1.000 | no clear difference | 44 / 11 / 45 |
| iterative_default[generator=gpt-6-luna] | Answer | -0.217 | [-0.434, +0.001] | 0.742 | no clear difference | 34 / 15 / 51 |
| iterative_default[generator=anthropic:claude-haiku-5-5] | Answer | +0.183 | [-0.031, +0.384] | 0.950 | no clear difference | 64 / 5 / 31 |
| agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash] | Answer | -0.056 | [-0.243, +0.134] | 1.000 | no clear difference | 41 / 5 / 54 |
| agent_search_tools[generator=openrouter:z-ai/glm-5.3] | Answer | -0.025 | [-0.230, +0.159] | 1.000 | no clear difference | 45 / 6 / 49 |
| agent_search_tools[generator=openrouter:moonshotai/kimi-k3] | Answer | -0.494 | [-0.785, -0.208] | 0.023 | worse | 40 / 4 / 56 |
| agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro] | Answer | -0.094 | [-0.292, +0.092] | 1.000 | no clear difference | 38 / 7 / 55 |
| agent_search_tools[generator=openrouter:qwen/qwen3.8-flash] | Answer | -0.139 | [-0.340, +0.044] | 1.000 | no clear difference | 39 / 5 / 56 |
| agent_search_tools[generator=openrouter:minimax/minimax-m3] | Answer | -0.175 | [-0.369, +0.025] | 0.924 | no clear difference | 39 / 4 / 57 |
| agent_search_tools[generator=gpt-6-luna] | Answer | -0.233 | [-0.449, -0.017] | 0.555 | no clear difference | 35 / 13 / 52 |
| agent_search_tools[generator=anthropic:claude-haiku-5-5] | Answer | +0.313 | [+0.133, +0.500] | 0.038 | better | 60 / 5 / 35 |
| hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=openrouter:minimax/minimax-m3] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] | Recall@5 | +0.000 | [+0.000, +0.000] | 1.000 | no clear difference | 0 / 100 / 0 |
| iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | Recall@5 | -0.009 | [-0.054, +0.035] | 1.000 | no clear difference | 13 / 73 / 14 |
| iterative_default[generator=openrouter:z-ai/glm-5.3] | Recall@5 | -0.034 | [-0.085, +0.016] | 1.000 | no clear difference | 14 / 67 / 19 |
| iterative_default[generator=openrouter:moonshotai/kimi-k3] | Recall@5 | -0.020 | [-0.068, +0.026] | 1.000 | no clear difference | 12 / 71 / 17 |
| iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | Recall@5 | -0.014 | [-0.062, +0.032] | 1.000 | no clear difference | 15 / 67 / 18 |
| iterative_default[generator=openrouter:qwen/qwen3.8-flash] | Recall@5 | -0.026 | [-0.073, +0.022] | 1.000 | no clear difference | 14 / 67 / 19 |
| iterative_default[generator=openrouter:minimax/minimax-m3] | Recall@5 | -0.017 | [-0.063, +0.028] | 1.000 | no clear difference | 14 / 68 / 18 |
| iterative_default[generator=gpt-6-luna] | Recall@5 | -0.044 | [-0.092, +0.000] | 1.000 | no clear difference | 12 / 69 / 19 |
| iterative_default[generator=anthropic:claude-haiku-5-5] | Recall@5 | -0.012 | [-0.062, +0.038] | 1.000 | no clear difference | 16 / 67 / 17 |
| agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash] | Recall@5 | +0.061 | [+0.007, +0.112] | 0.528 | no clear difference | 25 / 63 / 12 |
| agent_search_tools[generator=openrouter:z-ai/glm-5.3] | Recall@5 | +0.049 | [+0.003, +0.098] | 0.693 | no clear difference | 21 / 67 / 12 |
| agent_search_tools[generator=openrouter:moonshotai/kimi-k3] | Recall@5 | -0.200 | [-0.278, -0.120] | 0.023 | worse | 12 / 50 / 38 |
| agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro] | Recall@5 | +0.043 | [-0.008, +0.095] | 1.000 | no clear difference | 24 / 62 / 14 |
| agent_search_tools[generator=openrouter:qwen/qwen3.8-flash] | Recall@5 | +0.041 | [-0.005, +0.087] | 1.000 | no clear difference | 21 / 68 / 11 |
| agent_search_tools[generator=openrouter:minimax/minimax-m3] | Recall@5 | +0.022 | [-0.027, +0.074] | 1.000 | no clear difference | 22 / 62 / 16 |
| agent_search_tools[generator=gpt-6-luna] | Recall@5 | +0.045 | [-0.003, +0.090] | 1.000 | no clear difference | 20 / 69 / 11 |
| agent_search_tools[generator=anthropic:claude-haiku-5-5] | Recall@5 | +0.022 | [-0.032, +0.073] | 1.000 | no clear difference | 17 / 71 / 12 |

## Cost by stage

Mean $ per question, answering only (the judge's cost is excluded). `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | llm | tool | generate |
| --- | --- | --- | --- | --- |
| hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | $0.000001 | $0.000000 | $0.000000 | $0.000507 |
| hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3] | $0.000001 | $0.000000 | $0.000000 | $0.002196 |
| hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3] | $0.000001 | $0.000000 | $0.000000 | $0.003350 |
| hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | $0.000001 | $0.000000 | $0.000000 | $0.001711 |
| hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash] | $0.000001 | $0.000000 | $0.000000 | $0.000586 |
| hybrid_rerank_default[generator=openrouter:minimax/minimax-m3] | $0.000001 | $0.000000 | $0.000000 | $0.001318 |
| hybrid_rerank_default[generator=gpt-6-luna] | $0.000001 | $0.000000 | $0.000000 | $0.000374 |
| hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] | $0.000001 | $0.000000 | $0.000000 | $0.000818 |
| iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | $0.000002 | $0.000524 | $0.000000 | $0.000376 |
| iterative_default[generator=openrouter:z-ai/glm-5.3] | $0.000002 | $0.001848 | $0.000000 | $0.001364 |
| iterative_default[generator=openrouter:moonshotai/kimi-k3] | $0.000002 | $0.007249 | $0.000000 | $0.003134 |
| iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | $0.000002 | $0.001598 | $0.000000 | $0.001182 |
| iterative_default[generator=openrouter:qwen/qwen3.8-flash] | $0.000002 | $0.000567 | $0.000000 | $0.000404 |
| iterative_default[generator=openrouter:minimax/minimax-m3] | $0.000002 | $0.001151 | $0.000000 | $0.001170 |
| iterative_default[generator=gpt-6-luna] | $0.000002 | $0.000442 | $0.000000 | $0.000240 |
| iterative_default[generator=anthropic:claude-haiku-5-5] | $0.000002 | $0.000823 | $0.000000 | $0.000667 |
| agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash] | $0.000000 | $0.001367 | $0.000001 | $0.000513 |
| agent_search_tools[generator=openrouter:z-ai/glm-5.3] | $0.000000 | $0.002539 | $0.000001 | $0.001611 |
| agent_search_tools[generator=openrouter:moonshotai/kimi-k3] | $0.000000 | $0.008763 | $0.000001 | $0.003945 |
| agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro] | $0.000000 | $0.003425 | $0.000001 | $0.001090 |
| agent_search_tools[generator=openrouter:qwen/qwen3.8-flash] | $0.000000 | $0.001514 | $0.000001 | $0.000545 |
| agent_search_tools[generator=openrouter:minimax/minimax-m3] | $0.000000 | $0.003008 | $0.000001 | $0.001272 |
| agent_search_tools[generator=gpt-6-luna] | $0.000000 | $0.001422 | $0.000002 | $0.000586 |
| agent_search_tools[generator=anthropic:claude-haiku-5-5] | $0.000000 | $0.002094 | $0.000001 | $0.001167 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | tool | route | generate |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | 556.7 ms | 27.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1010.4 ms |
| hybrid_rerank_default[generator=openrouter:z-ai/glm-5.3] | 573.1 ms | 28.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3665.0 ms |
| hybrid_rerank_default[generator=openrouter:moonshotai/kimi-k3] | 555.8 ms | 28.3 ms | 0.0 ms | 0.0 ms | 0.0 ms | 2214.6 ms |
| hybrid_rerank_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | 579.1 ms | 24.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 8135.8 ms |
| hybrid_rerank_default[generator=openrouter:qwen/qwen3.8-flash] | 540.6 ms | 23.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 4372.1 ms |
| hybrid_rerank_default[generator=openrouter:minimax/minimax-m3] | 536.7 ms | 30.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 2599.7 ms |
| hybrid_rerank_default[generator=gpt-6-luna] | 643.2 ms | 36.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1729.8 ms |
| hybrid_rerank_default[generator=anthropic:claude-haiku-5-5] | 612.9 ms | 26.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3220.4 ms |
| iterative_default[generator=openrouter:deepseek/deepseek-v4.1-flash] | 588.0 ms | 0.0 ms | 3470.0 ms | 0.0 ms | 0.0 ms | 1112.0 ms |
| iterative_default[generator=openrouter:z-ai/glm-5.3] | 573.0 ms | 0.0 ms | 6077.3 ms | 0.0 ms | 0.0 ms | 2415.6 ms |
| iterative_default[generator=openrouter:moonshotai/kimi-k3] | 602.6 ms | 0.0 ms | 6112.5 ms | 0.0 ms | 0.0 ms | 2036.6 ms |
| iterative_default[generator=openrouter:xiaomi/mimo-v2.6-pro] | 686.7 ms | 0.0 ms | 18123.2 ms | 0.0 ms | 0.0 ms | 7591.0 ms |
| iterative_default[generator=openrouter:qwen/qwen3.8-flash] | 573.2 ms | 0.0 ms | 11667.6 ms | 0.0 ms | 0.0 ms | 4016.2 ms |
| iterative_default[generator=openrouter:minimax/minimax-m3] | 584.4 ms | 0.0 ms | 4876.5 ms | 0.0 ms | 0.0 ms | 3510.6 ms |
| iterative_default[generator=gpt-6-luna] | 810.0 ms | 0.0 ms | 6330.1 ms | 0.0 ms | 0.0 ms | 1535.6 ms |
| iterative_default[generator=anthropic:claude-haiku-5-5] | 567.0 ms | 0.0 ms | 6140.7 ms | 0.0 ms | 0.0 ms | 3598.7 ms |
| agent_search_tools[generator=openrouter:deepseek/deepseek-v4.1-flash] | 0.0 ms | 0.0 ms | 4574.1 ms | 836.0 ms | 0.0 ms | 2203.0 ms |
| agent_search_tools[generator=openrouter:z-ai/glm-5.3] | 0.0 ms | 0.0 ms | 3666.4 ms | 684.2 ms | 0.0 ms | 2979.5 ms |
| agent_search_tools[generator=openrouter:moonshotai/kimi-k3] | 0.0 ms | 0.0 ms | 4986.7 ms | 682.3 ms | 0.0 ms | 2529.3 ms |
| agent_search_tools[generator=openrouter:xiaomi/mimo-v2.6-pro] | 0.0 ms | 0.0 ms | 17348.1 ms | 1595.9 ms | 0.1 ms | 7676.1 ms |
| agent_search_tools[generator=openrouter:qwen/qwen3.8-flash] | 0.0 ms | 0.0 ms | 41706.0 ms | 995.6 ms | 0.1 ms | 14371.2 ms |
| agent_search_tools[generator=openrouter:minimax/minimax-m3] | 0.0 ms | 0.0 ms | 7532.4 ms | 1665.2 ms | 0.0 ms | 3173.2 ms |
| agent_search_tools[generator=gpt-6-luna] | 0.0 ms | 0.0 ms | 7422.0 ms | 1257.6 ms | 0.1 ms | 2452.8 ms |
| agent_search_tools[generator=anthropic:claude-haiku-5-5] | 0.0 ms | 0.0 ms | 5131.9 ms | 1341.0 ms | 0.0 ms | 3661.1 ms |
