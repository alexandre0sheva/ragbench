# RAGBench Leaderboard

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_retrieval_floor | — | — | — | 1.95 [1.74, 2.17] | 2.51 [2.17, 2.84] | 0.16 | — | $0.00021 | $0.00241 | 1402 ms | 2150 ms [2003 ms, 2482 ms] | ✓ | 0 | 5587 ms | Showing how much retrieval helps and how often the model makes things up without it |
| hybrid_rerank_default | 0.653 [0.597, 0.709] | 0.737 [0.676, 0.796] | 0.641 [0.588, 0.691] | 3.65 [3.50, 3.81] | 4.00 [3.86, 4.14] | 0.07 | 0.69 | $0.00109 | $0.00218 | 2485 ms | 3305 ms [3011 ms, 7941 ms] | ✓ | 0 | 33266 ms | Recall of hybrid plus rerank precision |
| agent_search_tools | 0.691 [0.637, 0.747] | 0.699 [0.637, 0.760] | 0.647 [0.596, 0.701] | 3.44 [3.25, 3.61] | 3.57 [3.38, 3.75] | 0.05 | 0.69 | $0.00264 | $0.00608 | 11136 ms | 16177 ms [14659 ms, 18698 ms] |  | 0 | 58368 ms | Questions that need computation, exact lookups, or several searches |
| claude_code | 0.632 [0.574, 0.692] | 0.693 [0.631, 0.758] | 0.633 [0.579, 0.687] | 3.55 [3.35, 3.76] | 3.50 [3.27, 3.73] | 0.02 | 0.94 | $0.01353 | $0.02636 | 23102 ms | 31459 ms |  | 0 | 1475871 ms | Measuring a coding-agent harness as a RAG system: how well, how fast and at what cost it finds the answer in your documents |
| codex | 0.473 [0.410, 0.536] | 0.514 [0.447, 0.581] | 0.484 [0.425, 0.544] | 2.92 [2.73, 3.12] | 2.54 [2.31, 2.77] | 0.04 | 0.89 | $0.01598 | $0.06145 | 28510 ms | 40058 ms [37863 ms, 43587 ms] |  | 0 | 1182367 ms | Measuring a coding-agent harness as a RAG system: how well, how fast and at what cost it finds the answer in your documents |
| opencode_glm | 0.656 [0.596, 0.714] | 0.737 [0.674, 0.798] | 0.670 [0.611, 0.726] | 3.64 [3.50, 3.77] | 3.03 [2.84, 3.20] | 0.03 | 0.95 | $0.01023 | $0.02275 | 12188 ms | 22541 ms |  | 1/150 | 1665373 ms | Measuring a coding-agent harness as a RAG system: how well, how fast and at what cost it finds the answer in your documents |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs hybrid_rerank_default

Each system minus the baseline `hybrid_rerank_default` (config), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| no_retrieval_floor | Answer | -1.706 | [-1.908, -1.491] | 0.005 | worse | 14 / 7 / 129 |
| agent_search_tools | Answer | -0.218 | [-0.397, -0.031] | 0.072 | no clear difference | 54 / 19 / 77 |
| claude_code | Answer | -0.100 | [-0.335, +0.144] | 0.854 | no clear difference | 75 / 9 / 66 |
| codex | Answer | -0.731 | [-0.943, -0.519] | 0.005 | worse | 32 / 3 / 115 |
| opencode_glm | Answer | -0.010 | [-0.179, +0.154] | 0.903 | no clear difference | 60 / 14 / 75 |
| agent_search_tools | Recall@5 | +0.038 | [+0.003, +0.074] | 0.114 | no clear difference | 29 / 106 / 15 |
| claude_code | Recall@5 | -0.021 | [-0.073, +0.028] | 0.826 | no clear difference | 28 / 91 / 31 |
| codex | Recall@5 | -0.179 | [-0.244, -0.119] | 0.004 | worse | 18 / 73 / 59 |
| opencode_glm | Recall@5 | -0.001 | [-0.054, +0.050] | 0.964 | no clear difference | 34 / 86 / 29 |

## Cost by stage

Mean $ per question, answering only (the judge's cost is excluded). `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | llm | tool | generate |
| --- | --- | --- | --- | --- |
| no_retrieval_floor | $0.000000 | $0.000000 | $0.000000 | $0.000028 |
| hybrid_rerank_default | $0.000001 | $0.000000 | $0.000000 | $0.000374 |
| agent_search_tools | $0.000000 | $0.001455 | $0.000002 | $0.000597 |
| claude_code | $0.000000 | $0.009125 | $0.000000 | $0.003991 |
| codex | $0.000000 | $0.000000 | $0.000000 | $0.015244 |
| opencode_glm | $0.000000 | $0.006879 | $0.000000 | $0.002540 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | tool | route | generate |
| --- | --- | --- | --- | --- | --- | --- |
| no_retrieval_floor | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1402.3 ms |
| hybrid_rerank_default | 662.3 ms | 44.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1777.6 ms |
| agent_search_tools | 0.0 ms | 0.0 ms | 7556.9 ms | 1144.8 ms | 0.1 ms | 2434.2 ms |
| claude_code | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| codex | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| opencode_glm | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
