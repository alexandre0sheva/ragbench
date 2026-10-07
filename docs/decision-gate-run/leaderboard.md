# RAGBench Leaderboard

> **Note:** Self-preference risk: the judge (gpt-6-luna) is also the generator of hybrid_rerank_default, corrective_default, no_retrieval_floor, decision_gate_default, decision_gate_strict, decision_gate_per_chunk, decision_gate_llm_json, decision_gate_no_sufficiency, decision_gate_sufficiency_only. Judges tend to rate answers from their own model higher, so those scores may be inflated. Use a different model under `evaluation.judge.model`, or set `evaluation.judge.independent: false` to accept this.

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.72 [4.59, 4.84] | 4.98 [4.95, 5.00] | 0.57 | 0.89 | $0.00053 | $0.00057 | 2024 ms | 2746 ms | ✓ | 0 | 10004 ms | Recall of hybrid plus rerank precision |
| corrective_default | 0.793 [0.733, 0.850] | 0.830 [0.774, 0.883] | 0.796 [0.742, 0.850] | 4.77 [4.67, 4.86] | 4.99 [4.97, 5.00] | 0.59 | 0.93 | $0.00060 | $0.00063 | 5243 ms | 8553 ms | ✓ | 0 | 8966 ms | Corpora where the first search often misses and a bad context should be noticed rather than answered from |
| no_retrieval_floor | — | — | — | 1.54 [1.40, 1.68] | 4.69 [4.51, 4.88] | 0.03 | — | $0.00012 | $0.00406 | 878 ms | 1139 ms | ✓ | 0 | 63 ms | Showing how much retrieval helps and how often the model makes things up without it |
| decision_gate_default | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.80 [4.70, 4.89] | 4.98 [4.94, 5.00] | 0.57 | 0.93 | $0.00094 | $0.00099 | 1918 ms | 2530 ms | ✓ | 0 | 9581 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_strict | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.65 [4.49, 4.80] | 4.97 [4.91, 5.00] | 0.55 | 0.89 | $0.00091 | $0.00099 | 1628 ms | 2802 ms | ✓ | 0 | 39020 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_per_chunk | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.45 [4.27, 4.60] | 4.96 [4.90, 4.99] | 0.52 | 0.82 | $0.00101 | $0.00116 | 2441 ms | 2836 ms |  | 0 | 128259 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_llm_json | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.71 [4.56, 4.84] | 4.96 [4.88, 5.00] | 0.56 | 0.89 | $0.00077 | $0.00083 | 4122 ms | 6618 ms |  | 0 | 147084 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_no_sufficiency | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.82 [4.72, 4.90] | 4.98 [4.94, 5.00] | 0.57 | 0.93 | $0.00086 | $0.00090 | 2314 ms | 2954 ms | ✓ | 0 | 19995 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_sufficiency_only | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.69 [4.55, 4.81] | 4.98 [4.95, 5.00] | 0.57 | 0.89 | $0.00130 | $0.00138 | 2250 ms | 2838 ms |  | 0 | 20578 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs no_retrieval_floor

Each system minus the baseline `no_retrieval_floor` (cheapest system), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default | Answer | +3.188 | [+2.985, +3.379] | 0.008 | better | 161 / 1 / 1 |
| corrective_default | Answer | +3.232 | [+3.048, +3.412] | 0.008 | better | 161 / 1 / 1 |
| decision_gate_default | Answer | +3.269 | [+3.092, +3.442] | 0.008 | better | 161 / 1 / 1 |
| decision_gate_strict | Answer | +3.118 | [+2.905, +3.330] | 0.008 | better | 154 / 8 / 1 |
| decision_gate_per_chunk | Answer | +2.911 | [+2.683, +3.142] | 0.008 | better | 154 / 7 / 2 |
| decision_gate_llm_json | Answer | +3.172 | [+2.974, +3.361] | 0.008 | better | 157 / 6 / 0 |
| decision_gate_no_sufficiency | Answer | +3.285 | [+3.115, +3.452] | 0.008 | better | 162 / 0 / 1 |
| decision_gate_sufficiency_only | Answer | +3.157 | [+2.944, +3.352] | 0.008 | better | 161 / 1 / 1 |

## Cost by stage

Mean $ per question, answering only (the judge's cost is excluded). `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | llm | grade | generate |
| --- | --- | --- | --- | --- |
| hybrid_rerank_default | $0.000000 | $0.000000 | $0.000000 | $0.000223 |
| corrective_default | $0.000001 | $0.000019 | $0.000198 | $0.000147 |
| no_retrieval_floor | $0.000000 | $0.000000 | $0.000000 | $0.000017 |
| decision_gate_default | $0.000000 | $0.000000 | $0.000657 | $0.000099 |
| decision_gate_strict | $0.000000 | $0.000000 | $0.000672 | $0.000072 |
| decision_gate_per_chunk | $0.000000 | $0.000000 | $0.000691 | $0.000111 |
| decision_gate_llm_json | $0.000000 | $0.000000 | $0.000540 | $0.000070 |
| decision_gate_no_sufficiency | $0.000000 | $0.000000 | $0.000566 | $0.000102 |
| decision_gate_sufficiency_only | $0.000000 | $0.000000 | $0.000773 | $0.000213 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | route | grade | generate |
| --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank_default | 223.4 ms | 16.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1087.0 ms |
| corrective_default | 212.7 ms | 0.0 ms | 585.7 ms | 0.0 ms | 2083.3 ms | 1104.2 ms |
| no_retrieval_floor | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1312.9 ms |
| decision_gate_default | 211.3 ms | 16.0 ms | 0.0 ms | 0.0 ms | 537.7 ms | 1679.2 ms |
| decision_gate_strict | 400.8 ms | 16.6 ms | 0.0 ms | 0.0 ms | 501.2 ms | 1350.4 ms |
| decision_gate_per_chunk | 217.8 ms | 18.1 ms | 0.0 ms | 0.0 ms | 847.5 ms | 1339.9 ms |
| decision_gate_llm_json | 227.2 ms | 14.1 ms | 0.0 ms | 0.0 ms | 2937.9 ms | 1334.7 ms |
| decision_gate_no_sufficiency | 217.6 ms | 17.2 ms | 0.0 ms | 0.0 ms | 339.0 ms | 1711.6 ms |
| decision_gate_sufficiency_only | 216.5 ms | 15.4 ms | 0.0 ms | 0.0 ms | 610.3 ms | 989.3 ms |
