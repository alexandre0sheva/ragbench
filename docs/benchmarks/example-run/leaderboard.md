# RAGBench Leaderboard

> **Note:** Self-preference risk: the judge (gpt-6-luna) is also the generator of bm25_default, vector_default, hybrid_default, rerank_default, hybrid_rerank_default, hyde_default, parent_doc_default, no_retrieval_floor, full_context_ceiling, sentence_window_default, vector_mmr, contextual_default, hierarchical_default, rag_fusion_default, decompose_default, corrective_default, iterative_default, agent_search_plain, agent_search_tools, grep_agent_default, adaptive_default, decision_gate_default, decision_gate_strict. Judges tend to rate answers from their own model higher, so those scores may be inflated. Use a different model under `evaluation.judge.model`, or set `evaluation.judge.independent: false` to accept this.

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 0.745 [0.681, 0.808] | 0.720 [0.654, 0.782] | 0.708 [0.647, 0.767] | 4.69 [4.55, 4.81] | 4.99 [4.97, 5.00] | 0.57 | 0.88 | $0.00049 | $0.00052 | 1163 ms | 1371 ms | ✓ | 0 | 306 ms | Cheap baseline and exact-term matching |
| vector_default | 0.790 [0.733, 0.847] | 0.738 [0.676, 0.796] | 0.718 [0.663, 0.774] | 4.67 [4.54, 4.79] | 4.94 [4.88, 4.99] | 0.58 | 0.91 | $0.00044 | $0.00047 | 1723 ms | 2374 ms | ✓ | 0 | 7214 ms | Semantic baseline |
| hybrid_default | 0.775 [0.715, 0.833] | 0.745 [0.679, 0.806] | 0.732 [0.671, 0.790] | 4.61 [4.46, 4.74] | 4.92 [4.82, 4.99] | 0.55 | 0.90 | $0.00048 | $0.00052 | 1653 ms | 1895 ms |  | 0 | 8774 ms | Balanced lexical + semantic retrieval |
| rerank_default | 0.789 [0.729, 0.846] | 0.747 [0.687, 0.805] | 0.732 [0.674, 0.786] | 4.74 [4.63, 4.84] | 4.95 [4.90, 4.99] | 0.59 | 0.92 | $0.00052 | $0.00055 | 1746 ms | 3054 ms | ✓ | 0 | 11246 ms | Higher precision context selection |
| hybrid_rerank_default | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.72 [4.59, 4.84] | 4.98 [4.95, 5.00] | 0.57 | 0.89 | $0.00053 | $0.00057 | 1964 ms | 3463 ms |  | 0 | 10788 ms | Recall of hybrid plus rerank precision |
| hyde_default | 0.787 [0.729, 0.845] | 0.725 [0.664, 0.784] | 0.710 [0.656, 0.765] | 4.68 [4.54, 4.80] | 4.97 [4.92, 5.00] | 0.56 | 0.91 | $0.00046 | $0.00050 | 2786 ms | 3110 ms |  | 0 | 38178 ms | Short or vaguely-worded questions |
| parent_doc_default | 0.789 [0.733, 0.845] | 0.738 [0.676, 0.796] | 0.717 [0.663, 0.773] | 4.71 [4.60, 4.81] | 4.93 [4.85, 4.99] | 0.57 | 0.89 | $0.00049 | $0.00053 | 2655 ms | 5704 ms |  | 0 | 5878 ms | Better answer context with precise retrieval |
| llm_heavy_default | 0.790 [0.728, 0.848] | 0.770 [0.710, 0.827] | 0.747 [0.692, 0.801] | 4.72 [4.60, 4.82] | 4.93 [4.86, 4.98] | 0.55 | 0.92 | $0.00512 | $0.00539 | 5273 ms | 7579 ms |  | 0 | 21480 ms | Higher-cost, quality-oriented experiments |
| no_retrieval_floor | — | — | — | 1.54 [1.40, 1.68] | 4.69 [4.51, 4.88] | 0.03 | — | $0.00012 | $0.00406 | 1054 ms | 1430 ms | ✓ | 0 | 60 ms | Showing how much retrieval helps and how often the model makes things up without it |
| full_context_ceiling | 0.741 [0.676, 0.804] | 0.728 [0.666, 0.791] | 0.715 [0.657, 0.772] | 4.78 [4.70, 4.84] | 4.84 [4.71, 4.94] | 0.60 | 1.00 | $0.00280 | $0.00297 | 1599 ms | 2038 ms | ✓ | 0 | 711 ms | Small corpora, and measuring how much a retrieval pipeline loses against reading everything |
| sentence_window_default | 0.780 [0.720, 0.838] | 0.692 [0.631, 0.755] | 0.693 [0.637, 0.749] | 4.42 [4.23, 4.59] | 4.82 [4.67, 4.94] | 0.54 | 0.92 | $0.00025 | $0.00028 | 1258 ms | 1518 ms | ✓ | 0 | 6029 ms | Precise matching with enough surrounding text to answer from |
| vector_mmr | 0.697 [0.635, 0.757] | 0.731 [0.668, 0.792] | 0.679 [0.620, 0.739] | 4.54 [4.40, 4.68] | 4.94 [4.89, 4.99] | 0.55 | 0.82 | $0.00040 | $0.00046 | 1320 ms | 1541 ms |  | 0 | 5836 ms | Semantic baseline |
| contextual_default | 0.784 [0.725, 0.840] | 0.753 [0.691, 0.812] | 0.741 [0.684, 0.798] | 4.69 [4.57, 4.81] | 4.93 [4.86, 4.99] | 0.58 | 0.91 | $0.00046 | $0.00050 | 1298 ms | 1690 ms | ✓ | 0 | 5960 ms | Corpora where chunks lose their meaning out of context (many similar documents, pronouns, section-relative facts) |
| hierarchical_default | 0.733 [0.669, 0.795] | 0.707 [0.642, 0.770] | 0.666 [0.607, 0.725] | 4.56 [4.40, 4.70] | 4.96 [4.91, 5.00] | 0.54 | 0.87 | $0.00033 | $0.00037 | 1287 ms | 1604 ms | ✓ | 0 | 5852 ms | "Which document?" questions and corpora too large to search chunk by chunk |
| rag_fusion_default | 0.808 [0.753, 0.863] | 0.744 [0.684, 0.802] | 0.744 [0.688, 0.800] | 4.75 [4.64, 4.86] | 4.96 [4.91, 5.00] | 0.59 | 0.93 | $0.00042 | $0.00045 | 3543 ms | 4093 ms | ✓ | 0 | 30327 ms | Questions worded differently from the documents |
| decompose_default | 0.771 [0.713, 0.831] | 0.728 [0.666, 0.787] | 0.723 [0.667, 0.781] | 4.72 [4.60, 4.83] | 4.93 [4.83, 4.99] | 0.58 | 0.91 | $0.00041 | $0.00044 | 2523 ms | 2888 ms | ✓ | 0 | 7849 ms | Multi-part, comparison and multi-hop questions |
| corrective_default | 0.793 [0.733, 0.850] | 0.830 [0.774, 0.883] | 0.796 [0.742, 0.850] | 4.77 [4.67, 4.86] | 4.99 [4.97, 5.00] | 0.59 | 0.93 | $0.00060 | $0.00063 | 5411 ms | 9595 ms |  | 0 | 8550 ms | Corpora where the first search often misses and a bad context should be noticed rather than answered from |
| iterative_default | 0.799 [0.742, 0.856] | 0.780 [0.721, 0.833] | 0.769 [0.713, 0.823] | 4.81 [4.71, 4.90] | 4.91 [4.83, 4.98] | 0.58 | 0.94 | $0.00057 | $0.00059 | 4903 ms | 6423 ms | ✓ | 0 | 11554 ms | Multi-hop questions whose second search depends on what the first one found |
| agent_search_plain | 0.814 [0.756, 0.871] | 0.766 [0.708, 0.826] | 0.758 [0.703, 0.813] | 4.58 [4.43, 4.72] | 4.79 [4.64, 4.90] | 0.53 | 0.94 | $0.00072 | $0.00080 | 3489 ms | 5345 ms |  | 0 | 13194 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools | 0.821 [0.767, 0.876] | 0.759 [0.699, 0.818] | 0.757 [0.703, 0.812] | 4.75 [4.63, 4.85] | 4.85 [4.72, 4.94] | 0.55 | 0.96 | $0.00082 | $0.00086 | 3606 ms | 5753 ms |  | 0 | 10094 ms | Questions that need computation, exact lookups, or several searches |
| grep_agent_default | 0.791 [0.736, 0.850] | 0.694 [0.634, 0.754] | 0.696 [0.640, 0.750] | 4.78 [4.69, 4.86] | 4.73 [4.56, 4.87] | 0.58 | 0.92 | $0.00109 | $0.00116 | 6334 ms | 8197 ms |  | 0 | 675 ms | Small or exact-match-heavy corpora, and testing whether you need retrieval infrastructure at all |
| adaptive_default | 0.771 [0.712, 0.828] | 0.724 [0.661, 0.784] | 0.719 [0.661, 0.774] | 4.70 [4.57, 4.82] | 4.99 [4.98, 5.00] | 0.56 | 0.91 | $0.00044 | $0.00047 | 2621 ms | 4119 ms |  | 0 | 7099 ms | Mixed workloads where no single architecture is best for every question |
| decision_gate_default | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.80 [4.70, 4.89] | 4.98 [4.94, 5.00] | 0.57 | 0.93 | $0.00094 | $0.00099 | 3226 ms | 4801 ms | ✓ | 0 | 9706 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_strict | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.65 [4.49, 4.80] | 4.97 [4.91, 5.00] | 0.55 | 0.89 | $0.00091 | $0.00099 | 1745 ms | 3086 ms |  | 0 | 10061 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs no_retrieval_floor

Each system minus the baseline `no_retrieval_floor` (cheapest system), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_default | Answer | +3.151 | [+2.948, +3.355] | 0.023 | better | 159 / 3 / 1 |
| vector_default | Answer | +3.137 | [+2.929, +3.340] | 0.023 | better | 158 / 2 / 3 |
| hybrid_default | Answer | +3.070 | [+2.855, +3.287] | 0.023 | better | 157 / 3 / 3 |
| rerank_default | Answer | +3.207 | [+3.015, +3.396] | 0.023 | better | 160 / 1 / 2 |
| hybrid_rerank_default | Answer | +3.188 | [+2.985, +3.379] | 0.023 | better | 161 / 1 / 1 |
| hyde_default | Answer | +3.144 | [+2.937, +3.346] | 0.023 | better | 158 / 4 / 1 |
| parent_doc_default | Answer | +3.177 | [+2.994, +3.356] | 0.023 | better | 162 / 1 / 0 |
| llm_heavy_default | Answer | +3.180 | [+2.977, +3.379] | 0.023 | better | 158 / 3 / 2 |
| full_context_ceiling | Answer | +3.239 | [+3.061, +3.413] | 0.023 | better | 160 / 2 / 1 |
| sentence_window_default | Answer | +2.880 | [+2.633, +3.110] | 0.023 | better | 154 / 3 / 6 |
| vector_mmr | Answer | +3.007 | [+2.794, +3.215] | 0.023 | better | 161 / 1 / 1 |
| contextual_default | Answer | +3.158 | [+2.958, +3.358] | 0.023 | better | 159 / 1 / 3 |
| hierarchical_default | Answer | +3.022 | [+2.820, +3.237] | 0.023 | better | 158 / 4 / 1 |
| rag_fusion_default | Answer | +3.218 | [+3.028, +3.409] | 0.023 | better | 160 / 1 / 2 |
| decompose_default | Answer | +3.185 | [+2.982, +3.384] | 0.023 | better | 160 / 1 / 2 |
| corrective_default | Answer | +3.232 | [+3.048, +3.412] | 0.023 | better | 161 / 1 / 1 |
| iterative_default | Answer | +3.275 | [+3.098, +3.454] | 0.023 | better | 161 / 1 / 1 |
| agent_search_plain | Answer | +3.044 | [+2.818, +3.259] | 0.023 | better | 157 / 2 / 4 |
| agent_search_tools | Answer | +3.216 | [+3.010, +3.413] | 0.023 | better | 157 / 1 / 5 |
| grep_agent_default | Answer | +3.247 | [+3.072, +3.428] | 0.023 | better | 162 / 0 / 1 |
| adaptive_default | Answer | +3.168 | [+2.964, +3.356] | 0.023 | better | 161 / 0 / 2 |
| decision_gate_default | Answer | +3.269 | [+3.092, +3.442] | 0.023 | better | 161 / 1 / 1 |
| decision_gate_strict | Answer | +3.118 | [+2.905, +3.330] | 0.023 | better | 154 / 8 / 1 |

## Cost by stage

Mean $ per question, answering only (the judge's cost is excluded). `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | llm | tool | grade | generate |
| --- | --- | --- | --- | --- | --- |
| bm25_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000198 |
| vector_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000172 |
| hybrid_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000193 |
| rerank_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000214 |
| hybrid_rerank_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000223 |
| hyde_default | $0.000001 | $0.000030 | $0.000000 | $0.000000 | $0.000170 |
| parent_doc_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000212 |
| llm_heavy_default | $0.000001 | $0.000734 | $0.000000 | $0.000000 | $0.004098 |
| no_retrieval_floor | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000017 |
| full_context_ceiling | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.002415 |
| sentence_window_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000078 |
| vector_mmr | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000151 |
| contextual_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000183 |
| hierarchical_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000119 |
| rag_fusion_default | $0.000001 | $0.000043 | $0.000000 | $0.000000 | $0.000143 |
| decompose_default | $0.000000 | $0.000025 | $0.000000 | $0.000000 | $0.000147 |
| corrective_default | $0.000001 | $0.000019 | $0.000000 | $0.000198 | $0.000147 |
| iterative_default | $0.000001 | $0.000180 | $0.000000 | $0.000000 | $0.000152 |
| agent_search_plain | $0.000000 | $0.000262 | $0.000001 | $0.000000 | $0.000224 |
| agent_search_tools | $0.000000 | $0.000346 | $0.000000 | $0.000000 | $0.000249 |
| grep_agent_default | $0.000000 | $0.000652 | $0.000000 | $0.000000 | $0.000296 |
| adaptive_default | $0.000000 | $0.000041 | $0.000000 | $0.000000 | $0.000158 |
| decision_gate_default | $0.000000 | $0.000000 | $0.000000 | $0.000657 | $0.000099 |
| decision_gate_strict | $0.000000 | $0.000000 | $0.000000 | $0.000672 | $0.000072 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | tool | route | grade | generate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 0.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1194.6 ms |
| vector_default | 171.3 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1100.9 ms |
| hybrid_default | 209.1 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1098.1 ms |
| rerank_default | 224.3 ms | 21.3 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1199.2 ms |
| hybrid_rerank_default | 228.0 ms | 15.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1087.0 ms |
| hyde_default | 657.2 ms | 0.0 ms | 1349.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1056.6 ms |
| parent_doc_default | 141.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1115.9 ms |
| llm_heavy_default | 481.5 ms | 9.7 ms | 3670.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 2896.7 ms |
| no_retrieval_floor | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1312.9 ms |
| full_context_ceiling | 0.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1450.3 ms |
| sentence_window_default | 142.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1089.9 ms |
| vector_mmr | 139.2 ms | 0.8 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1108.7 ms |
| contextual_default | 143.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1464.7 ms |
| hierarchical_default | 0.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1067.1 ms |
| rag_fusion_default | 721.5 ms | 0.0 ms | 1579.3 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1136.5 ms |
| decompose_default | 188.7 ms | 0.0 ms | 1105.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1006.1 ms |
| corrective_default | 204.5 ms | 0.0 ms | 585.7 ms | 0.0 ms | 0.0 ms | 2083.3 ms | 1104.2 ms |
| iterative_default | 248.2 ms | 0.0 ms | 3006.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1306.5 ms |
| agent_search_plain | 0.0 ms | 0.0 ms | 2304.3 ms | 294.9 ms | 0.0 ms | 0.0 ms | 1172.0 ms |
| agent_search_tools | 0.0 ms | 0.0 ms | 2305.9 ms | 239.7 ms | 0.0 ms | 0.0 ms | 1110.3 ms |
| grep_agent_default | 0.0 ms | 0.0 ms | 4218.3 ms | 4.3 ms | 0.0 ms | 0.0 ms | 1314.6 ms |
| adaptive_default | 137.6 ms | 8.0 ms | 447.2 ms | 25.0 ms | 0.0 ms | 0.0 ms | 1138.3 ms |
| decision_gate_default | 215.2 ms | 15.8 ms | 0.0 ms | 0.0 ms | 0.0 ms | 537.7 ms | 1679.2 ms |
| decision_gate_strict | 218.6 ms | 19.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 501.2 ms | 1350.4 ms |
