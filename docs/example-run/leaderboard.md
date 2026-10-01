# RAGBench Leaderboard

> **Note:** Self-preference risk: the judge (gpt-6-luna) is also the generator of bm25_default, vector_default, hybrid_default, rerank_default, hybrid_rerank_default, hyde_default, parent_doc_default, no_retrieval_floor, full_context_ceiling, sentence_window_default, vector_mmr, contextual_default, hierarchical_default, rag_fusion_default, decompose_default, corrective_default, iterative_default, agent_search_plain, agent_search_tools, grep_agent_default, adaptive_default. Judges tend to rate answers from their own model higher, so those scores may be inflated. Use a different model under `evaluation.judge.model`, or set `evaluation.judge.independent: false` to accept this.

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 0.745 [0.681, 0.808] | 0.720 [0.654, 0.782] | 0.708 [0.647, 0.767] | 4.69 [4.55, 4.81] | 4.99 [4.97, 5.00] | 0.57 | 0.88 | $0.00049 | $0.00052 | 1081 ms | 1378 ms | ✓ | 0 | 117997 ms | Cheap baseline and exact-term matching |
| vector_default | 0.790 [0.733, 0.847] | 0.738 [0.676, 0.796] | 0.718 [0.663, 0.774] | 4.67 [4.54, 4.79] | 4.94 [4.88, 4.99] | 0.58 | 0.91 | $0.00044 | $0.00047 | 1188 ms | 1637 ms | ✓ | 0 | 124238 ms | Semantic baseline |
| hybrid_default | 0.775 [0.715, 0.833] | 0.745 [0.679, 0.806] | 0.732 [0.671, 0.790] | 4.61 [4.46, 4.74] | 4.92 [4.82, 4.99] | 0.55 | 0.90 | $0.00048 | $0.00052 | 1232 ms | 1666 ms |  | 0 | 119307 ms | Balanced lexical + semantic retrieval |
| rerank_default | 0.789 [0.729, 0.846] | 0.747 [0.687, 0.805] | 0.732 [0.674, 0.786] | 4.74 [4.63, 4.84] | 4.95 [4.90, 4.99] | 0.59 | 0.92 | $0.00052 | $0.00055 | 1675 ms | 2201 ms | ✓ | 0 | 123635 ms | Higher precision context selection |
| hybrid_rerank_default | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.72 [4.59, 4.84] | 4.98 [4.95, 5.00] | 0.57 | 0.89 | $0.00053 | $0.00057 | 1432 ms | 2140 ms | ✓ | 0 | 88290 ms | Recall of hybrid plus rerank precision |
| hyde_default | 0.787 [0.729, 0.845] | 0.725 [0.664, 0.784] | 0.710 [0.656, 0.765] | 4.68 [4.54, 4.80] | 4.97 [4.92, 5.00] | 0.56 | 0.91 | $0.00046 | $0.00050 | 2798 ms | 3479 ms |  | 0 | 161818 ms | Short or vaguely-worded questions |
| parent_doc_default | 0.789 [0.733, 0.845] | 0.738 [0.676, 0.796] | 0.717 [0.663, 0.773] | 4.71 [4.60, 4.81] | 4.93 [4.85, 4.99] | 0.57 | 0.89 | $0.00049 | $0.00053 | 1207 ms | 1514 ms | ✓ | 0 | 122532 ms | Better answer context with precise retrieval |
| llm_heavy_default | 0.790 [0.728, 0.848] | 0.770 [0.710, 0.827] | 0.747 [0.692, 0.801] | 4.72 [4.60, 4.82] | 4.93 [4.86, 4.98] | 0.55 | 0.92 | $0.00512 | $0.00539 | 5975 ms | 8055 ms |  | 0 | 380425 ms | Higher-cost, quality-oriented experiments |
| no_retrieval_floor | — | — | — | 1.54 [1.40, 1.68] | 4.69 [4.51, 4.88] | 0.03 | — | $0.00012 | $0.00406 | 943 ms | 1093 ms | ✓ | 0 | 131282 ms | Showing how much retrieval helps and how often the model makes things up without it |
| full_context_ceiling | 0.741 [0.676, 0.804] | 0.728 [0.666, 0.791] | 0.715 [0.657, 0.772] | 4.78 [4.70, 4.84] | 4.84 [4.71, 4.94] | 0.60 | 1.00 | $0.00280 | $0.00297 | 1337 ms | 2257 ms | ✓ | 0 | 162644 ms | Small corpora, and measuring how much a retrieval pipeline loses against reading everything |
| sentence_window_default | 0.780 [0.720, 0.838] | 0.692 [0.631, 0.755] | 0.693 [0.637, 0.749] | 4.42 [4.23, 4.59] | 4.82 [4.67, 4.94] | 0.54 | 0.92 | $0.00025 | $0.00028 | 1195 ms | 1562 ms | ✓ | 0 | 131671 ms | Precise matching with enough surrounding text to answer from |
| vector_mmr | 0.697 [0.635, 0.757] | 0.731 [0.668, 0.792] | 0.679 [0.620, 0.739] | 4.54 [4.40, 4.68] | 4.94 [4.89, 4.99] | 0.55 | 0.82 | $0.00040 | $0.00046 | 1085 ms | 1533 ms | ✓ | 0 | 127569 ms | Semantic baseline |
| contextual_default | 0.784 [0.725, 0.840] | 0.753 [0.691, 0.812] | 0.741 [0.684, 0.798] | 4.69 [4.57, 4.81] | 4.93 [4.86, 4.99] | 0.58 | 0.91 | $0.00046 | $0.00050 | 1164 ms | 1605 ms | ✓ | 0 | 161942 ms | Corpora where chunks lose their meaning out of context (many similar documents, pronouns, section-relative facts) |
| hierarchical_default | 0.733 [0.669, 0.795] | 0.707 [0.642, 0.770] | 0.666 [0.607, 0.725] | 4.56 [4.40, 4.70] | 4.96 [4.91, 5.00] | 0.54 | 0.87 | $0.00033 | $0.00037 | 1091 ms | 1148 ms | ✓ | 0 | 147522 ms | "Which document?" questions and corpora too large to search chunk by chunk |
| rag_fusion_default | 0.807 [0.752, 0.864] | 0.743 [0.684, 0.800] | 0.742 [0.689, 0.796] | 4.75 [4.64, 4.86] | 4.96 [4.91, 5.00] | 0.59 | 0.93 | $0.00042 | $0.00045 | 3107 ms | 3488 ms | ✓ | 1/163 | 206080 ms | Questions worded differently from the documents |
| decompose_default | 0.771 [0.713, 0.831] | 0.728 [0.666, 0.787] | 0.723 [0.667, 0.781] | 4.72 [4.60, 4.83] | 4.93 [4.83, 4.99] | 0.58 | 0.91 | $0.00041 | $0.00044 | 2444 ms | 3368 ms | ✓ | 0 | 173806 ms | Multi-part, comparison and multi-hop questions |
| corrective_default | 0.793 [0.733, 0.850] | 0.830 [0.774, 0.883] | 0.796 [0.742, 0.850] | 4.77 [4.67, 4.86] | 4.99 [4.97, 5.00] | 0.59 | 0.93 | $0.00060 | $0.00063 | 5258 ms | 8715 ms |  | 0 | 235983 ms | Corpora where the first search often misses and a bad context should be noticed rather than answered from |
| iterative_default | 0.799 [0.742, 0.856] | 0.780 [0.721, 0.833] | 0.769 [0.713, 0.823] | 4.81 [4.71, 4.90] | 4.91 [4.83, 4.98] | 0.58 | 0.94 | $0.00057 | $0.00059 | 3254 ms | 3952 ms | ✓ | 0 | 251357 ms | Multi-hop questions whose second search depends on what the first one found |
| agent_search_plain | 0.814 [0.756, 0.871] | 0.766 [0.708, 0.826] | 0.758 [0.703, 0.813] | 4.58 [4.43, 4.72] | 4.79 [4.64, 4.90] | 0.53 | 0.94 | $0.00072 | $0.00080 | 3562 ms | 5403 ms |  | 0 | 224722 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools | 0.821 [0.767, 0.876] | 0.759 [0.699, 0.818] | 0.757 [0.703, 0.812] | 4.75 [4.63, 4.85] | 4.85 [4.72, 4.94] | 0.55 | 0.96 | $0.00082 | $0.00086 | 3316 ms | 5783 ms |  | 0 | 202125 ms | Questions that need computation, exact lookups, or several searches |
| grep_agent_default | 0.791 [0.736, 0.850] | 0.694 [0.634, 0.754] | 0.696 [0.640, 0.750] | 4.78 [4.69, 4.86] | 4.73 [4.56, 4.87] | 0.58 | 0.92 | $0.00109 | $0.00116 | 6557 ms | 9437 ms |  | 0 | 297447 ms | Small or exact-match-heavy corpora, and testing whether you need retrieval infrastructure at all |
| adaptive_default | 0.771 [0.712, 0.828] | 0.724 [0.661, 0.784] | 0.719 [0.661, 0.774] | 4.70 [4.57, 4.82] | 4.99 [4.98, 5.00] | 0.56 | 0.91 | $0.00044 | $0.00047 | 1853 ms | 3492 ms | ✓ | 0 | 149247 ms | Mixed workloads where no single architecture is best for every question |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs no_retrieval_floor

Each system minus the baseline `no_retrieval_floor` (cheapest system), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_default | Answer | +3.151 | [+2.948, +3.355] | 0.021 | better | 159 / 3 / 1 |
| vector_default | Answer | +3.137 | [+2.929, +3.340] | 0.021 | better | 158 / 2 / 3 |
| hybrid_default | Answer | +3.070 | [+2.855, +3.287] | 0.021 | better | 157 / 3 / 3 |
| rerank_default | Answer | +3.206 | [+3.013, +3.396] | 0.021 | better | 160 / 1 / 2 |
| hybrid_rerank_default | Answer | +3.188 | [+2.985, +3.379] | 0.021 | better | 161 / 1 / 1 |
| hyde_default | Answer | +3.144 | [+2.937, +3.346] | 0.021 | better | 158 / 4 / 1 |
| parent_doc_default | Answer | +3.177 | [+2.994, +3.356] | 0.021 | better | 162 / 1 / 0 |
| llm_heavy_default | Answer | +3.180 | [+2.977, +3.379] | 0.021 | better | 158 / 3 / 2 |
| full_context_ceiling | Answer | +3.239 | [+3.061, +3.413] | 0.021 | better | 160 / 2 / 1 |
| sentence_window_default | Answer | +2.880 | [+2.633, +3.110] | 0.021 | better | 154 / 3 / 6 |
| vector_mmr | Answer | +3.007 | [+2.794, +3.215] | 0.021 | better | 161 / 1 / 1 |
| contextual_default | Answer | +3.158 | [+2.958, +3.358] | 0.021 | better | 159 / 1 / 3 |
| hierarchical_default | Answer | +3.022 | [+2.820, +3.237] | 0.021 | better | 158 / 4 / 1 |
| rag_fusion_default | Answer | +3.215 | [+3.028, +3.406] | 0.021 | better | 159 / 1 / 2 |
| decompose_default | Answer | +3.185 | [+2.982, +3.384] | 0.021 | better | 160 / 1 / 2 |
| corrective_default | Answer | +3.232 | [+3.048, +3.412] | 0.021 | better | 161 / 1 / 1 |
| iterative_default | Answer | +3.275 | [+3.098, +3.454] | 0.021 | better | 161 / 1 / 1 |
| agent_search_plain | Answer | +3.044 | [+2.818, +3.259] | 0.021 | better | 157 / 2 / 4 |
| agent_search_tools | Answer | +3.216 | [+3.010, +3.413] | 0.021 | better | 157 / 1 / 5 |
| grep_agent_default | Answer | +3.247 | [+3.072, +3.428] | 0.021 | better | 162 / 0 / 1 |
| adaptive_default | Answer | +3.168 | [+2.964, +3.356] | 0.021 | better | 161 / 0 / 2 |

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
| llm_heavy_default | $0.000001 | $0.000734 | $0.000000 | $0.000000 | $0.004096 |
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
| agent_search_plain | $0.000000 | $0.000262 | $0.000001 | $0.000000 | $0.000223 |
| agent_search_tools | $0.000000 | $0.000346 | $0.000000 | $0.000000 | $0.000249 |
| grep_agent_default | $0.000000 | $0.000652 | $0.000000 | $0.000000 | $0.000296 |
| adaptive_default | $0.000000 | $0.000041 | $0.000000 | $0.000000 | $0.000158 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | tool | route | grade | generate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 0.8 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1195.5 ms |
| vector_default | 169.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1101.4 ms |
| hybrid_default | 208.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1098.6 ms |
| rerank_default | 229.0 ms | 17.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1194.9 ms |
| hybrid_rerank_default | 244.9 ms | 17.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1087.5 ms |
| hyde_default | 319.2 ms | 0.0 ms | 1350.1 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1057.1 ms |
| parent_doc_default | 170.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1116.8 ms |
| llm_heavy_default | 598.5 ms | 9.4 ms | 3671.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 2896.1 ms |
| no_retrieval_floor | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1313.4 ms |
| full_context_ceiling | 0.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1451.8 ms |
| sentence_window_default | 146.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1090.4 ms |
| vector_mmr | 142.9 ms | 0.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1109.2 ms |
| contextual_default | 152.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1465.2 ms |
| hierarchical_default | 1.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1067.8 ms |
| rag_fusion_default | 713.1 ms | 0.0 ms | 1580.1 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1136.7 ms |
| decompose_default | 193.9 ms | 0.0 ms | 1105.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1006.6 ms |
| corrective_default | 214.2 ms | 0.0 ms | 586.0 ms | 0.0 ms | 0.0 ms | 2084.7 ms | 1104.9 ms |
| iterative_default | 271.6 ms | 0.0 ms | 3008.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1303.8 ms |
| agent_search_plain | 0.0 ms | 0.0 ms | 2305.8 ms | 319.6 ms | 0.0 ms | 0.0 ms | 1170.8 ms |
| agent_search_tools | 0.0 ms | 0.0 ms | 2307.4 ms | 238.4 ms | 0.0 ms | 0.0 ms | 1111.2 ms |
| grep_agent_default | 0.0 ms | 0.0 ms | 4221.2 ms | 4.1 ms | 0.0 ms | 0.0 ms | 1315.5 ms |
| adaptive_default | 140.0 ms | 9.4 ms | 447.4 ms | 28.3 ms | 0.1 ms | 0.0 ms | 1139.0 ms |
