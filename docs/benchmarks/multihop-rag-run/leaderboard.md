# RAGBench Leaderboard

> **Note:** The LLM judge returned an unusable verdict for some questions and the heuristic judge scored them instead: llamaindex_default (0%). Their answer scores mix two judges; see `judge_fallback_rate` in `metrics_summary.csv`.

| System | Recall@5 | MRR@10 | nDCG@10 | Answer | Faithful | F1 | Ctx recall | $/Q | $/correct | Latency | p95 | Pareto | Errors | Wall Time | Best For |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 0.624 [0.585, 0.667] | 0.732 [0.690, 0.775] | 0.633 [0.595, 0.671] | 3.62 [3.50, 3.74] | 4.06 [3.96, 4.15] | 0.06 | 0.66 | $0.00104 | $0.00205 | 2020 ms | 2298 ms | ✓ | 0 | 310777 ms | Cheap baseline and exact-term matching |
| vector_default | 0.584 [0.546, 0.628] | 0.673 [0.627, 0.717] | 0.563 [0.527, 0.603] | 3.48 [3.36, 3.59] | 4.03 [3.93, 4.12] | 0.07 | 0.57 | $0.00106 | $0.00237 | 1786 ms | 2105 ms | ✓ | 0 | 315759 ms | Semantic baseline |
| hybrid_default | 0.565 [0.524, 0.606] | 0.648 [0.601, 0.692] | 0.549 [0.512, 0.585] | 3.25 [3.12, 3.38] | 3.90 [3.78, 4.00] | 0.06 | 0.56 | $0.00106 | $0.00304 | 2348 ms | 3255 ms |  | 0 | 341083 ms | Balanced lexical + semantic retrieval |
| rerank_default | 0.660 [0.620, 0.701] | 0.743 [0.701, 0.786] | 0.649 [0.614, 0.687] | 3.65 [3.54, 3.76] | 3.96 [3.87, 4.06] | 0.07 | 0.70 | $0.00108 | $0.00210 | 2055 ms | 2784 ms | ✓ | 0 | 315136 ms | Higher precision context selection |
| hybrid_rerank_default | 0.668 [0.629, 0.710] | 0.734 [0.692, 0.776] | 0.651 [0.614, 0.689] | 3.73 [3.61, 3.83] | 4.00 [3.90, 4.10] | 0.08 | 0.71 | $0.00109 | $0.00209 | 2242 ms | 3826 ms | ✓ | 0 | 307473 ms | Recall of hybrid plus rerank precision |
| hyde_default | 0.600 [0.562, 0.642] | 0.677 [0.633, 0.721] | 0.576 [0.539, 0.616] | 3.44 [3.32, 3.56] | 3.96 [3.86, 4.07] | 0.07 | 0.58 | $0.00111 | $0.00264 | 3420 ms | 4126 ms |  | 0 | 451016 ms | Short or vaguely-worded questions |
| parent_doc_default | 0.600 [0.562, 0.642] | 0.686 [0.643, 0.730] | 0.595 [0.560, 0.632] | 3.40 [3.26, 3.53] | 3.82 [3.70, 3.94] | 0.06 | 0.58 | $0.00123 | $0.00286 | 1724 ms | 2905 ms |  | 0 | 473211 ms | Better answer context with precise retrieval |
| llm_heavy_default | 0.667 [0.627, 0.708] | 0.762 [0.721, 0.805] | 0.666 [0.629, 0.705] | 3.83 [3.73, 3.94] | 4.11 [4.02, 4.20] | 0.06 | 0.71 | $0.00983 | $0.01647 | 6604 ms | 8092 ms | ✓ | 0 | 837212 ms | Higher-cost, quality-oriented experiments |
| no_retrieval_floor | — | — | — | 1.90 [1.76, 2.05] | 2.45 [2.20, 2.69] | 0.16 | — | $0.00021 | $0.00251 | 1300 ms | 1754 ms | ✓ | 0 | 225003 ms | Showing how much retrieval helps and how often the model makes things up without it |
| full_context_50k | 0.642 [0.602, 0.684] | 0.736 [0.695, 0.778] | 0.666 [0.633, 0.704] | 3.41 [3.29, 3.53] | 3.06 [2.91, 3.20] | 0.08 | 0.94 | $0.00581 | $0.01544 | 2329 ms | 3329 ms |  | 0 | 1940285 ms | Small corpora, and measuring how much a retrieval pipeline loses against reading everything |
| sentence_window_default | 0.582 [0.541, 0.623] | 0.702 [0.659, 0.747] | 0.594 [0.558, 0.632] | 3.44 [3.30, 3.56] | 4.17 [4.07, 4.27] | 0.10 | 0.59 | $0.00051 | $0.00108 | 2044 ms | 3254 ms | ✓ | 0 | 510923 ms | Precise matching with enough surrounding text to answer from |
| vector_mmr | 0.584 [0.544, 0.627] | 0.671 [0.625, 0.716] | 0.582 [0.545, 0.621] | 3.44 [3.31, 3.57] | 3.95 [3.84, 4.06] | 0.06 | 0.63 | $0.00102 | $0.00230 | 1671 ms | 2519 ms | ✓ | 0 | 418419 ms | Semantic baseline |
| contextual_default | 0.630 [0.591, 0.670] | 0.708 [0.666, 0.750] | 0.617 [0.581, 0.655] | 3.60 [3.48, 3.71] | 4.06 [3.96, 4.16] | 0.06 | 0.63 | $0.00104 | $0.00217 | 2129 ms | 2771 ms | ✓ | 0 | 2146490 ms | Corpora where chunks lose their meaning out of context (many similar documents, pronouns, section-relative facts) |
| hierarchical_default | 0.559 [0.518, 0.601] | 0.677 [0.628, 0.724] | 0.548 [0.509, 0.590] | 3.45 [3.31, 3.58] | 4.09 [3.98, 4.19] | 0.09 | 0.58 | $0.00076 | $0.00159 | 1714 ms | 2309 ms | ✓ | 0 | 902293 ms | "Which document?" questions and corpora too large to search chunk by chunk |
| raptor_collapsed | 0.549 [0.512, 0.593] | 0.653 [0.608, 0.697] | 0.535 [0.499, 0.575] | 3.36 [3.24, 3.49] | 4.09 [3.99, 4.19] | 0.08 | 0.53 | $0.00076 | $0.00180 | 1742 ms | 2245 ms | ✓ | 0 | 1620723 ms | Questions that need the gist of several related passages, and corpora where one fact is spread over many chunks |
| compression_llm_extract | 0.640 [0.601, 0.682] | 0.757 [0.714, 0.801] | 0.665 [0.630, 0.703] | 3.72 [3.57, 3.85] | 4.20 [4.08, 4.31] | 0.11 | 0.71 | $0.00112 | $0.00179 | 5666 ms | 7614 ms |  | 0 | 521450 ms | Long or noisy chunks where most of each passage is beside the point; smaller prompts on long contexts |
| late_interaction_rerank | 0.582 [0.543, 0.623] | 0.734 [0.689, 0.778] | 0.595 [0.559, 0.634] | 3.41 [3.28, 3.54] | 4.15 [4.05, 4.25] | 0.07 | 0.60 | $0.00076 | $0.00188 | 1936 ms | 2546 ms |  | 0 | 323149 ms | Questions whose wording must match precisely (names, terms of art) where one pooled vector per chunk blurs the detail |
| rag_fusion_default | 0.618 [0.580, 0.659] | 0.713 [0.669, 0.756] | 0.612 [0.576, 0.650] | 3.58 [3.47, 3.70] | 4.17 [4.09, 4.25] | 0.07 | 0.60 | $0.00085 | $0.00173 | 5088 ms | 6168 ms | ✓ | 0 | 539876 ms | Questions worded differently from the documents |
| decompose_default | 0.623 [0.584, 0.665] | 0.635 [0.589, 0.680] | 0.572 [0.535, 0.610] | 3.61 [3.49, 3.72] | 4.01 [3.91, 4.12] | 0.05 | 0.62 | $0.00087 | $0.00184 | 4536 ms | 6084 ms | ✓ | 0 | 506015 ms | Multi-part, comparison and multi-hop questions |
| corrective_default | 0.637 [0.599, 0.678] | 0.784 [0.743, 0.827] | 0.654 [0.618, 0.693] | 3.68 [3.57, 3.79] | 4.20 [4.11, 4.28] | 0.08 | 0.65 | $0.00100 | $0.00188 | 6528 ms | 9877 ms | ✓ | 0 | 542452 ms | Corpora where the first search often misses and a bad context should be noticed rather than answered from |
| iterative_default | 0.618 [0.576, 0.661] | 0.680 [0.637, 0.723] | 0.599 [0.561, 0.639] | 3.52 [3.40, 3.63] | 4.05 [3.95, 4.15] | 0.05 | 0.62 | $0.00123 | $0.00266 | 9515 ms | 10908 ms |  | 0 | 814833 ms | Multi-hop questions whose second search depends on what the first one found |
| agent_search_plain | 0.662 [0.622, 0.706] | 0.701 [0.658, 0.745] | 0.644 [0.605, 0.685] | 3.35 [3.22, 3.48] | 3.49 [3.36, 3.64] | 0.05 | 0.66 | $0.00226 | $0.00547 | 8633 ms | 10607 ms |  | 0 | 1114181 ms | Questions that need computation, exact lookups, or several searches |
| agent_search_tools | 0.693 [0.653, 0.735] | 0.702 [0.659, 0.745] | 0.649 [0.613, 0.688] | 3.53 [3.40, 3.64] | 3.64 [3.51, 3.77] | 0.05 | 0.69 | $0.00259 | $0.00568 | 10058 ms | 13071 ms |  | 0 | 1004362 ms | Questions that need computation, exact lookups, or several searches |
| grep_agent_default | 0.533 [0.492, 0.578] | 0.570 [0.525, 0.618] | 0.503 [0.467, 0.544] | 3.30 [3.15, 3.42] | 3.57 [3.42, 3.71] | 0.05 | 0.57 | $0.00309 | $0.00793 | 8169 ms | 8792 ms |  | 0 | 1067906 ms | Small or exact-match-heavy corpora, and testing whether you need retrieval infrastructure at all |
| adaptive_default | 0.644 [0.603, 0.687] | 0.716 [0.673, 0.761] | 0.629 [0.590, 0.669] | 3.57 [3.44, 3.69] | 4.06 [3.96, 4.16] | 0.06 | 0.66 | $0.00088 | $0.00175 | 2915 ms | 4143 ms | ✓ | 0 | 233730 ms | Mixed workloads where no single architecture is best for every question |
| decision_gate_default | 0.668 [0.629, 0.710] | 0.734 [0.692, 0.776] | 0.651 [0.614, 0.689] | 3.55 [3.40, 3.69] | 4.26 [4.16, 4.35] | 0.07 | 0.67 | $0.00181 | $0.00341 | 1986 ms | 3611 ms | ✓ | 0 | 537210 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| decision_gate_strict | 0.668 [0.629, 0.710] | 0.734 [0.692, 0.776] | 0.651 [0.614, 0.689] | 2.60 [2.41, 2.78] | 4.50 [4.38, 4.61] | 0.04 | 0.53 | $0.00154 | $0.00425 | 1314 ms | 2374 ms | ✓ | 0 | 455655 ms | Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal |
| graph_rag_local | 0.534 [0.495, 0.576] | 0.633 [0.586, 0.681] | 0.524 [0.485, 0.564] | 3.26 [3.13, 3.39] | 4.13 [4.03, 4.23] | 0.07 | 0.52 | $0.00092 | $0.00264 | 1890 ms | 2431 ms |  | 0 | 8025055 ms | Questions that connect entities across documents (`local`) and corpus-wide questions about themes and groups (`global`) |
| graph_rag_global | 0.131 [0.104, 0.161] | 0.188 [0.150, 0.228] | 0.133 [0.108, 0.162] | 2.04 [1.88, 2.19] | 3.33 [3.11, 3.55] | 0.13 | 0.15 | $0.00061 | $0.00381 | 2736 ms | 3087 ms |  | 0 | 7741583 ms | Questions that connect entities across documents (`local`) and corpus-wide questions about themes and groups (`global`) |
| langchain_default | 0.588 [0.549, 0.630] | 0.702 [0.657, 0.746] | 0.581 [0.544, 0.620] | 2.85 [2.73, 2.98] | 3.50 [3.35, 3.65] | 0.10 | 0.70 | $0.00088 | $0.00914 | 5015 ms | 8353 ms |  | 0 | 510232 ms | Benchmarking your own code or a framework pipeline against the built-in systems without deploying it |
| llamaindex_default | 0.591 [0.552, 0.634] | 0.671 [0.627, 0.716] | 0.570 [0.533, 0.607] | 2.60 [2.46, 2.73] | 3.08 [2.91, 3.24] | 0.22 | 0.71 | $0.00176 | $0.01430 | 4540 ms | 5798 ms |  | 0 | 608400 ms | Benchmarking your own code or a framework pipeline against the built-in systems without deploying it |
| haystack_default | 0.579 [0.543, 0.620] | 0.696 [0.652, 0.740] | 0.568 [0.533, 0.605] | 2.71 [2.57, 2.84] | 3.04 [2.87, 3.21] | 0.09 | 0.68 | $0.00115 | $0.01331 | 5952 ms | 10978 ms |  | 0 | 609302 ms | Benchmarking your own code or a framework pipeline against the built-in systems without deploying it |

Brackets are 95% bootstrap confidence intervals over the questions each system answered (2000 resamples, seed 0). Overlapping intervals do not by themselves mean two systems are tied: the paired comparison below uses the same questions for both.
✓ in Pareto: no other system is at least as good on answer score, cost and latency and better on one of them.

## Significance vs no_retrieval_floor

Each system minus the baseline `no_retrieval_floor` (cheapest system), on the questions both answered; positive means better. p-values are Holm-adjusted across the systems compared on each metric. "no clear difference" means the data cannot rank the two, not that they are equal.

| System | Metric | Difference | 95% CI | p (Holm) | Verdict | Wins / ties / losses |
| --- | --- | --- | --- | --- | --- | --- |
| bm25_default | Answer | +1.722 | [+1.560, +1.879] | 0.031 | better | 258 / 18 / 24 |
| vector_default | Answer | +1.573 | [+1.426, +1.730] | 0.031 | better | 251 / 17 / 32 |
| hybrid_default | Answer | +1.346 | [+1.196, +1.502] | 0.031 | better | 243 / 19 / 38 |
| rerank_default | Answer | +1.745 | [+1.603, +1.895] | 0.031 | better | 259 / 10 / 31 |
| hybrid_rerank_default | Answer | +1.824 | [+1.674, +1.973] | 0.031 | better | 263 / 13 / 24 |
| hyde_default | Answer | +1.533 | [+1.388, +1.684] | 0.031 | better | 250 / 14 / 36 |
| parent_doc_default | Answer | +1.495 | [+1.342, +1.671] | 0.031 | better | 238 / 18 / 44 |
| llm_heavy_default | Answer | +1.933 | [+1.788, +2.077] | 0.031 | better | 273 / 8 / 19 |
| full_context_50k | Answer | +1.510 | [+1.360, +1.663] | 0.031 | better | 251 / 13 / 36 |
| sentence_window_default | Answer | +1.535 | [+1.380, +1.692] | 0.031 | better | 250 / 20 / 30 |
| vector_mmr | Answer | +1.537 | [+1.387, +1.688] | 0.031 | better | 244 / 26 / 30 |
| contextual_default | Answer | +1.694 | [+1.540, +1.847] | 0.031 | better | 256 / 13 / 31 |
| hierarchical_default | Answer | +1.545 | [+1.395, +1.696] | 0.031 | better | 250 / 14 / 36 |
| raptor_collapsed | Answer | +1.463 | [+1.302, +1.625] | 0.031 | better | 243 / 15 / 42 |
| compression_llm_extract | Answer | +1.814 | [+1.648, +1.985] | 0.031 | better | 263 / 22 / 15 |
| late_interaction_rerank | Answer | +1.505 | [+1.341, +1.671] | 0.031 | better | 243 / 19 / 38 |
| rag_fusion_default | Answer | +1.682 | [+1.525, +1.840] | 0.031 | better | 259 / 10 / 31 |
| decompose_default | Answer | +1.704 | [+1.555, +1.862] | 0.031 | better | 262 / 9 / 29 |
| corrective_default | Answer | +1.780 | [+1.631, +1.932] | 0.031 | better | 260 / 14 / 26 |
| iterative_default | Answer | +1.616 | [+1.463, +1.767] | 0.031 | better | 250 / 18 / 32 |
| agent_search_plain | Answer | +1.452 | [+1.302, +1.619] | 0.031 | better | 243 / 17 / 40 |
| agent_search_tools | Answer | +1.626 | [+1.472, +1.775] | 0.031 | better | 255 / 15 / 30 |
| grep_agent_default | Answer | +1.394 | [+1.229, +1.552] | 0.031 | better | 236 / 20 / 44 |
| adaptive_default | Answer | +1.666 | [+1.506, +1.826] | 0.031 | better | 257 / 14 / 29 |
| decision_gate_default | Answer | +1.647 | [+1.492, +1.814] | 0.031 | better | 246 / 35 / 19 |
| decision_gate_strict | Answer | +0.695 | [+0.524, +0.869] | 0.031 | better | 164 / 68 / 68 |
| graph_rag_local | Answer | +1.357 | [+1.205, +1.514] | 0.031 | better | 240 / 19 / 41 |
| graph_rag_global | Answer | +0.136 | [+0.016, +0.260] | 0.031 | better | 139 / 58 / 103 |
| langchain_default | Answer | +0.953 | [+0.778, +1.139] | 0.031 | better | 233 / 9 / 58 |
| llamaindex_default | Answer | +0.694 | [+0.538, +0.853] | 0.031 | better | 214 / 15 / 71 |
| haystack_default | Answer | +0.807 | [+0.600, +1.022] | 0.031 | better | 231 / 13 / 56 |

## Cost by stage

Mean $ per question, answering only (the judge's cost is excluded). `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | llm | tool | grade | generate |
| --- | --- | --- | --- | --- | --- |
| bm25_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000342 |
| vector_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000357 |
| hybrid_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000365 |
| rerank_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000374 |
| hybrid_rerank_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000374 |
| hyde_default | $0.000002 | $0.000044 | $0.000000 | $0.000000 | $0.000356 |
| parent_doc_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000517 |
| llm_heavy_default | $0.000001 | $0.001344 | $0.000000 | $0.000000 | $0.007778 |
| no_retrieval_floor | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000028 |
| full_context_50k | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.005066 |
| sentence_window_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000119 |
| vector_mmr | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000334 |
| contextual_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000339 |
| hierarchical_default | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000211 |
| raptor_collapsed | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000213 |
| compression_llm_extract | $0.000001 | $0.000787 | $0.000000 | $0.000000 | $0.000063 |
| late_interaction_rerank | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000214 |
| rag_fusion_default | $0.000003 | $0.000072 | $0.000000 | $0.000000 | $0.000215 |
| decompose_default | $0.000001 | $0.000052 | $0.000000 | $0.000000 | $0.000236 |
| corrective_default | $0.000001 | $0.000022 | $0.000000 | $0.000216 | $0.000215 |
| iterative_default | $0.000002 | $0.000437 | $0.000000 | $0.000000 | $0.000239 |
| agent_search_plain | $0.000000 | $0.001129 | $0.000002 | $0.000000 | $0.000537 |
| agent_search_tools | $0.000000 | $0.001426 | $0.000002 | $0.000000 | $0.000588 |
| grep_agent_default | $0.000000 | $0.001989 | $0.000000 | $0.000000 | $0.000654 |
| adaptive_default | $0.000001 | $0.000080 | $0.000000 | $0.000000 | $0.000240 |
| decision_gate_default | $0.000001 | $0.000000 | $0.000000 | $0.001020 | $0.000229 |
| decision_gate_strict | $0.000001 | $0.000000 | $0.000000 | $0.000994 | $0.000100 |
| graph_rag_local | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000330 |
| graph_rag_global | $0.000001 | $0.000000 | $0.000000 | $0.000000 | $0.000240 |
| langchain_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000300 |
| llamaindex_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.001074 |
| haystack_default | $0.000000 | $0.000000 | $0.000000 | $0.000000 | $0.000454 |

## Latency by stage

Mean ms per question, answering only. `untracked` is cost or time that no recorded step accounts for.

| System | retrieve | rerank | llm | tool | route | grade | generate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_default | 49.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1712.4 ms |
| vector_default | 167.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1684.2 ms |
| hybrid_default | 584.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1686.9 ms |
| rerank_default | 529.4 ms | 61.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1732.9 ms |
| hybrid_rerank_default | 614.5 ms | 29.8 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1792.8 ms |
| hyde_default | 374.4 ms | 0.0 ms | 1592.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1760.8 ms |
| parent_doc_default | 184.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1820.3 ms |
| llm_heavy_default | 664.5 ms | 21.2 ms | 4221.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3877.0 ms |
| no_retrieval_floor | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1393.9 ms |
| full_context_50k | 43.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 23046.6 ms |
| sentence_window_default | 191.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1877.6 ms |
| vector_mmr | 179.1 ms | 0.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3094.0 ms |
| contextual_default | 209.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1596.1 ms |
| hierarchical_default | 16.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1590.7 ms |
| raptor_collapsed | 1.7 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1598.9 ms |
| compression_llm_extract | 258.0 ms | 0.0 ms | 3273.9 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1422.1 ms |
| late_interaction_rerank | 253.0 ms | 253.3 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1523.2 ms |
| rag_fusion_default | 1161.7 ms | 0.0 ms | 2065.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1687.4 ms |
| decompose_default | 624.3 ms | 0.0 ms | 1909.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1737.7 ms |
| corrective_default | 438.0 ms | 0.0 ms | 687.2 ms | 0.0 ms | 0.0 ms | 2132.7 ms | 1526.7 ms |
| iterative_default | 697.9 ms | 0.0 ms | 6225.5 ms | 0.0 ms | 0.0 ms | 0.0 ms | 1510.6 ms |
| agent_search_plain | 0.0 ms | 0.0 ms | 8216.0 ms | 1028.2 ms | 0.1 ms | 0.0 ms | 2877.7 ms |
| agent_search_tools | 0.0 ms | 0.0 ms | 7485.1 ms | 965.3 ms | 0.1 ms | 0.0 ms | 2374.2 ms |
| grep_agent_default | 0.0 ms | 0.0 ms | 8519.0 ms | 193.4 ms | 0.1 ms | 0.0 ms | 3110.9 ms |
| adaptive_default | 348.2 ms | 8.7 ms | 930.6 ms | 47.1 ms | 0.2 ms | 0.0 ms | 1520.9 ms |
| decision_gate_default | 553.5 ms | 24.2 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3497.5 ms | 1444.4 ms |
| decision_gate_strict | 580.5 ms | 25.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3598.7 ms | 684.4 ms |
| graph_rag_local | 677.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 2295.2 ms |
| graph_rag_global | 49.1 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3566.7 ms |
| langchain_default | 691.6 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 3611.3 ms |
| llamaindex_default | 94.8 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 5682.4 ms |
| haystack_default | 261.4 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms | 5515.6 ms |
