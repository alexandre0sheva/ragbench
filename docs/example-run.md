# Example run: 24 systems on the demo dataset

A complete live run of [`configs/all.yaml`](../configs/all.yaml) on the bundled demo dataset, kept as a worked example of what a run produces and as the source of the numbers in the README. It is also the one leaderboard that holds every system side by side, `decision_gate` included. Everything below comes from the files in [`docs/benchmarks/example-run/`](benchmarks/example-run/). For what each metric means see the [methodology](methodology.md); for how to read a report see [choosing an architecture](choosing-an-architecture.md); for a closer look at `decision_gate` see its [evaluation](decision-gate-evaluation.md).

## The run

| | |
| --- | --- |
| Run id | `all_systems_plus_decision_gate_20261007_184150` |
| Started / finished (UTC) | 2026-10-07T15:41:50Z / 2026-10-07T15:51:11Z (9 min of wall time, short because almost every call was served from the cache) |
| Mode | **live** (real OpenAI calls) |
| Models | `gpt-6-luna`, `gpt-6.1-sol`, `text-embedding-3-small`; prices as of 2026-10-01 |
| Dataset | 163 questions (138 with relevance labels), 60 documents (about 23k tokens), [`data/demo`](../data/demo) |
| Systems | 24 |
| Spend | $3.26 charged at standalone prices; $0.07 real spend for this run after $3.24 of cache reuse, because the same systems had been run before (the original 22 on 1 Oct 2026 for $2.91 of real spend, the gate for under $1 including its exploration); cap $6.00 |
| Versions | ragbench 0.3.0 plus `decision_gate`, Python 3.12.12, openai 3.26.0 |
| Questions that errored | 0 of 3912 question-system pairs (see [the earlier rejected call](#the-earlier-rejected-call)) |

## Recommendation

**Deploy `bm25_default`** under the balanced profile. 17 other systems are statistically tied with the best answer score, so cost and latency decided it. The full reasoning, ranking and per-category winners are in [`recommendation.md`](benchmarks/example-run/recommendation.md), and its config in [`winner.yaml`](benchmarks/example-run/winner.yaml).

## All systems

Sorted by answer score. Brackets are 95% bootstrap confidence intervals over the questions each system answered. ✓ in the last column means no other system is at least as good on answer score, cost and latency and better on one of them. Latency is the mean of a five-question probe and is noisy: compare latencies only when they differ by a factor.

| System | Recall@5 | MRR@10 | nDCG@10 | Answer score | Faithfulness | $ / question | p95 latency | Errors | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `iterative_default` | 0.799 [0.742, 0.856] | 0.780 [0.721, 0.833] | 0.769 [0.713, 0.823] | 4.81 [4.71, 4.90] | 4.91 | $0.00057 | 6423 ms | 0 | ✓ |
| `decision_gate_default` | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.80 [4.70, 4.89] | 4.98 | $0.00094 | 4801 ms | 0 | ✓ |
| `grep_agent_default` | 0.791 [0.736, 0.850] | 0.694 [0.634, 0.754] | 0.696 [0.640, 0.750] | 4.78 [4.69, 4.86] | 4.73 | $0.00109 | 8197 ms | 0 |  |
| `full_context_ceiling` | 0.741 [0.676, 0.804] | 0.728 [0.666, 0.791] | 0.715 [0.657, 0.772] | 4.78 [4.70, 4.84] | 4.84 | $0.00280 | 2038 ms | 0 | ✓ |
| `corrective_default` | 0.793 [0.733, 0.850] | 0.830 [0.774, 0.883] | 0.796 [0.742, 0.850] | 4.77 [4.67, 4.86] | 4.99 | $0.00060 | 9595 ms | 0 |  |
| `rag_fusion_default` | 0.808 [0.753, 0.863] | 0.744 [0.684, 0.802] | 0.744 [0.688, 0.800] | 4.75 [4.64, 4.86] | 4.96 | $0.00042 | 4093 ms | 0 | ✓ |
| `agent_search_tools` | 0.821 [0.767, 0.876] | 0.759 [0.699, 0.818] | 0.757 [0.703, 0.812] | 4.75 [4.63, 4.85] | 4.85 | $0.00082 | 5753 ms | 0 |  |
| `rerank_default` | 0.789 [0.729, 0.846] | 0.747 [0.687, 0.805] | 0.732 [0.674, 0.786] | 4.74 [4.63, 4.84] | 4.95 | $0.00052 | 3054 ms | 0 | ✓ |
| `hybrid_rerank_default` | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.72 [4.59, 4.84] | 4.98 | $0.00053 | 3463 ms | 0 |  |
| `decompose_default` | 0.771 [0.713, 0.831] | 0.728 [0.666, 0.787] | 0.723 [0.667, 0.781] | 4.72 [4.60, 4.83] | 4.93 | $0.00041 | 2888 ms | 0 | ✓ |
| `llm_heavy_default` | 0.790 [0.728, 0.848] | 0.770 [0.710, 0.827] | 0.747 [0.692, 0.801] | 4.72 [4.60, 4.82] | 4.93 | $0.00512 | 7579 ms | 0 |  |
| `parent_doc_default` | 0.789 [0.733, 0.845] | 0.738 [0.676, 0.796] | 0.717 [0.663, 0.773] | 4.71 [4.60, 4.81] | 4.93 | $0.00049 | 5704 ms | 0 |  |
| `adaptive_default` | 0.771 [0.712, 0.828] | 0.724 [0.661, 0.784] | 0.719 [0.661, 0.774] | 4.70 [4.57, 4.82] | 4.99 | $0.00044 | 4119 ms | 0 |  |
| `contextual_default` | 0.784 [0.725, 0.840] | 0.753 [0.691, 0.812] | 0.741 [0.684, 0.798] | 4.69 [4.57, 4.81] | 4.93 | $0.00046 | 1690 ms | 0 | ✓ |
| **`bm25_default`** | 0.745 [0.681, 0.808] | 0.720 [0.654, 0.782] | 0.708 [0.647, 0.767] | 4.69 [4.55, 4.81] | 4.99 | $0.00049 | 1371 ms | 0 | ✓ |
| `hyde_default` | 0.787 [0.729, 0.845] | 0.725 [0.664, 0.784] | 0.710 [0.656, 0.765] | 4.68 [4.54, 4.80] | 4.97 | $0.00046 | 3110 ms | 0 |  |
| `vector_default` | 0.790 [0.733, 0.847] | 0.738 [0.676, 0.796] | 0.718 [0.663, 0.774] | 4.67 [4.54, 4.79] | 4.94 | $0.00044 | 2374 ms | 0 | ✓ |
| `decision_gate_strict` | 0.774 [0.713, 0.832] | 0.737 [0.675, 0.797] | 0.727 [0.669, 0.784] | 4.65 [4.49, 4.80] | 4.97 | $0.00091 | 3086 ms | 0 |  |
| `hybrid_default` | 0.775 [0.715, 0.833] | 0.745 [0.679, 0.806] | 0.732 [0.671, 0.790] | 4.61 [4.46, 4.74] | 4.92 | $0.00048 | 1895 ms | 0 |  |
| `agent_search_plain` | 0.814 [0.756, 0.871] | 0.766 [0.708, 0.826] | 0.758 [0.703, 0.813] | 4.58 [4.43, 4.72] | 4.79 | $0.00072 | 5345 ms | 0 |  |
| `hierarchical_default` | 0.733 [0.669, 0.795] | 0.707 [0.642, 0.770] | 0.666 [0.607, 0.725] | 4.56 [4.40, 4.70] | 4.96 | $0.00033 | 1604 ms | 0 | ✓ |
| `vector_mmr` | 0.697 [0.635, 0.757] | 0.731 [0.668, 0.792] | 0.679 [0.620, 0.739] | 4.54 [4.40, 4.68] | 4.94 | $0.00040 | 1541 ms | 0 |  |
| `sentence_window_default` | 0.780 [0.720, 0.838] | 0.692 [0.631, 0.755] | 0.693 [0.637, 0.749] | 4.42 [4.23, 4.59] | 4.82 | $0.00025 | 1518 ms | 0 | ✓ |
| `no_retrieval_floor` | — | — | — | 1.54 [1.40, 1.68] | 4.69 | $0.00012 | 1430 ms | 0 | ✓ |

`no_retrieval_floor` has no retrieval metrics because it never retrieves; it shows what the model answers from its own knowledge. `full_context_ceiling` reads the whole corpus on every question.

### Context quality and declining to answer

The same systems, sorted the same way, on what the generator was given and on what happens to the 25 unanswerable and 138 answerable questions ([definitions](methodology.md#abstention-metrics)). Context precision is blank for `no_retrieval_floor`, which has no context.

| System | Context recall | Context precision | Abstain recall | False refusals | Abstain precision |
| --- | --- | --- | --- | --- | --- |
| `iterative_default` | 0.94 | 0.36 | 0.84 | 0.014 | 0.91 |
| `decision_gate_default` | 0.93 | 0.69 | 0.92 | 0.043 | 0.79 |
| `grep_agent_default` | 0.92 | 0.68 | 0.68 | 0.000 | 1.00 |
| `full_context_ceiling` | 1.00 | 0.02 | 0.96 | 0.000 | 1.00 |
| `corrective_default` | 0.93 | 0.34 | 0.80 | 0.029 | 0.83 |
| `rag_fusion_default` | 0.93 | 0.36 | 0.84 | 0.043 | 0.78 |
| `agent_search_tools` | 0.96 | 0.38 | 0.84 | 0.014 | 0.91 |
| `rerank_default` | 0.92 | 0.33 | 0.80 | 0.051 | 0.74 |
| `hybrid_rerank_default` | 0.89 | 0.32 | 0.80 | 0.058 | 0.71 |
| `decompose_default` | 0.91 | 0.34 | 0.84 | 0.043 | 0.78 |
| `llm_heavy_default` | 0.92 | 0.34 | 1.00 | 0.058 | 0.76 |
| `parent_doc_default` | 0.89 | 0.37 | 0.84 | 0.065 | 0.70 |
| `adaptive_default` | 0.91 | 0.33 | 0.88 | 0.065 | 0.71 |
| `contextual_default` | 0.91 | 0.33 | 0.84 | 0.065 | 0.70 |
| **`bm25_default`** | 0.88 | 0.31 | 0.88 | 0.072 | 0.69 |
| `hyde_default` | 0.91 | 0.33 | 0.88 | 0.058 | 0.73 |
| `vector_default` | 0.91 | 0.33 | 0.88 | 0.051 | 0.76 |
| `decision_gate_strict` | 0.89 | 0.79 | 1.00 | 0.087 | 0.68 |
| `hybrid_default` | 0.90 | 0.33 | 0.88 | 0.080 | 0.67 |
| `agent_search_plain` | 0.94 | 0.37 | 0.84 | 0.058 | 0.72 |
| `hierarchical_default` | 0.87 | 0.28 | 0.84 | 0.101 | 0.60 |
| `vector_mmr` | 0.82 | 0.24 | 0.88 | 0.101 | 0.61 |
| `sentence_window_default` | 0.92 | 0.36 | 0.88 | 0.109 | 0.59 |
| `no_retrieval_floor` | — | — | 1.00 | 0.935 | 0.16 |

## What separated the systems

Difference in answer score against `no_retrieval_floor`, on the questions both answered (paired bootstrap, p-values Holm-adjusted across systems):

| System | Difference | 95% CI | p (Holm) | Wins / ties / losses |
| --- | --- | --- | --- | --- |
| `iterative_default` | +3.27 | [+3.10, +3.45] | 0.023 | 161 / 1 / 1 |
| `decision_gate_default` | +3.27 | [+3.09, +3.44] | 0.023 | 161 / 1 / 1 |
| `grep_agent_default` | +3.25 | [+3.07, +3.43] | 0.023 | 162 / 0 / 1 |
| `full_context_ceiling` | +3.24 | [+3.06, +3.41] | 0.023 | 160 / 2 / 1 |
| `corrective_default` | +3.23 | [+3.05, +3.41] | 0.023 | 161 / 1 / 1 |
| `rag_fusion_default` | +3.22 | [+3.03, +3.41] | 0.023 | 160 / 1 / 2 |
| `agent_search_tools` | +3.22 | [+3.01, +3.41] | 0.023 | 157 / 1 / 5 |
| `rerank_default` | +3.21 | [+3.01, +3.40] | 0.023 | 160 / 1 / 2 |
| `hybrid_rerank_default` | +3.19 | [+2.99, +3.38] | 0.023 | 161 / 1 / 1 |
| `decompose_default` | +3.19 | [+2.98, +3.38] | 0.023 | 160 / 1 / 2 |
| `llm_heavy_default` | +3.18 | [+2.98, +3.38] | 0.023 | 158 / 3 / 2 |
| `parent_doc_default` | +3.18 | [+2.99, +3.36] | 0.023 | 162 / 1 / 0 |
| `adaptive_default` | +3.17 | [+2.96, +3.36] | 0.023 | 161 / 0 / 2 |
| `contextual_default` | +3.16 | [+2.96, +3.36] | 0.023 | 159 / 1 / 3 |
| `bm25_default` | +3.15 | [+2.95, +3.35] | 0.023 | 159 / 3 / 1 |
| `hyde_default` | +3.14 | [+2.94, +3.35] | 0.023 | 158 / 4 / 1 |
| `vector_default` | +3.14 | [+2.93, +3.34] | 0.023 | 158 / 2 / 3 |
| `decision_gate_strict` | +3.12 | [+2.90, +3.33] | 0.023 | 154 / 8 / 1 |
| `hybrid_default` | +3.07 | [+2.86, +3.29] | 0.023 | 157 / 3 / 3 |
| `agent_search_plain` | +3.04 | [+2.82, +3.26] | 0.023 | 157 / 2 / 4 |
| `hierarchical_default` | +3.02 | [+2.82, +3.24] | 0.023 | 158 / 4 / 1 |
| `vector_mmr` | +3.01 | [+2.79, +3.21] | 0.023 | 161 / 1 / 1 |
| `sentence_window_default` | +2.88 | [+2.63, +3.11] | 0.023 | 154 / 3 / 6 |

Every system beats the no-retrieval floor by a wide margin, so retrieval matters on this corpus. Among the retrieval systems the answer scores are close: the [recommendation](benchmarks/example-run/recommendation.md) lists the systems that cannot be told apart from the best one.

## By question category

The best system per category, next to the recommended one. Categories hold few questions, so a different winner here is a lead to check, not a significant result.

| Category | Best system | Its answer score | `bm25_default` |
| --- | --- | --- | --- |
| aggregation | `agent_search_plain` | 4.87 | 3.73 |
| comparison | `agent_search_tools` | 5.00 | 4.55 |
| date_arithmetic | `hierarchical_default` | 5.00 | 5.00 |
| direct_fact | `rag_fusion_default` | 5.00 | 4.88 |
| distractor | `rag_fusion_default` | 5.00 | 4.96 |
| exact_identifier | `hierarchical_default` | 5.00 | 5.00 |
| long_context | `sentence_window_default` | 5.00 | 5.00 |
| multi_hop | `agent_search_tools` | 4.92 | 4.28 |
| numeric_reasoning | `sentence_window_default` | 5.00 | 5.00 |
| paraphrase | `decompose_default` | 5.00 | 4.17 |
| temporal_conflict | `decision_gate_default` | 5.00 | 4.98 |
| unanswerable | `decision_gate_strict` | 4.72 | 4.48 |

## Caveats

- **The judge is also the generator.** `gpt-6-luna` writes most systems' answers and grades them, which tends to inflate scores; the report's banner says so. Use `evaluation.judge.model` to grade with a different model.
- **A small, easy corpus.** 60 short documents and 163 questions, 25 of them unanswerable. Differences of a few hundredths of a point are within the confidence intervals, the abstention numbers rest on very few questions, and the ranking may not carry over to your documents.
- **Caching.** 12662 of 12671 model lookups (100%) were served from the disk cache, because every system had been run before with the same configuration; the rows of the original systems are identical to the 1 Oct run. Costs are charged as if nothing was cached, so the numbers match a cold run (about $3.3); the real spend of this run was $0.07. Latency comes from the live probe, which bypasses the cache.
- **A system chosen after seeing results.** The default wording of `decision_gate` was picked between two candidates after a first run on this same dataset, so its rows are slightly optimistic; [its evaluation](decision-gate-evaluation.md#what-the-numbers-say) explains.

## The earlier rejected call

The 1 Oct run of the original 22 systems had one question-system pair fail: `rag_fusion_default` on `q_087`, where OpenAI answered one of that system's model calls with HTTP 400 `invalid_prompt` (its usage-policy filter flagged an ordinary question about a product release history). Errors are never cached, so this run asked again and the call succeeded; there are no errors in this run.

## Files

The small result files are kept in [`docs/benchmarks/example-run/`](benchmarks/example-run/). The full run directory also holds `per_question_results.jsonl` (every answer, retrieved passage, step and, for `decision_gate`, the probability of every judged chunk), `report_questions.json`, `report.html` and the cost breakdown; those are too large to commit, so they stay in `results/` on the machine that ran it (`ragbench report RUN` rebuilds the HTML from them).

| File | What it holds |
| --- | --- |
| [`config.yaml`](benchmarks/example-run/config.yaml) | the config the run used: `configs/all.yaml` with a spending cap (the run is named `all_systems_plus_decision_gate`) |
| [`winner.yaml`](benchmarks/example-run/winner.yaml) | a runnable config for the recommended system |
| [`recommendation.md`](benchmarks/example-run/recommendation.md) | the recommendation, ranking and per-category winners |
| [`recommendation.json`](benchmarks/example-run/recommendation.json) | the same, machine-readable |
| [`leaderboard.md`](benchmarks/example-run/leaderboard.md) | the leaderboard and the significance tables |
| [`metrics_summary.csv`](benchmarks/example-run/metrics_summary.csv) | every metric for every system, with confidence intervals |
| [`significance.csv`](benchmarks/example-run/significance.csv) | the paired comparisons against the baseline |
| [`stats.json`](benchmarks/example-run/stats.json) | the statistics behind the intervals and tests |
| [`failures.md`](benchmarks/example-run/failures.md) | failure types per system, with an example question each |
| [`qrels_audit.md`](benchmarks/example-run/qrels_audit.md) | relevance labels the run found doubtful |
| [`system_runtime.csv`](benchmarks/example-run/system_runtime.csv) | ingestion and question wall time per system |
| [`pareto.json`](benchmarks/example-run/pareto.json) | the quality / cost / latency frontier |
| [`routes.csv`](benchmarks/example-run/routes.csv) | which route the `adaptive` system chose |
| [`tool_usage.csv`](benchmarks/example-run/tool_usage.csv) | tool calls of the agentic systems |
| [`run_summary.json`](benchmarks/example-run/run_summary.json) | mode, models, spend, cache and error counts |
| [`run_manifest.json`](benchmarks/example-run/run_manifest.json) | versions, dataset and config hashes, timings |
