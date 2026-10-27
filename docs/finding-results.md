# Where the results are

RAGBench publishes several runs, and each has its own page and folder. This page says which one is current, where each kind of result lives, and in what order to read a leaderboard.

## Start here

**The current benchmark is [MultiHop-RAG](benchmarks/multihop-rag.md)** (run on 26–27 October 2026). It is RAGBench's most recent and largest run: every RAG solution RAGBench can run, on one public dataset, with one generator and an independent judge. The README's [published result](../README.md#a-published-result) is its short version. Older pages use the small demo dataset and are not comparable with it.

| You want | Go to |
| --- | --- |
| The current ranking of every system, with significance tests | [MultiHop-RAG: Leaderboard](benchmarks/multihop-rag.md#leaderboard) |
| The system to pick, and the ones tied with it | [MultiHop-RAG: Recommendation](benchmarks/multihop-rag.md#recommendation) and [Statistically tied with the best](benchmarks/multihop-rag.md#statistically-tied-with-the-best) |
| The headline findings in a few lines | [MultiHop-RAG: What it shows](benchmarks/multihop-rag.md#what-it-shows) |
| Which question types are hard, why answers fail | [MultiHop-RAG: Patterns across the runs](benchmarks/multihop-rag.md#patterns-across-the-runs) |
| Claude Code, Codex and OpenCode as RAG systems | [MultiHop-RAG: Agent harnesses](benchmarks/multihop-rag.md#agent-harnesses) |
| Which generator model to use, open-weight ones included | [MultiHop-RAG: Open-weight generators](benchmarks/multihop-rag.md#open-weight-generators) |
| LangChain, LlamaIndex and Haystack out of the box | [MultiHop-RAG: Frameworks](benchmarks/multihop-rag.md#frameworks-out-of-the-box) |
| Every published run's table on one page, filterable | the docs site's [Leaderboard](https://alexandre0sheva.github.io/ragbench/leaderboard/) and its explorer (built from the run folders when the site is built) |
| One run's files end to end, on the demo dataset | [Example run](example-run.md) (ragbench 0.4.0, October 2026) |
| `decision_gate` measured on its own | [Decision gate evaluation](decision-gate-evaluation.md) (demo dataset) |
| How a number is computed, and what it cannot tell you | [Methodology](methodology.md) |

## Which folder is which

Every published page is generated from the result files committed under [`docs/benchmarks/`](benchmarks/), never typed by hand (`python scripts/check_results.py` checks this in CI).

| Folder | What it holds | Explained on |
| --- | --- | --- |
| `multihop-rag-run/` | The main MultiHop-RAG run: every built-in system, the framework pipelines and GraphRAG on the same questions | [MultiHop-RAG](benchmarks/multihop-rag.md) |
| `multihop-rag-harnesses-run/` | The agent harnesses, on a stratified subset of those questions next to three systems of the main run; `corrections.json` corrects OpenCode's cost | [Agent harnesses](benchmarks/multihop-rag.md#agent-harnesses) |
| `multihop-rag-models-run/` | Three systems with each generator model, on a smaller subset | [Open-weight generators](benchmarks/multihop-rag.md#open-weight-generators) |
| `multihop-rag-optimize/` | One `ragbench optimize` run on a train / test split of the questions | [Optimizing a system](benchmarks/multihop-rag.md#optimizing-a-system-on-this-dataset) |
| `example-run/` | Every built-in system on the demo dataset | [Example run](example-run.md) |
| `decision-gate-run/` | `decision_gate` against its neighbours on the demo dataset | [Decision gate evaluation](decision-gate-evaluation.md) |

Each run folder has a `leaderboard.md` and a `report.html` rebuilt from its CSVs. They show that run alone; the page named in the last column puts it in context.

## Reading a leaderboard

1. **Check what was run.** The run table at the top of each page gives the dataset, the number of questions, the generator, the judge and the date. Compare systems only within one run, or within runs on the same questions.
2. **Read the recommendation and the tied list before the order.** The tied list names the systems a paired significance test cannot tell apart from the best; among those, the cheapest or fastest is usually the better pick. The order inside a tie is noise, and so is a difference inside the bracketed 95% confidence intervals.
3. **Look at the categories that matter to you.** An overall score can hide a weak category (on MultiHop-RAG, temporal questions and questions the documents cannot answer). The per-category table shows each system's strengths.
4. **Then cost and latency.** Both are per question; the Pareto front lists the systems that no other system beats on answer score, cost and latency at once.
5. **Read the marks and the limitations.** † marks a row from a run on fewer questions; a model-comparison row scored by a different judge is flagged; a cost the provider did not bill directly carries a correction. Each page ends with its limitations.

The metrics themselves (answer score, Recall@5, faithfulness, the confidence intervals and the significance tests) are defined in [Methodology](methodology.md).

## Your own runs

A run writes everything under `results/<run>/`: open `report.html` in a browser for the interactive report, `leaderboard.md` for the table and `recommendation.md` for the pick. `ragbench runs site` puts several runs on one filterable page. [Result files and the site](results.md) lists every file and its schema.
