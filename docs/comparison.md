# RAGBench and the other evaluation tools

RAGBench answers one question: **which RAG setup should I deploy for my documents and my questions, and can I trust that answer?** The tools below answer neighbouring questions, and most teams will want one of them as well. This page says what each one is for, where RAGBench overlaps with it, and when to reach for it instead.

It describes the projects as they were in October 2026, from their public documentation. They change quickly; check their own docs before deciding. Nothing here defines a metric (see [methodology.md](methodology.md)) or lists RAGBench's systems (see [systems.md](systems.md)).

## The short version

| Tool | Built for | Reach for it when |
| --- | --- | --- |
| **RAGAS** | A library of LLM-judged metrics for RAG outputs, plus test-set generation | You already run a pipeline and want its answers and retrieved contexts scored inside your own code or notebooks |
| **ARES** | Judges fine-tuned on synthetic data, with prediction-powered inference to put intervals on the scores from a few human labels | You have (or can afford) human labels and want the statistically tightest estimate of one system's quality |
| **TruLens** | Instrumenting a running app and scoring its traces with feedback functions, with a dashboard | You want to watch a pipeline in development or production and see where each answer went wrong |
| **DeepEval** | Unit tests for LLM apps: metrics as assertions in `pytest`, in CI | You want a regression suite that fails a build when a prompt change makes answers worse |
| **promptfoo** | Declarative matrices of prompts, models and test cases with assertions, plus red-teaming | You are comparing prompts or models on fixed cases, or probing for jailbreaks and leaks |
| **Harness²** | Optimizing an agent's harness (prompts, guardrails, tool guidance) per task, with no ground truth needed | You have an agent and no labelled data, and want the harness improved on the task in front of it |
| **RAGBench** | Running many RAG architectures, your own services and agent harnesses on **one** dataset, comparing them with statistics, and recommending one | You must choose between architectures or vendors, or want to know whether what you run beats a tuned baseline |

## What is different about RAGBench

- **It runs the systems, not just the scorer.** The evaluation libraries take the outputs of a pipeline you already built. RAGBench ships the pipelines (lexical, vector, hybrid, reranked, agentic, GraphRAG, decision-gated and more, [listed in systems.md](systems.md)), runs each on the same questions with the same generator, and so can tell you that *a plain hybrid retriever is enough here* instead of scoring the one thing you already have.
- **It benchmarks what you already run.** A deployed HTTP endpoint, a Python function, a LangChain, LlamaIndex or Haystack pipeline and an agent harness such as Claude Code, Codex or OpenCode all go on the same leaderboard as the built-in systems ([benchmarking-external-systems.md](benchmarking-external-systems.md)). The question "is my production RAG better than a tuned BM25?" gets a number with an interval.
- **The comparison is the product.** Every difference is a paired bootstrap on the same questions with Holm-adjusted p-values, trials expose run-to-run randomness, and the recommendation names the systems *statistically tied* with the best, then chooses among them by cost and latency under the constraints you set. Cost and latency are measured per stage and priced at standalone rates, so a cache cannot flatter a system.
- **Fairness rules are enforced.** The judge never sees more than the generator saw, an independent judge from another model family is the default, missing results are not scored as zero, and row order is deterministic ([methodology.md](methodology.md)).
- **It can improve the winner and prove it.** `ragbench optimize` edits a system's harness on a train split and adopts a change only when a held-out test split confirms it ([optimization.md](optimization.md)); the method never reads the test questions or the judge's verdicts on them.
- **It runs offline.** Every system, judge mode and the optimizer have a deterministic mock path, so you can see the whole pipeline, and CI can run it, with no key and no network.
- **A real result ships with it.** [The MultiHop-RAG benchmark](benchmarks/multihop-rag.md) puts every system RAGBench can run, three framework pipelines, three agent harnesses and six open-weight generators on one public dataset, with the money spent reported per provider.

## What RAGBench does not do

Be honest about the gaps; they are where the other tools win.

- **No production monitoring.** RAGBench evaluates a dataset offline. It does not sit in your app and score live traffic; TruLens and similar observability tools do. [`ragbench export-traces`](observability.md) can send a run's traces to Phoenix, Langfuse or any OpenTelemetry backend, which is the bridge.
- **No unit-test ergonomics.** There is no `assert` per test case in `pytest`. `ragbench compare-runs` flags regressions between two runs, which suits a nightly job better than a per-commit gate; DeepEval and promptfoo are built for the latter.
- **No red-teaming.** RAGBench does not probe for jailbreaks, prompt injection or data leaks. promptfoo has this built in.
- **No fine-tuned judges or human-label calibration.** The judges are prompted models (a panel, pointwise or pairwise, with claim-level faithfulness), not judges trained on your domain. ARES's approach gives tighter intervals when you have a few hundred human labels; RAGBench's intervals reflect question sampling and trial noise, not judge error against humans.
- **No labels, no retrieval metrics.** Retrieval is scored against the relevance labels in your dataset ([dataset-format.md](dataset-format.md)); without labels RAGBench still scores answers, and can write questions from your documents, but the questions come flagged `needs_review`.
- **You bring the cases.** Questions are your dataset's or written from your documents. RAGAS and ARES generate test sets as well; RAGBench's generator exists to get you started, not to replace questions your users actually ask.

## Using them together

- Pick the architecture with RAGBench, then keep a RAGAS, TruLens or DeepEval check on the one you deploy.
- Use `ragbench export-traces` to look at a benchmark run's traces in the observability tool you already have.
- Use promptfoo for the red-team pass on the system RAGBench recommended.
- If you have human labels, run ARES on the winner for a tighter estimate than a prompted judge gives.

## Harness2, the closest relative

[Harness²](https://github.com/google-research/harness2) is the closest in spirit to `ragbench optimize`, and the optimizer is a port of its Propose → Probe → Compose loop with one difference. Harness² refines a harness **per task with no ground truth**: it relies on contrasts between rollouts. RAGBench has a dataset, so it tunes on a train split and has to **prove** the gain on a held-out test split with its paired bootstrap, or the verdict is "no reliable gain" and the base is kept. If you have no questions and no labels, Harness² is the tool; if you have a dataset, RAGBench adds the guarantee. The loop is drawn in [the system wiki](system-wiki.md#the-optimize-loop).
