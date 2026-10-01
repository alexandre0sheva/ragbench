# RAGBench

[![CI](https://github.com/alexandre0sheva/ragbench/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandre0sheva/ragbench/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typed-PEP%20561-informational.svg)](https://peps.python.org/pep-0561/)

**RAGBench tells you which retrieval-augmented generation setup to deploy for *your* documents and *your* questions.** It runs many architectures (lexical, vector, hybrid, reranked, LLM-driven, tool-using agents and more) on the same data, scores retrieval, answers, cost and latency with confidence intervals, and ends with a recommendation and a runnable `winner.yaml`.

It is not a demo chatbot, and it needs no API key to try: without one it runs on deterministic mock models so you can see the whole pipeline first.

## Quickstart

```bash
pip install -e .
ragbench auto --docs ./my_docs --mock     # drop --mock for a real run: it estimates the cost and asks before spending
```

`auto` profiles your documents, writes questions from them if you have none (flagged `needs_review`; bring your own with `--questions`), runs a preset of systems, and prints the recommended one with the paths of `winner.yaml`, `report.html` and `recommendation.md`. If it stops, `ragbench auto --resume RUN_DIR` continues without paying again. Prefer to write the config yourself? `ragbench init my_ds --docs ./my_docs` scaffolds one; `ragbench demo` and `ragbench run --config configs/all.yaml --mock` run the bundled demo.

Not sure which architecture to try first, or what to look at in the report? Start with [Choosing an architecture](docs/choosing-an-architecture.md).

## What you get

Each run writes one directory: a **recommendation** (which system to deploy under your constraints, the systems statistically tied with it, a runnable `winner.yaml`), a self-contained **`report.html`** (leaderboard with confidence whiskers, quality-against-cost chart, category heatmap, cost and latency by stage, failure types, and a drill-down into every question with the retrieved passages and the agent's steps), `leaderboard.md`, and the CSV and JSONL files all of it is built from. `ragbench report RUN` rebuilds any report from those files, `ragbench runs` lists past runs and `ragbench compare-runs A B` flags regressions.

<!-- Task 32: report screenshots go here (docs/assets/report-desktop-light.png, report-desktop-dark.png, report-mobile.png) -->

## Systems

<!-- systems:start -->
| System | Description | Typical use |
| --- | --- | --- |
| `bm25` | Lexical BM25 over chunks | Cheap baseline and exact-term matching |
| `hybrid` | BM25 + vector search fused with (optionally weighted) Reciprocal Rank Fusion | Balanced lexical + semantic retrieval |
| `hybrid_rerank` | Hybrid BM25 + vector RRF retrieval followed by a reranking pass | Recall of hybrid plus rerank precision |
| `no_retrieval` | The model answers from its own knowledge with no documents at all (the floor baseline) | Showing how much retrieval helps and how often the model makes things up without it |
| `parent_doc` | Retrieve small child chunks, answer from their larger parent chunks | Better answer context with precise retrieval |
| `rerank` | Vector retrieval followed by a reranking pass | Higher precision context selection |
| `sentence_window` | Search single sentences, then hand the generator each matched sentence with the sentences around it | Precise matching with enough surrounding text to answer from |
| `vector` | Embedding search with cosine similarity over a pluggable vector backend (exact NumPy by default) | Semantic baseline |
| `decompose` | The LLM splits a complex question into sub-questions; each is searched (optionally in sequence, feeding earlier answers forward), then one answer is synthesized from all the evidence | Multi-part, comparison and multi-hop questions |
| `hyde` | Hypothetical Document Embeddings: the LLM writes a hypothetical answer used as the search probe | Short or vaguely-worded questions |
| `rag_fusion` | The LLM writes alternative queries; each is searched (hybrid or vector) and the rankings are merged with Reciprocal Rank Fusion | Questions worded differently from the documents |
| `contextual` | An LLM situates each chunk within its document at ingestion; the context is embedded and BM25-indexed with the chunk, then hybrid RRF retrieval | Corpora where chunks lose their meaning out of context (many similar documents, pronouns, section-relative facts) |
| `hierarchical` | Choose documents first from LLM-written summaries and titles, then retrieve chunks only inside the chosen documents | "Which document?" questions and corpora too large to search chunk by chunk |
| `adaptive` | Routes each question to the best of several configured pipelines (exact identifiers to BM25, comparisons to decomposition, calculations to a tool agent, the rest to the default) | Mixed workloads where no single architecture is best for every question |
| `full_context` | Puts whole documents (the whole corpus when it fits) in the prompt, most relevant first by BM25, up to a token budget (the ceiling baseline: do you need retrieval at all?) | Small corpora, and measuring how much a retrieval pipeline loses against reading everything |
| `agent_search` | A function-calling agent that searches the corpus and calls its configured tools (calculator, date_calc, corpus_grep, ...), then writes the cited answer | Questions that need computation, exact lookups, or several searches |
| `corrective` | Retrieve, grade each chunk with the LLM, and when too little is relevant rewrite the query and widen the search; optional answer self-check | Corpora where the first search often misses and a bad context should be noticed rather than answered from |
| `grep_agent` | An index-free agent: it lists, greps and reads the raw documents with tools. No chunking, no embeddings, no vector store | Small or exact-match-heavy corpora, and testing whether you need retrieval infrastructure at all |
| `iterative` | Search, ask the LLM what is known and what is missing, search for the missing part, and repeat until it says it is done or the hop limit is reached | Multi-hop questions whose second search depends on what the first one found |
| `llm_heavy` | LLM-driven ingestion metadata, query rewriting, and reranking | Higher-cost, quality-oriented experiments |
<!-- systems:end -->

Every system implements the same interface and runs on any dataset. Options and trade-offs: [docs/systems.md](docs/systems.md).

## Documentation

- [Choosing an architecture](docs/choosing-an-architecture.md): which systems to try for your corpus, questions, budget and privacy needs, and how to read the result
- [Configuration](docs/configuration.md): config sections, presets, sweeps, budgets, caching, concurrency, model providers
- [Dataset format](docs/dataset-format.md): documents, questions, qrels, importers, generating questions and labels
- [Methodology](docs/methodology.md): metrics, the judge, statistics, cost accounting, how to read the report, limitations
- [Systems](docs/systems.md) and [tools](docs/tools.md) *(generated tables)*, [command-line reference](docs/cli.md) *(generated)*
- [Extending RAGBench](docs/extending.md): new systems, tools, chunkers, rerankers, loaders, vector backends and model providers
- [Changelog](CHANGELOG.md) and the [release checklist](docs/release-checklist.md)

Live runs spend real money, and LLM-heavy and agentic systems spend the most; see [cost accounting](docs/methodology.md#cost-accounting) and how to cap spending in [configuration](docs/configuration.md#estimating-cost-and-capping-it).

## Roadmap

Not built yet:

- GraphRAG-style retrieval over an entity graph
- Persistent vector store adapters for Postgres / pgvector
- Multilingual evaluation: language-aware chunking, and question sets and judges per language
- Tools that call MCP servers
- A web dashboard server for comparing runs (today: the static `results/index.html` from `ragbench runs index`)

## Contributing and license

[CONTRIBUTING.md](CONTRIBUTING.md) explains the workflow and the checks a change must pass; please read the [Code of Conduct](CODE_OF_CONDUCT.md) too. Released under the [MIT license](LICENSE).
