# Choosing an architecture

There is no best RAG architecture, only one that is best for a particular corpus, set of questions, budget and set of constraints. This guide helps you decide **what to put in the comparison** and **what to look at when it finishes**. It does not define metrics (see [methodology.md](methodology.md)) or list options (see [systems.md](systems.md) and [configuration.md](configuration.md)).

The short version: run the cheap baselines and two or three candidates your situation points to, on your own documents and questions, and let the [recommendation](methodology.md#selection) decide. Architectures that look better on paper often tie with a plain baseline on a real corpus, and the report tells you when that happens.

```bash
ragbench auto --docs ./my_docs                 # a preset of systems, then a recommendation
ragbench run --config my.yaml --only bm25,vector,hybrid,decompose   # or choose the systems yourself
```

## 1. Say what you know about your situation

Match your situation to the candidates it suggests, then add them to a config (or pick the closest [preset](configuration.md#presets)). Each row is a hypothesis to test, not a rule.

| If your situation is... | Try | Why it might help |
| --- | --- | --- |
| **A small corpus** that fits in a model's context | `full_context` and `grep_agent` beside the baselines | They answer "do I need retrieval at all?": the first reads everything, the second needs no index. If they match the retrieval systems, skip the infrastructure. |
| **A large corpus**, or many documents that look alike | `hybrid`, `contextual`, `hierarchical` | Lexical plus semantic search copes with near-duplicates; contextual retrieval keeps a chunk's meaning when it is read alone; summary routing narrows the search to the right documents first. |
| **Exact identifiers**: error codes, SKUs, names, clause numbers | `bm25`, `hybrid`, `agent_search` with `corpus_grep` | Embeddings blur exact strings; keyword search and grep do not. |
| **Paraphrased or vaguely worded** questions | `vector`, `hyde`, `rag_fusion` | They search by meaning, or rewrite the question into wording that matches the documents. |
| **Multi-hop** questions and comparisons | `decompose`, `iterative` | They split the question, or search again with what the first search found. Expect to pay for extra model calls. |
| **Arithmetic, dates, totals** over the documents | `agent_search` with `calculator` and `date_calc` | Models are unreliable at exact arithmetic; a tool is exact, and the tool axis is benchmarkable (`tools: []` against `tools: [calculator, date_calc]`). |
| **Tables and spreadsheets** | `agent_search` with `lookup_table`; the CSV and JSON loaders | A cell value is looked up, not searched for in prose. |
| **Long documents with sections** | `parent_doc`, `sentence_window`, the `markdown` or `recursive` chunker | Precise matching with enough surrounding text to answer from, and cuts that follow the document's own structure. |
| **A mixed workload** where the kinds above all occur | `adaptive` | Routes each question to the pipeline that suits it. Worth it only if the category view shows different winners per category (step 4). |
| **Questions the documents cannot answer** | `corrective`, and `no_retrieval` as a floor | A system that grades its evidence can decline; the floor shows how often the bare model makes things up. |
| **A hard latency or cost limit** | The low-cost, fast systems first; set `selection.constraints` | The recommendation excludes any system that breaks a constraint and says which one. |
| **Data that must stay on your machine** | `local:` embeddings, an `openai_compatible:` endpoint on localhost; `recommend --local-models --no-network` | The decision can be restricted to systems whose every model and tool is local. |

Two systems belong in almost every comparison whatever your situation: `bm25` (the cheap lexical baseline) and `vector` (the semantic one). A fancier system that does not beat them by more than the noise is not worth its extra calls. Add `no_retrieval` to see how much retrieval is buying you at all.

## 2. Treat chunking as part of the architecture

How documents are cut moves retrieval quality as much as the choice of system. Before comparing systems, look at what a chunker does to your documents:

```bash
ragbench chunk-preview --docs ./my_docs --chunker '{type: markdown, chunk_size: 300}'
```

Then compare chunk sizes or chunkers with a [sweep](configuration.md#sweeps) rather than guessing, and let `ragbench inspect-dataset` suggest sizes for your corpus.

## 3. Match the effort to the stakes

`ragbench estimate` projects the cost before you spend; `evaluation.max_cost_usd` in the config (or `--max-cost` on `ragbench auto`) caps it. A reasonable path is the cheapest preset on a small sample of questions, then the candidates your situation pointed to on the full set. Agentic systems (loops, tools, routing) cost the most per question, so add them once the cheap systems have set a bar they must clear. If you have no questions yet, `ragbench auto` writes some, flagged `needs_review`; read them before trusting a result built on them.

## 4. Read the report in this order

1. **The banner.** Is it a mock run? Are the questions synthetic and unreviewed? Is a model unpriced? Any of these changes how much weight the numbers deserve.
2. **The recommendation.** The winner, and the systems statistically tied with it. If several tie, the choice among them is yours: take the cheapest or fastest, which is what the recommendation does by default.
3. **The leaderboard.** Check that the whiskers of the leader and the baselines overlap or not; a lead inside the noise is not a lead. Missing values show as blanks, never zeros.
4. **The categories.** The heatmap shows where each system is strong. If one system wins identifier questions and another wins multi-hop ones, `adaptive` (or two deployments) is worth trying. If one system wins everywhere, you do not need a router.
5. **Cost and latency.** Where the money and the seconds go, by stage. A system that wins by a hair and costs ten times more is rarely the right choice; the quality-against-cost chart shows the trade.
6. **Failures.** The failure types say what to fix: retrieval misses point at chunking or the retriever, over-refusals at too little context, hallucinations at generation or grounding.
7. **The questions.** Open the failing ones. The retrieved passages and the step trace show whether the evidence was never found, found but ranked too low, or found and ignored.
8. **The label audit.** If many answers were judged good while retrieval missed the "relevant" documents, your labels are incomplete, not the systems poor.

What each number means, and how far to trust it, is in [Reading the report](methodology.md#reading-the-report) and [Limitations](methodology.md#limitations).

## 5. A worked example: the bundled demo

The demo dataset (`ragbench demo`) is the documents of a fictional insurance company: 60 documents and 163 questions in 12 categories, among them exact identifiers, temporal conflicts, multi-hop, numeric reasoning, date arithmetic, aggregation, distractors, long-context and unanswerable questions. That mix is exactly the "mixed workload" row above, so a sensible plan is:

1. **Baselines:** `bm25`, `vector`, `hybrid`, plus `no_retrieval`.
2. **Candidates the categories point to:** `decompose` or `iterative` for the multi-hop and comparison questions, `agent_search` with `calculator` and `date_calc` for the numeric and date-arithmetic ones, `corrective` for the unanswerable ones, and `adaptive` to combine them. `configs/all.yaml` has every one of these.
3. **Run it:** `ragbench run --config configs/all.yaml --mock` costs nothing. Mock scores validate the pipeline, not the architectures, so the numbers are not a guide here; drop `--mock` (and look at `ragbench estimate` first) for a result that says something about quality.
4. **Read it:** in the live report, look at the category heatmap first. If the tool-using agent wins the numeric and date categories while the baselines win the identifier ones, that pattern is the case for `adaptive`; if the baselines tie everywhere, the simplest system is the answer.
5. **Decide:** `ragbench recommend --run results/<run> --max-cost 0.002` asks the question again under a cost limit, and `--export winner.yaml` writes the winner's config for you to point at your own documents.

## 6. After you choose

- Replace the `dataset:` in `winner.yaml` with your production documents and run it as the system you deploy.
- Keep the questions. When you change a model, a chunk size or a prompt, re-run and use `ragbench compare-runs OLD NEW` to see whether quality moved by more than the noise.
- Review synthetic questions, and add questions from real users as they arrive; a result is only as representative as its question set.
