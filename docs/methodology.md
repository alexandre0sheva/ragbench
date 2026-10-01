# Methodology

How RAGBench measures a RAG system, what each number means, and where it can mislead you. This page is the single owner of metric definitions, judge design, cost accounting, fairness rules and limitations; other docs link here instead of restating them.

Contents: [What a run measures](#what-a-run-measures) · [Retrieval metrics](#retrieval-metrics) · [Context metrics](#context-metrics) · [Answer metrics](#answer-metrics) · [The LLM judge](#the-llm-judge) · [Failure types](#failure-types) · [Agent, tool and routing metrics](#agent-tool-and-routing-metrics) · [Cost accounting](#cost-accounting) · [Latency measurement](#latency-measurement) · [Statistics](#statistics) · [Selection](#selection) · [Limitations](#limitations)

## What a run measures

Every system answers every question. Each answer is scored on four independent things, so a weakness shows up where it actually is:

| Layer | Question it answers | Needs | Metrics |
| --- | --- | --- | --- |
| Retrieval | Did the ranking contain the evidence? | `relevant_doc_ids` or qrels | Recall, Precision, Hit, MRR, nDCG at each `k` |
| Context | Did the generator *see* the evidence? | `relevant_doc_ids` or qrels | `context_recall`, `context_precision` |
| Answer | Is the answer right, faithful, and honest about what it does not know? | reference answers (judge, F1, EM), keywords, answerable flag | judge scores, `token_f1`, `exact_match`, `keyword_recall`, abstention metrics |
| Operations | What did it cost and how long did it take? | nothing | `$/Q`, ingestion cost, latency, steps, tool calls |

Every column of `metrics_summary.csv` is a mean over the questions that **succeeded** for that system. A question that raised is recorded with `failure_type: run_error` and excluded from every mean (`n_ok`, `n_error`). A metric that cannot be computed for a question (no reference answer, no keywords, a system that retrieves nothing) is missing for that question and the mean covers the others; if no question has it the cell is empty and reports show `—`. A missing value is never a zero.

## Retrieval metrics

Retrieval is scored at the **document** level, because every system chunks the corpus differently and chunk ids are not comparable. The chunk ranking a system returns is collapsed to a ranking of distinct documents (first occurrence wins), and a document is relevant if it is in the question's `relevant_doc_ids` (or has a positive grade in the qrels).

| Metric | Definition |
| --- | --- |
| `Recall@k` | Relevant documents among the top `k` documents ÷ all relevant documents |
| `Precision@k` | Relevant documents among the top `k` ÷ `k` |
| `Hit@k` | 1 if any relevant document is in the top `k` |
| `MRR@k` | 1 ÷ rank of the first relevant document within the top `k` (0 if none) |
| `nDCG@k` | Discounted cumulative gain of the top `k` with graded qrels (`2^grade − 1`), divided by the ideal ordering's |

**What `@k` means.** Retrieval metrics are computed on a ranking of `evaluation.retrieval_depth` chunks (default `max(k_values)`), independent of how many chunks the generator reads. The headline columns are recall at `primary_k` (default 5) and MRR / nDCG at the deepest `k`. The generator is given only the first `context_k` chunks of that ranking (a system's own `top_k` / `final_top_k` wins), so a system can rank the evidence 7th, score well at `@10`, and still never show it to its LLM. [Context metrics](#context-metrics) measure that gap. The keys are described in [configuration.md](configuration.md#retrieval-depth-context-size-and-failures).

Systems that retrieve nothing (`no_retrieval`) have no retrieval metrics at all: their cells are blank, they are never classified as a retrieval miss, and they stay out of the qrels audit. Agentic systems are scored on the passages their tool calls returned (one ranking per call, merged with Reciprocal Rank Fusion).

## Context metrics

These are computed on the chunks the generator was actually given (`context_chunk_ids` of the answer), against the same relevant documents as retrieval.

| Metric | Definition |
| --- | --- |
| `context_recall` | Relevant documents that at least one given chunk comes from ÷ all relevant documents. Missing for unanswerable questions (nothing to find). |
| `context_precision` | Given chunks that come from a relevant document ÷ chunks given. Missing when nothing was given or there are no relevant documents. |

`context_recall` below the retrieval recall at the deepest `k` means ranking quality is being thrown away before generation: raise `context_k`, or use a reranker so the evidence is on top. `context_precision` is the noise the generator has to read through; it is low by design for systems that give many chunks, and it ignores whether an irrelevant chunk was harmless.

## Answer metrics

### Deterministic metrics

No LLM, no network, identical on every run, and free. They cannot read meaning, but they cannot drift or favour a model family either.

| Metric | Definition |
| --- | --- |
| `token_f1` | F1 of token overlap between the answer and `reference_answer`. |
| `exact_match` | 1 if the normalized answer equals the normalized reference, else 0. |
| `keyword_recall` | Share of the question's `expected_keywords` that appear in the answer as whole tokens. |

Normalization lowercases, removes `[doc_id]` citations, punctuation (decimal points are kept) and digit-grouping commas (`1,200` = `1200`), and drops the articles *a*, *an*, *the*. Questions without a reference answer (or without keywords) and unanswerable questions (their reference only says the answer is not in the documents, which the abstention metrics judge) have no value. Expect `exact_match` to be near zero for free-text references: models answer in sentences. It is meant for short factual references; `token_f1` and `keyword_recall` are the useful ones for the rest.

### Abstention metrics

Whether a system says "I could not find the answer" when it should, and only then. A question is *answerable* when it has relevant documents; an answer is a *refusal* when it declines to answer (the sentence the prompts ask for, or a close paraphrase such as "the documents do not contain…"; an answer that merely contains "not available", as in "the plan is not available in Europe", is not a refusal).

| Metric | Definition | Higher is |
| --- | --- | --- |
| `abstain_precision` | Refusals that were right (the question had no answer) ÷ all refusals | better |
| `abstain_recall` | Unanswerable questions that were refused ÷ all unanswerable questions | better |
| `false_refusal_rate` | Answerable questions that were refused ÷ all answerable questions | worse |

A rate with an empty denominator (no unanswerable questions in the dataset, no refusals at all) is missing. A system that always answers has `abstain_recall` 0; one that always refuses has `false_refusal_rate` 1. Read the two together.

## The LLM judge

The judge scores five axes from 0 to 5 per question; `answer_score` is their unweighted mean, `faithfulness` is reported on its own.

| Axis | Asks |
| --- | --- |
| `correctness` | Does the answer state what the reference states? |
| `faithfulness` | Is every claim supported by the context the generator saw? |
| `completeness` | Does it cover everything the reference covers? |
| `relevance` | Does it address the question asked? |
| `citation_quality` | Are the cited `[doc_id]`s the ones holding the evidence? |

The judge also sets `is_supported_by_context` and `is_hallucinated`, which feed [failure classification](#failure-types). It sees the question, the reference answer, the answer, and **only the chunks that were given to the generator** (not the deeper ranking).

### Prompt (version `v2`)

Each axis has written anchors for 0, 3 and 5, the judge is told to ground every verdict only in the reference and the supplied context (never its own knowledge), and it must reply with one JSON object. The prompt version is stored with every judgment (`prompt_version` in `answer_judge.metadata` of `per_question_results.jsonl`); do not compare scores across versions.

### Configuration

```yaml
evaluation:
  judge_enabled: true
  judge:
    model: anthropic:claude-haiku-4-5   # default: evaluation.judge_model
    samples: 3                          # independent judgments per question, averaged
    temperature: 0.5                    # must be above 0 when samples > 1
    independent: true                   # warn when the judge is also a generator
```

With `samples > 1` the five axes are averaged, a flag is set when most samples set it (`is_hallucinated` at half), the cost of every sample is charged, and the population variance of `answer_score` across samples is stored as `answer_score_variance`. Use it to see how noisy the judge is on your data. Samples at temperature 0 would be identical, so that combination is rejected when the config loads.

### Bias and noise

- **Self-preference.** An LLM judge tends to rate answers from its own model family higher. When the judge model is also the generator of a system (same provider and model), the leaderboard and `report.html` carry a "Self-preference risk" warning naming those systems. Pick a judge from another family under `evaluation.judge.model`, or set `independent: false` to accept the risk. The warning is only raised when an LLM judge scored the run.
- **Noise.** A judge is not deterministic across models, prompts or sampling. With a few dozen questions a difference of a few tenths of a point is usually noise; see [Statistics](#statistics).
- **Reference dependence.** Correctness is judged against `reference_answer`; a wrong or incomplete reference misleads the judge.

### Failure handling and the heuristic judge

A reply counts as a judgment only if it is a JSON object with all five numeric scores (a missing axis is not read as 0). An unusable reply is retried once, with the bad reply and a format reminder appended so the retry is a different request. If that fails too, a deterministic **heuristic judge** (keyword and reference overlap, support check against the context, refusal detection) scores that one question, and the row is marked `fallback: true`. The run reports the share per system as `judge_fallback_rate` and adds a warning to the leaderboard when it is above 0, because those answer scores mix two judges. `parse_error` and `retries` are stored per question, and every call, retries included, is charged to the judge's cost.

The heuristic judge is also what scores everything in **mock mode** and when `judge_enabled: false`. That is by design, so `judge_fallback_rate` is empty then, and `judge` in the metadata reads `heuristic` with `fallback: false`. Heuristic scores validate the pipeline; they say nothing about quality.

## Failure types

Every question gets one `failure_type` (`failures.md`, `per_question_results.jsonl`, the failure chart in `report.html`). The classifier uses structured signals only: what the generator saw (`context_recall`), the ranking (`Hit@k`), refusal detection, the judge's scores and flags, and the question's `answer_type`.

| Type | When |
| --- | --- |
| `no_failure` | Answerable: correctness ≥ 3.5 and faithfulness ≥ 3. Unanswerable: refused. |
| `retrieval_miss` | The generator saw none of the evidence, and the ranking never held it either |
| `bad_reranking` | The generator saw none of the evidence, but a deeper rank held it (ranked below what the generator reads) |
| `possible_qrels_gap` | The generator saw no labeled evidence yet the answer is well supported (answer score and faithfulness ≥ 4): the labels may be incomplete, see `qrels_audit.md` |
| `over_refusal` | Refused although all the evidence was in the context |
| `insufficient_context` | Part of the evidence never reached the generator and the answer is a refusal or wrong. Also the fallback for a bad answer from a system whose context is not measured |
| `answer_hallucination` | The judge flagged a hallucination; an unanswerable question that was answered; or a badly wrong answer that the context does not support |
| `wrong_date` / `wrong_entity` | A wrong answer to a question with `answer_type` `date`, or `entity` / `person` / `name` / `organization`, whose expected keywords are missing from the answer |
| `partial_answer` | Everything else that scored below the bar |
| `format_error` | The answer is empty |
| `run_error` | The question raised; it is excluded from every mean |

When `context_recall` is not measured (a system that retrieves nothing), the ranking cut-off `Hit@primary_k` stands in for it. Classification is a triage aid, not a verdict: it is only as good as the judge, the qrels, and the `answer_type` labels.

## Agent, tool and routing metrics

Present only for systems that have them.

- **Agent loop** (`corrective`, `iterative`, `agent_search`, `grep_agent`). A loop runs a policy step by step under an `AgentBudget` (`max_steps`, `max_cost_usd`, `max_tokens`, `max_tool_calls`) and stops when the policy finishes, aborts, or a limit is reached. A spent budget ends the loop with `termination: budget` and a best-effort answer from what it has found, never an error; a misbehaving policy cannot loop past the step cap. Columns: `avg_steps` (loop iterations), `avg_llm_calls`, `budget_exhausted_rate` (share of questions that hit a cap). Per-question details are in the `agent` block of `per_question_results.jsonl`.
- **Tools** (see [tools.md](tools.md)). `avg_tool_calls`, `tool_error_rate`, `required_tool_used_rate` (on questions whose `requires_tools` names a tool, how often one was used) and the per-tool `tool_usage.csv`.
- **Routing** (`adaptive`). `routes.csv` per route (questions, share, mean answer score and cost) and `route_accuracy` when questions carry `routing_hint`.

## Cost accounting

- **`$/Q`** is the mean cost of answering one question **plus judging it**: embeddings, generation, query rewriting, reranking, tool calls and the judge's call (the `judge_cost` column in `cost_breakdown.csv` separates it). Ingestion cost is reported separately, per system, and is not amortized into `$/Q`. Stage tables in `leaderboard.md` ("Cost by stage") cover answering only; the judge is excluded. Step costs always add up to the answer's cost; anything a system fails to trace appears as an explicit `untracked` step.
- **Prices** come from the table in `src/ragbench/models/cost.py` (as of `PRICING_AS_OF`) and can be overridden per experiment with `pricing:` ([configuration.md](configuration.md#pricing)). A model with no price is charged $0 with a visible warning in every report, and prices change: costs from different dates are not comparable. `llm_heavy`, `contextual` and the agentic systems can be materially more expensive because they call an LLM during ingestion, query rewriting, reranking, or several times per question; the judge adds its own calls (more with `judge.samples`).
- **Budget caps and estimates use the same charges.** `evaluation.max_cost_usd` counts charged cost (the standalone prices above), and `ragbench estimate` projects it by running each system on the mock models and pricing the measured tokens ([how, and how far to trust it](configuration.md#estimating-cost-and-capping-it)). A run stopped by its cap compares only the systems that finished, because a mean over some questions is not comparable with a mean over all of them.
- **Caching never changes the bill a system is shown.** A cache hit, on disk or in process, is charged to the requesting system at the *current standalone price*, as if the call had been made; embedding cost is charged per text from one token estimate whether or not it was a hit, so a system's ingestion charge does not depend on run order. Systems in one run share corpus embeddings (`evaluation.embedding_cache`), and live runs keep LLM responses and corpus embeddings on disk. The spend actually avoided is reported separately (`cache.saved_cost_usd`, `real_spend_usd` in `run_summary.json`). This keeps `$/Q` the cost of deploying that system alone. See [Caching](configuration.md#caching) for the rules.

## Latency measurement

Latency is the wall time of answering one question, without judging. Measured while many questions run at once it includes queueing behind other requests, so live runs finish with a **probe**: `evaluation.latency_probe_questions` (default 5) questions per system, spread across the dataset, are re-asked one at a time with the disk cache bypassed. The Latency column then shows the probe's mean, with `p95`; `latency_source` says `probe` or `concurrent`. A cached LLM response replays the latency of the original call, and query embeddings are not cached by default, so cache hits do not make a system look faster. Mock runs skip the probe; their latencies are microseconds of CPU. Details: [configuration.md](configuration.md#latency-measurement).

## Statistics

A leaderboard of means invites over-reading: with 50 questions a 0.02 gap is noise. RAGBench therefore reports how uncertain each number is and only calls one system better than another when the data supports it. Everything is seeded (`evaluation.stats.seed`), so the same results produce the same intervals and p-values, and uses numpy only.

### Confidence intervals

The headline metrics (`Recall@primary_k`, `MRR` and `nDCG` at the deepest `k`, `answer_score`, `faithfulness`) carry a 95% **percentile-bootstrap** interval: the questions a system answered are resampled with replacement (`n_boot` times, default 2000) and the interval is the 2.5th to 97.5th percentile of the resampled means. They appear as `<metric>_ci_lo` / `<metric>_ci_hi` in `metrics_summary.csv` and as `mean [lo, hi]` in the leaderboards. A system that retrieves nothing has no retrieval interval.

For scale: if per-question answer scores spread with a standard deviation of about 1 point, the interval around a mean over 50 questions is roughly ±0.28 wide, and over 200 questions roughly ±0.14. Differences smaller than that are not evidence of anything.

The interval measures **which questions you happened to ask**. It does not include judge noise, run-to-run randomness in a system, or the chance that your questions are unrepresentative of real traffic. Overlapping intervals do not by themselves mean two systems are tied (the comparison below is sharper, because it uses the same questions for both), and non-overlapping intervals are not a significance test.

### Paired comparison against a baseline

`significance.csv` compares every system with one **baseline** system (`evaluation.stats.baseline`; default: the cheapest system by `$/Q`, the first of them on a tie) on the questions both answered, for `answer_score` and for recall at `primary_k`. Pairing matters: questions differ far more in difficulty than systems differ in quality, and comparing each question with itself removes that noise.

| Column | Meaning |
| --- | --- |
| `mean_diff`, `ci_lo`, `ci_hi` | The system's mean minus the baseline's, with its bootstrap interval. Positive means the system scored higher. |
| `p_value` | Two-sided bootstrap p-value: how often resampled differences fall on the other side of zero (doubled, never exactly 0; the floor is about `2 / (n_boot + 1)`). |
| `p_holm`, `significant` | The p-value after **Holm-Bonferroni** adjustment across the systems compared on that metric, and whether it is below 0.05. Comparing many systems makes some "wins" appear by luck; Holm keeps the chance of any false win across the family at 5%. Each metric is its own family. |
| `verdict` | `better`, `worse`, or `no clear difference`. The last means the data cannot rank the two systems, not that they are equal: with more questions it may resolve. |
| `wins`, `ties`, `losses` | Questions where the system scored higher, equal, or lower than the baseline. |

Metrics a system does not have (retrieval for a system that retrieves nothing) produce no row. The comparison is made against one baseline only; to compare two other systems, set `baseline` to one of them.

### Efficiency: cost per correct answer and the Pareto front

- **`cost_per_correct`** (`$/correct`): everything spent on a system's questions (the same per-question cost as `$/Q`: answering plus judging, ingestion excluded) divided by the number of answers whose `answer_score` is at least 4. Wrong answers cost money too, so they stay in the numerator. A system with no correct answer has none (blank): there is no finite price for something it never delivered. It answers "what does one good answer cost me?", which `$/Q` alone cannot.
- **`pareto_optimal`**: a system is on the Pareto front when no other system is at least as good on all of **answer score (higher), `$/Q` (lower) and latency (lower)** and strictly better on one. These are the systems worth choosing among; the rest are beaten outright by some other system. `pareto.json` lists the front and, for every system off it, which systems dominate it. Two systems with identical numbers are both kept, and a system missing one of the three values cannot be placed. The front compares means, so a system can be on it by a margin the intervals do not support; read it together with the intervals. In mock runs every cost is 0 and latencies are microseconds, so the front there only checks the plumbing.

### Configuration and outputs

```yaml
evaluation:
  stats:
    n_boot: 2000          # bootstrap resamples (min 100)
    seed: 0               # same results + seed = same intervals and p-values
    baseline: bm25        # system every other one is compared with; default: the cheapest
```

| File | Contents |
| --- | --- |
| `metrics_summary.csv` | adds `<metric>_ci_lo`, `<metric>_ci_hi`, `cost_per_correct`, `pareto_optimal` |
| `significance.csv` | one row per (metric, system) against the baseline: `n`, `mean_diff`, `ci_lo`, `ci_hi`, `p_value`, `p_holm`, `significant`, `verdict`, `wins`, `ties`, `losses` |
| `stats.json` | method, `n_boot`, `seed`, baseline (and where it came from), per-system question counts, means and intervals, `cost_per_correct` |
| `pareto.json` | the axes, the front, the points, and `dominated_by` |

`leaderboard.md` shows the intervals, a Pareto column and the significance table; `report.html` shows the intervals and marks Pareto-optimal systems with ★.

## Selection

`recommendation.md`, `recommendation.json` and `winner.yaml` answer "which one do I deploy?" from a run's results. `ragbench recommend --run <dir>` repeats it on any finished run with other constraints. The rule:

1. **Constraints first.** A system that breaks a hard limit (`max_cost_per_question`, `max_latency_ms_p95`, `min_faithfulness`, `min_answer_score`, `max_ingestion_cost`, `require_local_models`, `require_no_network`) is excluded, and the report says which limit and by how much. A metric the run did not measure cannot be shown to meet a limit, so it fails it. If nothing qualifies there is no winner, and the closest miss is named.
2. **Quality.** The best mean answer score among the rest (token F1, then recall, when there is no answer score) is the benchmark to beat.
3. **Statistical ties.** Every system whose quality is *not significantly worse* than the best, by the [paired bootstrap](#paired-comparison-against-a-baseline) with Holm adjustment across the systems tested, ties with it. A lead inside the noise is not a reason to pay more.
4. **Tie-break.** Among the tied, the profile's weights pick on cost and latency (the cheapest, the fastest, or the best mix), then the simplest system (fewest model calls per question, then steps).

Profiles bundle weights with this policy: `balanced` (quality 0.6, cost 0.2, latency 0.2), `cheapest_acceptable` (cost decides among the tied), `lowest_latency` (speed decides among the tied) and `max_quality` (ties are *not* merged: the highest mean wins whatever it costs). `selection.weights` overrides a profile's weights. The ranking lists the winner, then the tied systems best-first, then the rest by weighted score (min-max normalized across the feasible systems). `by_category` names the best system per question category and is descriptive only: categories are small.

`require_local_models` accepts a `local:` embedder or an `openai_compatible:` endpoint on localhost for every model a system calls; `require_no_network` additionally rejects any tool with network side effects. The rationale is generated from the data (differences with their intervals, cost ratios, which systems tie and why), mock runs and runs with under 30 questions carry a caveat, and `winner.yaml` is the winner's system block plus the providers, tools and limits it needs; its `dataset:` is the benchmark's, to be replaced with your production documents.

## Limitations

- **Document-level relevance.** A retrieved chunk counts if its document is relevant, whether or not that chunk holds the evidence. Context precision is therefore generous, and unlabeled but helpful documents count against a system (`qrels_audit.md` surfaces them).
- **LLM-judge noise and bias.** See [Bias and noise](#bias-and-noise). Scores are only comparable within one run (one judge, one prompt version).
- **Heuristic refusal detection.** Abstention metrics rely on recognizing refusals by wording. Models that decline in unusual phrasing are counted as answering.
- **Deterministic metrics are literal.** `exact_match` and `token_f1` penalize correct paraphrases and reward verbose answers that repeat the reference.
- **Failure classification is heuristic** and depends on labels you supply (`answer_type`, qrels).
- **Mock mode** uses hashing embeddings, a scripted LLM and the heuristic judge. It validates that pipelines run and that results are reproducible; it says nothing about real quality, and its banner is on every report.
- **Small datasets.** With tens of questions, per-category numbers rest on a handful of examples, and most differences between systems will be `no clear difference`. Add questions before trusting a ranking.
- **A recommendation is only as good as the benchmark.** It rests on your questions, your judge and the constraints you set; a different dataset can crown a different system. Treat it as the answer to "what did this evidence support", and check the ties.
- **Intervals cover question sampling only.** They do not include judge noise or a system's own randomness, and they assume your questions resemble the questions you care about.
