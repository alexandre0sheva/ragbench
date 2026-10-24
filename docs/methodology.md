# Methodology

How RAGBench measures a RAG system, what each number means, and where it can mislead you. This page is the single owner of metric definitions, judge design, cost accounting, fairness rules and limitations; other docs link here instead of restating them.

Contents: [What a run measures](#what-a-run-measures) · [Retrieval metrics](#retrieval-metrics) · [Context metrics](#context-metrics) · [Answer metrics](#answer-metrics) · [The LLM judge](#the-llm-judge) · [Failure types](#failure-types) · [Agent, tool and routing metrics](#agent-tool-and-routing-metrics) · [Cost accounting](#cost-accounting) · [Latency measurement](#latency-measurement) · [Statistics](#statistics) · [Selection](#selection) · [Reading the report](#reading-the-report) · [Synthetic questions and pooled labels](#synthetic-questions-and-pooled-labels) · [Limitations](#limitations)

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

Systems that retrieve nothing (`no_retrieval`) have no retrieval metrics at all: their cells are blank, they are never classified as a retrieval miss, and they stay out of the qrels audit. The same holds for an answerable question with no relevance labels (no `relevant_doc_ids`, no positive qrels; see [label-free mode](dataset-format.md#label-free-mode)): its retrieval and context metrics are blank, and if no question is labeled the retrieval columns are blank for every system. Agentic systems are scored on the passages their tool calls returned (one ranking per call, merged with Reciprocal Rank Fusion).

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

Whether a system says "I could not find the answer" when it should, and only then. A question is *answerable* when it has relevant documents, or when it says `answerable: true` ([dataset-format.md](dataset-format.md#questions)); an answer is a *refusal* when it declines to answer (the sentence the prompts ask for, or a close paraphrase such as "the documents do not contain…"; an answer that merely contains "not available", as in "the plan is not available in Europe", is not a refusal).

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

### Prompt (version `v3`)

Each axis has written anchors for 0, 3 and 5, the judge is told to ground every verdict only in the reference and the supplied context (never its own knowledge) and is told whether the question is answerable (with no reference it grades against the context alone), and it must reply with one JSON object. The prompt version is stored with every judgment (`prompt_version` in `answer_judge.metadata` of `per_question_results.jsonl`); do not compare scores across versions.

### Configuration

```yaml
evaluation:
  judge_enabled: true
  judge:
    model: anthropic:claude-haiku-5-5   # default: evaluation.judge_model
    samples: 3                          # independent judgments per question, averaged
    temperature: 0.5                    # must be above 0 when samples > 1
    independent: true                   # a judge never scores its own model family when another can
```

With `samples > 1` the five axes are averaged, a flag is set when most samples set it (`is_hallucinated` at half), the cost of every sample is charged, and the population variance of `answer_score` across samples is stored as `answer_score_variance`. Use it to see how noisy the judge is on your data. Samples at temperature 0 would be identical, so that combination is rejected when the config loads.

### Judge panels and agreement

`evaluation.judge.panel` replaces the one judge with two or more, ideally from different model families. Each judge scores every answer on its own (same prompt, same context, its own samples and retries), and the panel's verdict is the mean of each axis across judges (`aggregate: median` takes the median, which one outlying judge cannot move); a flag is set when most judges set it (`is_hallucinated` at half). A judge whose verdict was unusable is left out of that answer's aggregate; only when every judge failed does the heuristic score the answer (`fallback: true`). Every judge's calls are charged to the judge cost, and `ragbench estimate` counts them. The first judge of the panel is the lead: what needs a single model (grading relevance in `ragbench dataset label`) uses it.

With `independent: true` (the default) a judge **sits out** the answers of every system whose generator is from its own model family, so with `panel: [anthropic:claude-haiku-5-5, gpt-6-luna]` Haiku scores the GPT systems, GPT-6 Luna scores the Claude systems, and both score the rest. The family comes from the model name (`claude`, `gpt`, `gemini`, `llama`, `qwen`, `deepseek`, …) or, failing that, the OpenRouter vendor, so `openrouter:anthropic/claude-haiku-5.5` is Anthropic. When every judge is of a system's family, all of them score it and the self-preference warning names that system.

Each judged row records the panel in `answer_judge.metadata`: `judges` (who scored it), `sat_out`, `member_fallbacks`, and `panel_scores` (every judge's five axes and `answer_score`; empty for one that sat out or failed). From those, the run computes **`judge_agreement`**: Krippendorff's alpha (interval metric) between the judges, per axis and for `answer_score`, over every answer of the run, with a judge that sat out an answer counted as a missing value. Alpha is 1 when the judges give the same scores, 0 when they agree no better than chance, and negative when they disagree systematically. It is written to `run_summary.json` under `judge.agreement` and shown in the report banner; below 0.667 (the usual floor for even tentative conclusions) the reports warn that small score differences between systems are not trustworthy. A one-judge run has no `judge` block and no agreement number: use `samples` for its noise, and a human audit for its accuracy.

### Calibrating the judge against a human

Agreement between judges says they are consistent, not that they are right. `ragbench judge-audit RUN --sample 50` draws answers for a person to score by hand into `judge_audit_sheet.csv`: answers are grouped by system and by the judge's score band (below 2, 2 to 4, 4 and up) and drawn from the groups in turn (seeded), so every system and the whole score range are represented, failed questions excluded. The sheet shows what the judge saw (question, reference, the context the generator was given, the answer) and neither the system nor the judge's scores, so the person cannot anchor on them; `judge_audit_key.json` maps each row back to its answer. The person fills in any of the five axes from 0 to 5, with the same anchors as the judge prompt; blank cells are skipped.

`ragbench judge-audit RUN --labels filled.csv` then writes `judge_audit.md` and `judge_audit.json` and links them from the report banner. Per axis it reports the number of labels, Krippendorff's alpha between judge and human, the mean absolute difference, and the **bias** (judge minus human: above 0 the judge is lenient). The human `answer_score` is the mean of the five axes and is compared only where all five are filled. It also reports **verdict agreement**, the share of answers that judge and human both call correct (answer score at least 4) or both call not correct; the bias per system, since a judge lenient with one system favours it in the leaderboard; and, for a panel, each judge's alpha against the human. Fifty answers are enough to see a large bias; small differences need more.

### Bias and noise

- **Self-preference.** An LLM judge tends to rate answers from its own model family higher. When every judge that scored a system is from the family of that system's generator (a single judge always is, when the families match), the leaderboard and `report.html` carry a "Self-preference risk" warning naming those systems. Pick a judge from another family under `evaluation.judge.model` ([which one](models.md#picking-models)), add one to a [panel](#judge-panels-and-agreement) so it takes over those systems, or set `independent: false` to accept the risk. The warning is only raised when an LLM judge scored the run. The shipped configs judge their OpenAI generators with `anthropic:claude-haiku-5-5`.
- **Noise.** A judge is not deterministic across models, prompts or sampling. With a few dozen questions a difference of a few tenths of a point is usually noise; see [Statistics](#statistics).
- **Reference dependence.** Correctness is judged against `reference_answer`; a wrong or incomplete reference misleads the judge. An answerable question with no reference (label-free datasets often have none) is judged against the retrieved context alone: correctness and completeness then say how well the answer uses what the context offers, so they track faithfulness and cannot catch an answer that is faithful to the wrong passages. Add reference answers when you can.

### Failure handling and the heuristic judge

A reply counts as a judgment only if it is a JSON object with all five numeric scores (a missing axis is not read as 0). An unusable reply is retried once, with the bad reply and a format reminder appended so the retry is a different request. If that fails too, a deterministic **heuristic judge** (keyword and reference overlap, support check against the context, refusal detection) scores that one question, and the row is marked `fallback: true`. The run reports the share per system as `judge_fallback_rate` and adds a warning to the leaderboard when it is above 0, because those answer scores mix two judges. `parse_error` and `retries` are stored per question, and every call, retries included, is charged to the judge's cost.

The heuristic judge is also what scores everything in **mock mode** and when `judge_enabled: false`. That is by design, so `judge_fallback_rate` is empty then, and `judge` in the metadata reads `heuristic` with `fallback: false`. Heuristic scores validate the pipeline; they say nothing about quality.

## Pairwise arena

Pointwise scores ask a judge for an absolute number, which drifts between judges and prompts and, on a [label-free](dataset-format.md#label-free-mode) dataset, mostly measures faithfulness to the context. `evaluation.pairwise` adds a relative measure: after every system has answered, a judge is shown two answers to the same question and asked which one is better, and the verdicts add up to an **arena score**.

**What the judge sees.** The question, whether it is answerable, the reference answer when there is one (`null` otherwise), and the two answers as `answer_1` and `answer_2`: never which system wrote them, nor the context. It replies `1`, `2` or `tie` (prompt version `p1`, stored with every comparison); it is told that a clear refusal beats a confident wrong answer and not to prefer a position, a length or a style. An unusable reply is retried once with a format reminder; if that fails the comparison has no verdict.

**Both orders.** Every pair is judged twice, once with each answer first, and which one goes first in the first order is a seeded coin flip. A comparison scores its first system 1 when it won both times, 0 when it lost both, 0.5 for two ties, 0.75 for a win and a tie, and so on. A judge that picks whatever stands in one position gives each answer one win and one loss: the comparison counts as a draw, so position bias cancels instead of deciding the ranking. A comparison counts only when both orders gave a verdict; the others are left out (`failed`), never scored as a loss. Only each system's first-trial answer takes part, and a question a system failed on gives it no comparison there.

**Which pairs meet.** `pairs: all` (default) compares every pair of systems on every question; `vs_baseline` compares every system with the baseline of [the paired comparison](#paired-comparison-against-a-baseline) (`evaluation.stats.baseline`, else the cheapest); `adaptive` splits the questions into four rounds, pairs each question's systems around a shuffled ring in the first, and in later rounds spends each question's pairs where the arena so far is least sure (the largest p(1 − p) under the current strengths, discounted by how often the pair already met). `max_pairs_per_question` caps a question's pairs by a seeded sample (for `adaptive` it is the number of pairs per question, default the number of systems). Every schedule is seeded by `evaluation.stats.seed`.

**Who judges.** The judge of [the LLM judge](#the-llm-judge) section, asked pairwise. With a panel and `independent: true`, a pair is judged by the first panel judge whose model family wrote neither answer; when every judge shares a family with one side (a GPT system against a Claude system, judged by GPT-6 Luna and Haiku 5.5), all of them judge it, so their leanings offset, and the comparisons are counted as `same_family`. With one judge, a pair involving its own family is judged by it all the same and counted. In mock mode, and with `judge_enabled: false`, a heuristic stands in: the answer with the higher pointwise score wins (a tie within 0.25), so a mock arena repeats the mock leaderboard.

**The arena score.** A Bradley–Terry model gives each system a strength s such that s_x / (s_x + s_y) is the chance x beats y; it is fitted by maximum likelihood (Hunter's MM algorithm) with one virtual drawn comparison per pair of systems, which keeps a system that never wins or never loses at a finite strength. The arena score is that strength on the Elo scale, 1000 + 400 log10(s / geometric mean): 1000 is the average system and 400 points apart means 10:1 odds. Its 95% interval is a percentile bootstrap that resamples questions (with all their comparisons) and refits, `evaluation.stats.n_boot` times. Scores compare systems of one run; they are not comparable across runs with different systems, questions or judges. The **win rate** of x against y is the share of points x won in their comparisons, and `arena_win_rate` is a system's share against everyone it met.

**Position bias** is reported on the raw verdicts: `flip_rate` is the share of complete comparisons whose verdict followed the position (the same position won in both orders), and `first_position_rate` the share of decisive verdicts that went to the answer shown first (0.5 is unbiased). A live run with a `flip_rate` of 25% or more gets a warning: the arena is not skewed by those comparisons, but it rests on fewer.

**Cost.** Two judge calls per comparison, charged like pointwise judging (standalone prices, cache hits included) to the run rather than to a system, so `$/Q` is unchanged; they count toward `evaluation.max_cost_usd` (once it is reached no new comparison starts) and `ragbench estimate` lists them, priced at the lead judge's rate. All pairs of n systems is n(n − 1) calls per question: with many systems use `adaptive` or `max_pairs_per_question`.

**Outputs.** `pairwise.jsonl` (one line per comparison: question, the two systems in first-order order, judge, both verdicts, score, `flipped`, `same_family`), `arena.csv` (per system: score and interval, wins, ties, losses, comparisons, win rate), the columns `arena_score`, `arena_score_ci_lo`, `arena_score_ci_hi` and `arena_win_rate` in `metrics_summary.csv` (an "Arena" column in the leaderboards), and a `pairwise` block in `run_summary.json` with the schedule, judges, counts, cost, position bias and the win-rate matrix.

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
- **Decision gate** (`decision_gate`). `gate_abstain_rate` (the share of questions the gate answered with a refusal instead of calling the generator), `avg_chunks_kept` (chunks that passed the answerability threshold, at most the context size) and `avg_decision_calls` (decision requests per question). A refusal the gate produces is a refusal like any other, so correct and wrong ones are scored by the [abstention metrics](#abstention-metrics); retrieval is scored on the ungated ranking and the context metrics on the kept chunks. Per-question details, including the probability of every judged chunk, are in the `gate` block of `per_question_results.jsonl`.
- **Tools** (see [tools.md](tools.md)). `avg_tool_calls`, `tool_error_rate`, `required_tool_used_rate` (on questions whose `requires_tools` names a tool, how often one was used) and the per-tool `tool_usage.csv`.
- **Routing** (`adaptive`). `routes.csv` per route (questions, share, mean answer score and cost) and `route_accuracy` when questions carry `routing_hint`.

## Cost accounting

- **`$/Q`** is the mean cost of answering one question **plus judging it**: embeddings, generation, query rewriting, reranking, typed decision requests (`decision_cost`, billed on input tokens only for the Decisions API), tool calls and the judge's call (the `judge_cost` column in `cost_breakdown.csv` separates it). Ingestion cost is reported separately, per system, and is not amortized into `$/Q`. Stage tables in `leaderboard.md` ("Cost by stage") cover answering only; the judge is excluded. Step costs always add up to the answer's cost; anything a system fails to trace appears as an explicit `untracked` step.
- **Prices** come from the table in `src/ragbench/models/cost.py` (as of `PRICING_AS_OF`) and, for OpenRouter models, from the dated OpenRouter catalog; both can be overridden per experiment with `pricing:` ([configuration.md](configuration.md#pricing)). A price may depend on prompt length (Claude Haiku 5.5 above 100k prompt tokens, GPT-6 above 272K): each request is priced on its own at the tier its prompt falls in, all of its tokens at that tier's rates, as the providers bill. `ragbench estimate`, which measures summed tokens per question, picks the tier from the mean prompt of one call.
- **Billed versus estimated cost.** For OpenAI and Anthropic a call's cost is its token counts times the table's price: an estimate of the bill that ignores cached-input and batch discounts. An OpenRouter response states what it cost (`usage.cost`), and that billed figure is the call's cost; the table is then only the estimate before the run and the fallback for a response without one. A cache hit on an OpenRouter call is charged the cost the original call was billed. A model with no price is charged $0 with a visible warning in every report, and prices change: costs from different dates are not comparable. `llm_heavy`, `contextual` and the agentic systems can be materially more expensive because they call an LLM during ingestion, query rewriting, reranking, or several times per question; the judge adds its own calls (more with `judge.samples`).
- **Budget caps and estimates use the same charges.** `evaluation.max_cost_usd` counts charged cost (the standalone prices above) plus what the latency probe spends, and `ragbench estimate` projects it by running each system on the mock models and pricing the measured tokens ([how, and how far to trust it](configuration.md#estimating-cost-and-capping-it)). A run stopped by its cap compares only the systems that finished, because a mean over some questions is not comparable with a mean over all of them.
- **Caching never changes the bill a system is shown.** A cache hit, on disk or in process, is charged to the requesting system at the *current standalone price*, as if the call had been made; embedding cost is charged per text from one token estimate whether or not it was a hit, so a system's ingestion charge does not depend on run order. Systems in one run share corpus embeddings (`evaluation.embedding_cache`), and live runs keep LLM responses and corpus embeddings on disk. The spend actually avoided is reported separately (`cache.saved_cost_usd`, `real_spend_usd` in `run_summary.json`). This keeps `$/Q` the cost of deploying that system alone. See [Caching](configuration.md#caching) for the rules.

## Latency measurement

Latency is the wall time of answering one question, without judging. Measured while many questions run at once it includes queueing behind other requests, so live runs finish with a **probe**: `evaluation.latency_probe_questions` (default 5) questions per system, spread across the dataset, are re-asked one at a time with the disk cache bypassed. The Latency column then shows the probe's mean, with `p95`; `latency_source` says `probe` or `concurrent`.

A p95 from five calls is a rough number, so every run gives the p95 an exact, distribution-free 95% interval over the same measurements (`latency_ms_p95_ci_lo` / `latency_ms_p95_ci_hi`, shown as `p95 [lo, hi]`). Its bounds are two of the measured latencies, chosen so that, whatever the shape of the latency distribution, the interval contains the true p95 at least 95% of the time (the number of measurements below the true p95 is binomial). A bootstrap would claim more than it delivers here: it cannot reach above the slowest call seen, and for a p95 from a few dozen calls it covers far less often than 95%. The honest consequence is that **a p95 has no upper bound until there are 72 measurements**: below that `latency_ms_p95_ci_hi` is empty and the leaderboard shows no interval, only the lower bound in the CSV. The default 5-call probe therefore gives a lower bound only; read its p95 as "at least this slow". With `latency_probe_questions: auto` the probe sizes itself: it starts at 72 calls, adds 24 at a time until the interval lies within ±15% of the p95, and stops at 120 (re-asking questions when a system answered fewer). That is what a trustworthy p95 costs, so `auto` is for runs where latency decides the choice; `ragbench estimate` counts the probe (at the cap for `auto`), and probe spend counts toward `evaluation.max_cost_usd`: the probe stops when the cap is reached. A cached LLM response replays the latency of the original call, and query embeddings are not cached by default, so cache hits do not make a system look faster. Mock runs skip the probe; their latencies are microseconds of CPU. Details: [configuration.md](configuration.md#latency-measurement).

## Statistics

A leaderboard of means invites over-reading: with 50 questions a 0.02 gap is noise. RAGBench therefore reports how uncertain each number is and only calls one system better than another when the data supports it. Everything is seeded (`evaluation.stats.seed`), so the same results produce the same intervals and p-values, and uses numpy only.

### Confidence intervals

The headline metrics (`Recall@primary_k`, `MRR` and `nDCG` at the deepest `k`, `answer_score`, `faithfulness`) carry a 95% **percentile-bootstrap** interval: the questions a system answered are resampled with replacement (`n_boot` times, default 2000) and the interval is the 2.5th to 97.5th percentile of the resampled means. They appear as `<metric>_ci_lo` / `<metric>_ci_hi` in `metrics_summary.csv` and as `mean [lo, hi]` in the leaderboards. A system that retrieves nothing has no retrieval interval.

For scale: if per-question answer scores spread with a standard deviation of about 1 point, the interval around a mean over 50 questions is roughly ±0.28 wide, and over 200 questions roughly ±0.14. Differences smaller than that are not evidence of anything.

The interval measures **which questions you happened to ask**. With one trial a system's run-to-run randomness is in it only as the spread it adds between questions, mixed with how hard each question is; [repeated trials](#repeated-trials) separate the two. It does not cover the chance that your questions are unrepresentative of real traffic. Overlapping intervals do not by themselves mean two systems are tied (the comparison below is sharper, because it uses the same questions for both), and non-overlapping intervals are not a significance test.

### Repeated trials

Agentic systems, and any model sampled above temperature 0, do not give the same answer twice, so a ranking from one pass can flip on a re-run while its intervals look tight. `evaluation.trials: N` answers (and judges) every question N times. Each trial is a fresh answer: the trial index is part of the disk-cache key of every model call made while answering and judging it (trial 0 keeps the key of a one-trial run, so earlier caches still hit), while embeddings, which do not vary, are shared.

- **Scores.** A question's score is the mean over its trials, and a system's score is the mean of those; a trial that failed is left out of that question's mean, as a failed question is left out of a one-trial mean.
- **Intervals and comparisons** use a **cluster bootstrap**: questions are resampled with all their trials, which is the bootstrap of the per-question means. Trial noise is in it as the extra spread it gives those means, so the interval is wider for a system whose answers vary than for a steady system with the same expected scores. Resampling the trials again inside each question would count that noise twice (the interval would then claim 95% and cover about 99%, and too many systems would look tied). The paired comparison does the same on the shared questions; wins, ties and losses compare per-question means. `significance.csv`, the recommendation's tie test and `ragbench compare-runs` all use it. With one trial everything is exactly the bootstrap above.
- **`answer_score_trial_sd`**: the within-question standard deviation of `answer_score` across trials, averaged over questions. 0 for a system that always says the same thing.
- **`pass_at_k`**: the share of questions answered correctly (`answer_score` at least 4) on at least one of the k trials. **`pass_hat_k`** (pass^k): the share answered correctly on all k. pass^k is the reliability number: a deployed system is run once per request, and pass^k is how often every one of k users asking the same question gets a right answer. A trial that errored counts as not correct for both.
- **`rank_stability`**: the share of trials on which the system holds the rank it has over all trials, each trial ranked by its mean answer score (ties share the better rank). 1 means every single pass would have shown the same order for this system.

`n_ok` and `n_error` count answers (questions × trials), and `evaluation.max_error_rate` is a share of answers. `per_question_results.jsonl` and the retrieval, answer and cost tables hold one row per answer with a `trial` column; the report's question explorer shows each question's first trial. `ragbench estimate` multiplies the per-question cost and time by N. These columns appear only when `trials` is above 1.

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
| `metrics_summary.csv` | adds `<metric>_ci_lo`, `<metric>_ci_hi`, `cost_per_correct`, `pareto_optimal`, `latency_ms_p95_ci_lo`, `latency_ms_p95_ci_hi`; with trials, `answer_score_trial_sd`, `pass_at_k`, `pass_hat_k`, `rank_stability` |
| `significance.csv` | one row per (metric, system) against the baseline: `n`, `mean_diff`, `ci_lo`, `ci_hi`, `p_value`, `p_holm`, `significant`, `verdict`, `wins`, `ties`, `losses` |
| `stats.json` | method, `n_boot`, `seed`, baseline (and where it came from), per-system question counts, means and intervals, `cost_per_correct`; `trials` when above 1 |
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

## Reading the report

`report.html` is a decision document, top to bottom (`ragbench report RUN` rebuilds it, and every other report, from the run directory without re-running anything). It is built only from the files in the run directory (`report_data.json` next to it holds the same numbers), makes no network request (the one exception is reading `report_questions.json` from next to it, when the question data was too big to embed and the page is served over http), and every chart has a "Table view" with the same values.

1. **Banner**: mock or live, the dataset (questions, categories, how many have labels), spend (charged at standalone prices, and what caches avoided), and every run-level warning: mock scores, unpriced models, self-preference, judge fallbacks, concurrent latency and unreviewed synthetic questions.
2. **Recommendation**: the system to deploy under the profile and constraints, the systems statistically tied with it ([Selection](#selection)), the reasons, and its `winner.yaml` with a copy button.
3. **Quality against cost** (or latency, when every cost is $0): one mark per system, bubble area = latency. Color marks the recommended system, its ties and the Pareto-optimal systems; the rest are gray, and only the first few are labeled (hover or focus a mark for any). The line is the cost-quality frontier.
4. **Leaderboard**: whiskers are the 95% intervals of [Statistics](#statistics); ▲ / ▼ / ≈ mark a system as better, worse or not clearly different from the baseline on that metric (Holm-adjusted). Shading is the best value in a column, and is left off a column where every system is equal. "Columns" adds more metrics; "—" is a missing value, never 0.
   With `evaluation.pairwise` on, **Pairwise arena** follows: each system's [arena score](#pairwise-arena) and interval, points won and win–tie–loss record, who beat whom, and the position-bias rate.
5. **Categories**: mean answer score (or recall) by question category, darker = higher on a scale that spans the values shown, so read it within one chart. A category with few questions is noisy: the question count is in the column header.
6. **Cost and latency**: cost by stage (answering only, no judge) and p50 / p95 latency. **Agents and tools** appears only when a system has steps or tools. **Failures** counts questions by failure type ([Failure types](#failure-types)). **Label audit** counts answers judged well supported although retrieval missed labeled documents.
7. **Questions**: every question in a filterable table (search, category, system, failure type, "score below", "systems disagree" = best and worst answer scores at least 2 points apart). Open a row (Enter, or click) to see the systems side by side: the reference answer, each answer, the judge's scores and reasoning, the retrieved contexts (green: labeled relevant; amber: not labeled but the answer was judged right and grounded, the [label audit](#failure-types)'s candidates; dashed: ranked but never shown to the generator) with page and heading where the chunker recorded them, and the trace with each step's latency, cost and input and output previews. Keys: ↑ ↓ move, Enter opens, Esc closes, `/` searches. How much of this the page holds is capped by `report.max_embedded_mb` ([configuration.md](configuration.md#report)); the page says when it is showing a reduced copy.
8. **Reproducibility**: versions, config and dataset hashes, models, the price table's date and the configuration that ran.

## Synthetic questions and pooled labels

`ragbench generate-questions` and `ragbench label` ([dataset-format.md](dataset-format.md#generating-questions)) produce evaluation data with a model, so what they produce inherits that model's habits.

### Pooling bias

Pooled labels judge only the documents some benchmarked system retrieved in its top k. A relevant document that no system found is never judged and so never proposed, which flatters the systems that were pooled and understates recall for any system that would have found it. Adding a different kind of system to the run (lexical next to dense, say) widens the pool. A deeper `--top-k` helps at a cost that grows with it. The LLM grader's grades are a first draft to review, not ground truth.

Synthetic questions are shaped by the generator: they tend to be answerable from one passage, worded like the document unless paraphrased, and easier than real traffic. Scores on questions still flagged `metadata.needs_review` measure that taste as much as the systems; read the questions and clear the flag before relying on the numbers.

## Limitations

- **Document-level relevance.** A retrieved chunk counts if its document is relevant, whether or not that chunk holds the evidence. Context precision is therefore generous, and unlabeled but helpful documents count against a system (`qrels_audit.md` surfaces them).
- **LLM-judge noise and bias.** See [Bias and noise](#bias-and-noise). Scores are only comparable within one run (one judge or panel, one prompt version), and a judge's accuracy is unknown until a [human audit](#calibrating-the-judge-against-a-human) measures it.
- **Heuristic refusal detection.** Abstention metrics rely on recognizing refusals by wording. Models that decline in unusual phrasing are counted as answering.
- **Deterministic metrics are literal.** `exact_match` and `token_f1` penalize correct paraphrases and reward verbose answers that repeat the reference.
- **Failure classification is heuristic** and depends on labels you supply (`answer_type`, qrels).
- **Mock mode** uses hashing embeddings, a scripted LLM and the heuristic judge. It validates that pipelines run and that results are reproducible; it says nothing about real quality, and its banner is on every report.
- **Small datasets.** With tens of questions, per-category numbers rest on a handful of examples, and most differences between systems will be `no clear difference`. Add questions before trusting a ranking.
- **A recommendation is only as good as the benchmark.** It rests on your questions, your judge and the constraints you set; a different dataset can crown a different system. Treat it as the answer to "what did this evidence support", and check the ties.
- **Intervals cover question sampling only.** They do not include judge noise or a system's own randomness, and they assume your questions resemble the questions you care about.
