# Decision gate evaluation

`decision_gate` retrieves like `hybrid_rerank`, then asks a typed decision model whether each chunk helps and whether what is left is enough, and refuses to generate when it is not ([how it works](system-wiki.md#decision_gate), [options](systems.md#decision_gate)). This page measures it: on the bundled demo dataset, with real API calls, against the systems it is meant to be compared with, and it says when the result is worth the extra requests. The metrics are defined in [methodology.md](methodology.md); the files behind every number are in [`docs/decision-gate-run/`](decision-gate-run/), and [`configs/decision_gate.yaml`](../configs/decision_gate.yaml) reruns it. The [leaderboard of every system](#the-leaderboard-of-every-system) with the gate in it is further down.

## The run

| | |
| --- | --- |
| Run id | `decision_gate_demo_20261007_181133` |
| Mode | **live**: real OpenAI calls, including OpenAI's Decisions API (`POST /v1/decisions`, public beta since 2026-10-06) |
| Models | `gpt-6-luna` generates, judges and makes the decisions; `text-embedding-3-small` embeds |
| Dataset | the [demo dataset](../data/demo): 60 documents, 163 questions, of which 138 are answerable with relevance labels and 25 are unanswerable |
| Systems | `hybrid_rerank` (same ranking, no gate), `corrective` (nearest design), `no_retrieval` (the floor), and six `decision_gate` variants, each differing from the default in one way |
| Spend | $1.15 charged at standalone prices, about $0.30 real spend after cache reuse; about $0.9 in total for the whole investigation (probes, a smoke test, and earlier runs of the config as it evolved) |
| Versions | ragbench 0.3.0 plus this change, Python 3.12.12, prices as of 2026-10-01 |

Retrieval depth 10, generator context 5 chunks. Every `decision_gate` variant uses the chunker and retrieval options of `hybrid_rerank_default`, so its ranking, and therefore its Recall@5 (0.773), is the same by construction; what changes is what the generator is given and whether it is called at all. The `hybrid_rerank` and `corrective` rows match the [October 1 example run](example-run.md) (answer scores 4.72 and 4.77, the same refusal counts), so that run is the right place to read the other systems; the closest of them are quoted below.

## Results

Brackets are 95% bootstrap confidence intervals over the 163 questions. In the abstain recall column the denominator is the 25 unanswerable questions, in the false refusals column the 138 answerable ones. Latency is the mean of the live latency probe (five questions per system, asked one at a time, cache bypassed): it varied by roughly 25% between three runs of the same `hybrid_rerank` and the default gate ranged from 1.9 s to 3.2 s across two runs, so compare latencies only when they differ by a factor.

| System | Answer score | Ctx recall | Ctx precision | Abstain recall | False refusals | Abstain precision | $/Q | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_rerank (reference) | 4.72 [4.59, 4.84] | 0.89 | 0.32 | 0.80 (20/25) | 0.058 (8/138) | 0.71 | $0.00053 | 2.0 s |
| corrective (reference) | 4.77 [4.67, 4.86] | 0.93 | 0.34 | 0.80 (20/25) | 0.029 (4/138) | 0.83 | $0.00060 | 5.2 s |
| **`decision_gate` (default)** | **4.80** [4.70, 4.89] | 0.93 | 0.69 | 0.92 (23/25) | 0.043 (6/138) | 0.79 | $0.00094 | 1.9 s |
| `decision_gate`, strict wording | 4.65 [4.49, 4.80] | 0.89 | 0.79 | 1.00 (25/25) | 0.087 (12/138) | 0.68 | $0.00091 | 1.6 s |
| `decision_gate`, one request per chunk | 4.45 [4.27, 4.60] | 0.82 | 0.52 | 0.92 (23/25) | 0.116 (16/138) | 0.59 | $0.00101 | 2.4 s |
| `decision_gate`, prompted chat model (`llm_json`) | 4.71 [4.56, 4.84] | 0.89 | 0.80 | 0.96 (24/25) | 0.072 (10/138) | 0.71 | $0.00077 | 4.1 s |
| `decision_gate`, no sufficiency check | 4.82 [4.72, 4.90] | 0.93 | 0.69 | 0.92 (23/25) | 0.043 (6/138) | 0.79 | $0.00086 | 2.3 s |
| `decision_gate`, no chunk filter | 4.69 [4.55, 4.81] | 0.89 | 0.32 | 0.80 (20/25) | 0.058 (8/138) | 0.71 | $0.00130 | 2.3 s |
| `no_retrieval` (floor) | 1.54 [1.40, 1.68] | | | 1.00 (25/25) | 0.935 (129/138) | 0.16 | $0.00012 | 0.9 s |

$/Q includes the judge's call, as everywhere else in the reports; `decision_gate`'s decision requests cost $0.00066 of its $0.00094, about 6,600 input tokens per question.

## What the numbers say

**1. At the default settings the gate ties with the best neighbours on quality and gives the generator half the context.** Its answer score is +0.08 above `hybrid_rerank` (paired 95% interval [+0.01, +0.17], just clear of zero, and unadjusted for the several comparisons made here) and +0.04 above `corrective` ([−0.06, +0.13], a tie). The generator receives 2.3 chunks on average instead of 5 and context precision rises from 0.32 to 0.69, while context recall does not fall (0.93, against 0.89 for `hybrid_rerank` and 0.93 for `corrective`): kept chunks are drawn from the top ten, so a deeper chunk can take the place of a dropped one. The generator's own spend falls from $0.00022 to $0.00010 per question, which offsets about a fifth of what the decision requests cost.

**2. It declines more of the unanswerable questions, but the sample cannot say whether that is real.** It refused 23 of 25 against 20 of 25 for `hybrid_rerank` and `corrective`: three more questions, none the other way round (exact sign test p = 0.25). Its false refusals are 6 against 8 for `hybrid_rerank` and 4 for `corrective`; neither difference is significant. The honest reading is that the direction is right and 25 unanswerable questions are too few to be sure. Answers classed as hallucinated fell from five to three. Most of the refusals are the generator's own on a smaller, cleaner context: the gate itself abstained on 16 questions, the generator refused 13 more.

**3. Wording is a dial between refusing too much and too little, and the default is the plain one.** The strict wording ("does the passage state a fact that answers the question…") refuses every unanswerable question (25 of 25) but also 12 answerable ones, and scores 0.15 lower than the default (paired interval [+0.05, +0.27] in the default's favor). It is stricter about what counts as support, so it drops chunks that carry only part of an answer: chunk keep decisions have precision 0.72 and recall 0.78 against 0.59 and 0.84 for the plain wording, and multi-hop questions fall from 4.40 to 3.43. Use the strict pair when a confident wrong answer costs far more than a refusal; its config is in `configs/decision_gate.yaml`. The default was chosen after seeing this run, between these two candidates, so treat the demo numbers of the default as slightly optimistic: the real test is your own data.

**4. The probabilities are mostly a confident yes or no, so the threshold only decides the doubtful minority.** Only 9% of the plain wording's answerability probabilities (6% for the strict one) fall between 0.1 and 0.9. Even so, sweeping the answerability threshold from 0.1 to 0.9 over the 1,380 chunks judged for the labelled questions moves the plain wording from 437 chunks kept (keep precision 0.51, recall 0.88) to 304 (0.66 and 0.79), a spread about as wide as the one between the two wordings; I did not run thresholds end to end, only this keep sweep, which the saved probabilities make free. The sufficiency threshold is the more visible dial for the plain wording: counting only the gate's own abstains, raising it from 0.5 to 0.9 lifts abstain recall from 0.60 to 0.80 for false refusals from 0.01 to 0.04 (the generator refuses more on top). Nothing here says "0.7" means "right 70% of the time"; nobody has published calibration data for these models, so sweep both dials on your own questions.

**5. Judge all candidates in one request.** One request per chunk (isolated judgments, the same wording) scored 0.36 below the batched default (interval [+0.21, +0.52]): 16 false refusals against 6, context recall 0.82 against 0.93, and a less precise context (0.52 against 0.69). A likely reason is that a passage judged alone cannot see that the passage next to it completes the answer, which is what multi-passage questions need. The two cost about the same (the passages dominate the tokens) and the batched answerability request averaged 0.34 s against 0.62 s for the ten parallel ones.

**6. The typed endpoint beat a prompted chat model on the same questions, and was twice as fast.** `llm_json` puts the same predicates to the same `gpt-6-luna` and asks for JSON. The Decisions API scored 0.10 higher ([+0.01, +0.20]) with fewer false refusals (6 against 10), and its requests averaged 0.3 s against 1.9 s, which is most of the difference between 1.9 s and 4.1 s per answer. The chat-model route filtered chunks harder (context precision 0.80) and was a little cheaper per question ($0.00054 against $0.00066 in decisions). This compares one prompt for each mechanism, not prompting in general.

**7. The sufficiency check changed nothing here.** Switching it off gives the same refusals and an answer score of 4.82 against 4.80 (paired difference [−0.04, 0.00]). With the plain wording it fired on only four questions, and the generator refused all four anyway when the check was off, so it saved four generations and changed no outcome. It is on by default because it is half of the pattern, and a `sufficiency_threshold` of 0 removes a request per question. The mirror ablation shows the chunk filter carries the effect: with sufficiency alone, on the unfiltered top five chunks, the system has the same counts as `hybrid_rerank` (20 of 25 unanswerable refused, 8 false refusals).

**8. Where it loses.** Per category, the default matches `hybrid_rerank` almost everywhere and trails `corrective` on aggregation (3.80 against 4.31) and multi-hop questions (4.40 against 4.58): its false refusals are four aggregation questions, one multi-hop and one paraphrase, the questions whose evidence is spread over several documents. It cannot do what `corrective` does for a bad first search, because it only removes or refuses (Recall@5 stays at 0.773, against 0.793 for `corrective`).

## The leaderboard of every system

One joint run of every system in [`configs/all.yaml`](../configs/all.yaml) plus the default and strict gate ([`configs/all_plus_decision_gate.yaml`](../configs/all_plus_decision_gate.yaml)), run id `all_systems_plus_decision_gate_20261007_184150`, same dataset, same models, no failed questions. The disk cache served every call, so the other systems' rows reproduce the [October 1 example run](example-run.md) exactly (the repeat cost $0.07); the full generated leaderboard and metrics are in [`docs/decision-gate-run/`](decision-gate-run/all-systems-leaderboard.md). Sorted by answer score; ✓ in the last column means no other system is at least as good on answer score, cost and latency and better on one of them. The latency probe asks five questions per system and is noisy: the default gate measured 1.9 s in the table above and 3.2 s here, `hybrid_rerank` between 1.6 s and 2.0 s, so compare latencies only when they differ by a factor.

| # | System | Answer score | Recall@5 | Ctx precision | Abstain recall | False refusals | $/Q | Latency | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | iterative_default | 4.81 [4.71, 4.90] | 0.799 | 0.36 | 0.84 | 0.014 | $0.00057 | 4.9 s | ✓ |
| 2 | **decision_gate_default** | 4.80 [4.70, 4.89] | 0.774 | 0.69 | 0.92 | 0.043 | $0.00094 | 3.2 s | ✓ |
| 3 | grep_agent_default | 4.78 [4.69, 4.86] | 0.791 | 0.68 | 0.68 | 0.000 | $0.00109 | 6.3 s |  |
| 4 | full_context_ceiling | 4.78 [4.70, 4.84] | 0.741 | 0.02 | 0.96 | 0.000 | $0.00280 | 1.6 s | ✓ |
| 5 | corrective_default | 4.77 [4.67, 4.86] | 0.793 | 0.34 | 0.80 | 0.029 | $0.00060 | 5.4 s |  |
| 6 | rag_fusion_default | 4.75 [4.64, 4.86] | 0.808 | 0.36 | 0.84 | 0.043 | $0.00042 | 3.5 s | ✓ |
| 7 | agent_search_tools | 4.75 [4.63, 4.85] | 0.821 | 0.38 | 0.84 | 0.014 | $0.00082 | 3.6 s |  |
| 8 | rerank_default | 4.74 [4.63, 4.84] | 0.789 | 0.33 | 0.80 | 0.051 | $0.00052 | 1.7 s | ✓ |
| 9 | hybrid_rerank_default | 4.72 [4.59, 4.84] | 0.774 | 0.32 | 0.80 | 0.058 | $0.00053 | 2.0 s |  |
| 10 | decompose_default | 4.72 [4.60, 4.83] | 0.771 | 0.34 | 0.84 | 0.043 | $0.00041 | 2.5 s | ✓ |
| 11 | llm_heavy_default | 4.72 [4.60, 4.82] | 0.790 | 0.34 | 1.00 | 0.058 | $0.00512 | 5.3 s |  |
| 12 | parent_doc_default | 4.71 [4.60, 4.81] | 0.789 | 0.37 | 0.84 | 0.065 | $0.00049 | 2.7 s |  |
| 13 | adaptive_default | 4.70 [4.57, 4.82] | 0.771 | 0.33 | 0.88 | 0.065 | $0.00044 | 2.6 s |  |
| 14 | contextual_default | 4.69 [4.57, 4.81] | 0.784 | 0.33 | 0.84 | 0.065 | $0.00046 | 1.3 s | ✓ |
| 15 | bm25_default | 4.69 [4.55, 4.81] | 0.745 | 0.31 | 0.88 | 0.072 | $0.00049 | 1.2 s | ✓ |
| 16 | hyde_default | 4.68 [4.54, 4.80] | 0.787 | 0.33 | 0.88 | 0.058 | $0.00046 | 2.8 s |  |
| 17 | vector_default | 4.67 [4.54, 4.79] | 0.790 | 0.33 | 0.88 | 0.051 | $0.00044 | 1.7 s | ✓ |
| 18 | **decision_gate_strict** | 4.65 [4.49, 4.80] | 0.774 | 0.79 | 1.00 | 0.087 | $0.00091 | 1.7 s |  |
| 19 | hybrid_default | 4.61 [4.46, 4.74] | 0.775 | 0.33 | 0.88 | 0.080 | $0.00048 | 1.7 s |  |
| 20 | agent_search_plain | 4.58 [4.43, 4.72] | 0.814 | 0.37 | 0.84 | 0.058 | $0.00072 | 3.5 s |  |
| 21 | hierarchical_default | 4.56 [4.40, 4.70] | 0.733 | 0.28 | 0.84 | 0.101 | $0.00033 | 1.3 s | ✓ |
| 22 | vector_mmr | 4.54 [4.40, 4.68] | 0.697 | 0.24 | 0.88 | 0.101 | $0.00040 | 1.3 s |  |
| 23 | sentence_window_default | 4.42 [4.23, 4.59] | 0.780 | 0.36 | 0.88 | 0.109 | $0.00025 | 1.3 s | ✓ |
| 24 | no_retrieval_floor | 1.54 [1.40, 1.68] | — | — | 1.00 | 0.935 | $0.00012 | 1.1 s | ✓ |

**How to read it.** The top of the table is a statistical tie: the run's own recommendation lists most of the table as tied with `iterative_default` on answer score, the default gate among them, and under the balanced profile it picks `bm25_default` as the cheapest and fastest of them. On quality alone the gate is not distinguishable from the best, and it does not beat the plain systems by more than the noise (its paired +0.08 over `hybrid_rerank` is discussed above). What no other system offers together is its mix of a clean context (0.69 precision; of the other systems only `grep_agent` at 0.68 comes close, and `full_context` sits at 0.02 by design) with explicit abstention. Among the systems that decline at least 0.92 of the unanswerable questions, the default gate is the cheapest by a wide margin ($0.00094, against $0.0028 for `full_context` and $0.0051 for `llm_heavy`). `iterative` and `agent_search` with tools answer as well, cost less than the gate and refuse only two answerable questions in 138, but decline fewer of the unanswerable ones (0.84); `grep_agent` and `full_context` refuse no answerable question at all, at higher cost. The gate costs more per question than every system except `grep_agent`, `full_context` and `llm_heavy`.

## When to try it

**Try `decision_gate` when** the documents often cannot answer the question and you want the decision to decline to be an explicit rule on a probability, inspectable per chunk in the `gate` block of `per_question_results.jsonl`, rather than left to the generator; when the generator's context should be small and clean (long chunks, expensive or easily distracted generators, or citations that should point at the evidence only); or when you want a refusal that never costs a generation. Start from the defaults, `judge_mode: batched` and the Decisions API.

**Prefer `hybrid_rerank`** when cost matters most and a wrong answer is cheap. The gate costs 1.8 times as much per question here ($0.00094 against $0.00053) for a small gain (+0.08). The gate is `hybrid_rerank` plus a filter, so the comparison is always one config away.

**Prefer `corrective`** when the first search often misses. The gate cannot recover a chunk retrieval did not return; `corrective` rewrites and widens the search (Recall@5 0.793 against 0.773), refused fewer answerable questions (4 against 6), cost about 64% of the gate per question, and was slower (5.2 to 5.4 s against 1.9 to 3.2 s for the gate in two runs). The two are not combined today.

**Tune before trusting.** Compare the default with the strict wording on your own data, and read `abstain recall` and `false refusals` together: the strict wording gets 25 of 25 and 12 false refusals, the default 23 of 25 and 6. Keep `judge_mode: batched`, try the typed endpoint against `llm_json` if you cannot reach the Decisions API, and treat the sufficiency check as optional until it earns its request on your questions.

## What this evidence cannot say

- **One dataset, few unanswerable questions.** 25 unanswerable questions cannot separate 23 from 20 refusals, and the sufficiency check fired too rarely to judge. The demo corpus is a fictional insurance company; yours may behave differently.
- **One model family.** `gpt-6-luna` generates, judges and decides, so the judge may favor its own answers (the leaderboard says so). The Decisions API serves no other model today.
- **Document-level labels.** A chunk counts as relevant when its document does, so the keep precision and recall above are approximate.
- **A beta endpoint.** The Decisions API entered public beta on 2026-10-06; its price, latency and behavior may change, and no accuracy or calibration data has been published to check these results against. Jev (TypeSafe AI) is not benchmarked: it has a reserved backend but no client, because its API is not documented well enough to build one.
- **Latency is coarse.** The probe asks five questions per system; use the step latencies in `per_question_results.jsonl` for the stage view (a decision request averaged 0.2 to 0.3 s, a chat-model one 1.3 to 1.9 s).
- **Comparisons are paired but not multiplicity-adjusted.** The run's own recommendation applies a Holm correction across systems; the pairwise intervals quoted here do not, and the +0.08 over `hybrid_rerank` would likely not survive a correction for the number of comparisons made.

## Reproduce it

```bash
ragbench estimate --config configs/decision_gate.yaml      # about $1.4 with a cold cache; nothing is spent
ragbench run --config configs/decision_gate.yaml --mock    # offline: the mock decision model, same code paths
ragbench run --config configs/decision_gate.yaml           # live: needs OPENAI_API_KEY; the disk cache makes a repeat nearly free
ragbench run --config configs/all_plus_decision_gate.yaml  # the joint leaderboard of every system (about $3.5 with a cold cache)
```

The probabilities of every judged chunk are saved per question, so thresholds and the sufficiency check can be studied afterwards from `per_question_results.jsonl` without calling the model again; changing the wording or the answerability threshold changes which chunks reach the generator, so those need a new run.
