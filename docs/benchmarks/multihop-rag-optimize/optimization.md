# Optimization of `hybrid_rerank`

parallel mode, k = 2, feedback `contrast`, 2 finalist(s), selection `auto`.

## Verdict

**no reliable gain: the base harness is kept.**

On test (149 questions, 1 trial(s)): `p0-dom` 3.645 against the base's 3.682, difference -0.037 (95% CI -0.159 to +0.082), p = 0.5567, Holm-adjusted over 1 comparison(s) p = 0.5567; wins / ties / losses 53 / 32 / 64.

`optimized.yaml` holds `hybrid_rerank` unchanged.
- The splits have no dev questions, so `p0-dom` was picked on train (the best there); the test split still decides.

## Candidates

| candidate | kind | from | changes | train | dev | note |
| --- | --- | --- | --- | --- | --- | --- |
| `base` | base | - | - | 3.688 | - |  |
| `p0-inv` | INV | base | `top_k` | 3.660 | - |  |
| `p0-dom` | DOM | base | `grounding` | 3.746 | - | pick |
| `p1-inv` | INV | base | `top_k`, `chunk_size` | 3.562 | - |  |
| `p1-dom` | DOM | base | `few_shot` | 3.701 | - |  |
| `c0` | COMPOSE | base | `grounding`, `top_k`, `chunk_size` | 3.632 | - |  |
| `c1` | COMPOSE | base | `grounding`, `top_k` | 3.668 | - |  |

Train and dev columns are mean answer scores (0 to 5; dev is arena points when picked by arena). The base's train value averages its rollouts.

## Changes

### `p0-inv` (INV, from `base`)

> Many refusals occur because the generator sees only a few short chunks, and the question asks about multiple named sources or conditions. Increasing the number of chunks the generator reads gives it more opportunities to find all the pieces needed before concluding the answer is unavailable.

```diff
- top_k: 5
+ top_k: 10
```

### `p0-dom` (DOM, from `base`)

> Many questions in this collection are compound, asking for an entity that satisfies multiple conditions from different articles, but the current prompt only says to answer from context without explicitly telling the model to synthesize across multiple retrieved passages. Adding a collection-specific instruction to combine evidence from multiple sources and to identify when a required source is missing will reduce false refusals and improve multi-hop answers.

```diff
- grounding: 'Use only the provided context.'
+ grounding: 'Use only the provided context. When a question asks about multiple articles or sources, check whether each named source is present; if one is missing, say so explicitly. If the answer requires combining information from different passages, synthesize across them and cite each source used.'
```

### `p1-inv` (INV, from `base`)

> Many failures show the system refusing because the retrieved chunks are only tangentially related to the full question, even when the answer may exist in the collection. Increasing top_k and chunk_size gives the generator more context, making it less likely to miss the relevant passage and refuse prematurely.

```diff
- top_k: 5
+ top_k: 8
- chunk_size: 500
+ chunk_size: 800
```

### `p1-dom` (DOM, from `base`)

> Many questions ask for a single letter or initial derived from names in the documents, but the retrieval and answer often fail because the system treats them as general QA. Adding a few-shot example from a rollout that demonstrates extracting a letter/initial from the provided context teaches the model to focus on this collection's letter-answer pattern and reduces refusals.

```diff
- few_shot: None
+ few_shot: [{'question': 'Considering the analysis from an article by the BBC and the strategic forecast presented by CNN, which country, identified as a rising regional power in the BBC article and predicted to increase its defense spending according to CNN, has a single-letter stock market index symbol?', 'answer': 'I could not find the answer in the provided documents.'}]
```

### `c0` (COMPOSE, from `base`)

> The base system often refuses because it only sees the top 5 chunks, missing pieces needed to answer multi-source and temporal questions. Increasing top_k to 10 and chunk_size to 800 widens the context so more relevant evidence is available, while adding the p0-dom grounding rule makes the model explicitly check for missing named sources and synthesize across passages, reducing premature refusals.

```diff
- grounding: 'Use only the provided context.'
+ grounding: 'Use only the provided context. When a question asks about multiple articles or sources, check whether each named source is present; if one is missing, say so explicitly. If the answer requires combining information from different passages, synthesize across them and cite each source used.'
- top_k: 5
+ top_k: 10
- chunk_size: 500
+ chunk_size: 800
```

### `c1` (COMPOSE, from `base`)

> The base system refuses many questions because it retrieves too few chunks and the generator does not check whether each named source is actually present. The p0-dom grounding explicitly instructs the model to verify named sources and synthesize across passages, which addresses the recurring refusals where one required source is missing. Increasing top_k to 10 (p0-inv) gives the generator more material to find those sources or confirm their absence.

```diff
- grounding: 'Use only the provided context.'
+ grounding: 'Use only the provided context. When a question asks about multiple articles or sources, check whether each named source is present; if one is missing, say so explicitly. If the answer requires combining information from different passages, synthesize across them and cite each source used.'
- top_k: 5
+ top_k: 10
```

## Diversity

Candidate diversity 0.39 (1 minus the mean pairwise line-Jaccard similarity of the candidates' harnesses).

## Ledger

8 of 8 train probes and 6 of 6 proposer stages; charged $2.2308 of a $2.3500 cap, real spend $1.8685. Every step is in `ledger.json`.
