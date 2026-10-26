# Optimizing a system's harness

`ragbench optimize` improves one system's **harness**, the prompt texts and bounded knobs of its [`harness:` block](configuration.md#harness), without changing its model. It is a port of Harness²'s Propose → Probe → Compose loop with one difference that matters: RAGBench has a dataset, so it tunes on the **train** split and then has to **prove** the gain on a held-out **test** split before it keeps anything.

## Quick start

The dataset needs a split once (`ragbench split` writes `splits.json` next to the questions and never overwrites it):

```bash
ragbench split data/hard/questions.jsonl --train 0.5 --dev 0.1
```

Try the whole loop offline first. `--mock` uses the mock models and a scripted proposer, costs nothing, and writes the same files as a live run:

```bash
ragbench optimize --config configs/hard.yaml --system hybrid_rerank --k 2 --mock
```

Before a live run, see what it would cost:

```bash
ragbench estimate --config configs/hard.yaml --optimize hybrid_rerank --k 2 --proposer openrouter:deepseek/deepseek-v4-pro-0813
```

A live run needs a spending cap, and shows the estimate and asks before it starts (`--yes` skips the question):

```bash
ragbench optimize --config configs/hard.yaml --system hybrid_rerank --k 2 --proposer openrouter:deepseek/deepseek-v4-pro-0813 --max-cost-usd 2
```

Every option is in the [command reference](cli.md#ragbench-optimize).

## The loop

A **probe** is an ordinary RAGBench run of one harness on the train questions, judged as usual. The **proposer** is a chat model that reads evidence about how the system behaved and answers with one change to the harness. Which models to use for it is in [models.md](models.md).

**Parallel** (`--mode parallel`, the default: width):

1. The base harness is run `k` times on train (one run with `k` trials, so each rollout is a fresh sample).
2. From each rollout's evidence the proposer writes one **INV** candidate (a general change that would help on any document collection) and one **DOM** candidate (a change made for this collection): `2k` candidates.
3. Each candidate is probed on train.
4. The proposer reads evidence that contrasts every candidate with the base and writes `k` **compositions**, each merging the candidates' changes: keep what helped, drop what hurt.
5. Each composition is probed on train.

**Sequential** (`--mode sequential`: depth) runs `k` rounds of the same steps with one rollout, one INV, one DOM and one composition each. Every round starts from the best harness on train so far.

Both modes make exactly **`4k` train probes and `3k` proposer requests**, and `ledger.json` asserts it. A proposal that cannot be used gets one retry with the reason; a second failure keeps the parent's harness. A candidate whose harness was already probed (an inherited proposal, two proposals that agree) is counted and read back for free.

Then the result is decided on data the method never read:

- **Dev.** The candidates that beat the base on train (the best `--finalists`, default 2) and the base are run on dev. The best on dev is the pick, if it also beats the base there. On a label-free dataset `--select auto` compares them by pairwise judging instead of answer scores. Without dev questions the pick is the best on train.
- **Test.** The pick and the base are answered on test and compared question by question. The pick is adopted only when it is significantly better. Otherwise the verdict is **no reliable gain** and the base is kept, even after a gain on train and dev.

The decision rule and its statistics are in [methodology.md](methodology.md#verifying-an-optimized-harness).

## What the proposer sees

The proposer reads an **evidence tree** built from train runs only: the system's traces and answers, what went wrong, the other harnesses, and for compositions the contrasts between each candidate and the base. `--feedback` sets how much of the judge it sees:

- `contrast` (default): behaviour only. No grades, reference answers or winners: the proposer judges for itself which behaviour to keep.
- `labels`: the judge's grades and the reference answers of the train questions.
- `none`: the rollouts alone.

Dev and test questions are held out from every file and every request by the leakage guard, which stops the run rather than let one through. What is in the tree, and the guarantees, are in [methodology.md](methodology.md#what-a-harness-proposer-reads). Each tree is kept under `evidence/` in the output folder, so you can read exactly what the proposer read.

## Outputs

The output folder (`--out`, default `results/optimize_<system>_<time>`) holds:

- `optimization.md`: the verdict with the test comparison, every candidate (kind, parent, the fields it changed, train and dev results), each change as a diff with the proposer's rationale, candidate diversity, and the ledger's totals.
- `optimized.yaml`: your config with the adopted harness in the system's `harness:` block. It is unchanged when nothing was adopted. Run it as usual, or show the harness with [`ragbench harness show`](cli.md#ragbench-harness-show).
- `ledger.json`: every step (base rollouts, proposals, probes, compositions, dev selection, test verification) with its cost, and the budget identity.
- `probes/`, `evidence/`, `pools/`, `proposals/`: every probe's run folder (a normal run you can open with `ragbench report`), the evidence trees, and the saved proposals.

**Candidate diversity** is 1 minus the mean pairwise line-Jaccard similarity of the distinct candidates' harnesses. Near 0, the proposals collapsed onto one idea (Harness²'s collapse check), and a larger `k` or `--feedback labels` may help.

## Cost, caps and resuming

`--max-cost-usd` (or `evaluation.max_cost_usd`) caps the whole optimization. Each probe runs with what is left of the cap, and a probe that reaches it stops the loop: the verdict is "stopped" and the base is kept. Costs are counted at standalone prices, as in every run. The [disk cache](configuration.md#caching) makes repeated unchanged calls free in a live run (ingestion of an unchanged corpus, a re-probed harness), so the real spend in `ledger.json` is usually below the charged total.

Run the same command into the same `--out` folder to **resume**. Saved proposals are reused, so the proposer is not asked again. A probe whose stamp (the harness fingerprints, split and trials) matches a finished run is read back, and an unfinished one continues from its checkpoints. A different `--system`, `--k`, `--mode`, `--feedback`, `--finalists`, `--select`, `--test-trials` or config is refused, so one folder never mixes two optimizations. You can raise the cap between attempts.

## Limits

- Each probe indexes the corpus again. In a live run the cache serves unchanged indexing for free, but a `chunk_size` change re-embeds.
- The test comparison has the power of your test split. With a few dozen questions only a large gain can be shown, so "no reliable gain" can also mean "not enough questions".
- The proposer sees evidence, not code: it can only change the fields the system's harness exposes ([configuration.md](configuration.md#harness)).
