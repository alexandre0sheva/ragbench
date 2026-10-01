# Dataset Format

RAGBench evaluates retrieval at the document level. Do not add static `relevant_chunk_ids`; chunks are generated dynamically by each RAG system and may differ by chunker or strategy.

Contents: [Quick start](#quick-start-with-your-own-data) · [Documents](#documents) · [Inspecting a dataset](#inspecting-a-dataset) · [Questions](#questions) · [Optional qrels](#optional-qrels) · [Label-free mode](#label-free-mode) · [Importing existing data](#importing-existing-data) · [Generating questions](#generating-questions) · [Pooled labeling](#pooled-labeling) · [The demo dataset](#the-bundled-demo-dataset)

## Quick start with your own data

```bash
ragbench init my_ds --docs path/to/documents      # my_ds/ragbench.yaml + my_ds/questions.jsonl (placeholders)
ragbench inspect-dataset --docs path/to/documents --questions my_ds/questions.jsonl
ragbench run --config my_ds/ragbench.yaml --mock  # free pipeline check, then run without --mock
```

`ragbench init [DIR] --docs PATH [--questions FILE] [--preset quick|standard|thorough|agentic] [--force]` loads your documents (a wrong path or an unreadable file fails here, before anything is written) and writes:

- `DIR/ragbench.yaml`: the dataset paths, the systems of the preset (default `standard`) written out so you can edit them, an `evaluation:` section with the judge on and `max_cost_usd: 5.0` as a safety net for a first live run (remove it to run uncapped), and `selection: {profile: balanced}`. It refuses to replace an existing config without `--force`.
- `DIR/questions.jsonl`, only when you gave no `--questions` and the file does not exist: up to three **placeholder** questions built from your document titles (`answerable: true`, `metadata.template: true`, no labels), so the project runs before the real questions exist. Replace them. Questions files are never overwritten.

Paths in the config are relative to the directory you run `ragbench` from, like in every other config. To skip the steps in between, `ragbench auto --docs PATH` does the whole flow in one run directory: it [writes questions](#generating-questions) when you have none, runs a preset and ends with a recommendation (see the [README](../README.md#quickstart)). Next, write real questions ([Questions](#questions); labels are optional, see [label-free mode](#label-free-mode)), or convert ones you already have ([importing](#importing-existing-data)).

## Documents

Put your files in a directory and point `dataset.documents_path` at it (subfolders are searched). Supported formats:

| Extension | Becomes | Needs |
| --- | --- | --- |
| `.txt`, `.md`, `.markdown`, `.rst` | one document; a Markdown file's title is its first `# Heading` | nothing |
| `.html`, `.htm` | one document of the visible text: `<title>` (else the first `<h1>`) is the title; scripts, styles, comments and `<noscript>` are dropped; entities are decoded; paragraphs, list items and table rows keep their line breaks | nothing (standard library parser) |
| `.pdf` | one document; `metadata["page_spans"]` records the character range of every page, and each chunk gets `page` (and `page_end` when it crosses pages) in its metadata | `pip install 'ragbench[pdf]'` |
| `.docx` | one document; Word headings become `#` Markdown headings (so the `markdown` chunker can use them), tables become ` | `-separated lines | `pip install 'ragbench[docx]'` |
| `.csv`, `.tsv`, `.json` (an array of records), `.jsonl` | **one document per row / record**, written as `column: value` lines so the column names travel with the text | nothing |
| a single `.json` object | one document of the pretty-printed JSON | nothing |

```text
docs/
  doc_001.md
  handbook.docx
  contract.pdf
  faq.csv
  .ragbenchignore
```

**Document ids** are stable. A file named like `doc_001.md` uses its stem as the id; other files get a deterministic hash-based id. The documents of a table file are named `<file id>#<row>` (`doc_3fa9c1d2e0#7`), where the row counts data rows from 1 and is also stored as `metadata["row"]`. Qrels and `relevant_doc_ids` refer to these ids.

**Tables and records.** By default every column goes into the text. Pick columns and an id in the config:

```yaml
dataset:
  documents_path: data/faq
  tabular:
    text_columns: [question, answer]   # the document text (default: all columns)
    id_column: id                      # documents are named <file id>#<value of id> instead of #<row>
```

**Choosing files.** `dataset.include` and `dataset.exclude` take globs matched against each file's path relative to `documents_path` (`*` within a folder, `**` across folders; a pattern without `/` matches the file name anywhere). A `.ragbenchignore` file in the documents folder adds gitignore-style rules: `#` comments, blank lines, `dir/`, a leading `/` to anchor at the folder, `*`, `**`, `?`, and `!` to re-include (a file cannot be re-included once its folder is excluded, as in git):

```text
# .ragbenchignore
drafts/
*.tmp.md
!keep.tmp.md
```

**Encodings.** Files are read as UTF-8 (a UTF-8/16/32 byte-order mark is honoured). A file that is not valid UTF-8 is decoded as Windows-1252, which covers legacy Western-European text, and a warning says so; bytes that fit no encoding are replaced with `U+FFFD` with a warning. Re-save such files as UTF-8 when characters look wrong.

**Errors.** `dataset.on_error: raise` (default) stops at the first file that cannot be loaded (corrupt PDF, invalid JSON, a file above `dataset.max_file_mb`, default 25) and names it. `on_error: skip` skips such files and records a warning instead. The warnings (skips and encoding fallbacks) are listed by `ragbench inspect-dataset` and stored as `document_warnings` in each run's `run_summary.json`. A missing optional package (`pypdf`, `python-docx`) always stops the run with the `pip install` to run, because no file is at fault. `inspect-dataset` accepts `--include`, `--exclude` and `--on-error` to try these out.

Loaders are pluggable: register a function for an extension in `LOADERS` (see [extending.md](extending.md)).

## Inspecting a dataset

`ragbench inspect-dataset --docs PATH --questions FILE [--qrels FILE] [--include/--exclude GLOB] [--on-error raise|skip] [--no-estimate]` checks and profiles a dataset before you spend money on a run:

| Section | What it tells you |
| --- | --- |
| Documents | count; size in tokens (min, median, p95, max) and total; a language guess from common function words (`unknown` when it cannot tell) |
| Questions | count; words per question; category, difficulty and answer-type balance; how many are answerable |
| Qrels coverage | questions with relevance labels (ids or positive qrels); answerable questions without labels (retrieval is not scored on them); how many documents some question points to |
| Suggested chunk sizes | sizes (tokens) worth comparing, from where the corpus's text sits: half of the tokens lie in documents of the *token-weighted median* length or longer, and a chunk larger than a typical document is just the document. Feed them to a [sweep](configuration.md#sweeps) |
| Projected cost | what `--preset standard` would cost with the default models, from [`ragbench estimate`](configuration.md#estimating-cost-and-capping-it)'s mock-run measurement (a few seconds on a big corpus; `--no-estimate` skips it) |

Warnings never stop a run; the same dataset warnings are stored in each run's `run_summary.json` (`dataset_warnings`, next to `document_warnings` described above). They flag duplicate question ids, questions or qrels referencing documents that are not on disk, qrels for unknown question ids, `relevant_doc_ids` missing from the qrels file, documents with empty text, answerable questions without labels (and the label-free case as a whole), and these:

- **Near-duplicate documents**: pairs whose first 400 words overlap by at least 80% (Jaccard on five-word shingles; no MinHash). A system that returns the other copy scores a miss unless both copies are labeled relevant. The search skips shingles that occur in more than a fifth of the documents (boilerplate).
- **Question leakage**: a question of at least four words that appears word for word in a document (ignoring case, spacing and end punctuation). Keyword search finds it by string match, which flatters lexical systems.
- **Unreviewed synthetic questions**: rows still flagged `metadata.needs_review` (from [`generate-questions`](#generating-questions)), and template questions from its `--mock` mode.
- **A corpus that fits in one prompt**: under about 20,000 tokens, the `full_context` baseline (default budget 100,000 tokens) will likely beat every retrieval system, so the comparison between retrieval systems matters less than "do you need retrieval at all".

## Questions

Questions are stored as JSONL:

```json
{
  "id": "q_001",
  "question": "Who won the prestigious IIOTY award in 2023?",
  "reference_answer": "Maxine Thompson won the prestigious Insurance Innovator of the Year award in 2023.",
  "expected_keywords": ["Maxine Thompson", "Insurance Innovator of the Year", "2023"],
  "relevant_doc_ids": ["doc_005", "doc_014"],
  "category": "direct_fact",
  "difficulty": "easy",
  "answer_type": "single_fact"
}
```

For unanswerable questions, use an empty `relevant_doc_ids` list and make the reference answer explicit that the documents do not contain the answer.

`answerable` (optional `true`/`false`) says whether the documents can answer the question. When it is absent a question is answerable exactly when `relevant_doc_ids` is non-empty (the rule above, unchanged). Set it explicitly for a question that has an answer but no labels, or to mark an unanswerable one without relying on an empty list. In a [label-free](#label-free-mode) dataset questions that do not set it are answerable.

Which fields feed which metric ([methodology.md](methodology.md)): `reference_answer` drives `token_f1`, `exact_match` and the judge's correctness; `expected_keywords` drive `keyword_recall`; an empty `relevant_doc_ids` marks the question unanswerable for the abstention metrics unless `answerable` says otherwise; and `answer_type` set to `date` (or `entity`, `person`, `name`, `organization`) lets a wrong answer be classified as `wrong_date` (or `wrong_entity`) instead of a generic failure.

`requires_tools` is an optional list of tool names from the tool registry (see [tools.md](tools.md); an unknown name is an error) (for example `["calculator"]` or `["date_calc"]`) that an agent needs to answer the question. It is metadata only: it never changes retrieval or scoring, and is there so that later reports can show how much a tool-using system gains on the questions that need one.

`routing_hint` is an optional route name (`default`, `lexical`, `computation`, `multi_hop`, or a route name of your own) that an `adaptive` system should send the question to. It is metadata only: it never changes retrieval or scoring. When questions carry hints, the run reports the router's `route_accuracy` and per-route precision and recall in `routes.csv`; without hints you still get the distribution of questions over routes.

## Optional Qrels

`qrels.jsonl` supports graded relevance:

```json
{"query_id": "q_001", "doc_id": "doc_005", "relevance": 3}
```

Recommended relevance levels:

| Value | Meaning |
| --- | --- |
| 0 | Not relevant |
| 1 | Partially relevant |
| 2 | Relevant |
| 3 | Contains exact answer |

If qrels are missing, RAGBench derives binary relevance from `relevant_doc_ids`; if those are missing too, see [label-free mode](#label-free-mode).

## Label-free mode

Most people have documents and questions but no relevance labels. RAGBench runs on those: a question with no `relevant_doc_ids` and no qrels is allowed.

- **Retrieval and context metrics are skipped** for such a question (`Recall`, `MRR`, `nDCG`, `context_recall`... are blank, never 0), so a dataset with no labels at all has blank retrieval columns for every system. Answer quality, faithfulness, cost, latency and the [recommendation](methodology.md#selection) work as usual.
- **Answers are judged by the LLM judge** against `reference_answer` when there is one (the lexical `token_f1` / `keyword_recall` still use it). With no reference the judge grades against the retrieved context alone (faithfulness, relevance, citations; correctness and completeness track the context), see [methodology.md](methodology.md#bias-and-noise).
- **No labels at all means answerable.** When no question in the dataset has a label, questions that do not set `answerable` are treated as answerable, because "no relevant documents" cannot mean "unanswerable" when nothing is labeled. Mark real unanswerable questions `answerable: false`. In a dataset that is *partly* labeled the legacy rule holds (an unlabeled question without `answerable` is unanswerable), so set `answerable: true` on unlabeled questions that do have answers; `ragbench inspect-dataset` and the run warn about them.
- Validation **warns, it does not fail**.
- A question labeled only through the qrels file (no `relevant_doc_ids`; what `ragbench label --apply` produces) has relevant documents, so it counts as answerable.

Labels can be added later, one question at a time; the questions that have them then get retrieval metrics (the mean covers the labeled questions only, and `run_summary.json` records `dataset: {questions, labeled_questions, label_free}`).

## Importing existing data

`ragbench import --format csv|beir|qa-md --input PATH --output DIR [--docs DIR] [--split test] [--force]` writes `DIR/questions.jsonl` (and more, below) and prints the next command. It refuses to replace an earlier import without `--force` (for `beir` that includes its `docs/` folder). Questions without labels are written with an explicit `answerable: true`.

| Format | Input | Writes |
| --- | --- | --- |
| `csv` | a file with a `question` column (aliases `query`, `q`) and optional `answer`, `doc_ids` (separated by `;`, `\|` or `,`), `id`, `category`, `difficulty`, `answer_type`, `keywords` and `answerable` (`yes`/`no`) columns; other columns become `metadata` | `questions.jsonl`. Your documents are not part of a CSV: pass `--docs` to check the `doc_ids` against your folder |
| `beir` | a BEIR folder: `corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv` (`--split`, default `test`) | `docs/` (one `.md` file per corpus entry, `# title` first), `questions.jsonl` (the queries of the split with a document graded above 0), `qrels.jsonl` (graded, zeros included), `id_map.json` (BEIR id → RAGBench id) |
| `qa-md` | a Markdown file, or a folder of them, with `Q:` / `A:` pairs; an optional `Docs: doc_001, doc_002` line gives labels. Headings, bullets and bold around the markers (`## Q:`, `**Question:**`) are fine; text after `A:` continues until a blank line | `questions.jsonl` |

BEIR ids are made loader-safe (`doc_` + the id with anything but letters, digits, `_` and `-` replaced; ids that would collide, also on a case-insensitive file system, get a hash suffix) because a document's id comes from its file name. A BEIR corpus becomes one file per document, so import a subset of very large corpora. Qrels that name a document missing from the corpus and queries with no positive grade are left out with a warning; if nothing is left the import fails and says why.

## Generating questions

`ragbench generate-questions --docs PATH --out questions.jsonl [--n 100] [--mix single_hop=0.4,multi_hop=0.2,paraphrase=0.15,numeric=0.1,unanswerable=0.15] [--model REF] [--seed 0] [--config FILE] [--mock] [--max-cost USD] [--yes] [--force]` writes a questions file for a corpus that has none.

| Category | How it is written |
| --- | --- |
| `single_hop` | the model reads one document and writes a question, a short reference answer and expected keywords |
| `multi_hop` | the same from a *pair* of related documents (BM25 nearest neighbours; a pair is used once), so neither alone answers it; `relevant_doc_ids` has both |
| `paraphrase` | a single-hop question rewritten to avoid the document's own vocabulary |
| `numeric` | a question whose answer is a number stated in (or computed from) a document that contains numbers |
| `unanswerable` | a question plausible for the domain that no document answers; the model also names its key `entity`, and the question is rejected if that entity occurs anywhere in the corpus. `answerable: false`, no `relevant_doc_ids` |

The mix is split by largest remainder, so every category is within one question of its share. Documents are sampled stratified by length (short, medium and long documents alternate), and each prompt lists the questions already written from that document, so repeated draws differ. Candidates whose content words overlap an accepted question by 80% or more are dropped; a category that cannot reach its quota (no document with numbers, no related pairs, too many rejections) is reported, never padded with another. Rows are shuffled before they are numbered `q_001`…, so `evaluation.max_questions` takes a representative slice. The same `--seed` gives the same documents and, with the same model and the disk cache, the same questions.

Every row carries `metadata: {synthetic: true, needs_review: true, generator: <model>, source_doc_ids: [...]}` (plus `entity` for unanswerable ones and `paraphrase_of` for paraphrases). `inspect-dataset` counts the rows still flagged and says so; read them, fix or delete the bad ones, and remove `needs_review` when you are done. Scores on unreviewed synthetic questions measure the generator's taste as much as the systems ([methodology](methodology.md#pooling-bias)).

**Cost and mock mode.** A live run prints its estimated cost first (the documents each category reads, at the model's price) and asks before going on when it is above `evaluation.cost_confirm_threshold_usd` (default $1); with no terminal or `CI` set it refuses unless `--yes`. `--max-cost` stops writing once that much was charged and keeps what exists (exit status 1). Replies are cached on disk like every temperature-0 call. `--out` is never replaced without `--force`, because it may hold questions you reviewed. `--mock`, or a missing API key for the model, writes **template** questions instead ("What does the document say about <topic>?" over each document's own sentences), deterministic and flagged `mock: true`: they prove the pipeline runs, nothing more. `--config` supplies `providers:`, `pricing:` and `cache:` (needed for `openai_compatible:` models).

## Pooled labeling

`ragbench label --run RUN_DIR [--top-k 10] [--judge-model REF] [--out DIR] [--apply] [--mock] [--max-cost USD] [--max-workers 4] [--yes] [--force]` proposes relevance labels for a finished run.

1. **Pool.** For every answerable question, the first `--top-k` distinct documents each system retrieved (chunks of one document count once), united across *all* systems in the run. Unanswerable questions are skipped.
2. **Grade.** The judge model grades every pooled (question, document) pair 0-3 (0 not relevant, 1 partial, 2 relevant, 3 exact answer), reading the part of the document that best matches the question when it is long. A reply that is not a valid grade is asked again once; if it still is not, nothing is proposed for that pair. Calls go through the disk cache (re-running costs almost nothing) and the estimate is shown and confirmed first, as above. The run's own config supplies the dataset paths, `providers:`, `pricing:`, `cache:` and, unless `--judge-model` says otherwise, the judge model; run the command from the directory the run used.
3. **Write**, into the run directory (or `--out`):
   - `qrels.proposed.jsonl`: one `{query_id, doc_id, relevance}` row per graded pair, zeros included (judged, not relevant). A document that was not pooled and graded never appears.
   - `qrels_review.md`: the review described below.
   - with `--apply`, `qrels.merged.jsonl`: your labels plus the proposed grades for documents your labels do not mention. **Where both grade a document, yours wins.**

Your questions and qrels files are never modified. To use the merged labels, point `dataset.qrels_path` at the merged file; questions that had no labels are then labeled (and answerable), so they get retrieval metrics. Note what `relevant_doc_ids` does: recall is computed against a question's own `relevant_doc_ids` when it has any, and only nDCG sees the added qrels, so grades proposed for a question that already lists relevant documents refine its nDCG and leave its recall alone.

**The review** opens with counts (pairs, grade distribution, how often the grader disputes a label) and then lists what to check: existing labels the grader graded 0 (*disputed*), documents graded 2 or 3 that your labels lack (*possible missing label*), labeled documents that no system retrieved (*not judged*), the questions where the run itself flagged `possible_qrels_gap`, and, for questions with no labels, the proposed relevant documents. A question with no relevant document found is called out: in a dataset that is only partly labeled such a question reads as unanswerable, so give the ones that do have answers `answerable: true`. With `--mock` (or no API key) the grades come from a word-overlap stand-in and the review says so; they only prove the pipeline works.

## Qrels Audit

Runs generate `qrels_audit.md` and `qrels_audit.csv`. These files flag cases where an answer was judged highly supported even though retrieval metrics were low because retrieved documents were not labeled relevant. Treat those rows as candidates for human review.

## The bundled demo dataset

`ragbench demo` writes the demo dataset to `data/demo`: 60 documents, 163 questions and graded qrels set in the fictional "RAGBench Mutual" universe. The files in `data/demo` are the source of truth and ship inside the wheel; `ragbench demo` only copies them, keeps any file you edited, and restores the originals with `--overwrite`.

It is built to tell retrieval strategies apart. The corpus mixes very short FAQ pages, four handbooks of 2,400 to 3,100 words whose answers sit deep inside (where chunking matters), versioned documents that contradict each other (three remote work policies, two service level agreements, changelogs), near-duplicate product pages that differ in a few numbers, tables, code and configuration snippets, and a glossary and a newsletter written in a different register.

| Category | Tests | `requires_tools` |
| --- | --- | --- |
| `direct_fact` | one fact in one short document (the easy baseline) | |
| `paraphrase` | the question avoids the document's vocabulary | |
| `exact_identifier` | error codes, SKUs, contract and incident ids; favors lexical search | |
| `multi_hop` | the answer needs two documents | |
| `comparison` | contrasts two items, often near-duplicate pages | |
| `aggregation` | combines facts from three or more documents | `calculator` where it sums |
| `numeric_reasoning` | percentages, totals and prices computed from document numbers | `calculator` |
| `date_arithmetic` | days between two dates found in the documents | `date_calc` |
| `temporal_conflict` | several versions disagree; the latest (or the asked-for) version wins | |
| `distractor` | near-identical pages, and only one has the asked-for number | |
| `long_context` | a fact buried in a 2,400-word or longer handbook | |
| `unanswerable` | plausible, but the corpus never says; 15% of the questions | |

Key-evidence documents carry relevance 3 in the qrels, supporting documents 2, and later documents that merely restate an answer from the first 50 questions 1 (credit for nDCG only; recall uses `relevant_doc_ids`).
