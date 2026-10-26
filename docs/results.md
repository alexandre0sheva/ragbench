# Result files, schemas and the site

A finished run is a folder of plain files; the [example run](example-run.md) lists what each one holds. This page is about reading them from other tools, putting many runs on one page, and the documentation site that publishes the committed benchmark runs.

## Reading a run from your own tools

`run_summary.json` (one run in numbers: mode, models, spend, cache, errors, the names of the other files) and `metrics_summary.csv` (one row per system) are the two files other tools read most. Both are described by versioned [JSON Schemas](https://json-schema.org), shipped with the package:

```bash
ragbench runs schema                       # lists them
ragbench runs schema run_summary           # the schema, as JSON, on stdout
ragbench runs schema metrics_summary       # one row of metrics_summary.csv as an object
```

`run_summary.json` carries `schema_version`. `metrics_summary.csv` has no place for one without adding a column to every row, so it is versioned with the run's `run_summary.json`. A run written before the field existed has none, which means version 1.

- **A new version is a break.** Renaming or removing a field, or changing what it means or its unit, gets a new `schema_version` and a new schema file; the old file stays so a tool can read old runs. Adding an optional field does not change the version, so a reader should ignore fields it does not know. The schemas allow extra fields for this reason.
- **Columns are open-ended.** `metrics_summary.csv` has `retrieval_*@k` columns for each cutoff the run used, `*_ci_lo` / `*_ci_hi` intervals, and columns a system adds (steps, tool calls, gate abstentions). An empty cell is a metric that does not apply or was not measured: missing, never zero.
- A test validates a freshly written run and every committed benchmark run against these schemas, so they cannot drift from what the code writes.

## Many runs on one page

```bash
ragbench runs site results                 # results/site/index.html
ragbench runs site results --out public --title "Our RAG benchmark"
```

`ragbench runs site FOLDER` reads every complete run directory in `FOLDER` and writes one HTML file: a row per system per run, with filters by run, mode, system type and name, and a sortable column each. The data is inside the file and the page makes no network requests, so you can zip the folder or put it on any static host. Each run's `report.html` is copied next to the page and linked from its rows. Mock runs are hidden by default when a live run is present, because their scores validate the pipeline and say nothing about quality; the Mode filter shows them. [`ragbench runs index`](cli.md#ragbench-runs-index) is the other cross-run page: the history of runs with a score-over-time line per system.

## The documentation site

The site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) from the Markdown in this repository and published to GitHub Pages by CI on every push to `main`. To build it locally:

```bash
pip install -e ".[docs]"
python scripts/build_site.py               # build/site, with --strict: a broken link or anchor fails the build
python -m mkdocs serve                     # after the build script has staged the sources: live preview
```

The Markdown stays where GitHub renders it. `scripts/build_site.py` builds from a staged copy: `README.md` is the home page, `docs/` is the root, relative links are rewritten to follow the files, and a link to a file the site does not carry (configs, examples, source) goes to the repository on GitHub. The working plans under `docs/superpowers/` are not published.

**The Leaderboard page is generated.** `leaderboard.md` and its filterable twin are built from the result files committed under `docs/benchmarks/*-run/` (`metrics_summary.csv`, `run_summary.json`, `run_manifest.json`), never from numbers typed into a page. To publish a run, copy its small result files into `docs/benchmarks/<name>-run/` and push: CI rebuilds the page. A run that is mock is labelled as such.

**One-time setup for the repository owner:** in GitHub, Settings → Pages → Source: *GitHub Actions*. The `deploy-site` job of the CI workflow then publishes `build/site` on every push to `main`; pull requests only build it (so a broken link fails the review, not the deploy).
