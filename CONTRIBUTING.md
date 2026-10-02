# Contributing to RAGBench

Thanks for helping improve RAGBench. This project is a practical, evaluation-first RAG benchmark framework, so contributions should preserve reproducibility, local mock-mode execution, and clear reports.

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/alexandre0sheva/ragbench.git
cd ragbench
uv venv && source .venv/bin/activate      # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"                # or: pip install -e ".[dev]"
ragbench demo
pytest
```

RAGBench must work without `OPENAI_API_KEY`. Tests must not need the network, a paid API or a model download: use the mock models and the fixtures in `tests/conftest.py` (a tiny dataset, a scripted LLM), and `tests/fake_openai_server.py` for code that talks to a real SDK. Every `ragbench` command and option is in [docs/cli.md](docs/cli.md).

## The checks a change must pass

CI runs these; run them before you push.

```bash
ruff check .
mypy
pytest --cov                                 # fails below the coverage floor in pyproject.toml; raise the floor when coverage rises, never lower it
python scripts/generate_docs.py --check      # docs/systems.md, docs/cli.md, docs/tools.md and the README system table match the code
python scripts/audit_docs.py                 # no duplicated or broken documentation (see below)
ragbench compare --config configs/all.yaml --mock --max-workers 4
```

- **Golden snapshot.** `tests/golden/mock_metrics_v3.json` pins what every system retrieves and answers in a mock run. It must not change unless your change is meant to alter retrieval or answers; then run `python scripts/update_golden.py`, say so in the pull request, and note it in the changelog. `RAGBENCH_GOLDEN_STRICT=1 pytest tests/test_golden_mock.py` is the strict check.
- **Report structure.** `tests/golden/report_structure_v1.json` pins the sections and data keys of `report.html`. After an intended change: `RAGBENCH_UPDATE_SNAPSHOTS=1 pytest tests/test_report_snapshot.py`.
- **Every system and chunker is tested end to end** by `tests/test_e2e_matrix.py` as soon as it is registered: it must run, write every output, trace its cost, and be reproducible. Fix what it finds rather than excluding a component.

## Guidelines

- Keep changes modular and dataset-agnostic; match the style of the code around you (`from __future__ import annotations`, Pydantic v2, line length 140).
- Do not add required external services for default runs, and preserve mock mode.
- Add or update tests for behavior changes (tests first for logic such as metrics, parsing, routing, caching and statistics).
- Keep demo data fictional and safe to publish.
- Do not commit `results/`, `.env`, caches or local vector-store state.
- Every change adds a bullet under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).

## Documentation

Each piece of knowledge has one owner; everywhere else links to it. `scripts/audit_docs.py` (also a test) fails on broken links, option tables and system descriptions outside their owners, metric definitions outside the methodology, and counts that will go stale.

| Knowledge | Owner |
| --- | --- |
| Pitch, install, quickstart, links | `README.md` |
| Systems, their options, cost and latency profile | `docs/systems.md` (generated) |
| How each system, chunker, reranker and agent works (algorithms, diagrams) | `docs/system-wiki.md` |
| Commands and options | `docs/cli.md` (generated) |
| Config sections, presets, sweeps, budgets, caching, providers | `docs/configuration.md` |
| Dataset formats and onboarding | `docs/dataset-format.md` |
| Metrics, the judge, statistics, cost accounting, limitations | `docs/methodology.md` |
| Tools and agents | `docs/tools.md` (tool table generated) |
| Writing a new system, chunker, tool, provider or plugin | `docs/extending.md` |
| Which architecture to try and how to read the result | `docs/choosing-an-architecture.md` |
| Release history | `CHANGELOG.md` |
| Releasing | `docs/release-checklist.md` |

After adding or changing a system, option, command or tool, run `python scripts/generate_docs.py`.

## Adding a RAG system

See [docs/extending.md](docs/extending.md): declare the options and a `SystemSpec`, register the class, then regenerate the docs. Add a config example and at least one focused test; the end-to-end matrix covers the rest.

## Pull requests

Include a concise explanation of the evaluation impact and mention any API-cost or latency implications. The pull request template asks for the same.

## Maintainers: repository setup

- Enable GitHub Actions and Dependabot (`.github/dependabot.yml` groups routine bumps).
- Enable private vulnerability reporting (see [SECURITY.md](SECURITY.md)).
- Protect `main` once there are outside contributors, and require the CI jobs to pass before merging: *Lint and types*, *Generated docs are current*, *Tests* (Python 3.11 to 3.13), *Tests with every optional extra*, *CLI smoke test* and *Package build*. *Tests (Python 3.14)* and the dependency audit are advisory.
- Never commit `.env`, `results/`, `dist/`, `build/`, `*.egg-info/` or caches (`git status --short` before a first push).
