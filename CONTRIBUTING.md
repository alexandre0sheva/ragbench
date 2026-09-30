# Contributing to RAGBench

Thanks for helping improve RAGBench. This project is intended to be a practical, evaluation-first RAG benchmark framework, so contributions should preserve reproducibility, local mock-mode execution, and clear reports.

## Development Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/alexandre0sheva/ragbench.git
cd ragbench
uv venv && source .venv/bin/activate      # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"                # or: pip install -e ".[dev]"
ragbench demo
pytest
```

RAGBench must work without `OPENAI_API_KEY`. Tests should not require network calls or paid APIs.

## Common Commands

```bash
ruff check .
mypy
pytest
python scripts/generate_docs.py --check   # generated docs (docs/systems.md, docs/cli.md, README system table) are current
ragbench run --config configs/recommended.yaml --mock
ragbench compare --config configs/all.yaml --mock --max-workers 4
```

## Contribution Guidelines

- Keep changes modular and dataset-agnostic.
- Do not add required external services for default runs.
- Preserve mock mode for local testing and CI.
- Add or update tests for behavior changes.
- Keep demo data fictional and safe to publish.
- Avoid committing generated `results/`, `.env`, caches, or local vector-store state.

## Adding a RAG System

See [docs/extending.md](docs/extending.md): declare the options and a `SystemSpec`, register the class, then run `python scripts/generate_docs.py`. Add a config example and at least one focused test.

## Pull Requests

Before opening a pull request:

- Run `pytest`.
- Run `ruff check .` if Ruff is installed.
- Include a concise explanation of the evaluation impact.
- Mention any API-cost or latency implications.

