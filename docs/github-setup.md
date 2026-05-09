# GitHub Setup

Use these steps when creating the public GitHub repository.

## Before First Push

1. Create the repository on GitHub.
2. Replace placeholder URLs in `pyproject.toml` and `CITATION.cff`:

```toml
[project.urls]
Homepage = "https://github.com/<owner>/ragbench"
Repository = "https://github.com/<owner>/ragbench"
Issues = "https://github.com/<owner>/ragbench/issues"
Documentation = "https://github.com/<owner>/ragbench#readme"
```

3. Confirm ignored local files are not staged:

```bash
git status --short
```

Do not commit:

- `.env`
- `results/`
- `dist/`
- `build/`
- `src/*.egg-info/`
- `__pycache__/`

## Suggested First Commit

```bash
git init
git add .
git status --short
git commit -m "Initial RAGBench release"
```

## Local Verification

```bash
pip install -e ".[dev]"
ruff check .
pytest
ragbench demo
ragbench run --config configs/recommended.yaml --mock --max-workers 2
```

## GitHub Settings

Recommended repository settings:

- Enable GitHub Actions.
- Enable Dependabot security alerts.
- Enable private vulnerability reporting.
- Protect the `main` branch once the project has external contributors.
- Require CI to pass before merging pull requests.

