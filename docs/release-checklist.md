# Release Checklist

Use this checklist before tagging a public release.

- [ ] Update `CHANGELOG.md`.
- [ ] Confirm `pyproject.toml` version.
- [ ] Run `ruff check .`.
- [ ] Run `pytest`.
- [ ] Run `ragbench demo`.
- [ ] Run `ragbench run --config configs/recommended.yaml --mock`.
- [ ] Inspect generated `leaderboard.md`, `failures.md`, and `qrels_audit.md`.
- [ ] Confirm no `.env`, `results/`, caches, or private data are staged.
- [ ] Build package:

```bash
python -m build
```

