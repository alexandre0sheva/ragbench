# Release checklist

Use this before tagging a release. CI runs most of the automatic checks; the manual items are the ones CI cannot do.

## Prepare

- [ ] Rename `## [Unreleased]` in `CHANGELOG.md` to the version and date, and start a new empty `## [Unreleased]`.
- [ ] Set the version in `pyproject.toml` and `CITATION.cff` (and the date in the latter).
- [ ] **Refresh model names and prices.** Providers retire and add models every few months, and a stale price table makes every cost number wrong.
  - Re-read the **official** model and pricing pages of OpenAI and Anthropic (fetch them on the day; never work from memory), plus the pages of any other provider with a price row.
  - Update `MODEL_PRICING_USD_PER_1M` in `src/ragbench/models/cost.py` and put the source URLs in the comment above it. Keep only models that are currently served: the newest generation in each tier, and at most the previous one while it is still the cheaper option people pick. Dated snapshots are covered by prefix matching, so no rows for them. Mark any entry you could not confirm `# UNVERIFIED` instead of guessing, and do not delete a model you could not confirm is superseded.
  - Set `PRICING_AS_OF` to today's date. `ragbench doctor` and `ragbench estimate` warn when the table is more than 90 days old.
  - Re-check the defaults in `src/ragbench/models/defaults.py`, the `configs/*.yaml` files, the presets and every model named in the docs: each must point at the newest generation in its tier and have a price. `tests/test_cost_tracker.py` fails if one has no price.
  - Run `ragbench doctor --config configs/all.yaml`: every model should report a known price.

## Verify (all of this is also in CI)

- [ ] `ruff check .` and `mypy`.
- [ ] `pytest --cov` passes and meets the coverage floor.
- [ ] `RAGBENCH_GOLDEN_STRICT=1 pytest tests/test_golden_mock.py`: the golden snapshot did not change, or the change is intended and in the changelog.
- [ ] `python scripts/generate_docs.py --check` and `python scripts/audit_docs.py`.
- [ ] `ragbench auto --docs data/demo/docs --questions data/demo/questions.jsonl --preset quick --mock --yes` ends with a recommendation, and its `report.html` opens and renders in light and dark.
- [ ] `ragbench compare --config configs/all.yaml --mock --max-workers 4` completes; inspect `leaderboard.md`, `failures.md` and `qrels_audit.md`.

## Live run (maintainer, costs money)

- [ ] `ragbench estimate --config configs/all.yaml` and read the projected cost; proceed only if you accept it.
- [ ] `ragbench run --config configs/all.yaml` with a real key (in `.env` or your shell, never committed) and your cap set as `evaluation.max_cost_usd` in a copy of the config (`run` has no `--max-cost` flag; the shipped config stays uncapped). Check the banner says LIVE, no model is reported as unpriced, no judge fallbacks are warned about, and the latency probe ran.
- [ ] Open `report.html`, check the recommendation reads sensibly, and refresh the screenshots in `docs/assets/` if the report changed.

## Package

- [ ] Nothing private is staged: no `.env`, `results/`, caches, `dist/` or `build/`.
- [ ] Build and check the package:

```bash
python -m build
twine check --strict dist/*
```

- [ ] Install the wheel in a clean virtual environment and run `ragbench demo` and `ragbench run --preset quick --docs data/demo/docs --questions data/demo/questions.jsonl --mock`.
- [ ] Tag the release and publish.
