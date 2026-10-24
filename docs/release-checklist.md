# Release checklist

Use this before tagging a release. CI runs most of the automatic checks; the manual items are the ones CI cannot do.

## Prepare

- [ ] Rename `## [Unreleased]` in `CHANGELOG.md` to the version and date, and start a new empty `## [Unreleased]`.
- [ ] Set the version in `pyproject.toml` and `CITATION.cff` (and the date in the latter).
- [ ] **Refresh model names and prices.** Providers retire and add models every few months, and a stale price table makes every cost number wrong.
  - Re-read the **official** model and pricing pages of OpenAI and Anthropic (fetch them on the day; never work from memory), plus the pages of any other provider with a price row. Note every prompt-length tier (`tiers` in the table).
  - Run `ragbench models sync` and re-check the OpenRouter shortlist in `docs/models.md` against it (`ragbench models list --provider openrouter`): prices, open weights, tools, and the lowest reasoning effort. Regenerate the bundled snapshot `src/ragbench/models/data/openrouter_models.json` from the synced file for the shortlisted ids.
  - Update `MODEL_PRICING_USD_PER_1M` in `src/ragbench/models/cost.py` and put the source URLs in the comment above it. Keep only models that are currently served: the newest generation in each tier, and at most the previous one while it is still the cheaper option people pick. Dated snapshots are covered by prefix matching, so no rows for them. Mark any entry you could not confirm `# UNVERIFIED` instead of guessing, and do not delete a model you could not confirm is superseded.
  - Set `PRICING_AS_OF` to today's date. `ragbench doctor` and `ragbench estimate` warn when the table is more than 90 days old.
  - Re-check the defaults in `src/ragbench/models/defaults.py`, the `configs/*.yaml` files, the presets and every model named in the docs: each must point at the newest generation in its tier and have a price. `tests/test_cost_tracker.py` fails if one has no price.
  - Run `ragbench doctor --config configs/all.yaml`: every model should report a known price.

## Verify (all of this is also in CI)

- [ ] `ruff check .` and `mypy`.
- [ ] `pytest -n auto --cov` passes (the whole suite, slow tests included) and meets the coverage floor.
- [ ] `RAGBENCH_GOLDEN_STRICT=1 pytest tests/test_golden_mock.py`: the golden snapshot did not change, or the change is intended and in the changelog.
- [ ] `python scripts/generate_docs.py --check` and `python scripts/audit_docs.py`.
- [ ] `ragbench auto --docs data/demo/docs --questions data/demo/questions.jsonl --preset quick --mock --yes` ends with a recommendation, and its `report.html` opens and renders in light and dark.
- [ ] `ragbench compare --config configs/all.yaml --mock --max-workers 4` completes; inspect `leaderboard.md`, `failures.md` and `qrels_audit.md`.

## Live run (maintainer, costs money)

- [ ] Before **every** paid run, not only releases: repeat the model-and-price audit above, so the estimate and the cap are computed from today's prices.

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
- [ ] `docker build -t ragbench .` and `docker run --rm ragbench demo` work (CI builds the image on every push, so this only matters for a local check).

## Tag and publish

Publishing is automated by `.github/workflows/release.yml` and runs when a tag `vX.Y.Z` is pushed. The tag must equal `v` plus the version in `pyproject.toml`, `CITATION.cff` must carry the same version, and `CHANGELOG.md` needs a non-empty `## [X.Y.Z]` section (it becomes the GitHub release notes): `python scripts/release_notes.py vX.Y.Z` checks all three locally. The workflow then repeats the package checks, uploads the files that passed to PyPI with trusted publishing (OIDC, no token stored anywhere) and creates the GitHub release with those files attached.

- [ ] **Once, before the first release:** on PyPI, add a *pending trusted publisher* (Account settings → Publishing) for the project `ragbench`: owner `alexandre0sheva`, repository `ragbench`, workflow `release.yml`, environment `pypi`. In the GitHub repository settings create the environment `pypi` (add a required reviewer if you want a manual approval before each upload).
- [ ] Tag the commit and push the tag (pushing is the step that publishes; it cannot be undone, because PyPI never accepts the same version twice):

```bash
git tag -a vX.Y.Z -m "RAGBench X.Y.Z"
git push origin vX.Y.Z
```

- [ ] Watch the **Release** run. If a check fails before the upload, delete the tag (`git push --delete origin vX.Y.Z`, `git tag -d vX.Y.Z`), fix, and tag again.
- [ ] In a clean environment: `pip install ragbench==X.Y.Z` and `ragbench demo`.
