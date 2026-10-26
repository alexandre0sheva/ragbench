"""The versioned JSON schemas of `run_summary.json` and `metrics_summary.csv`: what a run writes validates, and the version is in the file."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from cli_support import invoke, mock_run

from ragbench.reporting.schemas import SCHEMA_VERSION, SCHEMAS, load_schema

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = sorted((ROOT / "docs" / "benchmarks").glob("*-run"))


def summary_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics_summary.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def typed(row: dict[str, str]) -> dict[str, object]:
    """A CSV row as `pandas` reads it: numbers are numbers, empty cells are missing."""
    out: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            continue
        try:
            out[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            out[key] = {"True": True, "False": False}.get(value, value)
    return out


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("schemas")
    return mock_run(root, root / "results" / "run", systems=["bm25", "hybrid_rerank"])


def test_every_schema_is_valid_json_schema_and_names_its_version():
    for name in SCHEMAS:
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["$id"].endswith(f"{name}.v{SCHEMA_VERSION}.schema.json") and schema["title"]


def test_a_run_writes_its_schema_version_first(run_dir):
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert next(iter(summary)) == "schema_version" and summary["schema_version"] == SCHEMA_VERSION == 1


def test_a_real_runs_summary_and_metrics_validate(run_dir):
    jsonschema.validate(json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")), load_schema("run_summary"))
    rows = summary_rows(run_dir)
    assert len(rows) == 2
    for row in rows:
        jsonschema.validate(typed(row), load_schema("metrics_summary"))


@pytest.mark.parametrize("committed", COMMITTED, ids=lambda p: p.name)
def test_the_committed_benchmark_runs_validate_too(committed):
    """Runs written before the version existed have none; that means version 1."""
    if (committed / "run_summary.json").is_file():
        jsonschema.validate(json.loads((committed / "run_summary.json").read_text(encoding="utf-8")), load_schema("run_summary"))
    for row in summary_rows(committed):
        jsonschema.validate(typed(row), load_schema("metrics_summary"))


def test_the_schemas_reject_what_would_break_a_reader(run_dir):
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(load_schema("run_summary"))
    assert not validator.is_valid({**summary, "mode": "imaginary"})
    assert not validator.is_valid({k: v for k, v in summary.items() if k != "run_id"})
    assert not validator.is_valid({**summary, "schema_version": 2})  # a reader of version 1 must not claim to understand 2
    row = typed(summary_rows(run_dir)[0])
    metrics = jsonschema.Draft202012Validator(load_schema("metrics_summary"))
    assert not metrics.is_valid({k: v for k, v in row.items() if k != "system"})
    assert not metrics.is_valid({**row, "answer_score": "high"})
    assert metrics.is_valid({**row, "a_future_column": 1.0})  # columns are open-ended: extras and k-values add some


def test_an_aborted_runs_early_summary_validates_as_well(tmp_path):
    early = {"schema_version": 1, "run_id": "r", "mode": "live", "run_wall_time_ms": 1.0, "num_systems": 0, "budget": {"stopped": True}}
    jsonschema.validate(early, load_schema("run_summary"))


def test_runs_schema_prints_the_schema_and_lists_the_names():
    shown = invoke("runs", "schema", "run_summary")
    assert shown.exit_code == 0 and json.loads(shown.stdout) == load_schema("run_summary")
    assert invoke("runs", "schema", "metrics_summary", "--version", "1").exit_code == 0
    listing = invoke("runs", "schema")
    assert listing.exit_code == 0 and "run_summary" in listing.output and "metrics_summary" in listing.output
    assert invoke("runs", "schema", "nope").exit_code == 2
    assert invoke("runs", "schema", "run_summary", "--version", "9").exit_code == 2
