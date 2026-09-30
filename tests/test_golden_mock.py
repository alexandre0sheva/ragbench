"""Regression net: a mock run over configs/all.yaml must keep reproducing the committed snapshot.

Strict mode (`RAGBENCH_GOLDEN_STRICT=1`) demands identical metrics, per-question document rankings and
answers; use it when refactoring. The default mode only requires metrics within a small tolerance so
the test does not flake when numpy/BLAS tie-breaking differs between platforms. Systems added after the
snapshot are ignored; regenerate with `python scripts/update_golden.py` when behavior is meant to change.
"""

from __future__ import annotations

import os

import pytest
from golden_support import collect, load_golden, run_all_systems_mock

STRICT = os.environ.get("RAGBENCH_GOLDEN_STRICT") == "1"
TOLERANCE = 0.03


@pytest.fixture(scope="module")
def current(tmp_path_factory):
    return collect(run_all_systems_mock(tmp_path_factory.mktemp("golden")))


def test_mock_run_matches_golden_snapshot(current):
    golden = load_golden()["systems"]
    assert set(golden) <= set(current["systems"]), "a system in the snapshot disappeared"
    for name, expected in golden.items():
        got = current["systems"][name]
        for metric, value in expected["metrics"].items():
            assert metric in got["metrics"], f"{name}: metric {metric} disappeared"
            tolerance = 1e-9 if STRICT else TOLERANCE
            assert got["metrics"][metric] == pytest.approx(value, abs=tolerance), f"{name}: {metric}"
        if STRICT or name == "bm25_default":  # BM25 is lexical and fully deterministic everywhere
            assert got["rankings"] == expected["rankings"], f"{name}: document rankings changed"
            assert got["answers_sha256"] == expected["answers_sha256"], f"{name}: generated answers changed"
