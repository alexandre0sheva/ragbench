"""Regenerate tests/golden/mock_metrics_v3.json from the current code (mock mode, no network)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from golden_support import GOLDEN_PATH, collect, run_all_systems_mock  # noqa: E402

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = collect(run_all_systems_mock(Path(tmp)))
    GOLDEN_PATH.write_text(json.dumps(snapshot, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH} ({len(snapshot['systems'])} systems)")
