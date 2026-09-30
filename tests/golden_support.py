"""Golden snapshot of a mock run over `configs/all.yaml`, used to prove refactors change no behavior.

`collect()` reduces a finished run directory to: per-system mean quality metrics (latency, cost and wall
time excluded), the per-question *document* ranking each system produced, and a hash of all generated
answers. Regenerate the committed snapshot with `python scripts/update_golden.py` when a change is
*supposed* to alter retrieval or answers (and say so in the changelog).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ragbench.evaluation.evaluator import run_benchmark
from ragbench.utils.text import unique_preserve_order

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "tests" / "golden" / "mock_metrics_v3.json"
EXCLUDED_COLUMNS = {
    "system_type",
    "avg_latency_ms",
    "avg_latency_concurrent_ms",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_source",
    "avg_cost_per_question",
    "system_wall_time_ms",
}


def run_all_systems_mock(workdir: Path) -> Path:
    """Run configs/all.yaml in mock mode against the bundled demo data, writing under `workdir`.

    The exact in-memory vector backend is used instead of Chroma: Chroma's HNSW index is not
    deterministic across processes (ties in the tail of a ranking shuffle), which would make a
    golden snapshot flaky. The snapshot guards pipeline logic, not the ANN library.
    """
    raw = (ROOT / "configs" / "all.yaml").read_text(encoding="utf-8")
    config_path = workdir / "all.yaml"
    config_path.write_text(
        raw.replace("data/demo", str(ROOT / "data" / "demo"))
        .replace("output_dir: results", f"output_dir: {workdir / 'results'}")
        .replace("vector_store: chroma", "vector_store: in_memory"),
        encoding="utf-8",
    )
    return run_benchmark(config_path, force_mock=True, max_workers=1)


def collect(run_dir: Path) -> dict:
    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    rows = [json.loads(line) for line in (run_dir / "per_question_results.jsonl").read_text(encoding="utf-8").splitlines()]
    systems: dict[str, dict] = {}
    for name, record in summary.iterrows():
        metrics = {col: round(float(record[col]), 6) for col in summary.columns if col not in EXCLUDED_COLUMNS and pd.notna(record[col])}
        mine = [row for row in rows if row["system"] == name]
        rankings = {row["question_id"]: unique_preserve_order(ctx["doc_id"] for ctx in row["retrieved_contexts"]) for row in mine}
        answers = hashlib.sha256("\n".join(row["answer"] for row in mine).encode("utf-8")).hexdigest()
        systems[str(name)] = {"metrics": metrics, "rankings": rankings, "answers_sha256": answers}
    return {"config": "configs/all.yaml", "systems": systems}


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
