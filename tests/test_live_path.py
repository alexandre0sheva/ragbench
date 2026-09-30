"""The live code path, end to end, in a *fresh interpreter*: real OpenAI SDK -> real HTTP -> a local fake server.

Mock-mode tests never touch the SDK, so problems that only appear with real clients and real threads (lazy imports
racing with requests, retry handling, caching of paid calls, the latency probe) are covered here. The fake key is
obviously not real and nothing leaves localhost.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fake_openai_server import FakeOpenAIServer

ROOT = Path(__file__).resolve().parents[1]


def _config(work: Path) -> Path:
    shutil.copytree(ROOT / "data" / "demo", work / "demo")
    systems = [
        {"type": "bm25", "name": "bm25", "chunker": {"type": "token", "chunk_size": 450, "chunk_overlap": 70}, "retrieval": {"top_k": 5}},
        {"type": "vector", "name": "vector", "retrieval": {"vector_store": "chroma", "top_k": 5}},
        {"type": "hybrid_rerank", "name": "hybrid_rerank", "retrieval": {"vector_store": "chroma", "final_top_k": 5}},
        {"type": "hyde", "name": "hyde", "retrieval": {"vector_store": "chroma", "top_k": 5}},
    ]
    config = {
        "run": {"name": "live_path", "output_dir": str(work / "results")},
        "dataset": {
            "documents_path": str(work / "demo" / "docs"),
            "questions_path": str(work / "demo" / "questions.jsonl"),
            "qrels_path": str(work / "demo" / "qrels.jsonl"),
        },
        "systems": systems,
        "evaluation": {"max_questions": 12, "max_workers": 4, "system_workers": 4, "latency_probe_questions": 2},
    }
    path = work / "live.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(config: Path, server: FakeOpenAIServer, work: Path) -> tuple[Path, dict]:
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-local-fake-key-not-real",
        "OPENAI_BASE_URL": server.base_url,
        "RAGBENCH_CACHE_DIR": str(work / "cache"),
    }
    server.reset()
    done = subprocess.run(
        [sys.executable, "-m", "ragbench.cli", "compare", "--config", str(config)], env=env, cwd=work, capture_output=True, text=True, timeout=240
    )
    assert done.returncode == 0, f"{done.stdout[-1500:]}\n{done.stderr[-1500:]}"
    out = sorted((work / "results").iterdir())[-1]
    return out, json.loads((out / "run_summary.json").read_text())


def test_live_run_with_parallel_systems_retries_caches_and_probes(tmp_path):
    config = _config(tmp_path)
    with FakeOpenAIServer() as server:
        out1, run1 = _run(config, server, tmp_path)
        cold = dict(server.stats)
        out2, run2 = _run(config, server, tmp_path)
        warm = dict(server.stats)

    # Real SDK traffic reached the local server, errors included: the 429s injected on the way were retried.
    assert run1["mode"] == "live" and run1["num_errors"] == 0 and run2["num_errors"] == 0, "worker threads must not hit import races or lose requests"
    assert cold["/v1/chat/completions"] > 50 and cold["/v1/embeddings"] > 5 and cold["429_injected"] > 0
    assert cold["max_completion_tokens_sent"] > 0 and cold.get("max_tokens_sent", 0) == 0
    summary = pd.read_csv(out1 / "metrics_summary.csv").set_index("system")
    assert list(summary.index) == ["bm25", "vector", "hybrid_rerank", "hyde"] and (summary["n_ok"] == 12).all()
    assert (summary["answer_score"] > 3.9).all(), "the fake judge's scores should flow through to the leaderboard"

    # The second run is served from the disk cache: paid LLM calls collapse to the latency probe's.
    assert warm["/v1/chat/completions"] < 0.2 * cold["/v1/chat/completions"]
    cache1, cache2 = run1["cache"], run2["cache"]
    assert cache1["enabled"] and cache2["hit_rate"] > 0.9
    assert cache2["charged_cost_usd"] == pytest.approx(cache1["charged_cost_usd"], rel=0.05)  # standalone charging, not cache-dependent
    assert cache2["real_spend_usd"] < 0.6 * cache2["charged_cost_usd"]
    # BM25 is deterministic (Chroma's ANN index is not, across processes), so it must be identical run to run.
    second = pd.read_csv(out2 / "metrics_summary.csv").set_index("system")
    cols = ["retrieval_recall@5", "retrieval_mrr@10", "answer_score", "faithfulness", "avg_cost_per_question"]
    pd.testing.assert_series_equal(summary.loc["bm25", cols], second.loc["bm25", cols])

    # Live runs get a clean one-at-a-time latency measurement.
    assert (summary["latency_source"] == "probe").all() and run1["execution"]["probe_calls"] == 8
    assert run1["execution"]["system_workers"] == 4 and run1["cache"]["real_spend_usd"] > 0
