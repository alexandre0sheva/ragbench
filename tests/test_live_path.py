"""The live code path, end to end, in a *fresh interpreter*: real OpenAI SDK -> real HTTP -> a local fake server.

Mock-mode tests never touch the SDK, so problems that only appear with real clients and real threads (lazy imports
racing with requests, retry handling, caching of paid calls, the latency probe) are covered here. The fake key is
obviously not real and nothing leaves localhost.
"""

from __future__ import annotations

import importlib.util
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
# Chroma is an optional extra. When it is installed the run exercises it: worker threads importing it lazily next to live
# requests is what once broke the real SDK, so that path stays covered where it can be.
VECTOR_STORE = "chroma" if importlib.util.find_spec("chromadb") else "numpy"


def _config(work: Path) -> Path:
    shutil.copytree(ROOT / "data" / "demo", work / "demo")
    systems = [
        {"type": "bm25", "name": "bm25", "chunker": {"type": "token", "chunk_size": 450, "chunk_overlap": 70}, "retrieval": {"top_k": 5}},
        {"type": "vector", "name": "vector", "retrieval": {"vector_store": VECTOR_STORE, "top_k": 5}},
        {"type": "hybrid_rerank", "name": "hybrid_rerank", "retrieval": {"vector_store": VECTOR_STORE, "final_top_k": 5}},
        {"type": "hyde", "name": "hyde", "retrieval": {"vector_store": VECTOR_STORE, "top_k": 5}},
        # Ingestion-time LLM calls (one per chunk / per document) made from worker threads, retried through the injected 429s and disk-cached.
        {"type": "contextual", "name": "contextual", "retrieval": {"vector_store": VECTOR_STORE, "top_k": 5}},
        {"type": "hierarchical", "name": "hierarchical", "retrieval": {"docs_k": 3, "top_k": 5}},
    ]
    config = {
        "run": {"name": "live_path", "output_dir": str(work / "results")},
        "dataset": {
            "documents_path": str(work / "demo" / "docs"),
            "questions_path": str(work / "demo" / "questions.jsonl"),
            "qrels_path": str(work / "demo" / "qrels.jsonl"),
        },
        "systems": systems,
        "evaluation": {"max_questions": 12, "max_workers": 4, "system_workers": 4, "ingest_workers": 4, "latency_probe_questions": 2},
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
    assert list(summary.index) == ["bm25", "vector", "hybrid_rerank", "hyde", "contextual", "hierarchical"] and (summary["n_ok"] == 12).all()
    ingestion = pd.read_csv(out1 / "cost_breakdown.csv").query("stage == 'ingestion'").set_index("system")
    assert ingestion.loc["contextual", "llm_cost"] > 0 and ingestion.loc["hierarchical", "llm_cost"] > 0 and ingestion.loc["bm25", "llm_cost"] == 0
    assert (summary["answer_score"] > 3.9).all(), "the fake judge's scores should flow through to the leaderboard"
    # Metrics v2 over the real client path: the judge's replies are complete judgments (no heuristic stand-in), the deterministic and
    # context metrics are filled in, and the judge defaulting to the generators' own model is called out in both reports.
    assert (summary["judge_fallback_rate"] == 0).all() and summary["token_f1"].notna().all() and summary["context_recall"].notna().all()
    assert (summary["answer_score_ci_lo"] <= summary["answer_score_ci_hi"]).all() and (out1 / "significance.csv").stat().st_size > 0, "statistics ride along on live runs too"
    assert "Self-preference risk" in (out1 / "leaderboard.md").read_text() and "Self-preference risk" in (out1 / "report.html").read_text()
    assert json.loads((out1 / "recommendation.json").read_text())["winner"] in summary.index and (out1 / "winner.yaml").exists(), "a live run ends with a recommendation too"

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

    # The spending guard counted every charge made on those worker threads: its total is the run's charged cost.
    assert run1["budget"]["stopped"] is False and run1["budget"]["spent_usd"] == pytest.approx(run1["cache"]["charged_cost_usd"], rel=0.01)

    # Live runs get a clean one-at-a-time latency measurement.
    assert (summary["latency_source"] == "probe").all() and run1["execution"]["probe_calls"] == 12
    assert run1["execution"]["system_workers"] == 4 and run1["cache"]["real_spend_usd"] > 0


def test_live_run_through_an_openai_compatible_endpoint_needs_no_openai_key(tmp_path):
    """A config that uses only `providers:` endpoints is a live run even with no OPENAI_API_KEY / ANTHROPIC_API_KEY at all."""
    shutil.copytree(ROOT / "data" / "demo", tmp_path / "demo")
    models = {"generator": "openai_compatible:fake/chat-model", "embedding": "openai_compatible:fake/embed-model"}
    with FakeOpenAIServer() as server:
        config = {
            "run": {"name": "compat_live", "output_dir": str(tmp_path / "results")},
            "dataset": {
                "documents_path": str(tmp_path / "demo" / "docs"),
                "questions_path": str(tmp_path / "demo" / "questions.jsonl"),
                "qrels_path": str(tmp_path / "demo" / "qrels.jsonl"),
            },
            "providers": {"fake": {"base_url": server.base_url, "limits": {"max_concurrent_requests": 3}}},
            "systems": [
                {"type": "bm25", "name": "bm25", "models": models},
                {"type": "vector", "name": "vector", "models": models, "retrieval": {"vector_store": "numpy", "top_k": 5}},
            ],
            "evaluation": {
                "max_questions": 6,
                "max_workers": 4,
                "judge": {"model": "openai_compatible:fake/judge-model", "samples": 2, "temperature": 0.3},
                "latency_probe_questions": 1,
            },
            "pricing": {"openai_compatible:fake/chat-model": {"input": 1.0, "output": 2.0}},
        }
        path = tmp_path / "compat.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        env = {**os.environ, "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "RAGBENCH_CACHE_DIR": str(tmp_path / "cache")}
        env.pop("OPENAI_BASE_URL", None)
        done = subprocess.run(
            [sys.executable, "-m", "ragbench.cli", "compare", "--config", str(path)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=240
        )
        stats = dict(server.stats)

    assert done.returncode == 0, f"{done.stdout[-1500:]}\n{done.stderr[-1500:]}"
    out = sorted((tmp_path / "results").iterdir())[-1]
    run = json.loads((out / "run_summary.json").read_text())
    assert run["mode"] == "live" and run["num_errors"] == 0
    assert "openai_compatible:fake/chat-model" in run["models_used"] and run["unknown_priced_models"] == []
    assert stats["/v1/chat/completions"] > 6 and stats["/v1/embeddings"] > 0, "traffic must have reached the endpoint over real HTTP"
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert (summary["n_ok"] == 6).all() and (summary["answer_score"] > 3.9).all()
    assert (summary["avg_cost_per_question"] > 0).all(), "the `pricing:` override prices the endpoint model"
    assert stats["judge_calls"] == 2 * 6 * 2, "two systems x six questions x two judge samples, and no retries because every reply is a valid judgment"
    assert (summary["judge_fallback_rate"] == 0).all()
    assert "Self-preference" not in (out / "leaderboard.md").read_text(), "the judge is a different model than the generators"
    row = json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])
    assert row["answer_judge"]["metadata"]["samples"] == 2 and row["answer_judge"]["metadata"]["prompt_version"] == "v3"
    assert (summary["latency_source"] == "probe").all()


def test_live_run_with_parallel_systems_stops_cleanly_at_its_budget(tmp_path):
    """The spending cap on real threads: a cap smaller than any system's ingestion stops the run with a clear message, not a traceback."""
    config = _config(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["evaluation"]["max_cost_usd"] = 1e-9
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with FakeOpenAIServer() as server:
        env = {
            **os.environ,
            "OPENAI_API_KEY": "sk-local-fake-key-not-real",
            "OPENAI_BASE_URL": server.base_url,
            "RAGBENCH_CACHE_DIR": str(tmp_path / "cache"),
        }
        done = subprocess.run(
            [sys.executable, "-m", "ragbench.cli", "compare", "--config", str(config), "--yes"], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=240
        )
    output = done.stdout + done.stderr
    assert done.returncode == 1 and "Traceback" not in output, output[-1500:]
    assert "budget" in output.lower() and "max_cost_usd" in output
    out = sorted((tmp_path / "results").iterdir())[-1]
    budget = json.loads((out / "run_summary.json").read_text())["budget"]
    assert budget["stopped"] is True and budget["spent_usd"] > 0 and set(budget["incomplete_systems"]) >= {"vector", "hybrid_rerank"}


def test_live_label_free_run_judges_answers_and_leaves_retrieval_blank(tmp_path):
    """Label-free questions over the real client path: no labels anywhere, half without a reference answer, and the judge is told so."""
    shutil.copytree(ROOT / "data" / "demo", tmp_path / "demo")
    demo = [json.loads(line) for line in (tmp_path / "demo" / "questions.jsonl").read_text().splitlines()]
    answerable = [row for row in demo if row["relevant_doc_ids"]][:6]
    rows = [{"id": row["id"], "question": row["question"], **({"reference_answer": row["reference_answer"]} if number % 2 == 0 else {})} for number, row in enumerate(answerable)]
    (tmp_path / "free.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    config = {
        "run": {"name": "label_free_live", "output_dir": str(tmp_path / "results")},
        "dataset": {"documents_path": str(tmp_path / "demo" / "docs"), "questions_path": str(tmp_path / "free.jsonl")},
        "systems": [
            {"type": "bm25", "name": "bm25", "retrieval": {"top_k": 5}},
            {"type": "vector", "name": "vector", "retrieval": {"vector_store": "numpy", "top_k": 5}},
        ],
        "evaluation": {"max_workers": 4, "latency_probe_questions": 0},
    }
    path = tmp_path / "free.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with FakeOpenAIServer() as server:
        out, run = _run(path, server, tmp_path)
        stats = dict(server.stats)

    assert run["mode"] == "live" and run["num_errors"] == 0
    assert {key: run["dataset"][key] for key in ("questions", "labeled_questions", "label_free")} == {"questions": 6, "labeled_questions": 0, "label_free": True}
    summary = pd.read_csv(out / "metrics_summary.csv").set_index("system")
    assert summary["retrieval_recall@5"].isna().all() and (summary["n_ok"] == 6).all()
    assert (summary["answer_score"] > 3.9).all() and (summary["judge_fallback_rate"] == 0).all()
    assert stats["judge_calls"] == 12 and stats["judge_calls_reference_free"] == 6, "the three questions without a reference reach the judge as answerable with a null reference"
    assert json.loads((out / "recommendation.json").read_text())["winner"] in summary.index, "a label-free run still ends with a recommendation"


def test_live_generate_questions_and_label_use_the_real_client_and_the_disk_cache(tmp_path):
    """`generate-questions` and `label` over real HTTP: valid questions, grades through the cache, and a re-run that costs (almost) nothing."""
    from ragbench.evaluation.evaluator import run_benchmark

    shutil.copytree(ROOT / "data" / "demo", tmp_path / "demo")
    docs, questions = tmp_path / "demo" / "docs", tmp_path / "questions.jsonl"
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-local-fake-key-not-real",
        "OPENAI_BASE_URL": "",
        "RAGBENCH_CACHE_DIR": str(tmp_path / "cache"),
        "CI": "",
    }

    def cli(server: FakeOpenAIServer, *args: str) -> subprocess.CompletedProcess[str]:
        server.reset()
        done = subprocess.run(
            [sys.executable, "-m", "ragbench.cli", *args, "--yes"],
            env={**env, "OPENAI_BASE_URL": server.base_url},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert done.returncode == 0, f"{done.stdout[-1500:]}\n{done.stderr[-1500:]}"
        return done

    with FakeOpenAIServer() as server:
        cli(server, "generate-questions", "--docs", str(docs), "--out", str(questions), "--n", "16", "--seed", "3")
        cold_generation = dict(server.stats)
        rows = [json.loads(line) for line in questions.read_text().splitlines()]
        assert len(rows) == 16 and all(row["metadata"]["needs_review"] and "mock" not in row["metadata"] for row in rows)
        assert {row["metadata"]["generator"] for row in rows} == {"gpt-6-luna"}  # a live run, not the template generator
        assert cold_generation["/v1/chat/completions"] >= 16

        cli(server, "generate-questions", "--docs", str(docs), "--out", str(questions), "--n", "16", "--seed", "3", "--force")
        assert server.stats["/v1/chat/completions"] < 0.2 * cold_generation["/v1/chat/completions"], "the same request is served from the disk cache"
        assert [json.loads(line) for line in questions.read_text().splitlines()] == rows

        # A finished run to label (mock models: the point here is the grading calls).
        config = {
            "run": {"name": "to_label", "output_dir": str(tmp_path / "results")},
            "dataset": {"documents_path": str(docs), "questions_path": str(questions)},
            "systems": [{"type": "bm25", "name": "bm25"}, {"type": "vector", "name": "vector", "retrieval": {"vector_store": "numpy"}}],
            "evaluation": {"max_questions": 5, "max_workers": 2, "latency_probe_questions": 0},
        }
        (tmp_path / "to_label.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        run_dir = run_benchmark(tmp_path / "to_label.yaml", force_mock=True, max_workers=2)

        cli(server, "label", "--run", str(run_dir), "--top-k", "3", "--apply")
        cold_label = dict(server.stats)
        proposed = [json.loads(line) for line in (run_dir / "qrels.proposed.jsonl").read_text().splitlines()]
        assert proposed and all(row["relevance"] in (0, 1, 2, 3) for row in proposed)
        assert cold_label["/v1/chat/completions"] >= len(proposed) and "> **Mock grader.**" not in (run_dir / "qrels_review.md").read_text()
        assert (run_dir / "qrels.merged.jsonl").exists()

        cli(server, "label", "--run", str(run_dir), "--top-k", "3")
        assert server.stats["/v1/chat/completions"] < 0.2 * cold_label["/v1/chat/completions"], "grades are cached"


def test_live_auto_run_checkpoints_on_worker_threads_and_resume_runs_only_the_missing_system(tmp_path):
    """`ragbench auto` over real HTTP with systems on parallel threads: checkpoints written from those threads, then a resume (cache off, so
    restoration and not caching is what saves the calls) that re-runs the one system whose checkpoint is gone."""
    from ragbench.evaluation.checkpoint import CheckpointStore

    shutil.copytree(ROOT / "data" / "demo" / "docs", tmp_path / "docs")
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "run": {"name": "base", "output_dir": str(tmp_path / "unused")},
                "dataset": {"documents_path": str(tmp_path / "docs"), "questions_path": str(tmp_path / "unused.jsonl")},
                "systems": [{"type": "bm25", "name": "bm25"}],
                "evaluation": {"max_workers": 3, "system_workers": 3, "latency_probe_questions": 2},
            }
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "OPENAI_API_KEY": "sk-local-fake-key-not-real", "RAGBENCH_CACHE_DIR": str(tmp_path / "cache"), "CI": ""}

    def cli(server: FakeOpenAIServer, *args: str) -> None:
        server.reset()
        done = subprocess.run(
            [sys.executable, "-m", "ragbench.cli", "auto", *args, "--yes", "--no-cache"],
            env={**env, "OPENAI_BASE_URL": server.base_url},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert done.returncode == 0, f"{done.stdout[-1500:]}\n{done.stderr[-1500:]}"

    with FakeOpenAIServer() as server:
        cli(server, "--docs", str(tmp_path / "docs"), "--config", str(tmp_path / "base.yaml"), "--preset", "quick", "--n-questions", "8", "--output-dir", str(tmp_path / "results"))
        cold = dict(server.stats)
        (run_dir,) = [path for path in (tmp_path / "results").iterdir() if path.is_dir()]
        store = CheckpointStore(run_dir)
        assert all(store.path_for(name).exists() for name in ("bm25", "vector", "hybrid_rerank"))
        assert json.loads((run_dir / "run_summary.json").read_text())["mode"] == "live" and json.loads((run_dir / "run_summary.json").read_text())["num_errors"] == 0
        assert cold["/v1/embeddings"] > 0 and cold["/v1/chat/completions"] > 16
        kept = {name: store.path_for(name).stat().st_mtime_ns for name in ("bm25", "hybrid_rerank")}
        store.path_for("vector").unlink()

        cli(server, "--resume", str(run_dir))
        resumed = dict(server.stats)

    assert {name: store.path_for(name).stat().st_mtime_ns for name in kept} == kept and store.path_for("vector").exists()
    assert resumed["/v1/chat/completions"] < 0.6 * cold["/v1/chat/completions"], "two of the three systems (and the question writing) were not paid for again"
    assert resumed["/v1/embeddings"] < cold["/v1/embeddings"]
    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    assert list(summary.index) == ["bm25", "vector", "hybrid_rerank"] and (summary["n_ok"] == 8).all()
    assert (summary["latency_source"] == "probe").all(), "restored systems keep the latency probe of the attempt that measured them"
    assert json.loads((run_dir / "recommendation.json").read_text())["winner"] in summary.index
