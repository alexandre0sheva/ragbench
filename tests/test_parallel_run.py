from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fake_systems import RankedFakeSystem, question, register_fakes, write_experiment
from typer.testing import CliRunner

from ragbench.cache import active_cache
from ragbench.cli import app
from ragbench.config.schema import EvaluationConfig, SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import BenchmarkEvaluator, BenchmarkRunError, ProgressListener, run_benchmark
from ragbench.rag_systems import SYSTEM_REGISTRY, create_rag_system
from ragbench.rag_systems.base import AnswerResult, IngestionResult
from ragbench.runtime import SerializedProgress, current_runtime, evenly_sample

ROOT = Path(__file__).resolve().parents[1]
# `pareto_optimal` is decided partly by measured latency, so it varies with timing like the latency columns do.
TIMING_COLUMNS = {"avg_latency_ms", "avg_latency_concurrent_ms", "system_wall_time_ms", "latency_ms_p50", "latency_ms_p95", "pareto_optimal"}


# --- config ----------------------------------------------------------------------------------


def test_new_evaluation_knobs_have_safe_defaults_and_validation():
    cfg = EvaluationConfig()
    assert (cfg.max_workers, cfg.system_workers, cfg.ingest_workers, cfg.latency_probe_questions) == (4, 1, 4, 5)
    for bad in ({"system_workers": 0}, {"ingest_workers": 0}, {"latency_probe_questions": -1}):
        with pytest.raises(ValueError):
            EvaluationConfig(**bad)


def test_evenly_sample_spreads_picks_across_the_list():
    assert evenly_sample(list(range(10)), 3) in ([0, 4, 9], [0, 5, 9])
    assert evenly_sample(list(range(10)), 1) == [0]
    assert evenly_sample(list(range(3)), 10) == [0, 1, 2]
    assert evenly_sample([], 5) == [] and evenly_sample([1, 2], 0) == []
    picks = evenly_sample(list(range(50)), 5)
    assert picks == sorted(set(picks)) and picks[0] == 0 and picks[-1] == 49


# --- systems in parallel ---------------------------------------------------------------------


def _all_systems_config(tmp_path: Path, **evaluation: object) -> Path:
    demo = tmp_path / "demo"
    shutil.copytree(ROOT / "data" / "demo", demo)
    raw = (ROOT / "configs" / "all.yaml").read_text(encoding="utf-8")
    text = raw.replace("data/demo", str(demo)).replace("output_dir: results", f"output_dir: {tmp_path / 'results'}")
    path = tmp_path / "all.yaml"
    path.write_text(text, encoding="utf-8")
    if evaluation:
        import yaml

        raw_cfg = yaml.safe_load(path.read_text())
        raw_cfg["evaluation"] = {**raw_cfg.get("evaluation", {}), **evaluation}
        path.write_text(yaml.safe_dump(raw_cfg))
    return path


def _workdir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    return path


def _rows(out: Path) -> list[dict]:
    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("latency_ms", None)
        for step in row["steps"]:
            step.pop("latency_ms", None)
    return rows


def test_parallel_systems_give_the_same_results_in_the_same_order_as_sequential(tmp_path):
    sequential = run_benchmark(_all_systems_config(_workdir(tmp_path, "a"), system_workers=1, max_workers=1), force_mock=True)
    parallel = run_benchmark(_all_systems_config(_workdir(tmp_path, "b"), system_workers=4, max_workers=2), force_mock=True)

    seq, par = pd.read_csv(sequential / "metrics_summary.csv"), pd.read_csv(parallel / "metrics_summary.csv")
    keep = [c for c in seq.columns if c not in TIMING_COLUMNS]
    pd.testing.assert_frame_equal(seq[keep], par[keep])  # same systems, same order, same numbers
    assert _rows(sequential) == _rows(parallel)  # every per-question row, answer, step and cost
    for name in ("retrieval_metrics.csv", "answer_metrics.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(sequential / name), pd.read_csv(parallel / name))
    cost_seq, cost_par = pd.read_csv(sequential / "cost_breakdown.csv"), pd.read_csv(parallel / "cost_breakdown.csv")
    pd.testing.assert_frame_equal(cost_seq.drop(columns=["latency_ms"]), cost_par.drop(columns=["latency_ms"]))
    assert json.loads((parallel / "run_summary.json").read_text())["execution"]["system_workers"] == 4


class BarrierSystem(RankedFakeSystem):
    """Every instance blocks in `ingest` until all of its siblings are also ingesting: only true parallelism gets through."""

    barrier = threading.Barrier(3, timeout=10)

    def ingest(self, documents: list[Document]) -> IngestionResult:
        type(self).barrier.wait()
        return super().ingest(documents)


def test_systems_really_ingest_concurrently_when_system_workers_allows_it(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_barrier", BarrierSystem)
    BarrierSystem.barrier = threading.Barrier(3, timeout=10)
    systems = [{"type": "fake_barrier", "name": f"s{i}", "retrieval": {"top_k": 3}} for i in range(3)]
    config = write_experiment(tmp_path, [question("q1", "What is topic 1?", ["doc_001"])], systems, {"system_workers": 3})

    out = run_benchmark(config, force_mock=True)

    names = pd.read_csv(out / "metrics_summary.csv")["system"].tolist()
    assert names == ["s0", "s1", "s2"]  # config order, whatever order they finished in

    # With one system worker the barrier can never be satisfied: proves the knob is what enables concurrency.
    BarrierSystem.barrier = threading.Barrier(3, timeout=0.3)
    sequential = write_experiment(_workdir(tmp_path, "seq"), [question("q1", "What is topic 1?", ["doc_001"])], systems, {"system_workers": 1})
    with pytest.raises(BenchmarkRunError):  # the broken barrier makes every ingestion fail, which fails the run
        run_benchmark(sequential, force_mock=True)


class OverlapDetectingProgress(ProgressListener):
    def __init__(self) -> None:
        self.inside = 0
        self.max_inside = 0
        self.events: list[str] = []

    def _enter(self, event: str) -> None:
        self.inside += 1
        self.max_inside = max(self.max_inside, self.inside)
        time.sleep(0.002)
        self.events.append(event)
        self.inside -= 1

    def run_started(self, num_systems, num_questions):
        self._enter("run")

    def system_started(self, name, index, total):
        self._enter(f"start:{name}")

    def ingestion_finished(self, name, num_chunks, latency_ms):
        self._enter(f"ingest:{name}")

    def question_finished(self, name, done, total):
        self._enter(f"q:{name}")

    def system_finished(self, name, wall_time_ms):
        self._enter(f"done:{name}")


def test_progress_callbacks_are_serialized_even_with_parallel_systems_and_questions(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    systems = [{"type": "fake_ranked", "name": f"s{i}", "retrieval": {"top_k": 3}} for i in range(4)]
    questions = [question(f"q{i}", f"What is topic {i}?", [f"doc_{i:03d}"]) for i in range(1, 7)]
    config = write_experiment(tmp_path, questions, systems, {"system_workers": 4, "max_workers": 3})
    progress = OverlapDetectingProgress()

    run_benchmark(config, force_mock=True, progress=progress)

    assert progress.max_inside == 1, "a listener was entered by two threads at once"
    assert sum(e.startswith("q:") for e in progress.events) == 4 * 6
    assert sum(e.startswith("done:") for e in progress.events) == 4


def test_serialized_progress_forwards_every_hook():
    calls: list[str] = []

    class Recorder(ProgressListener):
        def run_started(self, num_systems, num_questions):
            calls.append("run")

        def latency_probe(self, name, done, total):
            calls.append(f"probe:{name}:{done}/{total}")

    wrapped = SerializedProgress(Recorder())
    wrapped.run_started(2, 3)
    wrapped.latency_probe("a", 1, 5)
    wrapped.system_started("a", 1, 2)  # hooks the wrapped listener does not override are still callable
    assert calls == ["run", "probe:a:1/5"]


# --- latency probe ---------------------------------------------------------------------------


class ProbeSystem(RankedFakeSystem):
    """Records how it is called during the probe; latency grows with every call so percentiles are checkable."""

    calls: list[dict] = []
    lock = threading.Lock()
    active = 0
    peak = 0

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None) -> AnswerResult:
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
            number = len(cls.calls) + 1
            cls.calls.append({"question": question, "cache_active": active_cache() is not None, "workers": current_runtime().ingest_workers})
        time.sleep(0.003)
        try:
            result = super().answer_question(question, top_k=top_k, context_k=context_k)
            return result.model_copy(update={"latency_ms": 10.0 * number})
        finally:
            with cls.lock:
                cls.active -= 1


def _probe_run(tmp_path, monkeypatch, *, mode: str, probe_questions: int, use_cache: bool = False, workers: int = 4):
    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_probe", ProbeSystem)
    ProbeSystem.calls, ProbeSystem.active, ProbeSystem.peak = [], 0, 0
    questions = [question(f"q{i}", f"What is topic {i}?", [f"doc_{i:03d}"]) for i in range(1, 9)]
    config = write_experiment(
        tmp_path,
        questions,
        [{"type": "fake_probe", "name": "p", "retrieval": {"top_k": 3}}],
        {"max_workers": workers, "latency_probe_questions": probe_questions},
    )
    evaluator = BenchmarkEvaluator(config, force_mock=True, use_cache=use_cache)
    evaluator.mode = mode  # models stay mock (no key in tests); only the probe decision reads the mode
    return evaluator.run()


def test_probe_runs_sequentially_with_the_cache_bypassed_and_reports_clean_latency(tmp_path, monkeypatch):
    out = _probe_run(tmp_path, monkeypatch, mode="live", probe_questions=3)

    calls = ProbeSystem.calls
    assert len(calls) == 8 + 3  # the 8 real answers, then 3 probe answers
    probe_calls = calls[8:]
    assert ProbeSystem.peak > 1, "the main pass should have been concurrent"
    assert all(not call["cache_active"] for call in probe_calls)
    assert [c["question"] for c in probe_calls] == ["What is topic 1?", "What is topic 4?", "What is topic 8?"] or len({c["question"] for c in probe_calls}) == 3

    summary = pd.read_csv(out / "metrics_summary.csv").iloc[0]
    assert summary["latency_source"] == "probe"
    assert summary["latency_ms_p50"] == pytest.approx(100.0) and summary["latency_ms_p95"] == pytest.approx(109.0)  # probe latencies 90/100/110
    assert summary["avg_latency_ms"] == pytest.approx(100.0)  # leaderboard latency is the clean one
    assert summary["avg_latency_concurrent_ms"] == pytest.approx(45.0)  # mean of 10..80 from the concurrent pass is kept for reference

    # The probe never leaks into scores, rows, or per-question costs.
    assert summary["n_ok"] == 8 and len(pd.read_csv(out / "answer_metrics.csv")) == 8
    assert len(json.loads((out / "per_question_results.jsonl").read_text().splitlines()[0])["steps"]) >= 1
    run_summary = json.loads((out / "run_summary.json").read_text())
    assert run_summary["execution"]["latency_probe_questions"] == 3 and run_summary["execution"]["probe_calls"] == 3
    assert "p95" in (out / "leaderboard.md").read_text()


def test_probe_is_skipped_in_mock_mode_and_when_disabled(tmp_path, monkeypatch):
    mock = _probe_run(_workdir(tmp_path, "m"), monkeypatch, mode="mock", probe_questions=3)
    assert len(ProbeSystem.calls) == 8
    row = pd.read_csv(mock / "metrics_summary.csv").iloc[0]
    assert row["latency_source"] == "concurrent" and row["avg_latency_ms"] == pytest.approx(45.0)
    assert row["latency_ms_p50"] == pytest.approx(45.0) and row["latency_ms_p95"] == pytest.approx(10.0 * 7.65, rel=0.01) and "avg_latency_concurrent_ms" in row

    disabled = _probe_run(_workdir(tmp_path, "d"), monkeypatch, mode="live", probe_questions=0)
    assert len(ProbeSystem.calls) == 8
    assert pd.read_csv(disabled / "metrics_summary.csv").iloc[0]["latency_source"] == "concurrent"
    assert "Latency was measured while questions ran concurrently" in (disabled / "leaderboard.md").read_text()


def test_probe_with_fewer_questions_than_requested_and_a_failing_probe_call(tmp_path, monkeypatch):
    class FlakyProbe(ProbeSystem):
        def answer_question(self, question, top_k=None, context_k=None):
            result = super().answer_question(question, top_k, context_k)
            if len(ProbeSystem.calls) > 8 and len(ProbeSystem.calls) % 2 == 0:
                raise RuntimeError("probe hiccup")
            return result

    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_probe", FlakyProbe)
    ProbeSystem.calls, ProbeSystem.active, ProbeSystem.peak = [], 0, 0
    config = write_experiment(
        tmp_path,
        [question(f"q{i}", f"What is topic {i}?", [f"doc_{i:03d}"]) for i in range(1, 9)],
        [{"type": "fake_probe", "name": "p", "retrieval": {"top_k": 3}}],
        {"max_workers": 1, "latency_probe_questions": 50},
    )
    evaluator = BenchmarkEvaluator(config, force_mock=True, use_cache=False)
    evaluator.mode = "live"

    out = evaluator.run()  # a failing probe call must not fail the run

    row = pd.read_csv(out / "metrics_summary.csv").iloc[0]
    assert row["latency_source"] == "probe" and row["n_ok"] == 8
    assert json.loads((out / "run_summary.json").read_text())["execution"]["probe_calls"] < 8


def test_probe_cost_is_reported_as_real_spend_not_charged_to_systems(tmp_path, monkeypatch):
    out = _probe_run(tmp_path, monkeypatch, mode="live", probe_questions=2)
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["execution"]["probe_cost_usd"] == 0.0  # mock models are free; the field exists and feeds real_spend_usd
    assert "real_spend_usd" in summary["cache"]


# --- llm_heavy parallel ingestion ------------------------------------------------------------


class SlowEnrichmentLLM:
    """Counts concurrency of enrichment calls; answers deterministically from the prompt."""

    model_name = "mock-llm"

    def __init__(self) -> None:
        from ragbench.models.llms import MockLLM

        self.inner = MockLLM()
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def generate(self, messages, **kwargs):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.01)
        try:
            return self.inner.generate(messages, **kwargs)
        finally:
            with self.lock:
                self.now -= 1


def test_llm_heavy_enrichment_is_parallel_but_identical_to_sequential():
    from ragbench.runtime import RuntimeContext, activate_runtime

    docs = [Document(doc_id=f"doc_{i:03d}", path=f"{i}.md", title=f"T{i}", text=f"Topic {i} covers product {i} for underwriters and claims teams.") for i in range(1, 13)]
    cfg = SystemConfig(
        type="llm_heavy",
        name="heavy",
        chunker={"type": "token", "chunk_size": 6, "chunk_overlap": 0},
        retrieval={"vector_store": "in_memory"},
        llm_features={"enable_llm_ingestion": True, "enable_query_rewrite": False},
    )

    def ingest(workers: int):
        system = create_rag_system(cfg, force_mock=True)
        llm = SlowEnrichmentLLM()
        system.llm = llm  # type: ignore[assignment]
        with activate_runtime(RuntimeContext(ingest_workers=workers)):
            result = system.ingest(docs)
        return system, llm, result

    seq_system, seq_llm, seq_result = ingest(1)
    par_system, par_llm, par_result = ingest(4)

    assert seq_llm.peak == 1 and par_llm.peak > 1
    assert par_result.num_chunks == seq_result.num_chunks > 12
    assert [c.chunk_id for c in par_system.store.chunks] == [c.chunk_id for c in seq_system.store.chunks]
    assert [c.text for c in par_system.store.chunks] == [c.text for c in seq_system.store.chunks]  # enrichment landed on the right chunks


# --- CLI -------------------------------------------------------------------------------------


def test_cli_accepts_system_workers_and_reports_it(tmp_path, monkeypatch):
    register_fakes(monkeypatch)
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 1?", ["doc_001"])],
        [{"type": "fake_ranked", "name": f"s{i}", "retrieval": {"top_k": 3}} for i in range(2)],
    )

    result = CliRunner().invoke(app, ["run", "--config", str(config), "--mock", "--system-workers", "2", "--max-workers", "2"])

    assert result.exit_code == 0, result.output
    out = sorted((tmp_path / "results").iterdir())[-1]
    execution = json.loads((out / "run_summary.json").read_text())["execution"]
    assert (execution["system_workers"], execution["max_workers"]) == (2, 2)


# --- import warm-up (live-run thread-safety) ------------------------------------------------


def test_modules_to_warm_up_include_chromadb_only_when_a_system_uses_it(tmp_path, monkeypatch):
    register_fakes(monkeypatch)

    def modules(systems: list[dict]) -> list[str]:
        config = write_experiment(_workdir(tmp_path, f"w{len(list(tmp_path.iterdir()))}"), [question("q", "What is topic 1?", ["doc_001"])], systems)
        return BenchmarkEvaluator(config, force_mock=True)._modules_to_warm_up()

    assert modules([{"type": "bm25", "name": "a"}, {"type": "vector", "name": "b", "retrieval": {"vector_store": "in_memory"}}]) == ["httpx", "openai"]
    assert modules([{"type": "bm25", "name": "a"}, {"type": "hybrid", "name": "b"}]) == ["httpx", "openai"]  # hybrid defaults to numpy
    assert modules([{"type": "bm25", "name": "a"}, {"type": "hybrid", "name": "b", "retrieval": {"vector_store": "chroma"}}]) == ["httpx", "openai", "chromadb"]
    assert modules([{"type": "fake_ranked", "name": "a", "retrieval": {"top_k": 3}}]) == ["httpx", "openai"]  # spec-less custom systems


def test_imports_are_warmed_up_on_the_main_thread_before_any_worker_or_system_starts(tmp_path, monkeypatch):
    """Regression: OpenAI SDK 3.x peeks at `sys.modules['httpx']` per request; a worker thread doing the first import of
    httpx (via Chroma) at that moment made requests fail with "partially initialized module". Only live runs with
    parallel systems are exposed, so the ordering is pinned here rather than relying on a timing-dependent race."""
    events: list[tuple[str, str]] = []

    class RecordingSystem(RankedFakeSystem):
        def __init__(self, *args, **kwargs):
            events.append(("create", threading.current_thread().name))
            super().__init__(*args, **kwargs)

    register_fakes(monkeypatch)
    monkeypatch.setitem(SYSTEM_REGISTRY, "fake_recording", RecordingSystem)
    monkeypatch.setattr("ragbench.evaluation.evaluator.warm_up_imports", lambda modules: events.append(("warm:" + ",".join(modules), threading.current_thread().name)))
    systems = [{"type": "fake_recording", "name": f"s{i}", "retrieval": {"top_k": 3}} for i in range(3)]
    config = write_experiment(tmp_path, [question("q1", "What is topic 1?", ["doc_001"])], systems, {"system_workers": 3})

    run_benchmark(config, force_mock=True)

    assert events[0] == ("warm:httpx,openai", threading.main_thread().name)
    assert [kind for kind, _ in events[1:]] == ["create"] * 3
    assert all(thread != threading.main_thread().name for _, thread in events[1:]), "systems should be created on worker threads"


def test_warm_up_imports_reports_missing_optional_modules_without_failing():
    from ragbench.runtime import warm_up_imports

    assert warm_up_imports(["json", "definitely_not_an_installed_module_xyz"]) == ["definitely_not_an_installed_module_xyz"]
    import sys

    assert "httpx" in sys.modules or warm_up_imports(["httpx"]) == [] and "httpx" in sys.modules
