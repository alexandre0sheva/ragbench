from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import yaml
from fake_systems import RankedFakeSystem, question, register_fakes, write_experiment
from typer.testing import CliRunner

from ragbench.cache import CacheRuntime, DiskCache, activate_cache, active_cache, cache_key
from ragbench.cache import store as store_module
from ragbench.cli import app
from ragbench.config.schema import CacheConfig, SystemConfig
from ragbench.evaluation.evaluator import BenchmarkEvaluator, run_benchmark
from ragbench.models import cost
from ragbench.models.cached import CachedLLM
from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.embeddings import (
    EMBEDDING_CACHE,
    CachedEmbeddingModel,
    EmbeddingModel,
    EmbeddingResult,
    HashingEmbeddingModel,
)
from ragbench.models.llms import LLM, LLMResult, MockLLM, create_llm
from ragbench.rag_systems.trace import Tracer
from ragbench.utils.text import estimate_tokens

# --- keys ------------------------------------------------------------------------------------


def test_cache_key_is_canonical_and_namespaced():
    a = cache_key("llm", model="m", params={"temperature": 0, "max_tokens": 5}, messages=[{"role": "user", "content": "hi"}])
    b = cache_key("llm", messages=[{"content": "hi", "role": "user"}], params={"max_tokens": 5, "temperature": 0}, model="m")
    assert a == b and len(a) == 64
    assert a != cache_key("embeddings", model="m", params={"temperature": 0, "max_tokens": 5}, messages=[{"role": "user", "content": "hi"}])
    assert a != cache_key("llm", model="m2", params={"temperature": 0, "max_tokens": 5}, messages=[{"role": "user", "content": "hi"}])
    assert a != cache_key("llm", model="m", params={"temperature": 0, "max_tokens": 6}, messages=[{"role": "user", "content": "hi"}])


# --- DiskCache -------------------------------------------------------------------------------


def test_put_get_roundtrip_persists_across_instances_and_counts_stats(tmp_path):
    path = tmp_path / "c" / "cache.sqlite3"
    cache = DiskCache(path)
    assert cache.get("llm", "k1") is None
    cache.put("llm", "k1", b"value-1", meta={"model": "m"})
    cache.put("embeddings", "k1", b"other-namespace")
    assert cache.get("llm", "k1") == b"value-1" and cache.get("embeddings", "k1") == b"other-namespace"
    cache.record_saved("llm", 0.25)
    stats = cache.stats()
    assert (stats["hits"], stats["misses"]) == (2, 1)
    assert stats["saved_cost_usd"] == pytest.approx(0.25) and stats["size_bytes"] > 0
    assert stats["by_namespace"]["llm"] == {"hits": 1, "misses": 1, "saved_cost_usd": 0.25}
    cache.close()

    reopened = DiskCache(path)
    assert reopened.get("llm", "k1") == b"value-1"
    assert reopened.stats()["hits"] == 1  # hit counters are per instance (per run), entries persist
    assert reopened.describe() == {"llm": {"entries": 1, "bytes": 7}, "embeddings": {"entries": 1, "bytes": 15}}


def test_batch_helpers_and_overwrite(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite3")
    cache.put_many("embeddings", [(f"k{i}", f"v{i}".encode()) for i in range(1200)])  # larger than one SQL variable batch
    found = cache.get_many("embeddings", [f"k{i}" for i in range(1200)] + ["missing"])
    assert len(found) == 1200 and found["k1199"] == b"v1199" and "missing" not in found
    assert cache.stats()["by_namespace"]["embeddings"]["hits"] == 1200
    cache.put("embeddings", "k0", b"new")
    assert cache.get("embeddings", "k0") == b"new"


def test_disabled_cache_is_inert_and_creates_nothing(tmp_path):
    path = tmp_path / "never" / "cache.sqlite3"
    cache = DiskCache(path, enabled=False)
    cache.put("llm", "k", b"v")
    assert cache.get("llm", "k") is None and cache.get_many("llm", ["k"]) == {}
    assert not path.exists() and not path.parent.exists()


def test_ttl_expires_old_entries(tmp_path, monkeypatch):
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(store_module.time, "time", lambda: clock["now"])
    cache = DiskCache(tmp_path / "c.sqlite3", ttl_days=2)
    cache.put("llm", "k", b"v")
    clock["now"] += 86_400  # one day later: still fresh
    assert cache.get("llm", "k") == b"v"
    clock["now"] += 2 * 86_400  # three days: expired
    assert cache.get("llm", "k") is None
    assert cache.get_many("llm", ["k"]) == {}


def test_clear_by_namespace_or_everything(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite3")
    cache.put("llm", "a", b"1")
    cache.put("llm", "b", b"1")
    cache.put("embeddings", "a", b"1")
    assert cache.clear("llm") == 2
    assert cache.get("llm", "a") is None and cache.get("embeddings", "a") == b"1"
    assert cache.clear() == 1
    assert cache.describe() == {}


def test_corrupted_database_is_a_miss_with_one_warning_not_a_crash(tmp_path, caplog):
    path = tmp_path / "cache.sqlite3"
    path.write_bytes(b"this is definitely not a sqlite database" * 50)
    cache = DiskCache(path)

    with caplog.at_level(logging.WARNING):
        assert cache.get("llm", "k") is None
        cache.put("llm", "k", b"v")  # must not raise
        assert cache.get("llm", "k") is None
        assert cache.get_many("llm", ["k"]) == {}
    assert sum("disk cache" in record.getMessage().lower() for record in caplog.records) == 1
    assert cache.stats()["misses"] >= 2


def test_concurrent_writers_and_readers_do_not_corrupt_or_deadlock(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite3")

    def work(worker: int) -> int:
        ok = 0
        for i in range(60):
            key = f"w{worker}-{i}"
            cache.put("llm", key, key.encode())
            ok += cache.get("llm", key) == key.encode()
        return ok

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(work, range(16))) == 16 * 60
    assert cache.describe()["llm"]["entries"] == 16 * 60
    # A second connection (another process, in practice) sees the committed data.
    assert sqlite3.connect(tmp_path / "c.sqlite3").execute("select count(*) from entries").fetchone()[0] == 16 * 60


def test_runtime_activation_is_global_across_threads_and_restores(tmp_path):
    cache = DiskCache(tmp_path / "c.sqlite3")
    runtime = CacheRuntime(disk=cache, config=CacheConfig())
    assert active_cache() is None
    with activate_cache(runtime):
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert pool.submit(active_cache).result() is runtime  # worker threads see it (unlike a ContextVar)
    assert active_cache() is None


# --- CachedLLM -------------------------------------------------------------------------------


class FakeLLM(LLM):
    """A 'paid' model with a real price, that counts calls and takes measurable time."""

    model_name = "fake-paid-model"

    def __init__(self, delay: float = 0.0):
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.delay = delay

    def generate(self, messages, **kwargs) -> LLMResult:
        self.calls.append((messages, kwargs))
        time.sleep(self.delay)
        text = f"answer #{len(self.calls)}"
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=100,
            completion_tokens=20,
            cost=CostBreakdown(llm_prompt_tokens=100, llm_completion_tokens=20, llm_cost=estimate_model_cost(self.model_name, 100, 20)),
        )


@pytest.fixture()
def priced_model():
    cost.register_pricing({"fake-paid-model": {"input": 1000.0, "output": 2000.0}})  # $0.14 for 100 in + 20 out
    yield "fake-paid-model"
    cost.clear_pricing_overrides()


@pytest.fixture()
def runtime(tmp_path):
    rt = CacheRuntime(disk=DiskCache(tmp_path / "cache.sqlite3"), config=CacheConfig())
    with activate_cache(rt):
        yield rt


MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "what is 2+2?"}]


class ToolCallingLLM(FakeLLM):
    provider = "fake-provider"

    def generate(self, messages, **kwargs) -> LLMResult:
        from ragbench.models.llms import ToolCall

        result = super().generate(messages, **kwargs)
        result.tool_calls = [ToolCall(id="call_1", name="search", arguments={"query": "x"})]
        result.finish_reason = "tool_calls"
        return result


def test_tool_calls_and_finish_reason_survive_the_cache_and_tools_are_part_of_the_key(priced_model, runtime):
    tool = {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}
    inner = ToolCallingLLM()
    llm = CachedLLM(inner)

    first = llm.generate(MESSAGES, temperature=0, tools=[tool])
    second = llm.generate(MESSAGES, temperature=0, tools=[tool])
    without_tools = llm.generate(MESSAGES, temperature=0)

    assert len(inner.calls) == 2, "the tools list must change the key"
    assert second.cached and second.tool_calls == first.tool_calls and second.finish_reason == "tool_calls"
    assert not without_tools.cached


def test_the_provider_is_part_of_the_cache_key(priced_model, runtime):
    class Other(FakeLLM):
        provider = "other-provider"

    a, b = FakeLLM(), Other()
    CachedLLM(a).generate(MESSAGES, temperature=0)
    hit = CachedLLM(b).generate(MESSAGES, temperature=0)

    assert not hit.cached and len(b.calls) == 1
    assert CachedLLM(b).provider == "other-provider"


def test_identical_request_is_served_from_cache_with_standalone_cost(priced_model, runtime):
    inner = FakeLLM()
    llm = CachedLLM(inner)

    first = llm.generate(MESSAGES, temperature=0)
    second = llm.generate(MESSAGES, temperature=0)

    assert len(inner.calls) == 1
    assert second.text == first.text == "answer #1"
    assert first.cost.total_cost == pytest.approx(0.14)
    assert second.cost.total_cost == pytest.approx(first.cost.total_cost)  # charged as if it had been paid
    assert (second.prompt_tokens, second.completion_tokens) == (100, 20)
    assert not first.cached and second.cached
    stats = runtime.disk.stats()
    assert stats["hits"] == 1 and stats["saved_cost_usd"] == pytest.approx(0.14)
    assert stats["by_namespace"]["llm"]["saved_cost_usd"] == pytest.approx(0.14)


def test_hit_is_charged_at_the_current_price_not_the_price_when_cached(priced_model, runtime):
    llm = CachedLLM(FakeLLM())
    llm.generate(MESSAGES, temperature=0)
    cost.register_pricing({"fake-paid-model": {"input": 2000.0, "output": 4000.0}})

    assert llm.generate(MESSAGES, temperature=0).cost.total_cost == pytest.approx(0.28)


def test_any_change_to_the_request_is_a_miss(priced_model, runtime):
    inner = FakeLLM()
    llm = CachedLLM(inner)
    llm.generate(MESSAGES, temperature=0)
    llm.generate(MESSAGES, temperature=0, json_mode=True)
    llm.generate(MESSAGES, temperature=0, max_tokens=50)
    llm.generate([*MESSAGES, {"role": "user", "content": "again"}], temperature=0)
    llm.generate(MESSAGES)  # temperature omitted means 0 and is the same request as the first
    assert len(inner.calls) == 4
    # A different model name is a different entry.
    other = FakeLLM()
    other.model_name = "fake-paid-model-2"
    CachedLLM(other).generate(MESSAGES, temperature=0)
    assert len(other.calls) == 1


def test_nonzero_temperature_is_never_cached_unless_opted_in(priced_model, tmp_path):
    inner = FakeLLM()
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "a.sqlite3"), CacheConfig())):
        llm = CachedLLM(inner)
        llm.generate(MESSAGES, temperature=0.7)
        llm.generate(MESSAGES, temperature=0.7)
    assert len(inner.calls) == 2
    opted_in = FakeLLM()
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "b.sqlite3"), CacheConfig(llm_nonzero_temperature=True))):
        llm = CachedLLM(opted_in)
        llm.generate(MESSAGES, temperature=0.7)
        llm.generate(MESSAGES, temperature=0.7)
    assert len(opted_in.calls) == 1


def test_no_active_runtime_or_disabled_llm_caching_means_passthrough(priced_model, tmp_path):
    inner = FakeLLM()
    llm = CachedLLM(inner)
    llm.generate(MESSAGES, temperature=0)
    llm.generate(MESSAGES, temperature=0)
    assert len(inner.calls) == 2  # nothing active

    with activate_cache(CacheRuntime(DiskCache(tmp_path / "c.sqlite3"), CacheConfig(llm=False))):
        llm.generate(MESSAGES, temperature=0)
        llm.generate(MESSAGES, temperature=0)
    assert len(inner.calls) == 4


def test_inner_errors_are_not_cached(priced_model, runtime):
    class Flaky(FakeLLM):
        def generate(self, messages, **kwargs):
            if not self.calls:
                self.calls.append(1)  # type: ignore[arg-type]
                raise RuntimeError("provider down")
            return super().generate(messages, **kwargs)

    llm = CachedLLM(Flaky())
    with pytest.raises(RuntimeError):
        llm.generate(MESSAGES, temperature=0)
    assert llm.generate(MESSAGES, temperature=0).text.startswith("answer")


def test_mock_llm_is_never_wrapped_and_a_real_one_is(monkeypatch):
    assert isinstance(create_llm("gpt-x", force_mock=True), MockLLM)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import ragbench.models.providers.openai as openai_provider

    monkeypatch.setattr(openai_provider, "OpenAILLM", lambda model_name: FakeLLM())
    wrapped = create_llm("gpt-x")
    assert isinstance(wrapped, CachedLLM) and wrapped.model_name == "fake-paid-model"


def test_cached_hit_replays_the_original_latency(priced_model, runtime):
    llm = CachedLLM(FakeLLM(delay=0.03))
    first = llm.generate(MESSAGES, temperature=0)
    assert first.latency_ms is not None and first.latency_ms >= 25

    started = time.perf_counter()
    second = llm.generate(MESSAGES, temperature=0)
    real_elapsed_ms = (time.perf_counter() - started) * 1000

    assert real_elapsed_ms < 15 and second.latency_ms == pytest.approx(first.latency_ms)


def test_steps_and_answer_latency_include_replayed_llm_latency(priced_model, runtime, monkeypatch):
    register_fakes(monkeypatch)
    system = RankedFakeSystem(SystemConfig(type="fake_ranked", name="fake", retrieval={"top_k": 3}), force_mock=True)
    system.llm = CachedLLM(FakeLLM(delay=0.04))

    first = system.answer_question("What is topic 1?")
    second = system.answer_question("What is topic 1?")

    generate = next(s for s in second.steps if s.kind == "generate")
    assert generate.metadata["cached"] is True and generate.latency_ms >= 35
    assert second.latency_ms >= 35 and second.latency_ms == pytest.approx(first.latency_ms, rel=0.5)
    assert not next(s for s in first.steps if s.kind == "generate").metadata.get("cached")


def test_tracer_credit_makes_up_the_difference_between_elapsed_and_replayed_time():
    tracer = Tracer()
    result = LLMResult(text="x", model="m", prompt_tokens=1, completion_tokens=1, cost=CostBreakdown(), latency_ms=80.0, cached=True)
    with tracer.step("llm", "rewrite") as step:
        step.set_llm(result)
    assert tracer.steps[0].latency_ms == pytest.approx(80.0) and tracer.steps[0].metadata["cached"] is True
    assert tracer.latency_credit_ms == pytest.approx(80.0, abs=5)


# --- embeddings ------------------------------------------------------------------------------


class FakePaidEmbedding(EmbeddingModel):
    model_name = "fake-embedding"

    def __init__(self) -> None:
        self.doc_calls = 0
        self.query_calls = 0
        self.inner = HashingEmbeddingModel()

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        self.doc_calls += 1
        vectors = self.inner.embed_texts(texts).vectors
        tokens = sum(estimate_tokens(t, self.model_name) for t in texts)  # same counting the cache uses for hits
        price = estimate_model_cost(self.model_name, input_tokens=tokens)
        return EmbeddingResult(vectors, self.model_name, tokens, CostBreakdown(embedding_input_tokens=tokens, embedding_cost=price))

    def embed_query(self, text: str) -> EmbeddingResult:
        self.query_calls += 1
        return self.embed_texts([text])


@pytest.fixture()
def embedding_price():
    cost.register_pricing({"fake-embedding": {"input": 1_000_000.0}})  # $1 per token
    EMBEDDING_CACHE.clear()
    yield
    cost.clear_pricing_overrides()
    EMBEDDING_CACHE.clear()


TEXTS = ["alpha beta gamma", "delta epsilon", "zeta"]


def test_corpus_embeddings_survive_across_runs_via_the_disk_tier(embedding_price, runtime):
    inner = FakePaidEmbedding()
    model = CachedEmbeddingModel(inner)
    first = model.embed_texts(TEXTS)
    assert inner.doc_calls == 1

    EMBEDDING_CACHE.clear()  # a new process: the in-memory tier is empty, the disk tier is not
    second = model.embed_texts(TEXTS)

    assert inner.doc_calls == 1
    assert np.allclose(first.vectors, second.vectors) and second.vectors.dtype == np.float32
    assert first.cost.embedding_cost > 0
    assert second.cost.embedding_cost == pytest.approx(first.cost.embedding_cost)  # still charged at standalone price
    stats = runtime.disk.stats()["by_namespace"]["embeddings"]
    assert stats["hits"] == 3 and stats["saved_cost_usd"] == pytest.approx(first.cost.embedding_cost)


def test_partial_disk_hits_only_embed_the_missing_texts(embedding_price, runtime):
    inner = FakePaidEmbedding()
    model = CachedEmbeddingModel(inner)
    model.embed_texts(TEXTS[:2])
    EMBEDDING_CACHE.clear()

    result = model.embed_texts(TEXTS)

    assert inner.doc_calls == 2 and result.vectors.shape[0] == 3
    assert np.allclose(result.vectors, HashingEmbeddingModel().embed_texts(TEXTS).vectors)


def test_query_embeddings_are_not_cached_by_default_but_can_be(embedding_price, tmp_path):
    inner = FakePaidEmbedding()
    model = CachedEmbeddingModel(inner)
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "a.sqlite3"), CacheConfig())):
        model.embed_query("what is alpha?")
        model.embed_query("what is alpha?")
    assert inner.query_calls == 2  # latency stays honest

    inner2 = FakePaidEmbedding()
    model2 = CachedEmbeddingModel(inner2)
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "b.sqlite3"), CacheConfig(cache_query_embeddings=True))):
        first = model2.embed_query("what is alpha?")
        second = model2.embed_query("what is alpha?")
    assert inner2.query_calls == 1 and np.allclose(first.vectors, second.vectors) and second.vectors.shape == (1, 384)
    assert first.cost.embedding_cost > 0
    assert second.cost.embedding_cost == pytest.approx(first.cost.embedding_cost)


def test_disk_tier_can_be_switched_off_and_mock_embeddings_never_hit_the_disk(embedding_price, tmp_path):
    inner = FakePaidEmbedding()
    with activate_cache(CacheRuntime(DiskCache(tmp_path / "a.sqlite3"), CacheConfig(embeddings=False))) as rt:
        model = CachedEmbeddingModel(inner)
        model.embed_texts(TEXTS)
        EMBEDDING_CACHE.clear()
        model.embed_texts(TEXTS)
        assert inner.doc_calls == 2 and rt.disk.describe() == {}

    with activate_cache(CacheRuntime(DiskCache(tmp_path / "b.sqlite3"), CacheConfig())) as rt2:
        CachedEmbeddingModel(HashingEmbeddingModel()).embed_texts(TEXTS)
        assert rt2.disk.describe() == {}  # free to recompute, so not worth persisting


# --- evaluator / CLI -------------------------------------------------------------------------


def _experiment(tmp_path, extra=None):
    return write_experiment(
        tmp_path,
        [question("q1", "What is topic 2?", ["doc_002"])],
        [{"type": "bm25", "name": "bm25"}],
        extra=extra,
    )


def test_mock_runs_bypass_the_disk_cache_and_say_why(tmp_path, monkeypatch):
    import json

    cache_dir = tmp_path / "cachedir"
    monkeypatch.setenv("RAGBENCH_CACHE_DIR", str(cache_dir))

    out = run_benchmark(_experiment(tmp_path), force_mock=True)

    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["cache"]["enabled"] is False and "mock" in summary["cache"]["reason"]
    assert summary["cache"]["real_spend_usd"] == 0 and "embedding_cache" in summary
    assert not cache_dir.exists()
    assert active_cache() is None


def test_evaluator_opens_the_cache_for_live_runs_honoring_env_config_and_no_cache(tmp_path, monkeypatch):
    config = _experiment(tmp_path, {"cache": {"dir": str(tmp_path / "from_config"), "ttl_days": 7}})

    def opened(**kwargs):
        evaluator = BenchmarkEvaluator(config, **kwargs)
        evaluator.mode = "live"  # no real key in tests; only the decision logic is under test
        return evaluator._open_cache()

    monkeypatch.delenv("RAGBENCH_CACHE_DIR", raising=False)
    runtime = opened()
    assert runtime is not None and runtime.disk.path == tmp_path / "from_config" / "cache.sqlite3" and runtime.disk.ttl_days == 7

    monkeypatch.setenv("RAGBENCH_CACHE_DIR", str(tmp_path / "from_env"))
    assert opened().disk.path == tmp_path / "from_env" / "cache.sqlite3"  # type: ignore[union-attr]

    assert opened(use_cache=False) is None
    disabled = _experiment(tmp_path, {"cache": {"enabled": False}})
    evaluator = BenchmarkEvaluator(disabled)
    evaluator.mode = "live"
    assert evaluator._open_cache() is None


def test_cache_summary_reports_hits_savings_and_real_spend(tmp_path):
    from ragbench.cache import summarize_cache

    disk = DiskCache(tmp_path / "c.sqlite3")
    disk.put("llm", "a", b"x")
    disk.get("llm", "a")
    disk.get("llm", "missing")
    disk.record_saved("llm", 0.30)
    summary = summarize_cache(CacheRuntime(disk, CacheConfig()), charged_cost_usd=1.0, in_process_saved_usd=0.20)

    assert summary["enabled"] is True and summary["hits"] == 1 and summary["misses"] == 1 and summary["hit_rate"] == 0.5
    assert summary["saved_cost_usd"] == pytest.approx(0.30)
    assert summary["charged_cost_usd"] == 1.0 and summary["real_spend_usd"] == pytest.approx(0.5)  # 1.0 - 0.30 disk - 0.20 in-process
    assert summarize_cache(None, charged_cost_usd=2.0, in_process_saved_usd=0.0, reason="disabled in config")["real_spend_usd"] == 2.0


def test_cli_cache_stats_and_clear_and_no_cache_flag(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cli_cache"
    cache = DiskCache(cache_dir / "cache.sqlite3")
    cache.put("llm", "a", b"12345")
    cache.put("embeddings", "b", b"123")
    cache.close()
    runner = CliRunner()

    stats = runner.invoke(app, ["cache", "stats", "--cache-dir", str(cache_dir)])
    assert stats.exit_code == 0 and "llm" in stats.output and "embeddings" in stats.output

    declined = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir)], input="n\n")
    assert declined.exit_code != 0 and DiskCache(cache_dir / "cache.sqlite3").describe().keys() == {"llm", "embeddings"}

    one = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir), "--namespace", "llm", "--yes"])
    assert one.exit_code == 0 and list(DiskCache(cache_dir / "cache.sqlite3").describe()) == ["embeddings"]
    everything = runner.invoke(app, ["cache", "clear", "--cache-dir", str(cache_dir), "--yes"])
    assert everything.exit_code == 0 and DiskCache(cache_dir / "cache.sqlite3").describe() == {}

    empty = runner.invoke(app, ["cache", "stats", "--cache-dir", str(tmp_path / "nothing_here")])
    assert empty.exit_code == 0 and "empty" in empty.output.lower() and not (tmp_path / "nothing_here").exists()

    monkeypatch.setenv("RAGBENCH_CACHE_DIR", str(tmp_path / "env_dir"))
    env_stats = runner.invoke(app, ["cache", "stats"])
    assert env_stats.exit_code == 0 and "env_dir" in env_stats.output.replace("\n", "")

    config = _experiment(tmp_path)
    assert runner.invoke(app, ["run", "--config", str(config), "--mock", "--no-cache"]).exit_code == 0


def test_cache_config_defaults_and_validation():
    cfg = CacheConfig()
    assert (cfg.enabled, cfg.llm, cfg.embeddings, cfg.cache_query_embeddings, cfg.llm_nonzero_temperature, cfg.ttl_days) == (True, True, True, False, False, None)
    assert str(cfg.dir) == ".ragbench_cache"
    with pytest.raises(ValueError):
        CacheConfig(ttl_days=0)
    with pytest.raises(ValueError):
        CacheConfig(unknown_option=True)


def test_config_file_accepts_a_cache_section(tmp_path):
    from ragbench.config.loader import load_config

    path = _experiment(tmp_path, {"cache": {"enabled": False, "cache_query_embeddings": True}})
    cfg = load_config(path)
    assert cfg.cache.enabled is False and cfg.cache.cache_query_embeddings is True
    raw = yaml.safe_load(path.read_text())
    raw["cache"] = {"bogus": 1}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)



# --- end to end: a "live" run twice ----------------------------------------------------------


class _Counters:
    llm = 0
    doc_embeds = 0
    query_embeds = 0


class PaidMockLLM(MockLLM):
    """Mock content, real-looking price, call counting and a small delay (so latency replay is measurable)."""

    model_name = "fake-paid-model"

    def generate(self, messages, **kwargs) -> LLMResult:
        _Counters.llm += 1
        time.sleep(0.01)
        result = super().generate(messages, **kwargs)
        result.cost = CostBreakdown(
            llm_prompt_tokens=result.prompt_tokens,
            llm_completion_tokens=result.completion_tokens,
            llm_cost=estimate_model_cost(self.model_name, result.prompt_tokens, result.completion_tokens),
        )
        return result


class PaidHashingEmbedding(HashingEmbeddingModel):
    cacheable = True

    def __init__(self, *_args: Any) -> None:
        super().__init__()
        self.model_name = "fake-embedding"

    def _priced(self, texts: list[str]) -> EmbeddingResult:
        result = super().embed_texts(texts)
        tokens = sum(estimate_tokens(t, self.model_name) for t in texts)
        return EmbeddingResult(result.vectors, self.model_name, tokens, CostBreakdown(embedding_input_tokens=tokens, embedding_cost=estimate_model_cost(self.model_name, input_tokens=tokens)))

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        _Counters.doc_embeds += 1
        return self._priced(texts)

    def embed_query(self, text: str) -> EmbeddingResult:
        _Counters.query_embeds += 1
        return self._priced([text])


def test_second_live_style_run_is_served_from_the_cache_at_the_same_charged_cost(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import ragbench.models.providers.openai as openai_provider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(openai_provider, "OpenAILLM", lambda model_name: PaidMockLLM())
    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingModel", PaidHashingEmbedding)
    monkeypatch.setenv("RAGBENCH_CACHE_DIR", str(tmp_path / "shared_cache"))
    config = write_experiment(
        tmp_path,
        [question(f"q{i}", f"What is topic {i}?", [f"doc_{i:03d}"]) for i in range(1, 5)],
        [
            {"type": "vector", "name": "vec", "retrieval": {"vector_store": "in_memory"}},
            {"type": "hyde", "name": "hyde", "retrieval": {"vector_store": "in_memory"}},
        ],
        evaluation={"latency_probe_questions": 0},  # the probe deliberately bypasses the cache (tested in test_parallel_run)
        extra={"pricing": {"fake-paid-model": {"input": 1000.0, "output": 2000.0}, "fake-embedding": {"input": 500.0}}},
    )

    def run():
        _Counters.llm = _Counters.doc_embeds = _Counters.query_embeds = 0
        out = run_benchmark(config, max_workers=1)
        return out, json.loads((out / "run_summary.json").read_text()), pd.read_csv(out / "metrics_summary.csv").set_index("system")

    out1, summary1, metrics1 = run()
    calls1 = (_Counters.llm, _Counters.doc_embeds, _Counters.query_embeds)
    out2, summary2, metrics2 = run()
    calls2 = (_Counters.llm, _Counters.doc_embeds, _Counters.query_embeds)

    assert summary1["mode"] == summary2["mode"] == "live"
    assert calls1[0] > 0 and calls1[1] >= 1
    assert calls2[0] == 0 and calls2[1] == 0, "second run must not pay for LLM calls or corpus embeddings again"
    assert calls2[2] == calls1[2] > 0, "query embeddings stay uncached so latency is measured fresh"
    cache1, cache2 = summary1["cache"], summary2["cache"]
    assert cache1["enabled"] and cache1["hits"] == 0 and cache1["misses"] > 0
    assert cache2["hits"] > 0 and cache2["misses"] == 0 and cache2["hit_rate"] == 1.0
    assert cache2["saved_cost_usd"] > 0 and cache2["by_namespace"]["llm"]["hits"] > 0 and cache2["by_namespace"]["embeddings"]["hits"] > 0
    # Fairness: systems are charged the same either way; only the real spend differs.
    assert cache1["charged_cost_usd"] == pytest.approx(cache2["charged_cost_usd"], rel=1e-9)
    # What is still really paid in run 2 is the (deliberately uncached) query embeddings; everything else was avoided.
    assert cache2["real_spend_usd"] < 0.5 * cache2["charged_cost_usd"]
    assert cache1["real_spend_usd"] - cache2["real_spend_usd"] == pytest.approx(cache2["saved_cost_usd"], rel=1e-6)
    assert cache1["real_spend_usd"] > 0.5 * cache1["charged_cost_usd"]
    pd.testing.assert_series_equal(metrics1["avg_cost_per_question"], metrics2["avg_cost_per_question"], rtol=1e-9)
    pd.testing.assert_series_equal(metrics1["answer_score"], metrics2["answer_score"])
    # ... and cached LLM time is replayed, so the re-run is not reported as instant.
    assert metrics2.loc["hyde", "avg_latency_ms"] >= 0.6 * metrics1.loc["hyde", "avg_latency_ms"]
    assert "cache reused" in (out2 / "report.html").read_text()
    assert out1 != out2
