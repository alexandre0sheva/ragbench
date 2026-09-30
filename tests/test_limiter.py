from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import openai
import pytest

from ragbench.config.schema import LimitsConfig
from ragbench.models import retry as retry_module
from ragbench.models.embeddings import OpenAIEmbeddingModel
from ragbench.models.llms import OpenAILLM
from ragbench.runtime import (
    RateLimiter,
    RuntimeContext,
    activate_runtime,
    build_limiters,
    current_runtime,
    limiter_for,
    ordered_parallel_map,
)


class FakeClock:
    """Deterministic time: `sleep` advances the clock instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def _limiter(clock: FakeClock, **kwargs) -> RateLimiter:
    return RateLimiter(clock=clock.time, sleep=clock.sleep, **kwargs)


def test_max_concurrent_is_never_exceeded_under_32_threads():
    limiter = RateLimiter(max_concurrent=4)
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def work(_: int) -> None:
        with limiter.acquire():
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.005)
            with lock:
                state["now"] -= 1

    threads = [threading.Thread(target=work, args=(i,)) for i in range(32)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert state["peak"] == 4 and state["now"] == 0


def test_slot_is_released_when_the_body_raises():
    limiter = RateLimiter(max_concurrent=1)
    with pytest.raises(RuntimeError):
        with limiter.acquire():
            raise RuntimeError("boom")
    with limiter.acquire():  # would deadlock if the slot leaked
        pass


def test_requests_per_minute_uses_a_sliding_window():
    clock = FakeClock()
    limiter = _limiter(clock, requests_per_minute=3)

    for _ in range(3):
        with limiter.acquire():
            clock.now += 1  # each request takes a second
    assert clock.sleeps == []

    with limiter.acquire():  # the 4th must wait until the 1st is a minute old
        pass
    assert len(clock.sleeps) == 1 and clock.sleeps[0] == pytest.approx(57.0, abs=1.0)
    assert clock.now - 1000.0 >= 60


def test_tokens_per_minute_waits_for_budget_and_never_deadlocks_on_an_oversized_request():
    clock = FakeClock()
    limiter = _limiter(clock, tokens_per_minute=1000)
    with limiter.acquire(est_tokens=600):
        pass
    with limiter.acquire(est_tokens=300):  # 900 <= 1000: fits
        pass
    assert clock.sleeps == []
    with limiter.acquire(est_tokens=400):  # 1300 > 1000: wait for the first request to age out
        pass
    assert clock.sleeps and clock.now - 1000.0 >= 60

    fresh = _limiter(FakeClock(), tokens_per_minute=100)
    with fresh.acquire(est_tokens=10_000):  # bigger than the whole budget: allowed alone, never stuck
        pass


def test_limiter_without_limits_never_waits_and_reports_what_it_needs():
    clock = FakeClock()
    limiter = _limiter(clock)
    for _ in range(100):
        with limiter.acquire(est_tokens=10**6):
            pass
    assert clock.sleeps == [] and not limiter.limits_tokens
    assert RateLimiter(tokens_per_minute=5).limits_tokens


def test_build_limiters_from_config_and_runtime_context():
    assert build_limiters(LimitsConfig()) == {}
    limiters = build_limiters(LimitsConfig(max_concurrent_requests=2, requests_per_minute=60))
    assert set(limiters) == {"openai"}
    assert current_runtime().ingest_workers == 1 and limiter_for("openai") is None
    with activate_runtime(RuntimeContext(ingest_workers=6, limiters=limiters)):
        assert current_runtime().ingest_workers == 6 and limiter_for("openai") is limiters["openai"] and limiter_for("other") is None
    assert limiter_for("openai") is None
    with pytest.raises(ValueError):
        LimitsConfig(requests_per_minute=0)


# --- wiring into the OpenAI clients ----------------------------------------------------------


class CountingLimiter:
    limits_tokens = True

    def __init__(self) -> None:
        self.acquired: list[int] = []
        self.active = 0

    def acquire(self, est_tokens: int = 0):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            self.acquired.append(est_tokens)
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1

        return ctx()


def _rate_limit_error():
    import httpx

    response = httpx.Response(429, request=httpx.Request("POST", "https://x"), json={})
    return openai.RateLimitError("slow down", response=response, body=None)


def test_every_llm_attempt_including_retries_goes_through_the_limiter(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    limiter = CountingLimiter()
    script = [_rate_limit_error(), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None)]
    calls = {"n": 0}

    def create(**params):
        assert limiter.active == 1, "the request must be made while holding a slot"
        calls["n"] += 1
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with activate_runtime(RuntimeContext(limiters={"openai": limiter})):  # type: ignore[dict-item]
        OpenAILLM("gpt-5.4-nano", client=client).generate([{"role": "user", "content": "hello there"}], max_tokens=50)

    assert calls["n"] == 2 and len(limiter.acquired) == 2
    assert limiter.acquired[0] >= 50  # prompt estimate plus the requested completion tokens
    assert limiter.active == 0


def test_embedding_batches_run_in_parallel_but_keep_their_order_and_token_totals():
    import numpy as np

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def create(model, input):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        data = [SimpleNamespace(embedding=[float(int(text)), 1.0]) for text in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(input)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    texts = [str(i) for i in range(1, 41)]

    def run(workers: int):
        state.update(now=0, peak=0)
        with activate_runtime(RuntimeContext(ingest_workers=workers)):
            return OpenAIEmbeddingModel("text-embedding-3-small", batch_size=5, client=client).embed_texts(texts)

    sequential = run(1)
    assert state["peak"] == 1
    parallel = run(4)
    assert 2 <= state["peak"] <= 4
    assert np.allclose(parallel.vectors, sequential.vectors) and parallel.input_tokens == sequential.input_tokens == 40
    assert [int(row[0] * 1e6) for row in parallel.vectors] == sorted(int(row[0] * 1e6) for row in parallel.vectors)  # order preserved (rows are normalized but monotone)


# --- ordered_parallel_map --------------------------------------------------------------------


def test_ordered_parallel_map_preserves_order_and_reports_progress():
    done: list[tuple[int, int]] = []

    def slow_then_fast(n: int) -> int:
        time.sleep(0.03 if n == 0 else 0.0)
        return n * n

    result = ordered_parallel_map(slow_then_fast, range(8), workers=4, on_done=lambda d, t: done.append((d, t)))

    assert result == [n * n for n in range(8)]
    assert [d for d, _ in done] == list(range(1, 9)) and all(t == 8 for _, t in done)
    assert ordered_parallel_map(lambda x: x + 1, [1, 2, 3], workers=1) == [2, 3, 4]
    assert ordered_parallel_map(lambda x: x, [], workers=4) == []


def test_ordered_parallel_map_raises_the_first_error_and_stops_starting_new_work():
    started: list[int] = []

    def fn(n: int) -> int:
        started.append(n)
        if n == 0:
            raise ValueError("fatal")
        time.sleep(0.05)
        return n

    with pytest.raises(ValueError, match="fatal"):
        ordered_parallel_map(fn, range(40), workers=2)

    assert len(started) < 40
