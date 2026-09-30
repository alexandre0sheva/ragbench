from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from ragbench.models import retry as retry_module
from ragbench.models.errors import (
    PermanentModelError,
    RagbenchModelError,
    RateLimitError,
    TransientModelError,
    translate_openai_exception,
)
from ragbench.models.llms import OpenAILLM
from ragbench.models.retry import call_with_retry


def _flaky(errors: list[Exception], result: str = "ok"):
    state = {"calls": 0}

    def fn() -> str:
        state["calls"] += 1
        if errors:
            raise errors.pop(0)
        return result

    return fn, state


def test_retries_transient_and_rate_limit_with_exponential_backoff():
    sleeps: list[float] = []
    fn, state = _flaky([RateLimitError("429"), TransientModelError("503"), TransientModelError("503")])

    result = call_with_retry(fn, max_attempts=5, base_delay=1.0, max_delay=30.0, sleep=sleeps.append, jitter=lambda: 0.0)

    assert result == "ok"
    assert state["calls"] == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_is_capped_and_jittered():
    sleeps: list[float] = []
    fn, _ = _flaky([TransientModelError("x")] * 4)

    call_with_retry(fn, max_attempts=6, base_delay=10.0, max_delay=15.0, sleep=sleeps.append, jitter=lambda: 0.5)

    assert sleeps == [11.25, 15.0, 15.0, 15.0]


def test_permanent_errors_are_never_retried():
    sleeps: list[float] = []
    fn, state = _flaky([PermanentModelError("401")])

    with pytest.raises(PermanentModelError):
        call_with_retry(fn, sleep=sleeps.append)

    assert state["calls"] == 1
    assert sleeps == []


def test_unknown_exceptions_propagate_without_retry():
    fn, state = _flaky([ValueError("bug")])

    with pytest.raises(ValueError):
        call_with_retry(fn, sleep=lambda _: None)

    assert state["calls"] == 1


def test_retry_after_is_honored():
    sleeps: list[float] = []
    fn, _ = _flaky([RateLimitError("429", retry_after=7.0)])

    call_with_retry(fn, base_delay=1.0, sleep=sleeps.append, jitter=lambda: 0.0)

    assert sleeps == [7.0]


def test_exhausted_attempts_raise_the_last_error():
    fn, state = _flaky([TransientModelError("a"), TransientModelError("b"), TransientModelError("c")])

    with pytest.raises(TransientModelError, match="c"):
        call_with_retry(fn, max_attempts=3, sleep=lambda _: None)

    assert state["calls"] == 3


def test_default_sleep_is_resolved_at_call_time(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", slept.append)
    fn, _ = _flaky([TransientModelError("x")])

    call_with_retry(fn, base_delay=0.5, jitter=lambda: 0.0)

    assert slept == [0.5]


def _status_error(cls, status: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request, json={"error": {"message": "boom"}})
    return cls("boom", response=response, body=None)


def test_translate_openai_exceptions():
    request = httpx.Request("POST", "https://api.example.test")
    rate = translate_openai_exception(_status_error(openai.RateLimitError, 429, {"retry-after": "7"}))
    assert isinstance(rate, RateLimitError) and rate.retry_after == 7.0
    assert isinstance(translate_openai_exception(_status_error(openai.InternalServerError, 503)), TransientModelError)
    assert isinstance(translate_openai_exception(openai.APITimeoutError(request=request)), TransientModelError)
    assert isinstance(translate_openai_exception(openai.APIConnectionError(request=request)), TransientModelError)
    assert isinstance(translate_openai_exception(_status_error(openai.AuthenticationError, 401)), PermanentModelError)
    assert isinstance(translate_openai_exception(_status_error(openai.BadRequestError, 400)), PermanentModelError)
    # Anything that is not an OpenAI SDK error is left alone.
    other = ValueError("x")
    assert translate_openai_exception(other) is other


# --- OpenAILLM behavior against a fake client -------------------------------------------------


class _FakeChatCompletions:
    def __init__(self, script: list):
        self.script = script
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _completion(text: str = "hi"):
    message = SimpleNamespace(content=text)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _fake_client(script: list, responses=None):
    chat = _FakeChatCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=chat), responses=responses)
    return client, chat


def test_openai_llm_retries_rate_limits_then_succeeds(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    client, chat = _fake_client([_status_error(openai.RateLimitError, 429), _completion("done")])

    result = OpenAILLM("gpt-5.4-nano", client=client).generate([{"role": "user", "content": "q"}])

    assert result.text == "done"
    assert len(chat.calls) == 2


def test_openai_llm_does_not_fall_back_to_responses_on_server_errors(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    responses = SimpleNamespace(create=lambda **_: pytest.fail("Responses API must not be called for transient errors"))
    client, chat = _fake_client([_status_error(openai.InternalServerError, 500)] * 6, responses=responses)

    with pytest.raises(TransientModelError):
        OpenAILLM("gpt-5.4-nano", client=client).generate([{"role": "user", "content": "q"}])

    assert len(chat.calls) == 6  # bounded retries, no doubled spend


def test_openai_llm_falls_back_to_responses_only_for_unsupported_endpoint():
    error = _status_error(openai.NotFoundError, 404)
    error.message = "This model is only supported in v1/responses and not in v1/chat/completions."
    response = SimpleNamespace(output_text="from responses", usage=SimpleNamespace(input_tokens=3, output_tokens=2))
    responses = SimpleNamespace(create=lambda **_: response)
    client, _ = _fake_client([error], responses=responses)

    result = OpenAILLM("codex-x", client=client).generate([{"role": "user", "content": "q"}])

    assert result.text == "from responses"


def test_openai_llm_uses_max_completion_tokens_and_drops_rejected_temperature():
    rejected = _status_error(openai.BadRequestError, 400)
    rejected.message = "Unsupported value: 'temperature' does not support 0 with this model."
    client, chat = _fake_client([rejected, _completion("ok"), _completion("ok2")])
    llm = OpenAILLM("some-new-model", client=client)

    llm.generate([{"role": "user", "content": "q"}], temperature=0, max_tokens=50)
    llm.generate([{"role": "user", "content": "q"}], temperature=0, max_tokens=50)

    assert "temperature" in chat.calls[0] and "temperature" not in chat.calls[1]
    assert "temperature" not in chat.calls[2]  # learned for this model instance
    assert chat.calls[1]["max_completion_tokens"] == 50 and "max_tokens" not in chat.calls[1]


def test_known_reasoning_models_never_send_temperature():
    client, chat = _fake_client([_completion("ok")])

    OpenAILLM("o3-mini", client=client).generate([{"role": "user", "content": "q"}], temperature=0)

    assert "temperature" not in chat.calls[0]


def test_error_types_share_a_base_class():
    for cls in (RateLimitError, TransientModelError, PermanentModelError):
        assert issubclass(cls, RagbenchModelError)


def test_openai_embeddings_retry_transient_errors_per_batch(monkeypatch):
    import numpy as np

    from ragbench.models.embeddings import OpenAIEmbeddingModel

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    script: list = [_status_error(openai.RateLimitError, 429)]
    calls: list[list[str]] = []

    def create(model, input):
        calls.append(input)
        step = script.pop(0) if script else None
        if isinstance(step, Exception):
            raise step
        data = [SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=4 * len(input)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    model = OpenAIEmbeddingModel("text-embedding-3-small", batch_size=2, client=client)

    result = model.embed_texts(["a", "b", "c"])

    assert len(calls) == 3  # batch 1 retried once, batch 2 once
    assert result.vectors.shape == (3, 2) and np.allclose(result.vectors[0], [1.0, 0.0])
    assert result.input_tokens == 12
