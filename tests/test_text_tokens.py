from __future__ import annotations

import tiktoken

from ragbench.utils import text


def test_estimate_tokens_never_raises_for_unknown_models():
    text._get_encoding.cache_clear()

    assert text.estimate_tokens("hello brave new world", "gpt-6-luna") >= 4


def test_encoding_lookup_is_cached_per_model(monkeypatch):
    text._get_encoding.cache_clear()
    calls = {"n": 0}
    real = tiktoken.encoding_for_model

    def counting(model):
        calls["n"] += 1
        return real(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", counting)

    for _ in range(25):
        text.estimate_tokens("some text to count", "gpt-6-luna")

    assert calls["n"] == 1
    text._get_encoding.cache_clear()


def test_unknown_model_falls_back_to_default_encoding_without_repeated_failures(monkeypatch):
    text._get_encoding.cache_clear()
    calls = {"n": 0}

    def failing(model):
        calls["n"] += 1
        raise KeyError(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", failing)

    first = text.estimate_tokens("one two three four five", "gpt-6-luna")
    second = text.estimate_tokens("one two three four five", "gpt-6-luna")

    assert first == second and first > 0
    assert calls["n"] == 1
    text._get_encoding.cache_clear()


def test_heuristic_when_tiktoken_is_unavailable(monkeypatch):
    text._get_encoding.cache_clear()

    def broken(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(tiktoken, "encoding_for_model", broken)
    monkeypatch.setattr(tiktoken, "get_encoding", broken)

    assert text.estimate_tokens("one two three four", "x") == int(4 * 1.3)
    text._get_encoding.cache_clear()
