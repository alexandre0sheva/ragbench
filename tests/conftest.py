from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_the_real_environment(monkeypatch, tmp_path_factory):
    """Keep every test offline, free, and away from the project's cache directory.

    An empty `OPENAI_API_KEY` counts as "set" for the `.env` loader, so the real key in a developer's `.env` is never
    loaded, and a test that wants a key sets one explicitly. The cache directory points at a throwaway location.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("RAGBENCH_CACHE_DIR", str(tmp_path_factory.mktemp("ragbench_cache")))


@pytest.fixture
def tiny_dataset(tmp_path):
    """The `dataset:` section of a config over six short documents and eight questions (tests/dataset_support.py), written under `tmp_path`."""
    from dataset_support import write_tiny_dataset

    return write_tiny_dataset(tmp_path)


@pytest.fixture
def pair_dataset(tmp_path):
    """The smallest labeled dataset (two documents, two questions: q1, q2), written under `tmp_path`."""
    from dataset_support import write_pair_dataset

    return write_pair_dataset(tmp_path)


@pytest.fixture
def scripted_llm():
    """A factory for a chat model whose replies the test scripts: `scripted_llm(lambda prompt: "reply", price_per_call=0.01)`."""
    from scripted_llm import ScriptedLLM

    return ScriptedLLM
