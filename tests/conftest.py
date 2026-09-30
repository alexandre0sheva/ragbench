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
