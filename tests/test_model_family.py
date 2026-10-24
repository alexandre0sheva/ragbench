from __future__ import annotations

import pytest

from ragbench.models.family import model_family


@pytest.mark.parametrize(
    ("ref", "family"),
    [
        ("gpt-6-luna", "openai"),
        ("openai:gpt-6.1-sol", "openai"),
        ("anthropic:claude-haiku-5-5", "anthropic"),
        ("openrouter:anthropic/claude-haiku-5.5", "anthropic"),
        ("openrouter:openai/gpt-oss-120b", "openai"),
        ("openrouter:deepseek/deepseek-v4.1-flash", "deepseek"),
        ("openrouter:z-ai/glm-5.3", "zhipu"),
        ("openrouter:moonshotai/kimi-k3", "moonshot"),
        ("openrouter:xiaomi/mimo-v2.6-pro", "xiaomi"),
        ("openrouter:qwen/qwen3.8-flash", "qwen"),
        ("openrouter:minimax/minimax-m3", "minimax"),
        ("openrouter:mistralai/mistral-small-2603", "mistral"),
        ("openrouter:google/gemini-3.8-flash", "google"),
        ("openrouter:x-ai/grok-4.7", "xai"),
        ("openai_compatible:ollama/llama3.1:8b", "meta"),
        ("openai_compatible:vllm/Qwen3-8B-Instruct", "qwen"),
    ],
)
def test_family_comes_from_the_model_name_or_its_vendor(ref, family):
    assert model_family(ref) == family


def test_an_unrecognised_model_is_a_family_of_its_own():
    assert model_family("openrouter:acme/foo-7b") == "acme"
    assert model_family("openai_compatible:local/house-model") == "house-model"
