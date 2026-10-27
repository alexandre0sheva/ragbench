"""The `openrouter:` provider, the OpenRouter catalog and `ragbench models list|sync`: offline, against the fake server and a fixture."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fake_openai_server import FakeOpenAIServer

from ragbench.config.schema import OpenRouterConfig, OpenRouterModelSettings
from ragbench.models import cost
from ragbench.models import openrouter_catalog as catalog
from ragbench.models.errors import ModelInitError
from ragbench.models.llms import MockLLM, create_llm
from ragbench.models.providers.openrouter import OpenRouterLLM, manifest_settings, request_settings
from ragbench.runtime.context import RuntimeContext, activate_runtime

FIXTURE = Path(__file__).parent / "fixtures" / "openrouter_models_response.json"
QUESTION = [{"role": "user", "content": "Context:\nParis is the capital of France.\n\nQuestion: What is the capital of France?"}]


@pytest.fixture
def synced(tmp_path, monkeypatch):
    """The fixture response synced as the newest catalog."""
    path = tmp_path / "openrouter_models.json"
    monkeypatch.setenv(catalog.CATALOG_ENV, str(path))
    catalog.write_catalog(json.loads(FIXTURE.read_text()), today=date(2099, 1, 1))
    return path


@pytest.fixture
def server(monkeypatch):
    with FakeOpenAIServer(inject_429_every=0, usage_cost=0.0123, record_bodies=True) as fake:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-real")
        monkeypatch.setenv("OPENROUTER_BASE_URL", fake.base_url)
        yield fake


# --- catalog ------------------------------------------------------------------------------------------------------------


def test_sync_trims_the_api_response_into_a_dated_catalog(synced):
    loaded = catalog.load_catalog()
    assert loaded.fetched_at == "2099-01-01" and loaded.path == synced
    haiku = loaded.models["anthropic/claude-haiku-5.5"]
    # USD per token in the API, per 1M tokens here; the prompt-length override is a tier above 100k prompt tokens.
    assert haiku.price == {
        "input": 0.1,
        "output": 0.5,
        "tiers": [{"input": 0.1, "output": 0.5, "max_prompt_tokens": 100000}, {"input": 0.5, "output": 2.5}],
    }
    assert loaded.models["nvidia/switchyard"].price is None  # a router's variable price is not a price
    # A time-of-day discount is ignored: the base price, no tiers.
    assert loaded.models["deepseek/deepseek-v4.1-flash-1017"].price == {"input": 0.15, "output": 0.6}
    glm = loaded.models["z-ai/glm-5.3"]
    assert glm.open_weights and glm.reasoning_mandatory and glm.lowest_effort() == "low"
    assert loaded.models["deepseek/deepseek-v4.1-flash"].lowest_effort() == "none"


def test_bundled_snapshot_prices_the_shortlist_offline():
    loaded = catalog.load_catalog()
    assert loaded.path == catalog.BUNDLED_CATALOG
    for model in ("deepseek/deepseek-v4.1-flash", "z-ai/glm-5.3", "moonshotai/kimi-k3", "xiaomi/mimo-v2.6-pro", "qwen/qwen3.8-flash", "minimax/minimax-m3"):
        assert loaded.models[model].open_weights and loaded.models[model].supports_tools, model
        assert cost.price_known(f"openrouter:{model}"), model
    assert cost.estimate_model_cost("openrouter:deepseek/deepseek-v4.1-flash", 1_000_000, 1_000_000) == pytest.approx(0.0477 + 0.1908)  # audited 2026-10-26


def test_an_older_synced_file_does_not_shadow_the_bundled_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    monkeypatch.setenv(catalog.CATALOG_ENV, str(path))
    catalog.write_catalog(json.loads(FIXTURE.read_text()), today=date(2020, 1, 1))
    assert catalog.load_catalog().path == catalog.BUNDLED_CATALOG


def test_openrouter_prices_match_exact_ids_only_and_pricing_overrides_win(synced):
    # On OpenRouter a suffix is another model, not a dated snapshot: `-1017` must not inherit `deepseek-v4.1-flash`'s price.
    assert cost.estimate_model_cost("openrouter:deepseek/deepseek-v4.1-flash-1017", 1_000_000, 0) == pytest.approx(0.15)
    assert not cost.price_known("openrouter:deepseek/deepseek-v4.1-flash-9999")
    assert cost.estimate_model_cost("openrouter:anthropic/claude-haiku-5.5", 100_001, 0) == pytest.approx(100_001 * 0.5e-6)
    cost.register_pricing({"deepseek/deepseek-v4.1-flash-1017": {"input": 9.0, "output": 9.0}})
    try:
        assert cost.estimate_model_cost("openrouter:deepseek/deepseek-v4.1-flash-1017", 1_000_000, 0) == pytest.approx(9.0)
    finally:
        cost.clear_pricing_overrides()


# --- request settings ---------------------------------------------------------------------------------------------------


def test_reasoning_defaults_to_none_or_the_lowest_effort_a_model_accepts(synced):
    flash = catalog.model_info("deepseek/deepseek-v4.1-flash")
    glm = catalog.model_info("z-ai/glm-5.3")
    assert request_settings("deepseek/deepseek-v4.1-flash", OpenRouterModelSettings(), flash) == {"reasoning": {"effort": "none", "exclude": True}}
    assert request_settings("z-ai/glm-5.3", OpenRouterModelSettings(), glm)["reasoning"] == {"effort": "low", "exclude": True}
    assert request_settings("unknown/model", OpenRouterModelSettings(), None)["reasoning"] == {"effort": "none", "exclude": True}


def test_an_effort_the_model_does_not_accept_fails_before_any_request(synced):
    settings = OpenRouterModelSettings.model_validate({"reasoning": {"effort": "none"}})
    with pytest.raises(ModelInitError, match="accepts max, high, low"):
        request_settings("z-ai/glm-5.3", settings, catalog.model_info("z-ai/glm-5.3"))


def test_per_model_settings_replace_the_section_defaults_and_an_order_pins_the_provider():
    config = OpenRouterConfig.model_validate(
        {
            "reasoning": {"effort": "low"},
            "provider": {"data_collection": "deny"},
            "models": {"z-ai/glm-5.3": {"provider": {"order": ["z-ai"]}}},
        }
    )
    glm = request_settings("z-ai/glm-5.3", config.for_model("z-ai/glm-5.3"), None)
    assert glm == {"reasoning": {"effort": "low", "exclude": True}, "provider": {"order": ["z-ai"], "allow_fallbacks": False}}
    other = request_settings("qwen/qwen3.8-flash", config.for_model("qwen/qwen3.8-flash"), None)
    assert other["provider"] == {"data_collection": "deny"}
    assert manifest_settings(["gpt-6-luna", "openrouter:z-ai/glm-5.3"], config) == {"openrouter:z-ai/glm-5.3": glm}


# --- the provider over HTTP -----------------------------------------------------------------------------------------------


def test_a_request_carries_reasoning_and_pinning_and_is_charged_the_billed_cost(server, synced):
    config = OpenRouterConfig.model_validate({"provider": {"order": ["deepinfra"], "data_collection": "deny"}})
    with activate_runtime(RuntimeContext(openrouter=config)):
        llm = create_llm("openrouter:deepseek/deepseek-v4.1-flash")
    result = llm.generate(QUESTION, max_tokens=50)

    assert "Paris" in result.text and result.model == "openrouter:deepseek/deepseek-v4.1-flash"
    assert result.cost.llm_cost == pytest.approx(0.0123)  # `usage.cost` from the response, not the price table
    body = server.bodies[-1]
    assert body["model"] == "deepseek/deepseek-v4.1-flash" and body["max_tokens"] == 50
    assert body["reasoning"] == {"effort": "none", "exclude": True}
    assert body["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False, "data_collection": "deny"}
    assert body["usage"] == {"include": True} and "reasoning_effort" not in body


def test_without_usage_cost_the_price_table_is_the_cost(monkeypatch, synced):
    with FakeOpenAIServer(inject_429_every=0) as fake:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-real")
        monkeypatch.setenv("OPENROUTER_BASE_URL", fake.base_url)
        result = create_llm("openrouter:deepseek/deepseek-v4.1-flash").generate(QUESTION)
    expected = cost.estimate_model_cost("openrouter:deepseek/deepseek-v4.1-flash", result.prompt_tokens, result.completion_tokens)
    assert result.cost.llm_cost == pytest.approx(expected) and expected > 0


def test_a_model_without_json_output_is_asked_for_json_in_the_prompt(server, synced):
    info = catalog.OpenRouterModel(id="x/no-json", supported_parameters=("tools",))
    from openai import OpenAI

    llm = OpenRouterLLM("x/no-json", client=OpenAI(base_url=server.base_url, api_key="sk-or-not-real", max_retries=0), info=info)
    llm.generate(QUESTION, json_mode=True)
    body = server.bodies[-1]
    assert "response_format" not in body and "JSON object" in body["messages"][0]["content"]


def test_settings_are_part_of_the_cache_identity(synced):
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:9/v1", api_key="x")
    plain = OpenRouterLLM("deepseek/deepseek-v4.1-flash", client=client)
    pinned = OpenRouterLLM("deepseek/deepseek-v4.1-flash", OpenRouterModelSettings.model_validate({"provider": {"order": ["a"]}}), client=client)
    assert plain.provider.startswith("openrouter@") and plain.provider != pinned.provider
    assert plain.limiter_key == "openrouter"


def test_no_key_means_the_mock_and_a_missing_vendor_is_a_config_error(tmp_path):
    from ragbench.models.refs import validate_model_ref

    assert isinstance(create_llm("openrouter:deepseek/deepseek-v4.1-flash"), MockLLM)  # OPENROUTER_API_KEY is empty in tests
    with pytest.raises(ValueError, match="openrouter:<vendor>/<model>"):
        validate_model_ref("openrouter:deepseek-v4.1-flash", "llm", {})
    with pytest.raises(ValueError, match="chat models only"):
        validate_model_ref("openrouter:openai/text-embedding-3-small", "embedding", {})


def test_missing_openrouter_key_is_reported_before_a_live_run(tmp_path, monkeypatch):
    from ragbench.config.schema import ExperimentConfig
    from ragbench.models.refs import missing_credentials

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-real")
    config = ExperimentConfig.model_validate(
        {
            "run": {"name": "t", "output_dir": str(tmp_path)},
            "dataset": {"documents_path": str(tmp_path), "questions_path": str(tmp_path / "q.jsonl")},
            "systems": [{"type": "bm25", "models": {"generator": "openrouter:z-ai/glm-5.3"}}],
        }
    )
    assert missing_credentials(config) == ["model 'openrouter:z-ai/glm-5.3' needs OPENROUTER_API_KEY, which is not set"]


def test_tool_agents_fall_back_to_react_json_for_a_model_without_tools(monkeypatch):
    from ragbench.agents import tool_agent
    from ragbench.config.schema import SystemConfig
    from ragbench.rag_systems import create_rag_system

    system = create_rag_system(SystemConfig.model_validate({"type": "agent_search"}), force_mock=True)
    modes: list[str] = []
    original = tool_agent.ToolAgent.__init__

    def spy(self, llm, box, budget, harness, *, mode="native"):
        modes.append(mode)
        original(self, llm, box, budget, harness, mode=mode)

    monkeypatch.setattr(tool_agent.ToolAgent, "__init__", spy)
    system.ingest([])
    system.answer_question("anything?", top_k=3, context_k=3)
    system.llm.supports_tools = False
    system.answer_question("anything?", top_k=3, context_k=3)
    assert modes == ["native", "react_json"]


# --- CLI ----------------------------------------------------------------------------------------------------------------


def test_models_sync_from_a_file_and_list(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from ragbench.cli.app import app

    target = tmp_path / "catalog.json"
    monkeypatch.setenv(catalog.CATALOG_ENV, str(target))
    runner = CliRunner()
    synced = runner.invoke(app, ["models", "sync", "--from-file", str(FIXTURE)])
    assert synced.exit_code == 0, synced.output
    assert target.exists() and "6 models" in synced.output

    listed = runner.invoke(app, ["models", "list", "--provider", "openrouter", "--json"])
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.stdout)
    glm = next(row for row in rows if row["ref"] == "openrouter:z-ai/glm-5.3")
    assert glm["open_weights"] and glm["tools"] and glm["lowest_effort"] == "low" and glm["input"] == pytest.approx(0.079)

    everything = json.loads(runner.invoke(app, ["models", "list", "--json"]).stdout)
    refs = {row["ref"] for row in everything}
    assert {"anthropic:claude-haiku-5-5", "gpt-6-luna", "openrouter:z-ai/glm-5.3"} <= refs
    haiku = next(row for row in everything if row["ref"] == "anthropic:claude-haiku-5-5")
    assert haiku["long_prompt"] == ">100,000: 0.5/2.5"
    table = runner.invoke(app, ["models", "list", "--open-weights"])
    assert table.exit_code == 0 and "Model prices" in table.output


def test_models_sync_never_reaches_the_network_in_tests(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from ragbench.cli.app import app

    called: list[bool] = []
    monkeypatch.setattr(catalog, "fetch_models", lambda timeout=30.0: called.append(True) or json.loads(FIXTURE.read_text()))
    monkeypatch.setenv(catalog.CATALOG_ENV, str(tmp_path / "c.json"))
    result = CliRunner().invoke(app, ["models", "sync"])
    assert result.exit_code == 0, result.output
    assert called == [True]


def test_a_cache_hit_replays_the_billed_cost(server, synced, tmp_path):
    from ragbench.cache import CacheRuntime, DiskCache, activate_cache
    from ragbench.config.schema import CacheConfig

    llm = create_llm("openrouter:deepseek/deepseek-v4.1-flash")
    with activate_cache(CacheRuntime(disk=DiskCache(tmp_path / "cache.sqlite3"), config=CacheConfig())):
        first = llm.generate(QUESTION)
        second = llm.generate(QUESTION)
    assert not first.cached and second.cached and len(server.bodies) == 1
    # The table would say much less for these few tokens; the hit is charged what the original call was billed.
    assert first.cost.llm_cost == second.cost.llm_cost == pytest.approx(0.0123)
