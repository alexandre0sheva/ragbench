"""Model refs, provider selection, tool-calling translation and local embeddings. No network, no model downloads."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fake_openai_server import FakeOpenAIServer

from ragbench.config.schema import ExperimentConfig, ProviderConfig
from ragbench.models import cost
from ragbench.models.errors import ModelInitError, RateLimitError, TransientModelError
from ragbench.models.llms import LLMResult, MockLLM, ToolCall, create_llm, parse_model_ref
from ragbench.models.refs import collect_model_refs, missing_credentials, resolve_run_mode, validate_model_ref, warm_up_modules
from ragbench.registry import UnknownComponentError
from ragbench.runtime import RuntimeContext, activate_runtime

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic"


def _ns(value: Any) -> Any:
    """JSON -> attribute-style objects, the shape the SDKs hand back."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: item if key == "input" else _ns(item) for key, item in value.items()})  # tool input stays a dict
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


class FakeAnthropic:
    def __init__(self, *responses: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(step, Exception):
            raise step
        return _ns(step)


def _config(tmp_path: Path, systems: list[dict[str, Any]], *, judge: str | None = None, providers: dict | None = None, judge_enabled: bool = True) -> ExperimentConfig:
    raw: dict[str, Any] = {
        "run": {"name": "t", "output_dir": str(tmp_path)},
        "dataset": {"documents_path": str(tmp_path), "questions_path": str(tmp_path / "q.jsonl")},
        "systems": systems,
        "evaluation": {"judge_enabled": judge_enabled, **({"judge_model": judge} if judge else {})},
    }
    if providers:
        raw["providers"] = providers
    return ExperimentConfig.model_validate(raw)


# --- model refs ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("gpt-6-luna", ("openai", "gpt-6-luna")),
        ("openai:gpt-6-luna", ("openai", "gpt-6-luna")),
        ("anthropic:claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
        ("openai_compatible:ollama/llama3.1:8b", ("openai_compatible", "ollama/llama3.1:8b")),
        ("local:BAAI/bge-small-en-v1.5", ("local", "BAAI/bge-small-en-v1.5")),
        ("llama3:8b", ("openai", "llama3:8b")),  # an unrecognised prefix is part of a bare model name
    ],
)
def test_parse_model_ref(ref, expected):
    assert parse_model_ref(ref) == expected


def test_a_mistyped_provider_prefix_is_reported_with_a_suggestion():
    with pytest.raises(UnknownComponentError, match="anthropic"):
        parse_model_ref("anthropics:claude-sonnet-5-5")
    with pytest.raises(ValueError, match="empty"):
        parse_model_ref("anthropic:")


def test_validate_model_ref_checks_provider_kind_and_endpoint():
    endpoints = {"ollama": ProviderConfig(base_url="http://localhost:11434/v1")}
    validate_model_ref("openai_compatible:ollama/llama3.1", "llm", endpoints)
    validate_model_ref("openai_compatible:ollama/nomic-embed-text", "embedding", endpoints)
    validate_model_ref("local:BAAI/bge-small-en-v1.5", "embedding", endpoints)
    with pytest.raises(ValueError, match="unknown endpoint 'vllm'.*ollama"):
        validate_model_ref("openai_compatible:vllm/x", "llm", endpoints)
    with pytest.raises(ValueError, match="<endpoint>/<model>"):
        validate_model_ref("openai_compatible:ollama", "llm", endpoints)
    with pytest.raises(ValueError, match="embedding"):
        validate_model_ref("local:BAAI/bge-small-en-v1.5", "llm", endpoints)
    with pytest.raises(ValueError, match="chat"):
        validate_model_ref("anthropic:claude-sonnet-5-5", "embedding", endpoints)


def test_config_validation_reports_bad_model_refs_with_their_location(tmp_path):
    system = {"type": "vector", "models": {"generator": "openai_compatible:nowhere/llama", "embedding": "text-embedding-3-small"}}
    with pytest.raises(ValueError, match=r"systems\[0\].*generator.*unknown endpoint 'nowhere'"):
        _config(tmp_path, [system])
    with pytest.raises(ValueError, match=r"judge_model.*anthropics"):
        _config(tmp_path, [{"type": "bm25"}], judge="anthropics:claude-haiku-4-5")
    cfg = _config(
        tmp_path,
        [{"type": "vector", "models": {"generator": "anthropic:claude-haiku-4-5", "embedding": "local:BAAI/bge-small-en-v1.5"}}],
        judge="openai_compatible:ollama/llama3.1",
        providers={"ollama": {"base_url": "http://localhost:11434/v1"}},
    )
    assert cfg.providers["ollama"].api_key_env is None


def test_provider_endpoint_names_cannot_break_the_ref_syntax():
    with pytest.raises(ValueError):
        ProviderConfig.model_validate({"base_url": "http://x/v1", "type": "anthropic"})
    from ragbench.config.schema import ExperimentConfig as Experiment

    with pytest.raises(ValueError, match="endpoint name"):
        Experiment.model_validate(
            {
                "run": {},
                "dataset": {"documents_path": ".", "questions_path": "q"},
                "systems": [{"type": "bm25"}],
                "providers": {"bad/name": {"base_url": "http://x/v1"}},
            }
        )


# --- provider selection -------------------------------------------------------------------------


def test_hosted_provider_without_a_key_keeps_the_mock_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert isinstance(create_llm("anthropic:claude-haiku-4-5"), MockLLM)
    assert isinstance(create_llm("gpt-6-luna"), MockLLM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    assert isinstance(create_llm("anthropic:claude-haiku-4-5", force_mock=True), MockLLM)


def test_create_llm_selects_the_provider_and_keeps_wrapping_clients_for_the_cache(monkeypatch):
    from ragbench.models.cached import CachedLLM
    from ragbench.models.providers import anthropic as anthropic_module

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setattr(anthropic_module, "build_client", lambda timeout: FakeAnthropic({}))

    plain = create_llm("openai:gpt-6-luna")
    claude = create_llm("anthropic:claude-sonnet-5-5")

    assert isinstance(plain, CachedLLM) and plain.model_name == "gpt-6-luna" and plain.inner.provider == "openai"
    assert isinstance(claude, CachedLLM) and claude.model_name == "claude-sonnet-5-5" and claude.inner.provider == "anthropic"


def test_openai_compatible_needs_a_configured_endpoint_and_names_its_model_with_the_ref(monkeypatch):
    endpoints = {"ollama": ProviderConfig(base_url="http://localhost:11434/v1")}
    llm = create_llm("openai_compatible:ollama/llama3.1:8b", providers=endpoints)
    assert llm.model_name == "openai_compatible:ollama/llama3.1:8b"
    assert llm.inner.api_model == "llama3.1:8b" and llm.inner.provider.startswith("openai_compatible:ollama@http://localhost:11434")
    assert llm.inner.limiter_key == "openai_compatible:ollama"
    with pytest.raises(ModelInitError, match="endpoint 'vllm'"):
        create_llm("openai_compatible:vllm/x", providers=endpoints)


def test_openai_compatible_reads_the_key_from_the_named_environment_variable(monkeypatch):
    endpoints = {"hosted": ProviderConfig(base_url="http://localhost:9/v1", api_key_env="HOSTED_API_KEY")}
    monkeypatch.delenv("HOSTED_API_KEY", raising=False)
    with pytest.raises(ModelInitError, match="HOSTED_API_KEY"):
        create_llm("openai_compatible:hosted/m", providers=endpoints)
    monkeypatch.setenv("HOSTED_API_KEY", "secret-not-real")
    assert create_llm("openai_compatible:hosted/m", providers=endpoints).inner.client.api_key == "secret-not-real"


def test_runtime_context_supplies_the_endpoints_when_none_are_passed():
    endpoints = {"ollama": ProviderConfig(base_url="http://localhost:11434/v1")}
    with activate_runtime(RuntimeContext(providers=endpoints)):
        assert create_llm("openai_compatible:ollama/m").inner.api_model == "m"


def test_openai_compatible_talks_to_a_real_http_server_with_the_real_sdk():
    from ragbench.models.embeddings import create_embedding_model

    with FakeOpenAIServer(inject_429_every=0) as server:
        endpoints = {"fake": ProviderConfig(base_url=server.base_url)}
        llm = create_llm("openai_compatible:fake/chat-model", providers=endpoints)
        result = llm.generate([{"role": "user", "content": "Context:\nParis is the capital of France.\n\nQuestion: What is the capital of France?"}], max_tokens=50)
        embedder = create_embedding_model("openai_compatible:fake/embed-model", providers=endpoints)
        vectors = embedder.embed_texts(["alpha beta", "gamma delta"])

        assert "Paris" in result.text and result.model == "openai_compatible:fake/chat-model"
        assert result.cost.total_cost == 0.0  # endpoints are priced at $0 unless `pricing:` says otherwise
        assert vectors.vectors.shape[0] == 2 and vectors.cost.total_cost == 0.0
        assert server.stats["max_tokens_sent"] == 1 and server.stats.get("max_completion_tokens_sent", 0) == 0
        assert server.stats["/v1/embeddings"] == 1


# --- OpenAI-compatible tool translation is covered in test_openai_wire; Anthropic here ----------


def _roundtrip() -> dict[str, Any]:
    return json.loads((FIXTURES / "tool_roundtrip.json").read_text())


def test_anthropic_request_translation_matches_the_recorded_wire_format():
    from ragbench.models.providers.anthropic import to_anthropic_request

    fixture = _roundtrip()
    system, messages, tools = to_anthropic_request(fixture["openai_messages"], fixture["tools"])

    assert system == fixture["anthropic_request"]["system"]
    assert messages == fixture["anthropic_request"]["messages"]
    assert tools == fixture["anthropic_request"]["tools"]


def test_anthropic_response_becomes_text_plus_tool_calls_and_survives_a_round_trip():
    from ragbench.models.providers.anthropic import AnthropicLLM, to_anthropic_request

    fixture = _roundtrip()
    client = FakeAnthropic(fixture["anthropic_response"])
    llm = AnthropicLLM("claude-sonnet-5-5", client=client)

    result = llm.generate(fixture["openai_messages"], tools=fixture["tools"], max_tokens=300)

    expected = fixture["expected_result"]
    assert result.text == expected["text"] and result.finish_reason == expected["finish_reason"]
    assert (result.prompt_tokens, result.completion_tokens) == (expected["prompt_tokens"], expected["completion_tokens"])
    assert result.tool_calls == [ToolCall(id="toolu_03", name="search", arguments={"query": "refund window days"})]
    assert result.cost.llm_cost == pytest.approx(412 / 1e6 * 2.0 + 57 / 1e6 * 10.0)  # claude-sonnet-5-5: $2 / $10 per 1M

    # Feeding the assistant turn back in reproduces the tool_use block the API returned.
    _, messages, _ = to_anthropic_request([*fixture["openai_messages"][:3], result.assistant_message()])
    sent_blocks = messages[-1]["content"]
    assert sent_blocks[-1] == {"type": "tool_use", "id": "toolu_03", "name": "search", "input": {"query": "refund window days"}}

    request = client.calls[0]
    assert request["model"] == "claude-sonnet-5-5" and request["system"] == fixture["anthropic_request"]["system"]
    assert request["tools"] == fixture["anthropic_request"]["tools"]
    assert "temperature" not in request and "temperature" not in request.get("extra_body", {}), "Claude 5 models take no sampling parameters"


def test_anthropic_sets_effort_and_thinking_headroom_only_where_supported_and_temperature_only_on_older_models():
    from ragbench.models.providers.anthropic import THINKING_HEADROOM, AnthropicLLM

    reply = json.loads((FIXTURES / "text_response.json").read_text())
    new, old = FakeAnthropic(reply), FakeAnthropic(reply)
    AnthropicLLM("claude-sonnet-5-5", client=new).generate([{"role": "user", "content": "hi"}], max_tokens=200)
    AnthropicLLM("claude-haiku-4-5-20251001", client=old).generate([{"role": "user", "content": "hi"}], max_tokens=200)

    assert new.calls[0]["output_config"] == {"effort": "low"} and new.calls[0]["max_tokens"] == 200 + THINKING_HEADROOM
    assert "output_config" not in old.calls[0] and old.calls[0]["max_tokens"] == 200
    assert old.calls[0]["extra_body"] == {"temperature": 0.0}


def test_anthropic_json_mode_adds_an_instruction_and_unwraps_code_fences():
    from ragbench.models.providers.anthropic import AnthropicLLM

    client = FakeAnthropic(json.loads((FIXTURES / "text_response.json").read_text()))
    result = AnthropicLLM("claude-haiku-4-5", client=client).generate(
        [{"role": "system", "content": "judge"}, {"role": "user", "content": "go"}], json_mode=True
    )

    assert json.loads(result.text) == {"score": 4}
    assert "JSON" in client.calls[0]["system"] and client.calls[0]["system"].startswith("judge")
    assert client.calls[0]["max_tokens"] >= 1024  # Anthropic requires max_tokens; a default is filled in


def test_anthropic_rejected_temperature_is_dropped_once_and_remembered():
    from ragbench.models.providers import anthropic as module

    bad = module.PermanentModelError("temperature is not supported for this model")
    ok = json.loads((FIXTURES / "text_response.json").read_text())
    client = FakeAnthropic(bad, ok)
    llm = module.AnthropicLLM("claude-haiku-4-5", client=client)

    llm.generate([{"role": "user", "content": "hi"}])
    llm.generate([{"role": "user", "content": "hi"}])

    assert "extra_body" in client.calls[0] and "extra_body" not in client.calls[1] and "extra_body" not in client.calls[2]


def test_anthropic_transient_errors_are_retried_with_the_shared_retry_wrapper(monkeypatch):
    from ragbench.models import retry as retry_module
    from ragbench.models.providers.anthropic import AnthropicLLM

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    ok = json.loads((FIXTURES / "text_response.json").read_text())
    client = FakeAnthropic(RateLimitError("slow down"), TransientModelError("overloaded"), ok)

    result = AnthropicLLM("claude-haiku-4-5", client=client).generate([{"role": "user", "content": "hi"}])

    assert len(client.calls) == 3 and result.finish_reason == "stop"


def test_anthropic_sdk_exceptions_map_onto_the_retry_taxonomy():
    anthropic = pytest.importorskip("anthropic")
    import httpx

    from ragbench.models.errors import PermanentModelError, translate_anthropic_exception

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    def status_error(cls, status):
        return cls("boom", response=httpx.Response(status, request=request, headers={"retry-after": "7"}), body=None)

    limited = translate_anthropic_exception(status_error(anthropic.RateLimitError, 429))
    assert isinstance(limited, RateLimitError) and limited.retry_after == 7.0
    assert isinstance(translate_anthropic_exception(status_error(anthropic.InternalServerError, 529)), TransientModelError)
    assert isinstance(translate_anthropic_exception(status_error(anthropic.AuthenticationError, 401)), PermanentModelError)


def test_anthropic_provider_against_the_real_sdk_over_a_mock_transport(monkeypatch):
    """The real `anthropic` SDK must accept what we send (`output_config`, `extra_body`, tools) and parse what comes back."""
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from ragbench.models import retry as retry_module
    from ragbench.models.providers.anthropic import AnthropicLLM

    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)
    fixture = _roundtrip()
    seen: list[dict[str, Any]] = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
        return httpx2.Response(200, json=fixture["anthropic_response"])

    client = anthropic.Anthropic(api_key="sk-ant-test-not-real", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    result = AnthropicLLM("claude-sonnet-5-5", client=client).generate(fixture["openai_messages"], tools=fixture["tools"], max_tokens=300)

    assert len(seen) == 2, "529 overloaded must be retried"
    body = seen[1]
    assert body["model"] == "claude-sonnet-5-5" and body["output_config"] == {"effort": "low"} and "temperature" not in body
    assert body["messages"] == fixture["anthropic_request"]["messages"] and body["tools"] == fixture["anthropic_request"]["tools"]
    assert result.tool_calls == [ToolCall(id="toolu_03", name="search", arguments={"query": "refund window days"})]


def test_missing_optional_extras_raise_an_actionable_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes `import anthropic` fail
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")

    with pytest.raises(ImportError, match=r"ragbench\[anthropic\]"):
        create_llm("anthropic:claude-haiku-4-5")
    from ragbench.models.embeddings import create_embedding_model

    with pytest.raises(ImportError, match=r"ragbench\[local\]"):
        create_embedding_model("local:BAAI/bge-small-en-v1.5").embed_texts(["x"])


def test_cli_shows_the_missing_extra_hint_intact_and_exits_2(tmp_path, monkeypatch):
    """`ragbench[anthropic]` looks like Rich markup; the CLI must print it verbatim."""
    from typer.testing import CliRunner

    from ragbench.cli import app

    root = Path(__file__).resolve().parents[1] / "data" / "demo"
    config = tmp_path / "c.yaml"
    config.write_text(
        json.dumps(
            {
                "run": {"name": "x", "output_dir": str(tmp_path / "out")},
                "dataset": {"documents_path": str(root / "docs"), "questions_path": str(root / "questions.jsonl"), "qrels_path": str(root / "qrels.jsonl")},
                "systems": [{"type": "bm25", "models": {"generator": "anthropic:claude-haiku-4-5"}}],
                "evaluation": {"max_questions": 1, "judge_enabled": False},
            }
        )
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setitem(sys.modules, "anthropic", None)

    result = CliRunner().invoke(app, ["run", "--config", str(config), "--no-cache"])

    assert result.exit_code == 2
    assert "ragbench[anthropic]" in result.output.replace("\n", "")


# --- mock, tools, result helpers ----------------------------------------------------------------


def test_mock_llm_accepts_tools_and_tool_messages_without_calling_any():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    result = MockLLM().generate(messages, tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}])

    assert result.tool_calls == [] and result.finish_reason == "stop"


def test_assistant_message_uses_the_openai_shape_with_json_string_arguments():
    result = LLMResult(
        text="",
        model="m",
        prompt_tokens=1,
        completion_tokens=1,
        cost=cost.CostBreakdown(),
        tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "x"})],
    )

    assert result.assistant_message() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'}}],
    }
    assert LLMResult(text="hi", model="m", prompt_tokens=1, completion_tokens=1, cost=cost.CostBreakdown()).assistant_message() == {"role": "assistant", "content": "hi"}


def test_a_third_party_provider_registered_in_the_registry_is_usable_in_refs_and_config(tmp_path):
    from ragbench.registry import LLM_PROVIDERS

    class PluginLLM(MockLLM):
        provider = "plugin"

        def __init__(self, model: str):
            super().__init__()
            self.model_name = f"plugin-{model}"

    LLM_PROVIDERS.add("acme", lambda model, *, providers: PluginLLM(model))
    try:
        assert parse_model_ref("acme:fast-1") == ("acme", "fast-1")
        assert create_llm("acme:fast-1").model_name == "plugin-fast-1"
        _config(tmp_path, [{"type": "bm25", "models": {"generator": "acme:fast-1"}}])  # validates
        with pytest.raises(ValueError, match="chat models only"):
            validate_model_ref("acme:x", "embedding", {})
    finally:
        LLM_PROVIDERS.mapping.pop("acme")


# --- local embeddings ---------------------------------------------------------------------------


def _fake_sentence_transformers(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class FakeSentenceTransformer:
        def __init__(self, name: str, **kwargs: Any) -> None:
            calls.append({"load": name, **kwargs})

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            calls.append({"encode": len(texts), **kwargs})
            return np.array([[3.0, 4.0]] * len(texts), dtype=np.float32)  # deliberately not unit length

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


def test_local_embeddings_are_lazy_normalized_free_and_batched(monkeypatch):
    from ragbench.models.embeddings import create_embedding_model

    calls = _fake_sentence_transformers(monkeypatch)
    embedder = create_embedding_model("local:BAAI/bge-small-en-v1.5")
    assert embedder.model_name == "local:BAAI/bge-small-en-v1.5" and calls == [], "construction must not load (download) the model"

    result = embedder.embed_texts(["one two", "three"])

    assert [c for c in calls if "load" in c] == [{"load": "BAAI/bge-small-en-v1.5"}]
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0) and result.vectors.shape == (2, 2)
    assert result.cost.total_cost == 0.0 and result.input_tokens > 0
    encode = next(c for c in calls if "encode" in c)
    assert encode["batch_size"] > 0 and encode["show_progress_bar"] is False


def test_local_embeddings_load_the_model_once_across_threads(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from ragbench.models.providers.local_embeddings import LocalEmbeddingModel

    calls = _fake_sentence_transformers(monkeypatch)
    model = LocalEmbeddingModel("BAAI/bge-small-en-v1.5")
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: model.embed_texts([f"text {i}"]), range(16)))

    assert sum("load" in c for c in calls) == 1


# --- pricing ------------------------------------------------------------------------------------


def test_endpoint_and_local_models_are_free_by_default_and_not_flagged_unknown():
    cost.reset_unknown_priced_models()

    assert cost.estimate_model_cost("local:BAAI/bge-small-en-v1.5", input_tokens=1_000_000) == 0.0
    assert cost.estimate_model_cost("openai_compatible:ollama/llama3.1", 1_000_000, 1_000_000) == 0.0
    assert cost.unknown_priced_models() == []


def test_endpoint_prices_can_be_set_by_ref_endpoint_model_or_model_but_never_inherit_the_openai_table():
    cost.reset_unknown_priced_models()
    try:
        assert cost.estimate_model_cost("openai_compatible:router/gpt-6-luna", 1_000_000, 0) == 0.0, "the OpenAI table must not price a proxy's model"
        for key in ("openai_compatible:router/gpt-6-luna", "router/gpt-6-luna", "gpt-6-luna"):
            cost.clear_pricing_overrides()
            cost.register_pricing({key: {"input": 0.5, "output": 1.0}})
            assert cost.estimate_model_cost("openai_compatible:router/gpt-6-luna", 1_000_000, 1_000_000) == 1.5, key
    finally:
        cost.clear_pricing_overrides()


def test_provider_prefixed_names_resolve_to_the_bare_model_price():
    assert cost.estimate_model_cost("anthropic:claude-haiku-4-5", 1_000_000, 0) == cost.estimate_model_cost("claude-haiku-4-5", 1_000_000, 0) == 1.0
    assert cost.estimate_model_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0  # dated snapshot of a priced alias
    assert cost.estimate_model_cost("openai:gpt-6-luna", 1_000_000, 0) == 0.10


# --- run mode, credentials, warm-up -------------------------------------------------------------


def test_run_mode_is_live_when_any_used_model_can_actually_be_reached(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    claude = _config(tmp_path, [{"type": "bm25", "models": {"generator": "anthropic:claude-haiku-4-5"}}], judge_enabled=False)
    local_only = _config(
        tmp_path,
        [{"type": "vector", "models": {"generator": "openai_compatible:ollama/m", "embedding": "local:BAAI/bge-small-en-v1.5"}}],
        judge="openai_compatible:ollama/m",
        providers={"ollama": {"base_url": "http://localhost:11434/v1"}},
    )
    assert resolve_run_mode(claude, force_mock=False) == "mock"  # no ANTHROPIC_API_KEY: same no-key mock mode as before
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    assert resolve_run_mode(claude, force_mock=False) == "live"
    assert resolve_run_mode(claude, force_mock=True) == "mock"
    assert resolve_run_mode(local_only, force_mock=False) == "live"  # local servers need no key
    assert missing_credentials(local_only) == []


def test_a_live_run_refuses_to_silently_mix_real_and_mock_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    cfg = _config(tmp_path, [{"type": "bm25", "models": {"generator": "anthropic:claude-haiku-4-5"}}], judge="gpt-6-luna")

    assert resolve_run_mode(cfg, force_mock=False) == "live"
    (problem,) = missing_credentials(cfg)
    assert "gpt-6-luna" in problem and "OPENAI_API_KEY" in problem


def test_unused_embedding_defaults_and_a_disabled_judge_do_not_demand_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    cfg = _config(tmp_path, [{"type": "bm25", "models": {"generator": "anthropic:claude-haiku-4-5"}}], judge_enabled=False)

    refs = collect_model_refs(cfg)
    assert refs.llm == ["anthropic:claude-haiku-4-5"] and refs.embedding == []
    assert missing_credentials(cfg) == []


def test_only_providers_that_are_referenced_get_their_sdk_warmed_up(tmp_path):
    openai_only = _config(tmp_path, [{"type": "bm25"}])
    mixed = _config(
        tmp_path,
        [{"type": "vector", "models": {"generator": "anthropic:claude-haiku-4-5", "embedding": "local:BAAI/bge-small-en-v1.5"}}],
    )
    assert "anthropic" not in warm_up_modules(openai_only) and "sentence_transformers" not in warm_up_modules(openai_only)
    assert {"anthropic", "sentence_transformers"} <= set(warm_up_modules(mixed))
