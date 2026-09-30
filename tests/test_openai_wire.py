"""The real `openai` SDK against a mock HTTP transport: checks what actually goes over the wire and how replies are parsed.

Other tests use fake client objects; these make sure the request payloads and response handling work with the SDK itself.
"""

from __future__ import annotations

import json
import threading

import httpx
import openai
import pytest

from ragbench.models import retry as retry_module
from ragbench.models.errors import PermanentModelError, TransientModelError
from ragbench.models.providers.openai import OpenAIEmbeddingModel, OpenAILLM
from ragbench.runtime import RuntimeContext, activate_runtime


def _completion(text: str = "hello", prompt: int = 11, completion: int = 7) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "x", "code": None}}, headers=headers)


class Server:
    """Records every request and answers from a script (a list of responses or callables)."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.requests: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append((request.url.path, json.loads(request.content or b"{}")))
            step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return step(request) if callable(step) else step

    def client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key="sk-test-not-real",
            base_url="https://api.example.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self)),
            max_retries=0,
        )


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda _: None)


def test_chat_request_payload_and_response_parsing_with_the_real_sdk():
    server = Server(httpx.Response(200, json=_completion("the answer", 11, 7)))
    llm = OpenAILLM("gpt-6-luna", client=server.client())

    result = llm.generate([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], temperature=0, max_tokens=42, json_mode=True)

    path, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "gpt-6-luna" and body["temperature"] == 0 and body["max_completion_tokens"] == 42 and "max_tokens" not in body
    assert body["response_format"] == {"type": "json_object"} and body["messages"][1] == {"role": "user", "content": "hi"}
    assert result.text == "the answer" and (result.prompt_tokens, result.completion_tokens) == (11, 7)
    assert result.cost.llm_prompt_tokens == 11 and result.model == "gpt-6-luna"


def test_rate_limit_is_retried_using_the_retry_after_header_then_succeeds(monkeypatch):
    server = Server(_error(429, "slow down", {"retry-after": "3"}), _error(503, "overloaded"), httpx.Response(200, json=_completion("ok")))
    slept: list[float] = []
    monkeypatch.setattr(retry_module.time, "sleep", slept.append)

    result = OpenAILLM("gpt-6-luna", client=server.client()).generate([{"role": "user", "content": "q"}])

    assert result.text == "ok" and len(server.requests) == 3
    assert slept[0] >= 3.0  # honors Retry-After


def test_server_errors_exhaust_bounded_retries_and_never_touch_the_responses_api():
    server = Server(_error(500, "boom"))

    with pytest.raises(TransientModelError):
        OpenAILLM("gpt-6-luna", client=server.client()).generate([{"role": "user", "content": "q"}])

    assert len(server.requests) == 6 and {path for path, _ in server.requests} == {"/v1/chat/completions"}


def test_auth_errors_fail_immediately_without_retries():
    server = Server(_error(401, "bad key"))

    with pytest.raises(PermanentModelError):
        OpenAILLM("gpt-6-luna", client=server.client()).generate([{"role": "user", "content": "q"}])

    assert len(server.requests) == 1


def test_a_model_that_rejects_temperature_is_retried_once_without_it_and_remembered():
    server = Server(
        _error(400, "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported."),
        httpx.Response(200, json=_completion("ok")),
    )
    llm = OpenAILLM("some-reasoning-model", client=server.client())

    llm.generate([{"role": "user", "content": "q"}], temperature=0)
    llm.generate([{"role": "user", "content": "q"}], temperature=0)

    bodies = [body for _, body in server.requests]
    assert "temperature" in bodies[0] and "temperature" not in bodies[1] and "temperature" not in bodies[2]


def test_a_responses_only_model_falls_back_to_the_responses_api():
    responses_reply = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "codex-like",
        "status": "completed",
        "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "from responses", "annotations": []}]}],
        "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
    }

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return _error(404, "This model is only supported in v1/responses and not in v1/chat/completions.")
        return httpx.Response(200, json=responses_reply)

    server = Server(route)
    result = OpenAILLM("codex-like", client=server.client()).generate(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}], max_tokens=30, json_mode=True
    )

    assert [path for path, _ in server.requests] == ["/v1/chat/completions", "/v1/responses"]
    body = server.requests[1][1]
    assert body["max_output_tokens"] == 30 and "Return JSON only." in body["instructions"] and body["input"] == [{"role": "user", "content": "q"}]
    assert result.text == "from responses" and (result.prompt_tokens, result.completion_tokens) == (9, 4)


def _embedding_reply(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [{"object": "embedding", "index": i, "embedding": [float(t), 1.0]} for i, t in enumerate(texts)]
    return httpx.Response(200, json={"object": "list", "data": data, "model": "text-embedding-3-small", "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)}})


def test_embeddings_batches_payload_order_and_token_totals_with_parallel_workers():
    server = Server(_embedding_reply)
    texts = [str(i) for i in range(1, 26)]

    with activate_runtime(RuntimeContext(ingest_workers=4)):
        result = OpenAIEmbeddingModel("text-embedding-3-small", batch_size=10, client=server.client()).embed_texts(texts)

    assert sorted(len(body["input"]) for _, body in server.requests) == [5, 10, 10]
    assert all(body["model"] == "text-embedding-3-small" and path == "/v1/embeddings" for path, body in server.requests)
    assert result.vectors.shape == (25, 2) and result.input_tokens == 25
    first_components = [float(row[0]) for row in result.vectors]
    assert first_components == sorted(first_components), "batch order was not preserved"


def test_embedding_rate_limit_retries_only_the_failed_batch():
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return _error(429, "slow down")
        return _embedding_reply(request)

    result = OpenAIEmbeddingModel("text-embedding-3-small", batch_size=2, client=Server(flaky).client()).embed_texts(["1", "2", "3"])

    assert calls["n"] == 3 and result.vectors.shape == (3, 2)  # batch 1 twice, batch 2 once


def _tool_call_completion() -> dict:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"query\": \"refund window\"}"}},
                        {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": "not json"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
    }


_SEARCH_TOOL = {"type": "function", "function": {"name": "search", "description": "Search.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}


def test_tools_are_sent_and_tool_calls_parsed_with_the_real_sdk():
    from ragbench.models.llms import ToolCall

    server = Server(httpx.Response(200, json=_tool_call_completion()))
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
    ]

    result = OpenAILLM("gpt-5.6-terra", client=server.client()).generate(history, tools=[_SEARCH_TOOL])

    body = server.requests[0][1]
    assert body["tools"] == [_SEARCH_TOOL] and body["messages"] == history
    assert result.finish_reason == "tool_calls" and result.text == ""
    assert result.tool_calls[0] == ToolCall(id="call_1", name="search", arguments={"query": "refund window"})
    assert result.tool_calls[1].arguments == {"_raw_arguments": "not json"}, "malformed arguments are surfaced, not dropped"
    assert result.assistant_message()["tool_calls"][0]["function"]["arguments"] == '{"query": "refund window"}'


def test_gpt6_models_are_sent_reasoning_effort_none_and_it_is_dropped_if_an_endpoint_rejects_it():
    server = Server(httpx.Response(200, json=_completion("ok")))
    OpenAILLM("gpt-6-luna", client=server.client()).generate([{"role": "user", "content": "q"}], max_tokens=20)
    assert server.requests[0][1]["reasoning_effort"] == "none" and server.requests[0][1]["max_completion_tokens"] == 20

    picky = Server(_error(400, "Unrecognized request argument supplied: reasoning_effort"), httpx.Response(200, json=_completion("ok")))
    llm = OpenAILLM("gpt-6-luna", client=picky.client())
    llm.generate([{"role": "user", "content": "q"}])
    llm.generate([{"role": "user", "content": "q"}])
    bodies = [body for _, body in picky.requests]
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[1] and "reasoning_effort" not in bodies[2]


def test_a_responses_only_model_cannot_be_given_tools():
    def route(request: httpx.Request) -> httpx.Response:
        return _error(404, "This model is only supported in v1/responses and not in v1/chat/completions.")

    with pytest.raises(PermanentModelError, match="tool calling"):
        OpenAILLM("codex-like", client=Server(route).client()).generate([{"role": "user", "content": "q"}], tools=[_SEARCH_TOOL])
