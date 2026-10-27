"""The OpenRouter relay of `cli_agent`: OpenCode's requests go through it, and the cost recorded is what OpenRouter billed (`usage.cost`), not
OpenCode's own estimate. Everything here talks to a fake upstream on 127.0.0.1; nothing reaches OpenRouter."""

from __future__ import annotations

import json
import stat
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from dataset_support import DOCUMENTS

from ragbench.config.schema import OpenRouterConfig, OpenRouterModelSettings, OpenRouterRouting, SystemConfig
from ragbench.documents.schema import Document
from ragbench.rag_systems import create_rag_system
from ragbench.rag_systems.openrouter_relay import OpenRouterRelay, _stream_cost
from ragbench.runtime.context import RuntimeContext, activate_runtime

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cli_agent"


class FakeOpenRouter:
    """Answers `/chat/completions` with a billed `usage.cost` (as one JSON reply, or as server-sent events when the request streams); a model
    named `broken` gets HTTP 500 and no cost. Every request's path, headers and body are kept."""

    def __init__(self, costs: list[float]) -> None:
        self.requests: list[dict] = []
        self._costs = list(costs)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                if body.get("model") == "broken":
                    return self._send(500, "application/json", b'{"error": {"message": "upstream failed"}}')
                usage = {"prompt_tokens": 100, "completion_tokens": 10, "cost": outer._costs.pop(0)}
                if not body.get("stream"):
                    reply = {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": usage}
                    return self._send(200, "application/json", json.dumps(reply).encode())
                events = [{"choices": [{"delta": {"content": "o"}}]}, {"choices": [{"delta": {"content": "k"}}]}, {"choices": [], "usage": usage}]
                stream = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"
                self._send(200, "text/event-stream", stream)

            def _send(self, status: int, content_type: str, raw: bytes) -> None:
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def upstream(monkeypatch):
    fake = FakeOpenRouter([0.004, 0.0065, 0.001])
    monkeypatch.setenv("OPENROUTER_BASE_URL", fake.base_url)  # never openrouter.ai
    yield fake
    fake.close()


def _post(url: str, body: dict) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer sk-or-test"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_the_relay_forwards_requests_and_adds_up_what_openrouter_billed_per_key(upstream):
    relay = OpenRouterRelay(extra=lambda model: {"provider": {"data_collection": "deny"}, "reasoning": {"effort": "low"}} if model == "z-ai/glm-5.3" else {})
    try:
        key, other = relay.open(), relay.open()
        status, raw = _post(f"{relay.base_url(key)}/chat/completions", {"model": "z-ai/glm-5.3", "messages": [], "reasoning": {"effort": "high"}})
        assert status == 200 and json.loads(raw)["choices"][0]["message"]["content"] == "ok"
        status, raw = _post(f"{relay.base_url(key)}/chat/completions", {"model": "z-ai/glm-5.3", "messages": [], "stream": True})
        assert status == 200 and raw.count(b"data:") == 4 and b'"cost": 0.0065' in raw  # the stream reaches the harness whole
        _post(f"{relay.base_url(other)}/chat/completions", {"model": "anthropic/claude-haiku-4.5", "messages": [], "usage": {"include": False}})
        status, _ = _post(f"{relay.base_url(other)}/chat/completions", {"model": "broken", "messages": []})
        assert status == 500

        bill = relay.take(key)
        assert (bill.cost_usd, bill.billed, bill.unbilled) == (pytest.approx(0.0105), 2, 0)
        other_bill = relay.take(other)
        assert (other_bill.cost_usd, other_bill.billed, other_bill.unbilled) == (pytest.approx(0.001), 1, 1)  # the failed request had no cost

        first, _, haiku, _ = upstream.requests
        assert first["path"] == "/api/v1/chat/completions" and first["headers"]["Authorization"] == "Bearer sk-or-test"
        assert first["headers"]["Accept-Encoding"] == "identity"  # so the relay can read the usage
        assert first["body"]["usage"] == {"include": True} and first["body"]["provider"] == {"data_collection": "deny"}
        assert first["body"]["reasoning"] == {"effort": "low"}  # RAGBench's setting wins over the harness's own
        assert haiku["body"]["usage"] == {"include": True} and "reasoning" not in haiku["body"]
        assert _post(f"{relay.base_url('nope')}/chat/completions", {})[0] == 404  # a closed or unknown key
    finally:
        relay.close()


def test_a_stream_is_read_to_its_end_for_the_cost_after_the_harness_has_gone():
    written: list[bytes] = []

    def write(line: bytes) -> bool:
        written.append(line)
        return False  # the harness was stopped at its step cap after the first line

    lines = [b'data: {"choices": []}\n', b"\n", b'data: {"choices": [], "usage": {"cost": 0.002}}\n', b"data: [DONE]\n"]
    assert _stream_cost(lines, write) == 0.002 and written == lines[:1]


# -- OpenCode through the relay ------------------------------------------------------------------------------------------------------------

FAKE_OPENCODE = """#!{python}
import json, sys, urllib.request
from pathlib import Path
workspace = Path(sys.argv[sys.argv.index("--dir") + 1])
config = json.loads((workspace / "opencode.json").read_text())
base = config["provider"]["openrouter"]["options"]["baseURL"]
for stream in (False, True):
    body = {{"model": "z-ai/glm-5.3", "messages": [{{"role": "user", "content": "q"}}], "stream": stream}}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={{"Content-Type": "application/json"}})
    urllib.request.urlopen(request, timeout=10).read()
sys.stdout.write(Path({fixture!r}).read_text().replace("{{workspace}}", str(workspace)))
"""


def test_opencode_on_openrouter_is_charged_what_openrouter_billed_not_its_own_estimate(upstream, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    opencode = bin_dir / "opencode"
    opencode.write_text(FAKE_OPENCODE.format(python=sys.executable, fixture=str(FIXTURES / "opencode_recorded.jsonl")), encoding="utf-8")
    opencode.chmod(opencode.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir))
    routing = OpenRouterConfig(models={"z-ai/glm-5.3": OpenRouterModelSettings(provider=OpenRouterRouting(data_collection="deny"))})
    with activate_runtime(RuntimeContext(openrouter=routing)):
        system = create_rag_system(SystemConfig(type="cli_agent", name="oc", retrieval={"preset": "opencode", "model": "openrouter:z-ai/glm-5.3"}))
        system.ingest([Document(doc_id=doc_id, path=f"corpus/{doc_id}.md", title=doc_id, text=text) for doc_id, text in DOCUMENTS.items()])
        result = system.answer_question("Who is the security lead?")

    turns = [step for step in result.steps if step.kind in ("llm", "generate")]
    assert len(turns) == 7 and result.metadata["external"]["cost_known"] is True
    # OpenCode said $0.00658 (its list-price estimate); OpenRouter billed $0.0105 for the two requests, spread over the turns by tokens.
    assert result.cost.llm_cost == pytest.approx(0.0105) and sum(step.cost.llm_cost for step in turns) == pytest.approx(0.0105)
    tokens = [step.prompt_tokens + step.completion_tokens for step in turns]
    assert turns[0].cost.llm_cost == pytest.approx(0.0105 * tokens[0] / sum(tokens))
    sent = upstream.requests[0]["body"]
    assert sent["provider"] == {"data_collection": "deny"} and sent["reasoning"] == {"effort": "low"} and sent["usage"] == {"include": True}
