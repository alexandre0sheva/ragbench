"""A local OpenAI-compatible server backed by RAGBench's own mock models: real HTTP, real SDK, no network, no cost.

Point the SDK at it with `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` and any obviously fake key. Every Nth request
is answered with HTTP 429 so retry handling is exercised too.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ragbench.models.embeddings import HashingEmbeddingModel
from ragbench.models.llms import MockLLM

JUDGE_REPLY = json.dumps(
    {
        "correctness": 4,
        "faithfulness": 4,
        "completeness": 4,
        "relevance": 4,
        "citation_quality": 4,
        "is_supported_by_context": True,
        "is_hallucinated": False,
        "reasoning": "fake judge",
    }
)


class FakeOpenAIServer:
    def __init__(self, inject_429_every: int = 11, latency_s: float = 0.003) -> None:
        self.stats: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._llm = MockLLM()
        self._embedder = HashingEmbeddingModel()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                with outer._lock:
                    outer.stats[self.path] += 1
                    outer.stats["total"] += 1
                    number = outer.stats["total"]
                time.sleep(latency_s)
                if inject_429_every and number % inject_429_every == 0:
                    with outer._lock:
                        outer.stats["429_injected"] += 1
                    return self._reply(429, {"error": {"message": "slow down", "type": "rate_limit_exceeded"}}, {"retry-after": "0"})
                if self.path.endswith("/chat/completions"):
                    with outer._lock:
                        outer.stats["max_completion_tokens_sent"] += "max_completion_tokens" in body
                        outer.stats["max_tokens_sent"] += "max_tokens" in body
                    json_mode = body.get("response_format", {}).get("type") == "json_object"
                    system = " ".join(m["content"] for m in body["messages"] if m["role"] == "system")
                    if json_mode and "evaluation judge" in system:
                        text, prompt_tokens, completion_tokens = JUDGE_REPLY, 100, 30
                    else:
                        result = outer._llm.generate(body["messages"], json_mode=json_mode)
                        text, prompt_tokens, completion_tokens = result.text, result.prompt_tokens, result.completion_tokens
                    return self._reply(
                        200,
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body["model"],
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
                        },
                    )
                if self.path.endswith("/embeddings"):
                    texts = body["input"]
                    embedded = outer._embedder.embed_texts(texts)
                    data = [{"object": "embedding", "index": i, "embedding": embedded.vectors[i].tolist()} for i in range(len(texts))]
                    return self._reply(
                        200,
                        {"object": "list", "model": body["model"], "data": data, "usage": {"prompt_tokens": embedded.input_tokens, "total_tokens": embedded.input_tokens}},
                    )
                self._reply(404, {"error": {"message": "unknown route"}})

            def _reply(self, status: int, payload: dict, headers: dict[str, str] | None = None) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def reset(self) -> None:
        with self._lock:
            self.stats.clear()

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
