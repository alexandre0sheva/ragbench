"""A stand-in for "your deployed RAG": a tiny HTTP service that answers questions over a folder of documents.

    python examples/fake_rag_server.py --docs data/demo/docs --port 8765
    ragbench run --config configs/external_example.yaml --mock

It needs only the standard library and no key: retrieval is word overlap over paragraphs, and the answer is the best-matching
sentence. RAGBench's tests start it in a thread to exercise `type: external_http` end to end.

`POST /ask` takes `{"question": ..., "top_k": 5}` (or `{"query": ..., "k": 5}`; `GET /ask?question=...&top_k=5` works too) and
returns `{"answer", "contexts": [{"doc_id", "text", "score"}], "usage": {"prompt_tokens", "completion_tokens"}, "cost_usd"}`.
`POST /documents` with `{"documents": [{"doc_id", "text", ...}]}` replaces the corpus (what `post_documents_to` sends).
Query parameters on `/ask` change the reply for tests: `ids=path|opaque|none` (document ids as file names, as ids RAGBench
cannot map, or left out), `usage=tokens|none` (token counts only, or no usage and no cost), `shape=nested` (another JSON
layout), `fail=N` (answer 503 to the first N requests), `sleep=S` (wait S seconds first).
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

WORD = re.compile(r"[a-z0-9$][a-z0-9$\-]*")
STOP = {"the", "a", "an", "of", "is", "are", "what", "who", "which", "when", "how", "does", "do", "in", "for", "and", "to", "per", "did", "was"}
PRICE_PER_TOKEN = (0.10 / 1_000_000, 0.50 / 1_000_000)  # what the fake "bills" for its own model, USD per prompt / completion token


def _terms(text: str) -> set[str]:
    return {word for word in WORD.findall(text.lower()) if word not in STOP}


class FakeRag:
    """The service's state: the corpus split into paragraphs, the requests it saw, and how many failures are left to serve."""

    def __init__(self, documents: dict[str, str] | None = None) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.failures_served: dict[str, int] = {}
        self.load(documents or {})

    def load(self, documents: dict[str, str]) -> None:
        with self.lock:
            self.documents = dict(documents)
            self.passages = [(doc_id, part.strip()) for doc_id, text in sorted(documents.items()) for part in text.split("\n\n") if part.strip() and not part.startswith("#")]

    def ask(self, question: str, top_k: int, options: dict[str, str]) -> dict[str, Any]:
        wanted = _terms(question)
        scored = sorted(((len(wanted & _terms(text)), -index, doc_id, text) for index, (doc_id, text) in enumerate(self.passages)), reverse=True)
        hits = [(score, doc_id, text) for score, _, doc_id, text in scored[:top_k] if score > 0]
        if hits:
            sentences = re.split(r"(?<=[.!?])\s+", hits[0][2])
            answer = max(sentences, key=lambda sentence: (len(wanted & _terms(sentence)), -len(sentence)))
        else:
            answer = "I could not find the answer in the provided documents."
        contexts = [{"doc_id": self._doc_id(doc_id, options.get("ids", "")), "text": text, "score": float(score)} for score, doc_id, text in hits]
        for context in contexts:
            if context["doc_id"] is None:
                del context["doc_id"]
        prompt = len(question.split()) + sum(len(text.split()) for _, _, text in hits)
        completion = len(answer.split())
        reply: dict[str, Any] = {"answer": answer, "contexts": contexts}
        if options.get("usage") != "none":
            reply["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion}
            if options.get("usage") != "tokens":
                reply["cost_usd"] = round(prompt * PRICE_PER_TOKEN[0] + completion * PRICE_PER_TOKEN[1], 12)
        if options.get("shape") == "nested":
            return {"data": {"output": {"text": answer}, "sources": [{"meta": {"id": c.get("doc_id")}, "content": c["text"]} for c in contexts]}}
        return reply

    @staticmethod
    def _doc_id(doc_id: str, style: str) -> str | None:
        if style == "path":
            return f"docs/{doc_id}.md"
        if style == "opaque":
            return f"kb-{sum(map(ord, doc_id)) * 7919:x}"
        return None if style == "none" else doc_id


def _handler(rag: FakeRag) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the base class's name)
            pass

        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self, payload: dict[str, Any]) -> None:
            url = urlparse(self.path)
            options = {key: values[-1] for key, values in parse_qs(url.query).items()}
            with rag.lock:
                rag.requests.append({"method": self.command, "path": url.path, "headers": dict(self.headers.items()), "body": payload})
            if "sleep" in options:
                time.sleep(float(options["sleep"]))
            if "fail" in options:
                with rag.lock:
                    served = rag.failures_served.get(self.path, 0)
                    rag.failures_served[self.path] = served + 1
                if served < int(options["fail"]):
                    return self._reply(503, {"error": "temporarily unavailable"})
            if url.path == "/documents":
                rag.load({str(d["doc_id"]): str(d["text"]) for d in payload.get("documents", [])})
                return self._reply(200, {"documents": len(rag.documents)})
            if url.path != "/ask":
                return self._reply(404, {"error": f"no route {url.path}"})
            question = payload.get("question") or payload.get("query")
            if not question:
                return self._reply(400, {"error": "missing question"})
            self._reply(200, rag.ask(str(question), int(payload.get("top_k") or payload.get("k") or 5), options))

        def do_POST(self) -> None:  # noqa: N802 (http.server's naming)
            length = int(self.headers.get("Content-Length") or 0)
            self._handle(json.loads(self.rfile.read(length) or b"{}"))

        def do_GET(self) -> None:  # noqa: N802
            self._handle({key: values[-1] for key, values in parse_qs(urlparse(self.path).query).items()})

    return Handler


def make_server(documents: dict[str, str] | None = None, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, FakeRag]:
    rag = FakeRag(documents)
    server = ThreadingHTTPServer((host, port), _handler(rag))
    server.daemon_threads = True
    return server, rag


def read_folder(folder: Path) -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.md"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, help="folder of .md documents to answer from (else wait for POST /documents)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server, _ = make_server(read_folder(args.docs) if args.docs else None, args.host, args.port)
    print(f"Fake RAG listening on http://{args.host}:{server.server_address[1]}/ask", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
