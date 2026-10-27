"""A local relay between an agent harness and OpenRouter that records what OpenRouter billed.

A harness such as OpenCode prices its own tokens from a static price table, and for an `openrouter:` model that table holds the cheapest
upstream's price. OpenRouter routes each request to one of the model's upstream providers and bills that provider's price, which for an
open-weight model can be several times the cheapest one, so the harness's cost under-reports the bill. The harness is pointed at this relay
instead (its OpenRouter base URL): every request is forwarded to OpenRouter with `usage: {include: true}` and the `openrouter:` section's
routing and reasoning settings, and the `usage.cost` of every response, OpenRouter's own bill, is added up per question.

The relay listens on 127.0.0.1 only and forwards to OpenRouter only (`OPENROUTER_BASE_URL` when set, for tests and proxies). Each question
gets its own path prefix (`http://127.0.0.1:<port>/<key>`), so concurrent questions are billed apart.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ragbench.models.providers.openrouter import BASE_URL, BASE_URL_ENV

logger = logging.getLogger(__name__)

UPSTREAM_TIMEOUT_S = 600
DRAIN_TIMEOUT_S = 30.0  # how long `take` waits for a question's in-flight requests (a stopped harness's last response) to finish
# Not forwarded: hop-by-hop headers, and the ones the relay sets itself (`accept-encoding: identity`, so that it can read the usage).
_SKIP_REQUEST = {"host", "content-length", "connection", "keep-alive", "accept-encoding", "transfer-encoding", "proxy-connection", "te", "upgrade"}
_SKIP_RESPONSE = {"content-length", "connection", "keep-alive", "transfer-encoding", "content-encoding"}


@dataclass
class Bill:
    """What OpenRouter billed for one question's requests. `unbilled` counts responses that carried no cost (a failed request, or one cut off
    before its usage arrived), so a bill with `unbilled > 0` is a lower bound."""

    cost_usd: float = 0.0
    billed: int = 0
    unbilled: int = 0
    in_flight: int = 0


class OpenRouterRelay:
    """Forwards a harness's OpenRouter requests and adds up their billed cost per key; `extra(model)` gives the request parameters every request
    for a model goes out with, in place of the harness's own."""

    def __init__(self, extra: Callable[[str], dict[str, Any]] | None = None, upstream: str | None = None):
        self.upstream = (upstream or os.environ.get(BASE_URL_ENV) or BASE_URL).rstrip("/")
        self._extra = extra or (lambda model: {})
        self._bills: dict[str, Bill] = {}
        self._changed = threading.Condition()
        relay = self

        class Handler(_RelayHandler):
            owner = relay

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="openrouter-relay", daemon=True).start()

    def open(self) -> str:
        """A new key: the base URL the harness should use for one question."""
        key = secrets.token_hex(8)
        with self._changed:
            self._bills[key] = Bill()
        return key

    def base_url(self, key: str) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/{key}"

    def take(self, key: str, timeout_s: float = DRAIN_TIMEOUT_S) -> Bill:
        """The key's bill, once its in-flight requests are done (or `timeout_s` has passed); the key is closed."""
        with self._changed:
            self._changed.wait_for(lambda: self._bills[key].in_flight == 0, timeout=timeout_s)
            return self._bills.pop(key)

    def forget(self, key: str) -> None:
        """Close a key without waiting (the question failed before its bill was taken)."""
        with self._changed:
            self._bills.pop(key, None)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # -- called by the handler ----------------------------------------------------------------------------------------------------------

    def _start(self, key: str) -> bool:
        with self._changed:
            bill = self._bills.get(key)
            if bill is None:
                return False
            bill.in_flight += 1
            return True

    def _finish(self, key: str, cost: float | None) -> None:
        with self._changed:
            bill = self._bills.get(key)
            if bill is None:  # `take` gave up waiting for this request
                return
            bill.in_flight -= 1
            if cost is None:
                bill.unbilled += 1
            else:
                bill.billed += 1
                bill.cost_usd += cost
            self._changed.notify_all()

    def _body(self, raw: bytes) -> bytes:
        """The request body with billing usage on and the model's settings in force (a body that is not a JSON object passes unchanged)."""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            return raw
        if not isinstance(body, dict):
            return raw
        usage = body.get("usage")
        body["usage"] = {**(usage if isinstance(usage, dict) else {}), "include": True}
        body.update(self._extra(str(body.get("model") or "")))
        return json.dumps(body).encode()


class _RelayHandler(BaseHTTPRequestHandler):
    owner: OpenRouterRelay

    def do_POST(self) -> None:
        self._relay()

    def do_GET(self) -> None:
        self._relay()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler's signature
        logger.debug("openrouter relay: " + format, *args)

    def _relay(self) -> None:
        relay = self.owner
        key, _, rest = self.path.lstrip("/").partition("/")
        if not relay._start(key):
            self.send_error(404, "unknown relay key")
            return
        cost: float | None = None
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            data = relay._body(raw) if self.command == "POST" else None
            headers = {name: value for name, value in self.headers.items() if name.lower() not in _SKIP_REQUEST}
            request = urllib.request.Request(f"{relay.upstream}/{rest}", data=data, headers={**headers, "Accept-Encoding": "identity"}, method=self.command)
            try:
                response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_S)  # noqa: S310 - a fixed https upstream
            except urllib.error.HTTPError as error:
                response = error
            with response:
                cost = self._send(response)
        except Exception as exc:  # noqa: BLE001 - the harness sees a failed request; the relay keeps serving
            logger.warning("openrouter relay: %s %s failed: %s", self.command, rest, exc)
            try:
                self.send_error(502, "relay error")
            except OSError:
                pass
        finally:
            relay._finish(key, cost)

    def _send(self, response: Any) -> float | None:
        """Forward the upstream response and return the `usage.cost` it carried. A server-sent event stream is forwarded line by line and read to
        its end even if the harness has gone (stopped at its step cap), so the cost in its last event is not lost."""
        self.send_response(response.status)
        for name, value in response.headers.items():
            if name.lower() not in _SKIP_RESPONSE:
                self.send_header(name, value)
        streaming = "text/event-stream" in (response.headers.get("Content-Type") or "")
        if not streaming:
            body = response.read()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)
            return _cost(_json(body))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        return _stream_cost(response, self._write)

    def _write(self, data: bytes) -> bool:
        try:
            self.wfile.write(data)
            self.wfile.flush()
            return True
        except OSError:  # the harness went away; keep reading upstream for the usage
            return False


def _stream_cost(lines: Iterable[bytes], write: Callable[[bytes], bool]) -> float | None:
    cost = None
    connected = True
    for line in lines:
        if connected:
            connected = write(line)
        if line.startswith(b"data:"):
            found = _cost(_json(line[5:].strip()))
            cost = found if found is not None else cost
    return cost


def _json(data: bytes) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def _cost(payload: Any) -> float | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    value = usage.get("cost") if isinstance(usage, dict) else None
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
