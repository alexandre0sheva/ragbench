from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from contextlib import nullcontext
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment

from ragbench.documents.schema import Document
from ragbench.models.errors import RateLimitError, TransientModelError
from ragbench.models.retry import call_with_retry
from ragbench.rag_systems.external_rag import ExternalRAG, ExternalReply, as_float, as_int
from ragbench.rag_systems.options import ExternalHttpOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.runtime.limiter import RateLimiter
from ragbench.utils.jsonpath import find, first

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
INGEST_BATCH = 100


def fill_env(text: str) -> str:
    """Replace every `${VAR}` with the environment variable; a variable that is not set fails, naming it (never a value)."""

    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"environment variable {name} is not set (used as ${{{name}}} in an external_http system)")
        return os.environ[name]

    return _ENV_VAR.sub(value, text)


@SYSTEMS.register("external_http")
class ExternalHttpRAG(ExternalRAG):
    spec = SystemSpec(
        type="external_http",
        title="External HTTP endpoint",
        summary="Your own RAG service: each question is sent to an HTTP endpoint and the JSON reply (answer, passages, usage, cost) is mapped with JSONPath",
        best_for="Benchmarking a RAG you already run (any stack) against the built-in systems",
        cost_profile="low",
        latency_profile="medium",
        requires_llm=False,
        agentic=False,
        options=ExternalHttpOptions,
        chunker=None,
        external=True,
    )
    options: ExternalHttpOptions

    def __init__(self, config, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self._template = SandboxedEnvironment(undefined=StrictUndefined).from_string(self.options.request)
        limits = self.options.limits
        self._limiter = RateLimiter(max_concurrent=limits.max_concurrent_requests, requests_per_minute=limits.requests_per_minute) if limits.is_set else None

    def _ingest_external(self, documents: list[Document]) -> None:
        if self.options.post_documents_to is None:
            return
        for start in range(0, len(documents), INGEST_BATCH):
            batch = [document.model_dump(mode="json") for document in documents[start : start + INGEST_BATCH]]
            self._send(self.options.post_documents_to, "POST", {"documents": batch})

    def _call(self, question: str, top_k: int) -> ExternalReply:
        rendered = self._template.render(question=question, top_k=top_k)
        try:
            payload = json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.name}: the request template did not render JSON ({exc}): {rendered[:200]}") from None
        reply = self._send(self.options.url, self.options.method, payload)
        mapping = self.options.response
        answer = first(reply, mapping.answer)
        if answer is None:
            raise ValueError(f"{self.name}: the reply has no value at {mapping.answer} (set retrieval.response.answer); it starts {json.dumps(reply)[:200]}")
        contexts = None
        if mapping.contexts is not None:
            contexts = [
                {"text": first(item, mapping.text), "doc_id": first(item, mapping.doc_id), "score": first(item, mapping.score)}
                for item in find(reply, mapping.contexts)
            ]
        return ExternalReply(
            answer=str(answer),
            contexts=contexts,
            prompt_tokens=as_int(first(reply, mapping.prompt_tokens)),
            completion_tokens=as_int(first(reply, mapping.completion_tokens)),
            cost_usd=as_float(first(reply, mapping.cost_usd)),
        )

    def _send(self, url_template: str, method: str, payload: dict[str, Any]) -> Any:
        """One JSON request with retries. Errors name the URL as configured (with `${VAR}`, so a secret in it is never shown)."""
        url = fill_env(url_template)
        headers = {"Accept": "application/json", **{name: fill_env(value) for name, value in self.options.headers.items()}}
        data = None
        if method == "GET":
            parts = urlsplit(url)
            query = "&".join(part for part in (parts.query, urlencode({k: v if isinstance(v, str) else json.dumps(v) for k, v in payload.items()})) if part)
            url = urlunsplit(parts._replace(query=query))
        else:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        def attempt() -> Any:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._limiter.acquire() if self._limiter is not None else nullcontext(), urllib.request.urlopen(request, timeout=self.options.timeout_s) as response:
                    body = response.read()
            except urllib.error.HTTPError as exc:
                detail = f"{url_template} answered HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}"
                if exc.code == 429:
                    raise RateLimitError(detail) from None
                if exc.code >= 500:
                    raise TransientModelError(detail) from None
                raise RuntimeError(f"{self.name}: {detail}") from None
            except TimeoutError:
                raise TransientModelError(f"{url_template} timed out after {self.options.timeout_s:g} s") from None
            except urllib.error.URLError as exc:
                raise TransientModelError(f"{url_template} could not be reached: {exc.reason}") from None
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise RuntimeError(f"{self.name}: {url_template} did not answer JSON: {body[:200].decode('utf-8', 'replace')}") from None

        try:
            return call_with_retry(attempt, max_attempts=self.options.retries + 1, base_delay=0.5, max_delay=10.0)
        except (RateLimitError, TransientModelError) as exc:
            raise RuntimeError(f"{self.name}: {exc} (after {self.options.retries + 1} attempt(s))") from None
