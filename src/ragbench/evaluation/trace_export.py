"""Trace export: a finished run's per-question traces as spans for Phoenix, Langfuse or any OpenTelemetry backend.

Everything is built from the run's files (`per_question_results.jsonl`, `run_manifest.json`), so exporting needs no API and spends nothing.
Each answered question is one trace: a root `ragbench.answer` span (question, answer, system, score, cost) with one child per step of the
system (retrieve, rerank, llm, tool, ...). Ids are derived from the run, system, question and trial, so exporting twice never duplicates
a trace in a backend. A run records how long each step took, not when it began: a system's traces start at the run's start and follow
one another (different systems ran side by side), and a trace's steps follow one another, so durations are real and offsets are laid out.

Two attribute conventions: `openinference` (Phoenix, Langfuse, Arize: `openinference.span.kind`, `input.value`, `llm.token_count.*`,
`retrieval.documents.*`) and `otel` (the OpenTelemetry GenAI names: `gen_ai.operation.name`, `gen_ai.usage.*`). Never exported: the
reference answer and the judge's reasoning, because a trace backend is somebody else's server.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ragbench.rag_systems.external_http_rag import fill_env
from ragbench.utils.extras import require_extra
from ragbench.utils.jsonl import read_jsonl

Convention = Literal["openinference", "otel"]
ROOT_NAME = "ragbench.answer"
SERVICE_NAME = "ragbench"
BATCH_SPANS = 512
SEND_TIMEOUT_S = 15.0
OPENINFERENCE_KINDS: dict[str, str] = {
    "retrieve": "RETRIEVER",
    "rerank": "RERANKER",
    "llm": "LLM",
    "generate": "LLM",
    "tool": "TOOL",
    "embed": "EMBEDDING",
    "route": "CHAIN",
    "grade": "CHAIN",
    "external": "CHAIN",
}
GENAI_OPERATIONS: dict[str, str] = {
    "retrieve": "retrieval",
    "rerank": "rerank",
    "llm": "chat",
    "generate": "chat",
    "tool": "execute_tool",
    "embed": "embeddings",
}
_LLM_KINDS = ("llm", "generate")
Attribute = str | int | float | bool


class TraceExportError(ValueError):
    """The run cannot be exported as asked (no rows, an unknown system) or the collector did not take the spans."""


@dataclass
class Span:
    trace_id: int
    span_id: int
    parent_id: int | None
    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Attribute] = field(default_factory=dict)
    error: str | None = None
    resource: dict[str, str] = field(default_factory=dict)


# -- building spans ---------------------------------------------------------------------------------------------------------------------------


def _digest(*parts: object, size: int) -> int:
    value = int.from_bytes(hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()[:size], "big")
    return value or 1  # an all-zero id is invalid in OpenTelemetry


def _run_start_ns(run_dir: Path) -> int:
    manifest = run_dir / "run_manifest.json"
    started: str | None = None
    if manifest.is_file():
        try:
            started = json.loads(manifest.read_text(encoding="utf-8")).get("started_utc")
        except (OSError, ValueError):
            started = None
    try:
        moment = datetime.fromisoformat(str(started).replace("Z", "+00:00")) if started else datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC)
    except ValueError:
        moment = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=UTC)
    return int(moment.timestamp()) * 1_000_000_000


def _ns(milliseconds: Any) -> int:
    try:
        return max(0, int(float(milliseconds or 0) * 1_000_000))
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _total_cost(cost: Any) -> float:
    return _number(cost.get("total_cost")) or 0.0 if isinstance(cost, dict) else 0.0


def _root_attributes(row: dict[str, Any], run: str, convention: Convention) -> dict[str, Attribute]:
    question, answer = str(row.get("question") or ""), str(row.get("answer") or "")
    attributes: dict[str, Attribute] = {"ragbench.run": run, "ragbench.system": str(row.get("system")), "ragbench.question_id": str(row.get("question_id"))}
    if convention == "openinference":
        attributes |= {"openinference.span.kind": "CHAIN", "input.value": question, "output.value": answer, "session.id": run}
    else:
        attributes |= {"gen_ai.operation.name": "invoke_workflow", "ragbench.input": question, "ragbench.output": answer}
    for key in ("system_type", "category", "failure_type", "difficulty"):
        if row.get(key):
            attributes[f"ragbench.{key}"] = str(row[key])
    if row.get("trial") is not None:
        attributes["ragbench.trial"] = int(row["trial"])
    score = _number((row.get("answer_judge") or {}).get("answer_score"))
    if score is not None:
        attributes["ragbench.answer_score"] = score
    attributes["ragbench.cost_usd"] = _total_cost(row.get("cost"))
    return attributes


def _step_attributes(step: dict[str, Any], convention: Convention) -> dict[str, Attribute]:
    kind = str(step.get("kind"))
    metadata = step.get("metadata") or {}
    tokens = (int(step.get("prompt_tokens") or 0), int(step.get("completion_tokens") or 0))
    cost = _total_cost(step.get("cost"))
    attributes: dict[str, Attribute] = {"ragbench.step.kind": kind, "ragbench.step.name": str(step.get("name"))}
    if cost:
        attributes["ragbench.cost_usd"] = cost
    doc_ids = [str(doc) for doc in metadata.get("doc_ids") or []]
    shown = {key: value for key, value in metadata.items() if key != "doc_ids"}
    if convention == "openinference":
        attributes["openinference.span.kind"] = OPENINFERENCE_KINDS.get(kind, "CHAIN")
        for label, key in (("input.value", "input_preview"), ("output.value", "output_preview")):
            if step.get(key) is not None:
                attributes[label] = str(step[key])
        if kind in _LLM_KINDS and any(tokens):
            attributes |= {"llm.token_count.prompt": tokens[0], "llm.token_count.completion": tokens[1], "llm.token_count.total": sum(tokens)}
        if kind in _LLM_KINDS and cost:
            attributes["llm.cost.total"] = cost
        if kind == "tool":
            attributes["tool.name"] = str(step.get("name"))
        for index, doc in enumerate(doc_ids):
            attributes[f"retrieval.documents.{index}.document.id"] = doc
        if shown:
            attributes["metadata"] = json.dumps(shown, ensure_ascii=False, default=str)
    else:
        attributes["gen_ai.operation.name"] = GENAI_OPERATIONS.get(kind, "invoke_workflow")
        for label, key in (("ragbench.input", "input_preview"), ("ragbench.output", "output_preview")):
            if step.get(key) is not None:
                attributes[label] = str(step[key])
        if kind in _LLM_KINDS and any(tokens):
            attributes |= {"gen_ai.usage.input_tokens": tokens[0], "gen_ai.usage.output_tokens": tokens[1]}
        if kind == "tool":
            attributes["gen_ai.tool.name"] = str(step.get("name"))
        if doc_ids:
            attributes["ragbench.doc_ids"] = ", ".join(doc_ids)
        if shown:
            attributes["ragbench.metadata"] = json.dumps(shown, ensure_ascii=False, default=str)
    return attributes


def _trace_spans(row: dict[str, Any], run: str, start_ns: int, convention: Convention, resource: dict[str, str]) -> list[Span]:
    system, question_id, trial = row.get("system"), row.get("question_id"), row.get("trial")
    trace_id = _digest(run, system, question_id, trial, size=16)
    steps = row.get("steps") or []
    durations = [_ns(step.get("latency_ms")) for step in steps]
    duration = max(_ns(row.get("latency_ms")), sum(durations))
    root = Span(trace_id, _digest(trace_id, "root", size=8), None, ROOT_NAME, start_ns, start_ns + duration, _root_attributes(row, run, convention), row.get("error") or None, resource)
    spans = [root]
    cursor = start_ns
    for index, (step, length) in enumerate(zip(steps, durations, strict=True)):
        metadata = step.get("metadata") or {}
        error = str(metadata["error"]) if metadata.get("error") else None
        spans.append(Span(trace_id, _digest(trace_id, index, size=8), root.span_id, f"{step.get('kind')}:{step.get('name')}", cursor, cursor + length, _step_attributes(step, convention), error, resource))
        cursor += length
    return spans


def build_spans(
    run_dir: Path,
    *,
    convention: Convention = "openinference",
    systems: Collection[str] | None = None,
    question_ids: Collection[str] | None = None,
    limit: int | None = None,
    project: str | None = None,
) -> list[Span]:
    """The spans of a finished run, one trace per answered question (and trial), in the run's row order.

    `systems` and `question_ids` narrow the export, `limit` keeps the first N traces, `project` names the project a backend files them under
    (the run's name by default).
    """
    path = run_dir / "per_question_results.jsonl"
    if not path.is_file():
        raise TraceExportError(f"{run_dir} has no per_question_results.jsonl, so there are no traces to export.")
    rows = read_jsonl(path)
    present = sorted({str(row.get("system")) for row in rows})
    unknown = sorted(set(systems or ()) - set(present))
    if unknown:
        raise TraceExportError(f"No system {', '.join(unknown)} in this run. Systems: {', '.join(present) or 'none'}.")
    run = run_dir.name
    resource = {"service.name": SERVICE_NAME, "openinference.project.name": project or run}
    origin = _run_start_ns(run_dir)
    cursors: dict[str, int] = {}
    spans: list[Span] = []
    traces = 0
    for row in rows:
        system = str(row.get("system"))
        if (systems and system not in systems) or (question_ids and str(row.get("question_id")) not in question_ids):
            continue
        if limit is not None and traces >= limit:
            break
        made = _trace_spans(row, run, cursors.get(system, origin), convention, resource)
        cursors[system] = made[0].end_ns
        spans.extend(made)
        traces += 1
    if not spans:
        raise TraceExportError("Nothing to export: no question of the run matches --system / --question.")
    return spans


def span_record(span: Span) -> dict[str, Any]:
    """One span as a JSON line (the OTLP/JSON field names, flat)."""
    return {
        "trace_id": format(span.trace_id, "032x"),
        "span_id": format(span.span_id, "016x"),
        "parent_span_id": None if span.parent_id is None else format(span.parent_id, "016x"),
        "name": span.name,
        "start_time_unix_nano": span.start_ns,
        "end_time_unix_nano": span.end_ns,
        "status": {"code": "ERROR" if span.error else "OK", **({"message": span.error} if span.error else {})},
        "attributes": span.attributes,
        "resource": span.resource,
    }


def write_spans(spans: list[Span], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for span in spans:
            f.write(json.dumps(span_record(span), ensure_ascii=False) + "\n")


def trace_count(spans: list[Span]) -> int:
    return sum(1 for span in spans if span.parent_id is None)


# -- sending ----------------------------------------------------------------------------------------------------------------------------------


def parse_headers(pairs: list[str]) -> dict[str, str]:
    """`NAME=VALUE` pairs with `${ENV_VAR}` filled in (a variable that is not set fails, naming it, never a value)."""
    headers: dict[str, str] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name.strip():
            raise TraceExportError(f"A header must look like NAME=VALUE (got '{name or pair}').")
        headers[name.strip()] = fill_env(value, f"the header {name.strip()}")
    return headers


def _batches(spans: list[Span]) -> Iterator[list[Span]]:
    for start in range(0, len(spans), BATCH_SPANS):
        yield spans[start : start + BATCH_SPANS]


def send_spans(spans: list[Span], endpoint: str, headers: dict[str, str] | None = None) -> int:
    """Send the spans to an OTLP/HTTP endpoint (`.../v1/traces`) in batches; return how many went. Needs the `otel` extra."""
    require_extra("opentelemetry.sdk", "Sending traces", "otel", package="opentelemetry-sdk")
    require_extra("opentelemetry.exporter.otlp.proto.http", "Sending traces", "otel", package="opentelemetry-exporter-otlp-proto-http")
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope
    from opentelemetry.trace import SpanContext, SpanKind, Status, StatusCode, TraceFlags

    scope = InstrumentationScope(SERVICE_NAME)
    resources: dict[tuple[tuple[str, str], ...], Resource] = {}

    def readable(span: Span) -> ReadableSpan:
        resource = resources.setdefault(tuple(sorted(span.resource.items())), Resource.create(span.resource))
        context = SpanContext(span.trace_id, span.span_id, is_remote=False, trace_flags=TraceFlags(TraceFlags.SAMPLED))
        parent = None if span.parent_id is None else SpanContext(span.trace_id, span.parent_id, is_remote=False, trace_flags=TraceFlags(TraceFlags.SAMPLED))
        status = Status(StatusCode.ERROR, span.error[:500]) if span.error else Status(StatusCode.OK)
        return ReadableSpan(
            span.name, context, parent, resource, dict(span.attributes), kind=SpanKind.INTERNAL, status=status, start_time=span.start_ns, end_time=span.end_ns, instrumentation_scope=scope
        )

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers or None, timeout=SEND_TIMEOUT_S)
    sent = 0
    try:
        for batch in _batches(spans):
            if exporter.export([readable(span) for span in batch]) is not SpanExportResult.SUCCESS:
                raise TraceExportError(f"The collector at {endpoint} did not accept the spans (rejected or unreachable) after {sent} of {len(spans)} were sent; see the log above for its answer.")
            sent += len(batch)
    finally:
        exporter.shutdown()
    return sent
