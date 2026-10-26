"""Trace export: a finished run's per-question traces as OpenInference / OpenTelemetry spans (offline JSONL, or OTLP/HTTP to Phoenix, Langfuse, ...)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli_support import invoke, mock_run

from ragbench.evaluation import trace_export
from ragbench.evaluation.trace_export import TraceExportError, build_spans, span_record
from ragbench.utils.jsonl import read_jsonl, write_jsonl

STARTED = "2026-10-26T10:00:00Z"
REFERENCE = "THE-REFERENCE-ANSWER-THAT-MUST-STAY-HOME"


def step(kind: str, name: str, *, latency: float = 10.0, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "name": name, "latency_ms": latency, "cost": {"total_cost": 0.0}, "prompt_tokens": 0, "completion_tokens": 0, "input_preview": None, "output_preview": None, "metadata": {}, **extra}


def row(system: str, question_id: str, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "system": system,
        "system_type": "bm25",
        "question_id": question_id,
        "question": f"What is {question_id}?",
        "category": "direct_fact",
        "reference_answer": REFERENCE,
        "answer": f"{system} answers {question_id}",
        "answer_judge": {"answer_score": 4.5, "reasoning": "the judge's reasoning"},
        "failure_type": "no_failure",
        "cost": {"total_cost": 0.002},
        "latency_ms": 100.0,
        "error": None,
        "steps": [
            step("retrieve", "bm25_search", latency=20.0, output_preview="2 chunks from 2 docs: d1, d2", metadata={"doc_ids": ["d1", "d2"]}),
            step("rerank", "cross_encoder", latency=5.0, metadata={"doc_ids": ["d2", "d1"]}),
            step("generate", "answer", latency=60.0, input_preview="the prompt", output_preview="the answer", prompt_tokens=300, completion_tokens=40, cost={"total_cost": 0.002}),
        ],
    }
    return {**base, **extra}


def write_run(root: Path, rows: list[dict[str, Any]], *, name: str = "run") -> Path:
    run_dir = root / name
    run_dir.mkdir()
    write_jsonl(run_dir / "per_question_results.jsonl", rows)
    (run_dir / "run_manifest.json").write_text(json.dumps({"started_utc": STARTED}), encoding="utf-8")
    (run_dir / "metrics_summary.csv").write_text("system\nalpha\nbeta\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    rows = [row("alpha", "q1"), row("alpha", "q2"), row("beta", "q1", steps=[step("generate", "answer", latency=30.0)], latency_ms=30.0)]
    return write_run(tmp_path, rows)


def by_name(spans, name: str):
    return [s for s in spans if s.name == name]


# -- the span tree ------------------------------------------------------------------------------------------------------------------------------


def test_every_answer_is_one_trace_with_a_root_and_one_child_per_step(run_dir):
    spans = build_spans(run_dir)
    assert len({s.trace_id for s in spans}) == 3  # (alpha, q1), (alpha, q2), (beta, q1)
    roots = [s for s in spans if s.parent_id is None]
    assert len(roots) == 3 and all(s.name == "ragbench.answer" for s in roots)
    first = [s for s in spans if s.trace_id == roots[0].trace_id]
    assert [s.name for s in first] == ["ragbench.answer", "retrieve:bm25_search", "rerank:cross_encoder", "generate:answer"]
    assert all(s.parent_id == roots[0].span_id for s in first[1:])
    assert len({s.span_id for s in spans}) == len(spans)


def test_ids_are_the_same_on_every_export_so_a_backend_never_holds_duplicates(run_dir):
    assert [(s.trace_id, s.span_id) for s in build_spans(run_dir)] == [(s.trace_id, s.span_id) for s in build_spans(run_dir)]
    assert all(0 < s.trace_id < 2**128 and 0 < s.span_id < 2**64 for s in build_spans(run_dir))


def test_the_trials_of_one_question_are_separate_traces(tmp_path):
    rows = [row("alpha", "q1", trial=0), row("alpha", "q1", trial=1)]
    spans = build_spans(write_run(tmp_path, rows))
    assert len({s.trace_id for s in spans}) == 2


def test_steps_run_one_after_another_inside_the_root_and_keep_their_real_duration(run_dir):
    spans = [s for s in build_spans(run_dir) if s.trace_id == build_spans(run_dir)[0].trace_id]
    root, *children = spans
    assert (root.end_ns - root.start_ns) == 100_000_000  # the answer's latency
    assert [(c.end_ns - c.start_ns) for c in children] == [20_000_000, 5_000_000, 60_000_000]
    assert children[0].start_ns == root.start_ns and children[1].start_ns == children[0].end_ns and children[2].start_ns == children[1].end_ns
    assert children[-1].end_ns <= root.end_ns


def test_a_root_is_never_shorter_than_its_steps(tmp_path):
    spans = build_spans(write_run(tmp_path, [row("alpha", "q1", latency_ms=1.0)]))
    root, *children = spans
    assert root.end_ns >= children[-1].end_ns


def test_traces_start_at_the_runs_start_and_follow_each_other_per_system(run_dir):
    roots = {(s.attributes["ragbench.system"], s.attributes["ragbench.question_id"]): s for s in build_spans(run_dir) if s.parent_id is None}
    start = 1_793_008_800_000_000_000  # 2026-10-26T10:00:00Z
    assert roots[("alpha", "q1")].start_ns == start
    assert roots[("alpha", "q2")].start_ns == roots[("alpha", "q1")].end_ns
    assert roots[("beta", "q1")].start_ns == start  # systems ran side by side


def test_a_failed_question_is_an_error_span_with_the_message(tmp_path):
    failed = row("alpha", "q1", error="RuntimeError: boom", steps=[], answer="", failure_type="run_error")
    spans = build_spans(write_run(tmp_path, [failed]))
    assert len(spans) == 1 and spans[0].error == "RuntimeError: boom"


def test_a_step_that_raised_is_an_error_span(tmp_path):
    bad = row("alpha", "q1", steps=[step("tool", "search", metadata={"error": "TimeoutError: slow"})])
    _, child = build_spans(write_run(tmp_path, [bad]))
    assert child.error == "TimeoutError: slow"


# -- what is exported -------------------------------------------------------------------------------------------------------------------------


def test_openinference_names_the_span_kinds_and_the_question_and_answer(run_dir):
    spans = build_spans(run_dir, convention="openinference")
    root = spans[0]
    assert root.attributes["openinference.span.kind"] == "CHAIN"
    assert root.attributes["input.value"] == "What is q1?" and root.attributes["output.value"] == "alpha answers q1"
    assert root.attributes["session.id"] == "run"
    kinds = {s.name: s.attributes["openinference.span.kind"] for s in spans[:4]}
    assert kinds == {"ragbench.answer": "CHAIN", "retrieve:bm25_search": "RETRIEVER", "rerank:cross_encoder": "RERANKER", "generate:answer": "LLM"}


def test_openinference_llm_steps_carry_token_counts_and_cost_and_retrievers_their_documents(run_dir):
    _, retrieve, _, generate = build_spans(run_dir, convention="openinference")[:4]
    assert generate.attributes["llm.token_count.prompt"] == 300 and generate.attributes["llm.token_count.completion"] == 40
    assert generate.attributes["llm.token_count.total"] == 340 and generate.attributes["llm.cost.total"] == pytest.approx(0.002)
    assert generate.attributes["input.value"] == "the prompt" and generate.attributes["output.value"] == "the answer"
    assert retrieve.attributes["retrieval.documents.0.document.id"] == "d1" and retrieve.attributes["retrieval.documents.1.document.id"] == "d2"


def test_every_step_kind_has_an_openinference_kind(run_dir):
    kinds = {k: trace_export.OPENINFERENCE_KINDS[k] for k in ("retrieve", "rerank", "llm", "tool", "embed", "route", "grade", "generate", "external")}
    assert kinds["tool"] == "TOOL" and kinds["embed"] == "EMBEDDING" and kinds["llm"] == "LLM" and set(kinds.values()) <= {"CHAIN", "RETRIEVER", "RERANKER", "LLM", "TOOL", "EMBEDDING", "AGENT"}


def test_otel_convention_uses_the_genai_attributes(run_dir):
    spans = build_spans(run_dir, convention="otel")
    assert not any(key.startswith("openinference.") for s in spans for key in s.attributes)
    generate = spans[3]
    assert generate.attributes["gen_ai.operation.name"] == "chat"
    assert generate.attributes["gen_ai.usage.input_tokens"] == 300 and generate.attributes["gen_ai.usage.output_tokens"] == 40
    assert spans[1].attributes["gen_ai.operation.name"] == "retrieval"
    assert spans[0].attributes["ragbench.input"] == "What is q1?"


def test_the_run_and_the_systems_result_ride_on_the_root_under_ragbench(run_dir):
    attributes = build_spans(run_dir)[0].attributes
    assert attributes["ragbench.run"] == "run" and attributes["ragbench.system"] == "alpha" and attributes["ragbench.question_id"] == "q1"
    assert attributes["ragbench.answer_score"] == 4.5 and attributes["ragbench.failure_type"] == "no_failure"
    assert attributes["ragbench.cost_usd"] == pytest.approx(0.002) and attributes["ragbench.category"] == "direct_fact"


@pytest.mark.parametrize("convention", ["openinference", "otel"])
def test_the_reference_answer_and_the_judges_reasoning_are_never_exported(run_dir, convention):
    text = json.dumps([span_record(s) for s in build_spans(run_dir, convention=convention)])
    assert REFERENCE not in text and "the judge's reasoning" not in text


def test_attribute_values_are_only_what_otlp_can_carry(run_dir):
    for s in build_spans(run_dir):
        assert all(isinstance(v, str | int | float | bool) for v in s.attributes.values()), s.attributes


# -- choosing what to export -----------------------------------------------------------------------------------------------------------------


def test_systems_and_questions_narrow_the_export_and_limit_counts_traces(run_dir):
    assert {s.attributes["ragbench.system"] for s in build_spans(run_dir, systems=["beta"]) if s.parent_id is None} == {"beta"}
    only = [s for s in build_spans(run_dir, question_ids=["q2"]) if s.parent_id is None]
    assert [s.attributes["ragbench.question_id"] for s in only] == ["q2"]
    assert len([s for s in build_spans(run_dir, limit=2) if s.parent_id is None]) == 2


def test_an_unknown_system_or_a_run_without_rows_is_an_error_naming_what_exists(run_dir, tmp_path):
    with pytest.raises(TraceExportError, match="gamma.*alpha, beta"):
        build_spans(run_dir, systems=["gamma"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TraceExportError, match="per_question_results.jsonl"):
        build_spans(empty)


# -- the command ---------------------------------------------------------------------------------------------------------------------------------


def test_jsonl_is_written_offline_into_the_run_by_default(run_dir):
    result = invoke("export-traces", str(run_dir))
    assert result.exit_code == 0, result.output
    lines = read_jsonl(run_dir / "traces.jsonl")
    assert len(lines) == 10  # 3 roots + 3 + 3 + 1 steps
    assert all(len(line["trace_id"]) == 32 and len(line["span_id"]) == 16 for line in lines)
    roots = [line for line in lines if line["parent_span_id"] is None]
    assert len(roots) == 3 and roots[0]["attributes"]["openinference.span.kind"] == "CHAIN"
    assert roots[0]["resource"]["openinference.project.name"] == "run"
    assert "traces.jsonl" in result.output


def test_jsonl_goes_to_the_path_given_and_json_prints_the_summary(run_dir, tmp_path):
    target = tmp_path / "out" / "spans.jsonl"
    result = invoke("export-traces", str(run_dir), "--out", str(target), "--json")
    assert result.exit_code == 0 and target.exists() and not (run_dir / "traces.jsonl").exists()
    summary = json.loads(result.stdout)
    assert summary["format"] == "jsonl" and summary["traces"] == 3 and summary["spans"] == 10 and summary["path"] == str(target)


def test_an_unknown_format_or_system_is_a_usage_error(run_dir):
    assert invoke("export-traces", str(run_dir), "--format", "zipkin").exit_code == 2
    result = invoke("export-traces", str(run_dir), "--system", "gamma")
    assert result.exit_code == 2 and "alpha, beta" in result.output


def test_otlp_needs_an_endpoint_and_says_so(run_dir, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    for kind in ("otlp", "openinference"):
        result = invoke("export-traces", str(run_dir), "--format", kind)
        assert result.exit_code == 2 and "--endpoint" in result.output


def test_otlp_without_the_extra_names_the_extra(run_dir, monkeypatch):
    monkeypatch.setattr("ragbench.utils.extras.extra_installed", lambda module: False)
    result = invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", "http://127.0.0.1:9/v1/traces")
    assert result.exit_code == 2 and "ragbench[otel]" in result.output


def test_a_real_mock_run_exports_every_system_and_step(tmp_path):
    run = mock_run(tmp_path, tmp_path / "results" / "mock_run", systems=["bm25", "hybrid_rerank"])
    result = invoke("export-traces", str(run), "--limit", "4", "--json")
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["traces"] == 4 and summary["spans"] > 4
    spans = read_jsonl(Path(summary["path"]))
    assert {s["attributes"]["openinference.span.kind"] for s in spans} >= {"CHAIN", "RETRIEVER", "LLM"}


# -- sending over OTLP/HTTP -----------------------------------------------------------------------------------------------------------------------

otel = pytest.importorskip


class Collector:
    """A local OTLP/HTTP endpoint: records the decoded requests and answers with the status asked for."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[tuple[dict[str, str], Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (the http.server name)
                from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest, ExportTraceServiceResponse

                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    import gzip

                    body = gzip.decompress(body)
                request = ExportTraceServiceRequest()
                request.ParseFromString(body)
                outer.requests.append(({k.lower(): v for k, v in self.headers.items()}, request))
                self.send_response(status)
                self.send_header("Content-Type", "application/x-protobuf")
                payload = ExportTraceServiceResponse().SerializeToString() if status == 200 else b"rejected"
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/traces"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def spans(self) -> list[Any]:
        return [span for _, request in self.requests for resource in request.resource_spans for scope in resource.scope_spans for span in scope.spans]


@pytest.fixture
def collector():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    server = Collector()
    yield server
    server.close()


def attribute_map(attributes) -> dict[str, Any]:
    return {a.key: getattr(a.value, a.value.WhichOneof("value")) for a in attributes}


def test_otlp_delivers_every_span_with_our_ids_and_times(run_dir, collector):
    result = invoke("export-traces", str(run_dir), "--format", "openinference", "--endpoint", collector.url, "--json")
    assert result.exit_code == 0, result.output
    expected = build_spans(run_dir)
    sent = collector.spans()
    assert len(sent) == len(expected) == json.loads(result.stdout)["spans"]
    assert {s.span_id.hex() for s in sent} == {format(s.span_id, "016x") for s in expected}
    assert {s.trace_id.hex() for s in sent} == {format(s.trace_id, "032x") for s in expected}
    root = next(s for s in sent if s.name == "ragbench.answer" and attribute_map(s.attributes)["ragbench.question_id"] == "q1" and attribute_map(s.attributes)["ragbench.system"] == "alpha")
    assert root.start_time_unix_nano == 1_793_008_800_000_000_000 and root.end_time_unix_nano - root.start_time_unix_nano == 100_000_000
    assert attribute_map(root.attributes)["openinference.span.kind"] == "CHAIN"
    children = [s for s in sent if s.parent_span_id == root.span_id]
    assert len(children) == 3


def test_the_resource_names_the_project_and_the_service(run_dir, collector):
    assert invoke("export-traces", str(run_dir), "--format", "openinference", "--endpoint", collector.url, "--project", "my-bench").exit_code == 0
    resource = collector.requests[0][1].resource_spans[0].resource
    attributes = attribute_map(resource.attributes)
    assert attributes["openinference.project.name"] == "my-bench" and attributes["service.name"] == "ragbench"


def test_the_otlp_format_sends_the_genai_attributes(run_dir, collector):
    assert invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", collector.url).exit_code == 0
    kinds = [attribute_map(s.attributes) for s in collector.spans()]
    assert any(a.get("gen_ai.operation.name") == "chat" for a in kinds) and not any("openinference.span.kind" in a for a in kinds)


def test_headers_are_sent_filled_from_the_environment_and_never_printed(run_dir, collector, monkeypatch):
    monkeypatch.setenv("LANGFUSE_AUTH", "Basic c2VjcmV0")
    result = invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", collector.url, "--header", "Authorization=${LANGFUSE_AUTH}", "--header", "x-team=bench")
    assert result.exit_code == 0, result.output
    headers = collector.requests[0][0]
    assert headers["authorization"] == "Basic c2VjcmV0" and headers["x-team"] == "bench"
    assert "c2VjcmV0" not in result.output and "c2VjcmV0" not in result.stderr


def test_a_header_with_an_unset_variable_fails_before_anything_is_sent(run_dir, collector, monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    result = invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", collector.url, "--header", "Authorization=${NOT_SET_ANYWHERE}")
    assert result.exit_code == 2 and "NOT_SET_ANYWHERE" in result.output and not collector.requests
    assert invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", collector.url, "--header", "no-equals-sign").exit_code == 2


def test_big_exports_go_out_in_batches(run_dir, collector, monkeypatch):
    monkeypatch.setattr(trace_export, "BATCH_SPANS", 4)
    assert invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", collector.url).exit_code == 0
    assert len(collector.requests) == 3 and len(collector.spans()) == 10


def test_the_endpoint_can_come_from_the_standard_environment_variable(run_dir, collector, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", collector.url)
    assert invoke("export-traces", str(run_dir), "--format", "otlp").exit_code == 0 and collector.spans()


def test_a_collector_that_rejects_the_spans_fails_the_command(run_dir):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    rejecting = Collector(status=400)
    try:
        result = invoke("export-traces", str(run_dir), "--format", "otlp", "--endpoint", rejecting.url)
    finally:
        rejecting.close()
    assert result.exit_code == 1 and "rejected" in result.output.lower()
