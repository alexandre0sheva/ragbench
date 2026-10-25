"""`external_http` and `python_callable`: benchmark a RAG that runs outside RAGBench (a deployed endpoint, or a function of your own)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml
from dataset_support import DOCUMENTS
from external_support import INGESTED, SIMPLE_RAG

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.evaluation.judge_panel import scoring_judges
from ragbench.models.family import model_family
from ragbench.models.refs import answering_model, system_model_refs
from ragbench.rag_systems import create_rag_system
from ragbench.utils.jsonl import read_jsonl
from ragbench.utils.jsonpath import JsonPathError, find

QUESTION = "How much does HarborShield cost per month?"


def _documents() -> list[Document]:
    return [Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text) for doc_id, text in DOCUMENTS.items()]


@pytest.fixture
def server(rag_server):
    url, rag = rag_server
    rag.load(DOCUMENTS)
    rag.requests.clear()
    rag.failures_served.clear()
    return url, rag


def _http(url: str, **options):
    system = create_rag_system(SystemConfig(type="external_http", name="my_rag", retrieval={"url": url, **options}))
    system.ingest(_documents())
    return system


# -- JSONPath ------------------------------------------------------------------------------------------------------------------------------


def test_jsonpath_reads_fields_indexes_and_wildcards():
    data = {"a": {"b": [{"c": 1}, {"c": 2}], "d-e": "x"}}
    assert find(data, "$.a.b[*].c") == [1, 2]
    assert find(data, "$.a.b[1].c") == [2]
    assert find(data, "$['a']['d-e']") == ["x"]
    assert find(data, "$") == [data]
    assert find(data, "$.a.missing") == [] and find(data, "$.a.b[7]") == []
    with pytest.raises(JsonPathError, match="must start with"):
        find(data, "a.b")


# -- external_http --------------------------------------------------------------------------------------------------------------------------


def test_an_endpoint_answer_becomes_one_external_step_with_its_contexts_and_cost(server):
    url, _ = server
    result = _http(f"{url}/ask").answer_question(QUESTION, top_k=3)

    assert "$200" in result.answer
    assert result.retrieval_result.chunks[0].doc_id == "doc_001" and len(result.retrieval_result.chunks) <= 3
    assert [(step.kind, step.name) for step in result.steps] == [("external", "external_http")]
    assert result.cost.llm_cost > 0 and result.steps[0].cost.total_cost == pytest.approx(result.cost.total_cost)
    assert result.token_usage["prompt_tokens"] > 0
    assert result.metadata["context_chunk_ids"] == [chunk.chunk_id for chunk in result.retrieval_result.chunks]
    assert result.metadata["external"] == {"unmapped_doc_ids": [], "cost_known": True}


def test_the_request_template_and_env_secrets_in_headers_are_sent_but_never_recorded(server, monkeypatch):
    url, rag = server
    monkeypatch.setenv("MY_RAG_TOKEN", "s3cret-value")
    system = _http(f"{url}/ask", headers={"Authorization": "Bearer ${MY_RAG_TOKEN}"}, request='{"query": {{ question | tojson }}, "k": {{ top_k }}}')
    result = system.answer_question(QUESTION, top_k=2)

    sent = rag.requests[-1]
    assert sent["headers"]["Authorization"] == "Bearer s3cret-value"
    assert sent["body"] == {"query": QUESTION, "k": 2}
    assert "s3cret" not in result.model_dump_json() and "s3cret" not in repr(system.options)


def test_a_missing_secret_names_the_variable(server, monkeypatch):
    url, _ = server
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE"):
        _http(f"{url}/ask", headers={"X-Key": "${NOT_SET_ANYWHERE}"}).answer_question(QUESTION)


def test_get_sends_the_rendered_request_as_query_parameters(server):
    url, rag = server
    result = _http(f"{url}/ask", method="GET").answer_question(QUESTION, top_k=1)

    assert rag.requests[-1]["method"] == "GET" and "$200" in result.answer


def test_jsonpath_mappings_read_another_layout(server):
    url, _ = server
    mapping = {"answer": "$.data.output.text", "contexts": "$.data.sources[*]", "text": "$.content", "doc_id": "$.meta.id"}
    result = _http(f"{url}/ask?shape=nested", response=mapping).answer_question(QUESTION)

    assert "$200" in result.answer and result.retrieval_result.chunks[0].doc_id == "doc_001"


def test_a_response_without_the_answer_field_fails_the_question(server):
    url, _ = server
    with pytest.raises(ValueError, match=r"\$\.nothing"):
        _http(f"{url}/ask", response={"answer": "$.nothing"}).answer_question(QUESTION)


def test_transient_failures_are_retried(server, monkeypatch):
    monkeypatch.setattr("ragbench.models.retry.time.sleep", lambda _: None)
    url, _ = server
    assert "$200" in _http(f"{url}/ask?fail=2", retries=2).answer_question(QUESTION).answer
    with pytest.raises(RuntimeError, match="503"):
        _http(f"{url}/ask?fail=5", retries=1).answer_question(QUESTION)


def test_a_slow_endpoint_times_out(server):
    url, _ = server
    with pytest.raises(RuntimeError, match="timed out|timeout"):
        _http(f"{url}/ask?sleep=0.5", timeout_s=0.05, retries=0).answer_question(QUESTION)


def test_ingest_posts_the_documents_when_asked(server):
    url, rag = server
    rag.load({})
    system = _http(f"{url}/ask", post_documents_to=f"{url}/documents")

    assert [r["path"] for r in rag.requests] == ["/documents"]
    assert {d["doc_id"] for d in rag.requests[0]["body"]["documents"]} == set(DOCUMENTS)
    assert "$200" in system.answer_question(QUESTION).answer


def test_ids_that_name_the_file_are_mapped_back_to_the_corpus(server):
    url, _ = server
    result = _http(f"{url}/ask?ids=path").answer_question(QUESTION)

    assert result.retrieval_result.chunks[0].doc_id == "doc_001" and result.metadata["external"]["unmapped_doc_ids"] == []


def test_ids_that_match_no_document_are_reported_as_unmapped(server):
    url, _ = server
    result = _http(f"{url}/ask?ids=opaque").answer_question(QUESTION)

    assert result.metadata["external"]["unmapped_doc_ids"] and result.metadata["external"]["unmapped_doc_ids"][0].startswith("kb-")


def test_cost_from_token_usage_and_the_declared_model_or_unknown(server):
    url, _ = server
    priced = _http(f"{url}/ask?usage=tokens", model="gpt-6-luna").answer_question(QUESTION)
    assert priced.cost.llm_cost > 0 and priced.metadata["external"]["cost_known"]

    unknown = _http(f"{url}/ask?usage=tokens").answer_question(QUESTION)
    assert unknown.cost.llm_cost == 0 and not unknown.metadata["external"]["cost_known"]
    assert not _http(f"{url}/ask?usage=none", model="gpt-6-luna").answer_question(QUESTION).metadata["external"]["cost_known"]


def test_options_are_validated_at_load_time():
    with pytest.raises(ValueError, match="url"):
        SystemConfig(type="external_http", retrieval={})
    with pytest.raises(ValueError, match="answr"):
        SystemConfig(type="external_http", retrieval={"url": "http://x/ask", "response": {"answr": "$.a"}})
    with pytest.raises(ValueError, match="JSONPath"):
        SystemConfig(type="external_http", retrieval={"url": "http://x/ask", "response": {"answer": "answer"}})


# -- python_callable ------------------------------------------------------------------------------------------------------------------------


def _callable(**options):
    system = create_rag_system(SystemConfig(type="python_callable", name="mine", retrieval=options))
    system.ingest(_documents())
    return system


def test_a_callable_returning_a_dict_is_answered_and_ingested():
    result = _callable(target="external_support:answer", ingest_target="external_support:ingest").answer_question(QUESTION, top_k=1)

    assert INGESTED == list(DOCUMENTS)
    assert result.answer == f"echo: {QUESTION}" and result.retrieval_result.chunks[0].doc_id == "doc_001"
    assert [(s.kind, s.name) for s in result.steps] == [("external", "python_callable")] and result.cost.llm_cost == pytest.approx(0.002)


def test_a_callable_can_be_named_by_file_path_and_return_an_answer_result():
    by_path = _callable(target=f"{SIMPLE_RAG}:answer", ingest_target=f"{SIMPLE_RAG}:ingest").answer_question(QUESTION)
    assert "$200" in by_path.answer and by_path.retrieval_result.chunks[0].doc_id == "doc_001"

    result = _callable(target="external_support:answer_result").answer_question("When does ClaimPilot ship?")
    assert result.answer == "Q3." and result.cost.llm_cost == pytest.approx(0.004) and [s.kind for s in result.steps] == ["external"]


def test_a_bad_target_fails_early_and_a_wrong_return_type_fails_the_question():
    with pytest.raises(ValueError, match="no_such_module"):
        SystemConfig(type="python_callable", retrieval={"target": "no_such_module:answer"}) and create_rag_system(
            SystemConfig(type="python_callable", retrieval={"target": "no_such_module:answer"}))
    with pytest.raises(ValueError, match="module:function"):
        SystemConfig(type="python_callable", retrieval={"target": "just_a_name"})
    with pytest.raises(TypeError, match="int"):
        _callable(target="external_support:not_a_result").answer_question(QUESTION)


# -- model refs and the judge ---------------------------------------------------------------------------------------------------------------


def test_ragbench_calls_no_model_for_an_external_system_and_its_judge_family_is_unknown_unless_declared():
    plain = SystemConfig(type="external_http", retrieval={"url": "http://x/ask"})
    declared = SystemConfig(type="external_http", retrieval={"url": "http://x/ask", "model": "claude-haiku-5-5"})

    assert system_model_refs(plain).llm == [] and system_model_refs(plain).embedding == []
    assert answering_model(plain) is None and answering_model(declared) == "claude-haiku-5-5"
    assert answering_model(SystemConfig(type="bm25")) == "gpt-6-luna"
    assert model_family(None) == "unknown"
    assert scoring_judges(["gpt-6-luna", "anthropic:claude-haiku-5-5"], None, independent=True) == ["gpt-6-luna", "anthropic:claude-haiku-5-5"]


# -- end to end -----------------------------------------------------------------------------------------------------------------------------


def _run(tmp_path, dataset, systems) -> tuple[pd.DataFrame, dict, list[dict]]:
    config = {"run": {"name": "external", "output_dir": str(tmp_path / "results")}, "dataset": dataset, "systems": systems,
              "evaluation": {"max_workers": 2, "latency_probe_questions": 0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = run_benchmark(path, force_mock=True, use_cache=False, run_dir=tmp_path / "out")
    summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
    return pd.read_csv(run / "metrics_summary.csv").set_index("system"), summary, read_jsonl(run / "per_question_results.jsonl")


def test_mock_e2e_compares_bm25_with_an_endpoint_served_by_the_fixture(server, tmp_path, tiny_dataset):
    url, _ = server
    metrics, summary, rows = _run(tmp_path, tiny_dataset, [{"type": "bm25", "name": "bm25"}, {"type": "external_http", "name": "deployed", "retrieval": {"url": f"{url}/ask"}}])

    assert set(metrics.index) == {"bm25", "deployed"} and metrics.loc["deployed", "n_error"] == 0
    assert metrics.loc["deployed", "retrieval_recall@5"] > 0 and metrics.loc["deployed", "avg_cost_per_question"] > 0
    deployed = [r for r in rows if r["system"] == "deployed"]
    assert all([step["kind"] for step in r["steps"]] == ["external"] for r in deployed)
    assert not any("external" in notice.lower() for notice in summary["notices"]) and "external" not in summary


def test_unmapped_ids_leave_retrieval_blank_and_unknown_cost_is_reported(server, tmp_path, tiny_dataset):
    url, _ = server
    systems = [{"type": "bm25", "name": "bm25"}, {"type": "external_http", "name": "opaque", "retrieval": {"url": f"{url}/ask?ids=opaque&usage=none"}}]
    metrics, summary, rows = _run(tmp_path, tiny_dataset, systems)

    assert pd.isna(metrics.loc["opaque", "retrieval_recall@5"]) and pd.isna(metrics.loc["opaque", "retrieval_mrr@10"])
    assert not pd.isna(metrics.loc["bm25", "retrieval_recall@5"])
    assert all(r["retrieval_metrics"] == {} for r in rows if r["system"] == "opaque")
    notices = " ".join(summary["notices"])
    assert "opaque" in notices and "document ids" in notices and "cost" in notices.lower()
    assert summary["external"]["unmapped_doc_ids"]["opaque"]["questions"] > 0 and summary["external"]["unknown_cost"] == ["opaque"]


def test_estimate_never_calls_an_external_system_and_says_it_is_not_included(server, tmp_path, tiny_dataset):
    from ragbench.config.loader import load_config
    from ragbench.evaluation.estimate import estimate_run

    url, rag = server
    config = {"run": {"name": "est", "output_dir": str(tmp_path)}, "dataset": tiny_dataset,
              "systems": [{"type": "bm25", "name": "bm25"}, {"type": "external_http", "name": "deployed", "retrieval": {"url": f"{url}/ask", "post_documents_to": f"{url}/documents"}}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    estimate = estimate_run(load_config(path))

    assert rag.requests == []
    deployed = next(row for row in estimate.systems if row.system == "deployed")
    assert deployed.external and deployed.error is None and deployed.total_usd == 0
    assert any("deployed" in warning and "not included" in warning for warning in estimate.warnings)


def test_where_an_external_system_runs_its_models_is_unknown_to_the_local_only_constraints(tiny_dataset):
    from ragbench.config.schema import ExperimentConfig
    from ragbench.selection.recommend import _locality

    config = ExperimentConfig.model_validate({"run": {"name": "x"}, "dataset": tiny_dataset,
                                              "systems": [{"type": "external_http", "name": "deployed", "retrieval": {"url": "http://127.0.0.1:9/ask"}}]})
    assert _locality(config, "deployed") == (None, None)


def test_the_shipped_example_config_runs_against_the_fake_service(rag_server, tmp_path, monkeypatch):
    from external_support import ROOT

    url, rag = rag_server
    rag.load({path.stem: path.read_text(encoding="utf-8") for path in sorted((ROOT / "data" / "demo" / "docs").glob("*.md"))})
    raw = yaml.safe_load((ROOT / "configs" / "external_example.yaml").read_text(encoding="utf-8"))
    raw["systems"][1]["retrieval"]["url"] = f"{url}/ask"
    raw["evaluation"].update(max_questions=12, latency_probe_questions=0)
    raw["run"]["output_dir"] = str(tmp_path)
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(ROOT)  # the callable's target is a path relative to the working directory
    run = run_benchmark(path, force_mock=True, use_cache=False, run_dir=tmp_path / "out")

    metrics = pd.read_csv(run / "metrics_summary.csv").set_index("system")
    assert list(metrics.index) == ["bm25_default", "my_deployed_rag", "my_inprocess_rag"] and (metrics["n_error"] == 0).all()
    assert (metrics["retrieval_recall@5"] > 0).all() and (metrics.loc[["my_deployed_rag", "my_inprocess_rag"], "avg_cost_per_question"] > 0).all()


def test_an_external_system_builds_no_model_client_and_reports_no_model_used(monkeypatch):
    from ragbench.evaluation.evaluator import _model_names
    from ragbench.models.llms import MockLLM

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")  # with a key, the base class would build a real OpenAI client
    system = create_rag_system(SystemConfig(type="external_http", retrieval={"url": "http://127.0.0.1:9/ask"}))
    assert isinstance(system.llm, MockLLM) and not system.force_mock
    assert _model_names(system) == set()


def _kinds(result) -> list[tuple[str, str]]:
    return [(step.kind, step.name) for step in result.steps]


def test_a_callable_that_returns_steps_keeps_them_and_they_add_up_to_its_cost():
    result = _callable(target="external_support:answer_result_with_steps").answer_question("When does ClaimPilot ship?")

    assert _kinds(result) == [("retrieve", "framework_retriever"), ("generate", "framework_llm")]
    assert result.cost.embedding_cost == pytest.approx(0.0001) and result.cost.llm_cost == pytest.approx(0.004)
    assert sum(step.cost.total_cost for step in result.steps) == pytest.approx(result.cost.total_cost)
    assert result.metadata["external"]["cost_known"] and result.retrieval_result.steps == result.steps[:1]


def test_a_callable_that_records_on_the_active_tracer_keeps_those_steps_and_their_cost():
    result = _callable(target="external_support:answer_traced").answer_question(QUESTION)

    assert _kinds(result) == [("retrieve", "my_search"), ("generate", "my_answer")]
    assert result.cost.total_cost == pytest.approx(0.0032) and result.metadata["external"]["cost_known"]
    assert result.token_usage == {"prompt_tokens": 30, "completion_tokens": 4}


def test_steps_can_be_plain_dicts_in_the_reply():
    result = _callable(target="external_support:answer_with_step_dicts").answer_question(QUESTION)

    assert _kinds(result) == [("retrieve", "lookup"), ("generate", "llm")] and result.cost.llm_cost == pytest.approx(0.001)
    assert "untracked" not in [step.name for step in result.steps]


def test_the_terminal_summary_says_when_an_external_system_was_not_scored_or_priced(server, tmp_path, tiny_dataset):
    from cli_support import invoke

    url, _ = server
    config = {"run": {"name": "cli", "output_dir": str(tmp_path / "results")}, "dataset": tiny_dataset,
              "systems": [{"type": "bm25", "name": "bm25"}, {"type": "external_http", "name": "opaque", "retrieval": {"url": f"{url}/ask?ids=opaque&usage=none"}}],
              "evaluation": {"max_workers": 2, "latency_probe_questions": 0}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = invoke("run", "--config", str(path), "--mock")

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "Retrieval not scored for opaque" in output and "Cost unknown for opaque" in output
