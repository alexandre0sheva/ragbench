"""Framework adapters (`ragbench.adapters`): LangChain, LlamaIndex and Haystack pipelines as `python_callable` targets.

The recorder and the missing-extra errors need no framework. Everything else runs against the real frameworks (with their own fake models and
embeddings), so it skips where an extra is not installed; the `frameworks` CI job installs all three.
"""

from __future__ import annotations

import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Any

import pytest
import yaml
from dataset_support import DOCUMENTS
from external_support import ROOT

from ragbench.adapters import offline
from ragbench.adapters._common import Call, Recorder, passage, source_ref
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models import cost as pricing
from ragbench.models.errors import MissingExtraError
from ragbench.rag_systems import create_rag_system

QUESTIONS = ["How much does HarborShield cost per month?", "When does ClaimPilot ship?", "What does error HS-4127 mean?", "Which office hosts the claims team?"]


def _documents() -> list[Document]:
    return [Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text) for doc_id, text in DOCUMENTS.items()]


def _system(target=None, *, ingest=None, mock=True, **options):
    """A `python_callable` system around a target built in the test (a path-string target would import the example file instead). The
    pipelines here run on the offline models, which only a `--mock` run may score, so the system is a mock run's unless `mock=False`."""
    config = SystemConfig(type="python_callable", name="framework", retrieval={"target": "external_support:answer", "model": "gpt-6-luna", **options})
    system = create_rag_system(config, force_mock=mock)
    if target is not None:
        system._answer = target
    system.ingest(_documents())
    if ingest is not None:
        ingest(_documents())
    return system


def _kinds(result) -> list[tuple[str, str]]:
    return [(step.kind, step.name) for step in result.steps]


def _example(name: str):
    spec = importlib.util.spec_from_file_location(f"ragbench_example_{name}", ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# -- no framework needed ---------------------------------------------------------------------------------------------------------------------


def test_a_document_names_its_file_by_the_first_metadata_key_it_has():
    assert source_ref({"source": "docs/a.md", "file_path": "x"}) == "x"  # `file_path` outranks `source`
    assert source_ref({"doc_id": "doc_7", "file_path": "x"}) == "doc_7"
    assert source_ref({}, None, "fallback") == "fallback" and source_ref(None) is None
    assert passage("text", "a.md", 3) == {"text": "text", "doc_id": "a.md", "score": 3.0} and passage("t", None, True)["score"] is None


def test_a_call_is_priced_at_the_model_the_framework_reported_else_at_the_declared_one_else_not_at_all():
    call = Call("generate", "llm", 5.0, model="gpt-6-luna", prompt_tokens=1000, completion_tokens=100)
    expected = pricing.estimate_model_cost("gpt-6-luna", 1000, 100)
    assert expected > 0

    reported = Recorder(model=None)
    reported.add(call)
    assert reported.steps()[0].cost.llm_cost == pytest.approx(expected)

    declared = Recorder(model="gpt-6-luna")
    declared.add(Call("generate", "llm", 5.0, model="a-model-nobody-priced", prompt_tokens=1000, completion_tokens=100))  # reported, but without a price
    step = declared.steps()[0]
    assert step.cost.llm_cost == pytest.approx(expected) and step.metadata["framework_model"] == "a-model-nobody-priced"

    unpriced = Recorder(model=None)
    unpriced.add(Call("generate", "llm", 5.0, model="a-model-nobody-priced", prompt_tokens=1000, completion_tokens=100))
    assert unpriced.steps()[0].cost.total_cost == 0 and unpriced.steps()[0].prompt_tokens == 1000

    retrieval = Recorder(model="gpt-6-luna")
    retrieval.add(Call("retrieve", "search", 1.0, model="gpt-6-luna", prompt_tokens=5))
    assert retrieval.steps()[0].cost.total_cost == 0  # only a model call costs


@pytest.mark.parametrize(
    ("module", "call", "extra", "package"),
    [
        ("langchain_core", lambda m: m.from_chain(object()), "langchain", "langchain-core"),
        ("llama_index.core", lambda m: m.from_query_engine(object()), "llamaindex", "llama-index-core"),
        ("haystack", lambda m: m.from_pipeline(object(), lambda q, k: {}), "haystack", "haystack-ai"),
    ],
)
def test_a_missing_extra_gives_the_one_line_install_command(monkeypatch, module, call, extra, package):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == module else real(name, *args))
    adapter = importlib.import_module(f"ragbench.adapters.{ {'langchain_core': 'langchain', 'llama_index.core': 'llamaindex', 'haystack': 'haystack'}[module] }")

    with pytest.raises(MissingExtraError) as error:
        call(adapter)
    assert f"pip install 'ragbench[{extra}]'" in str(error.value) and package in str(error.value) and "\n" not in str(error.value)


def test_a_live_run_refuses_a_pipeline_that_answers_with_the_offline_models():
    """Without RAGBENCH_REFERENCE_LIVE=1 the reference pipelines fall back to the mock models; a live run must fail them, not score them."""

    def offline_answer(question, top_k):
        text, prompt_tokens, completion_tokens = offline.answer("HarborShield costs $200 per month.", question)
        return {"answer": text, "contexts": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}

    with pytest.raises(RuntimeError, match="offline mock models in a live run"):
        _system(offline_answer, mock=False).answer_question(QUESTIONS[0], top_k=3)
    assert _system(offline_answer).answer_question(QUESTIONS[0], top_k=3).answer  # under --mock it is the point


def test_the_reference_pipelines_spend_nothing_unless_asked_to(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    assert not offline.live()  # a key alone does not make a mock run live
    monkeypatch.setenv("RAGBENCH_REFERENCE_LIVE", "1")
    assert offline.live()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert not offline.live()  # asking for live without a key falls back to the mock


# -- LangChain -------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def langchain_parts():
    pytest.importorskip("langchain_core")
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.vectorstores import InMemoryVectorStore

    embeddings = _example_embeddings()
    store = InMemoryVectorStore(embeddings)
    store.add_documents([LangChainDocument(page_content=text, metadata={"source": f"{doc_id}.md"}) for doc_id, text in DOCUMENTS.items()])
    reply = AIMessage(content="HarborShield costs $200 per month.", usage_metadata={"input_tokens": 120, "output_tokens": 9, "total_tokens": 129},
                      response_metadata={"model_name": "gpt-6-luna"})
    return store.as_retriever(search_kwargs={"k": 3}), GenericFakeChatModel(messages=cycle([reply]))


def _example_embeddings():
    from langchain_core.embeddings import Embeddings

    class Hashing(Embeddings):
        def embed_documents(self, texts):
            return offline.embed(texts)

        def embed_query(self, text):
            return offline.embed([text])[0]

    return Hashing()


def test_a_langchain_retriever_and_model_become_a_retrieve_step_and_a_priced_generate_step(langchain_parts):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    from ragbench.adapters.langchain import from_retriever

    retriever, model = langchain_parts
    generate = RunnableLambda(lambda inputs: f"{inputs['question']}\n" + "\n".join(doc.page_content for doc in inputs["context"])) | model | StrOutputParser()
    result = _system(from_retriever(retriever, generate, model="gpt-6-luna")).answer_question(QUESTIONS[0], top_k=3)

    assert result.answer == "HarborShield costs $200 per month."
    assert _kinds(result) == [("retrieve", "langchain_retriever"), ("generate", "langchain_llm")]
    generate_step = result.steps[1]
    assert (generate_step.prompt_tokens, generate_step.completion_tokens) == (120, 9) and result.token_usage == {"prompt_tokens": 120, "completion_tokens": 9}
    assert generate_step.cost.llm_cost == pytest.approx(pricing.estimate_model_cost("gpt-6-luna", 120, 9)) and result.cost.total_cost == pytest.approx(generate_step.cost.total_cost)
    assert "untracked" not in [step.name for step in result.steps] and result.retrieval_result.steps == result.steps[:1]
    assert len(result.retrieval_result.chunks) == 3 and result.retrieval_result.chunks[0].doc_id == "doc_001"
    assert result.metadata["external"] == {"unmapped_doc_ids": [], "cost_known": True}


def test_a_langchain_chain_returning_its_context_and_one_returning_a_string_both_report_their_passages(langchain_parts):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    from ragbench.adapters.langchain import from_chain

    retriever, model = langchain_parts

    def retrieve_and_answer(inputs, config):  # `create_retrieval_chain` shape: {"input"} -> {"input", "context", "answer"}
        documents = retriever.invoke(inputs["input"], config=config)
        return {"input": inputs["input"], "context": documents, "answer": model.invoke(inputs["input"], config=config).content}

    as_dict = _system(from_chain(RunnableLambda(retrieve_and_answer))).answer_question(QUESTIONS[1], top_k=3)
    assert _kinds(as_dict) == [("retrieve", "langchain_retriever"), ("generate", "langchain_llm")] and as_dict.retrieval_result.chunks[0].doc_id

    as_string = _system(from_chain(retriever | RunnableLambda(lambda documents: documents[0].page_content) | model | StrOutputParser(), input_key=None)).answer_question(QUESTIONS[1], top_k=3)
    assert as_string.answer == "HarborShield costs $200 per month." and len(as_string.retrieval_result.chunks) == 3  # the retriever's documents, read from its callback


def test_langchain_questions_answered_at_once_do_not_share_steps(langchain_parts):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    from ragbench.adapters.langchain import from_retriever

    retriever, model = langchain_parts
    system = _system(from_retriever(retriever, RunnableLambda(lambda inputs: inputs["question"]) | model | StrOutputParser(), model="gpt-6-luna"))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda question: system.answer_question(question, top_k=3), QUESTIONS * 4))
    assert all(_kinds(result) == [("retrieve", "langchain_retriever"), ("generate", "langchain_llm")] for result in results)


# -- LlamaIndex ------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def llamaindex_example():
    pytest.importorskip("llama_index.core")
    module = _example("llamaindex_default")
    module.ingest(_documents())
    return module


def test_a_llamaindex_query_engine_becomes_a_retrieve_step_and_a_priced_generate_step(llamaindex_example):
    system = _system(llamaindex_example.answer)
    result = system.answer_question(QUESTIONS[0], top_k=3)

    assert "$200" in result.answer
    assert _kinds(result) == [("retrieve", "llamaindex_retriever"), ("generate", "llamaindex_llm")]
    generate_step = result.steps[1]
    assert generate_step.prompt_tokens > 0 and generate_step.completion_tokens > 0 and result.token_usage["prompt_tokens"] == generate_step.prompt_tokens
    assert generate_step.cost.llm_cost == pytest.approx(pricing.estimate_model_cost("gpt-6-luna", generate_step.prompt_tokens, generate_step.completion_tokens))
    assert result.retrieval_result.chunks[0].doc_id == "doc_001" and result.metadata["external"]["unmapped_doc_ids"] == []
    assert "untracked" not in [step.name for step in result.steps]


def test_llamaindex_questions_answered_at_once_do_not_share_steps(llamaindex_example):
    system = _system(llamaindex_example.answer)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda question: system.answer_question(question, top_k=3), QUESTIONS * 4))
    assert all(_kinds(result) == [("retrieve", "llamaindex_retriever"), ("generate", "llamaindex_llm")] for result in results)
    assert len({result.answer for result in results}) == len(QUESTIONS)


def test_the_llamaindex_reference_pipeline_runs_live_on_ragbench_clients_without_the_openai_integration(monkeypatch):
    """Live, the pipeline is LlamaIndex's own, but its model and embeddings go through RAGBench's (here the paid fakes: no network)."""
    pytest.importorskip("llama_index.core")
    from paid_fakes import install as install_paid_fakes

    install_paid_fakes(monkeypatch)
    monkeypatch.setenv("RAGBENCH_REFERENCE_LIVE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    module = _example("llamaindex_default")
    module.ingest(_documents())
    assert type(module._llm).__name__ == "OpenAI" and module._llm.metadata.is_chat_model
    result = _system(module.answer, mock=False).answer_question(QUESTIONS[0], top_k=3)  # a live run: the offline models must not be touched

    assert _kinds(result) == [("retrieve", "llamaindex_retriever"), ("generate", "llamaindex_llm")]
    generate = result.steps[-1]
    assert generate.prompt_tokens > 0 and generate.completion_tokens > 0 and result.cost.total_cost > 0


def test_llamaindex_events_outside_a_question_are_ignored(llamaindex_example):
    engine = llamaindex_example._index.as_query_engine(llm=llamaindex_example._llm, similarity_top_k=2)  # the handler is registered, but no question is being recorded
    assert "$200" in str(engine.query(QUESTIONS[0]))


# -- Haystack --------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def haystack_example():
    pytest.importorskip("haystack")
    module = _example("haystack_default")
    module.ingest(_documents())
    return module


def test_a_haystack_pipeline_becomes_a_retrieve_step_and_a_priced_generate_step(haystack_example):
    result = _system(haystack_example.answer).answer_question(QUESTIONS[0], top_k=3)

    assert "$200" in result.answer
    assert _kinds(result) == [("retrieve", "haystack_retriever"), ("generate", "haystack_generator")]
    generate_step = result.steps[1]
    assert generate_step.prompt_tokens > 0 and generate_step.cost.llm_cost == pytest.approx(
        pricing.estimate_model_cost("gpt-6-luna", generate_step.prompt_tokens, generate_step.completion_tokens)
    )
    assert result.retrieval_result.chunks[0].doc_id == "doc_001" and len(result.retrieval_result.chunks) == 3
    assert result.metadata["external"]["unmapped_doc_ids"] == [] and "untracked" not in [step.name for step in result.steps]


def test_haystack_questions_answered_at_once_do_not_share_steps(haystack_example):
    system = _system(haystack_example.answer)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda question: system.answer_question(question, top_k=3), QUESTIONS * 4))
    assert all(_kinds(result) == [("retrieve", "haystack_retriever"), ("generate", "haystack_generator")] for result in results)


def test_a_haystack_chat_generator_and_a_custom_answer_reader_are_supported():
    pytest.importorskip("haystack")
    from haystack import Document as HaystackDocument
    from haystack import Pipeline, component
    from haystack.dataclasses import ChatMessage

    from ragbench.adapters.haystack import default_answer, from_pipeline

    @component
    class Search:
        @component.output_types(documents=list[HaystackDocument])
        def run(self, query: str, top_k: int):
            return {"documents": [HaystackDocument(content="HarborShield costs $200 per month.", meta={"file_path": "docs/doc_001.md"}, score=0.9)][:top_k]}

    @component
    class Chat:
        @component.output_types(replies=list[ChatMessage])
        def run(self, documents: Any, query: str):
            return {"replies": [ChatMessage.from_assistant("$200", meta={"model": "gpt-6-luna", "usage": {"prompt_tokens": 50, "completion_tokens": 3}})]}

    pipeline = Pipeline()
    pipeline.add_component("search", Search())
    pipeline.add_component("chat", Chat())
    pipeline.connect("search.documents", "chat.documents")
    answer = from_pipeline(pipeline, lambda question, top_k: {"search": {"query": question, "top_k": top_k}, "chat": {"query": question}}, retriever="search", generator="chat")

    result = _system(answer).answer_question(QUESTIONS[0], top_k=2)
    assert result.answer == "$200" and _kinds(result) == [("retrieve", "haystack_retriever"), ("generate", "haystack_generator")]
    assert result.steps[1].prompt_tokens == 50 and result.steps[1].cost.llm_cost > 0
    assert [chunk.doc_id for chunk in result.retrieval_result.chunks] == ["doc_001"] and result.retrieval_result.chunks[0].score == pytest.approx(0.9)

    with pytest.raises(ValueError, match="no `replies` or `answers`"):
        default_answer({"meta": []})

    again = from_pipeline(pipeline, lambda question, top_k: {"search": {"query": question, "top_k": top_k}, "chat": {"query": question}}, retriever="search", generator="chat",
                          answer=lambda outputs: "custom: " + outputs["replies"][0].text)
    assert again(QUESTIONS[0], 2)["answer"] == "custom: $200"
    assert [step.kind for step in again(QUESTIONS[0], 2)["steps"]] == ["retrieve", "generate"]  # the components were already wrapped: still one step each


# -- the reference pipelines, end to end -----------------------------------------------------------------------------------------------------


def test_the_shipped_framework_config_runs_all_three_pipelines_with_no_key(tmp_path, monkeypatch):
    for module in ("langchain_core", "llama_index.core", "haystack"):
        pytest.importorskip(module)
    raw = yaml.safe_load((ROOT / "configs" / "frameworks_example.yaml").read_text(encoding="utf-8"))
    raw["evaluation"].update(max_questions=10, latency_probe_questions=0)
    raw["run"]["output_dir"] = str(tmp_path)
    path = tmp_path / "frameworks.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(ROOT)
    run = run_benchmark(path, force_mock=True, use_cache=False, run_dir=tmp_path / "out")

    import pandas as pd

    metrics = pd.read_csv(run / "metrics_summary.csv").set_index("system")
    assert list(metrics.index) == ["bm25_default", "langchain_default", "llamaindex_default", "haystack_default"] and (metrics["n_error"] == 0).all()
    assert (metrics["retrieval_recall@5"] > 0).all()
    assert (metrics.loc[["langchain_default", "llamaindex_default", "haystack_default"], "avg_cost_per_question"] > 0).all()


# -- NLTK data -------------------------------------------------------------------------------------------------------------------------------


def test_llamaindex_keeps_its_nltk_data_private_when_the_default_location_is_shared(tmp_path):
    """LlamaIndex stores NLTK data inside site-packages; on a runner whose umask makes that group-writable NLTK refuses to read or download
    there, and the sentence splitter fails for every question. The adapter then points NLTK at a private directory of its own."""
    pytest.importorskip("llama_index.core")
    import os
    import stat

    from ragbench.adapters.llamaindex import private_nltk_dir

    cache = tmp_path / "cache"
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)
    chosen = private_nltk_dir(shared, {}, cache)
    assert chosen is not None and chosen.is_relative_to(cache) and stat.S_IMODE(chosen.stat().st_mode) == 0o700 and chosen.stat().st_uid == os.getuid()

    inner = tmp_path / "inner"  # private itself, but a file in it is group-writable
    (inner / "corpora").mkdir(parents=True)
    inner.chmod(0o700)
    (inner / "corpora").chmod(0o700)
    (inner / "corpora" / "stopwords.zip").write_bytes(b"x")
    (inner / "corpora" / "stopwords.zip").chmod(0o664)
    assert private_nltk_dir(inner, {}, cache) == chosen

    private = tmp_path / "private"
    (private / "corpora").mkdir(parents=True)
    private.chmod(0o700)
    (private / "corpora").chmod(0o700)
    assert private_nltk_dir(private, {}, cache) is None  # a private default is left alone, with whatever data it holds

    missing = tmp_path / "not_yet_created"  # judged by the umask it will be created under
    assert private_nltk_dir(missing, {}, cache, umask=0o002) == chosen
    assert private_nltk_dir(missing, {}, cache, umask=0o022) is None
    assert private_nltk_dir(shared, {"NLTK_DATA": "/somewhere/the/user/chose"}, cache) is None  # the user's own choice always wins
