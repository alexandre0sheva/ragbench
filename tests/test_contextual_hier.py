"""`contextual` (LLM-situated chunks) and `hierarchical` (summary-routed) retrieval, plus the MockLLM responder table they rely on."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.rag_systems import create_rag_system
from ragbench.utils.jsonl import write_jsonl

PER_CALL = 0.001


class ScriptedLLM(MockLLM):
    """Answers the ingestion prompts (those carrying a `Document title:` line) with `respond(prompt)`, records those calls and charges a
    fixed price per call. Every other prompt (question answering) goes to the plain mock."""

    def __init__(self, respond):
        super().__init__()
        self.respond = respond
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        if "Document title:" not in prompt:
            return super().generate(messages, **kwargs)
        with self._lock:
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
        text = self.respond(prompt)
        if isinstance(text, Exception):
            raise text
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
        return LLMResult(text=text, model="scripted", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")


def _doc(doc_id: str, text: str, title: str | None = None) -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=title or doc_id, text=text)


def _system(type_: str, llm=None, **sections):
    system = create_rag_system(SystemConfig(type=type_, **sections), force_mock=True)
    if llm is not None:
        system.llm = llm
    return system


# --- the MockLLM responder table ------------------------------------------------------------------------------------------


def test_a_registered_responder_takes_over_matching_prompts_and_can_be_removed():
    llm = MockLLM()
    name = MockLLM.register_responder(lambda prompt, json_mode: "ZEBRA-PROMPT" in prompt, lambda prompt: "zebra answer", name="zebra")
    try:
        assert llm.generate([{"role": "user", "content": "hello ZEBRA-PROMPT"}]).text == "zebra answer"
        assert llm.generate([{"role": "user", "content": "Context:\nThe sky is blue.\n\nQuestion: what colour is the sky"}]).text != "zebra answer"
    finally:
        MockLLM.unregister_responder(name)

    assert llm.generate([{"role": "user", "content": "hello ZEBRA-PROMPT"}]).text != "zebra answer"


def test_the_built_in_prompt_kinds_still_get_their_old_answers():
    llm = MockLLM()

    assert json.loads(llm.generate([{"role": "user", "content": "x"}], json_mode=True).text) == {"score": 3, "reasoning": "Mock JSON response."}
    assert "Reference information about" in llm.generate([{"role": "user", "content": "Write a hypothetical passage.\nQuestion: refund policy"}]).text
    assert json.loads(llm.generate([{"role": "user", "content": "Rewrite the question\nrefund policy"}]).text) == {"queries": ["refund policy"]}
    assert "could not find the answer from my own knowledge" in llm.generate([{"role": "user", "content": "No documents are provided.\nQuestion: x"}]).text
    answer = llm.generate([{"role": "user", "content": "Context:\n[doc_001 | a]\nHarborShield costs $200 per month.\n\nQuestion: How much does HarborShield cost?"}]).text
    assert "$200" in answer


def test_the_mock_situates_a_chunk_with_the_document_title_and_summarises_with_the_first_sentences():
    llm = MockLLM()
    situate = llm.generate([{"role": "user", "content": "Document title: Severity Policy\n<chunk>x</chunk>\nPlease give a short succinct context to situate this chunk within the overall document."}]).text
    summary = llm.generate([{"role": "user", "content": "Write a short summary of this document so it can be found later.\nDocument title: T\nDocument text:\nFirst sentence here. Second one follows. Third is ignored."}]).text

    assert "Severity Policy" in situate
    assert summary == "First sentence here. Second one follows."


# --- contextual -------------------------------------------------------------------------------------------------------------


def _policy_docs() -> list[Document]:
    return [
        _doc("doc_s1", "# Alpha escalation policy\n\nThe limit is 30 minutes.", title="Alpha escalation policy"),
        _doc("doc_s2", "# Beta escalation policy\n\nThe limit is 30 minutes.", title="Beta escalation policy"),
        _doc("doc_x", "# Cafeteria\n\nLunch is served until 2 pm every day.", title="Cafeteria"),
    ]


def _situating(prompt: str) -> str:
    title = prompt.split("Document title:", 1)[1].splitlines()[0].strip()
    return f"This passage belongs to the {title} document."


def test_contextual_calls_the_llm_once_per_chunk_and_prepends_the_context_before_indexing():
    llm = ScriptedLLM(_situating)
    system = _system("contextual", llm)

    ingestion = system.ingest(_policy_docs())

    assert len(llm.calls) == ingestion.num_chunks == 3
    assert all(call["kwargs"].get("temperature") == 0 and not call["kwargs"].get("json_mode") for call in llm.calls), "temperature 0 so the LLM cache applies"
    indexed = {chunk.doc_id: chunk for chunk in system.bm25_store.chunks}
    assert indexed["doc_s1"].text.startswith("This passage belongs to the Alpha escalation policy document.")
    assert "The limit is 30 minutes." in indexed["doc_s1"].text
    assert indexed["doc_s1"].metadata["context"].startswith("This passage") and indexed["doc_s1"].metadata["original_text"].endswith("30 minutes.")
    assert [c.text for c in system.vector_store.chunks] == [c.text for c in system.bm25_store.chunks], "both indexes see the same enriched text"


def test_contextual_ingestion_cost_is_the_sum_of_the_llm_calls_and_is_reported():
    llm = ScriptedLLM(_situating)
    system = _system("contextual", llm)

    ingestion = system.ingest(_policy_docs())

    assert ingestion.cost.llm_cost == pytest.approx(3 * PER_CALL)
    assert ingestion.cost.llm_prompt_tokens == 30 and ingestion.metadata["contexts_generated"] == 3 and ingestion.metadata["context_failures"] == 0


def test_the_context_disambiguates_chunks_whose_own_text_is_identical():
    """Both documents end with the same sentence in a chunk of its own, so only the context says which document it belongs to."""
    page = "This page covers escalation timing for the support desk. The limit is 30 minutes."
    docs = [
        _doc("doc_s1", f"# Alpha escalation policy\n\n{page}", title="Alpha escalation policy"),
        _doc("doc_s2", f"# Beta escalation policy\n\n{page}", title="Beta escalation policy"),
        _doc("doc_x", "# Cafeteria\n\nLunch is served until two pm every day.", title="Cafeteria"),
    ]
    chunker = {"type": "word", "chunk_size": 7, "chunk_overlap": 0}
    contextual = _system("contextual", ScriptedLLM(_situating), chunker=chunker)
    plain = _system("hybrid", chunker=chunker)
    contextual.ingest(docs)
    plain.ingest(docs)

    def first_limit_chunk(system, query: str) -> str:
        return next(c.doc_id for c in system.fetch_context(query, top_k=3).chunks if "limit is 30 minutes" in c.text)

    for doc_id, query in (("doc_s1", "Alpha escalation policy limit minutes"), ("doc_s2", "Beta escalation policy limit minutes")):
        assert first_limit_chunk(contextual, query) == doc_id
    assert first_limit_chunk(plain, "Beta escalation policy limit minutes") == "doc_s1", "without context the identical chunks cannot be told apart"


def test_retrieval_shows_the_original_chunk_text_not_the_context_prefix():
    system = _system("contextual", ScriptedLLM(_situating), retrieval={"top_k": 2})
    system.ingest(_policy_docs())

    chunk = system.fetch_context("Alpha escalation policy limit", top_k=2).chunks[0]

    assert "This passage belongs" not in chunk.text and "The limit is 30 minutes." in chunk.text
    assert chunk.metadata["context"].startswith("This passage belongs to the Alpha escalation")
    answer = system.answer_question("Alpha escalation policy limit")
    assert "This passage belongs" not in answer.answer and answer.retrieval_result.chunks[0].metadata["original_text"] == chunk.text


def test_contextual_answers_with_the_hybrid_retrieval_steps():
    system = _system("contextual", ScriptedLLM(_situating))
    system.ingest(_policy_docs())

    answer = system.answer_question("Alpha escalation policy limit")

    assert [(s.kind, s.name) for s in answer.steps] == [("retrieve", "bm25_search"), ("retrieve", "vector_search"), ("generate", "answer")]
    assert system.fetch_context("x").metadata["retriever"] == "hybrid_rrf"


def test_a_failing_context_call_degrades_that_chunk_only_and_is_counted():
    def flaky(prompt: str):
        return RuntimeError("boom") if "Cafeteria" in prompt else _situating(prompt)

    system = _system("contextual", ScriptedLLM(flaky))

    ingestion = system.ingest(_policy_docs())

    cafeteria = next(c for c in system.bm25_store.chunks if c.doc_id == "doc_x")
    assert cafeteria.text == cafeteria.metadata["original_text"] and cafeteria.metadata["context"] == ""
    assert ingestion.metadata["context_failures"] == 1 and ingestion.metadata["contexts_generated"] == 2


def test_when_every_context_call_fails_ingestion_raises_instead_of_pretending():
    system = _system("contextual", ScriptedLLM(lambda prompt: RuntimeError("invalid api key")))

    with pytest.raises(RuntimeError, match="invalid api key"):
        system.ingest(_policy_docs())


def test_a_long_document_is_shown_to_the_llm_as_a_window_around_the_chunk():
    sentences = [f"Sentence number {i} talks about topic{i}." for i in range(400)]
    llm = ScriptedLLM(lambda prompt: "ctx")
    system = _system(
        "contextual", llm, chunker={"type": "word", "chunk_size": 40, "chunk_overlap": 0}, retrieval={"document_max_chars": 1500}
    )

    system.ingest([_doc("doc_long", " ".join(sentences))])

    assert len(llm.calls) > 10
    assert all(len(call["prompt"]) < 4000 for call in llm.calls), "the whole document is never sent"
    for call in llm.calls:
        before, after = call["prompt"].split("<chunk>")
        chunk_text = after.split("</chunk>")[0].strip()
        assert chunk_text in before, "the excerpt is a window around the chunk, so it contains the chunk"


def test_contextual_accepts_the_hybrid_options_and_rejects_unknown_ones():
    system = _system("contextual", retrieval={"rrf_k": 30, "bm25_weight": 2.0, "document_max_chars": 5000})
    assert system.options.rrf_k == 30 and system.options.document_max_chars == 5000
    with pytest.raises(ValueError, match="document_max_chars|unknown option"):
        SystemConfig(type="contextual", retrieval={"document_max_chars": 10})
    with pytest.raises(ValueError, match="unknown option"):
        SystemConfig(type="contextual", retrieval={"docs_k": 3})


# --- hierarchical -----------------------------------------------------------------------------------------------------------

TOPICS = {
    "doc_refund": ("Returns handbook", "Customers may send items back within fourteen days. Unused goods qualify. Receipts are checked at the counter."),
    "doc_parking": ("Facilities guide", "Parking permits renew every January. Bicycles use the basement racks. Visitors register at reception."),
    "doc_payroll": ("Finance manual", "Salaries are paid monthly. Expense claims need receipts. Travel budgets reset each April."),
    "doc_security": ("Security standard", "Passwords rotate quarterly. Laptops use disk encryption. Incidents are reported within an hour."),
    "doc_catering": ("Events cookbook", "Lunch menus change weekly. Vegetarian options are always offered. Coffee is free."),
}
SUMMARIES = {
    "Returns handbook": "Refund policy and how customers return purchased goods.",
    "Facilities guide": "Building access, parking and bicycle storage.",
    "Finance manual": "Pay, expenses and travel budgets.",
    "Security standard": "Password, laptop and incident rules.",
    "Events cookbook": "Menus and catering for staff events.",
}


def _topic_docs() -> list[Document]:
    return [_doc(doc_id, f"# {title}\n\n{body}", title=title) for doc_id, (title, body) in TOPICS.items()]


def _summarising(prompt: str) -> str:
    title = prompt.split("Document title:", 1)[1].splitlines()[0].strip()
    return SUMMARIES[title]


def _hier(llm=None, **retrieval):
    return _system("hierarchical", llm or ScriptedLLM(_summarising), chunker={"type": "word", "chunk_size": 12, "chunk_overlap": 0}, retrieval=retrieval)


def test_hierarchical_writes_one_summary_per_document_and_reports_the_cost():
    llm = ScriptedLLM(_summarising)
    system = _hier(llm)

    ingestion = system.ingest(_topic_docs())

    assert len(llm.calls) == 5 and ingestion.cost.llm_cost == pytest.approx(5 * PER_CALL)
    assert ingestion.num_chunks >= 5 and ingestion.metadata["summaries_generated"] == 5 and ingestion.metadata["docs_without_summary"] == 0
    assert all(call["kwargs"].get("temperature") == 0 for call in llm.calls)


def test_the_first_stage_narrows_the_search_to_docs_k_documents():
    system = _hier(docs_k=2, chunks_per_doc=3, top_k=5)
    system.ingest(_topic_docs())

    result = system.fetch_context("refund for returned goods", top_k=10)

    routed = result.metadata["routed_docs"]
    assert len(routed) == 2 and routed[0] == "doc_refund"
    assert {chunk.doc_id for chunk in result.chunks} <= set(routed)
    assert all(chunk.metadata["routed_doc_rank"] in (1, 2) for chunk in result.chunks)


def test_a_document_is_found_through_its_summary_when_its_chunks_never_say_the_words():
    system = _hier(docs_k=1, chunks_per_doc=2)
    system.ingest(_topic_docs())

    result = system.fetch_context("what is the refund policy", top_k=5)

    assert result.metadata["routed_docs"] == ["doc_refund"] and all(chunk.doc_id == "doc_refund" for chunk in result.chunks)
    assert not any("refund" in chunk.text.lower() for chunk in result.chunks), "only the summary mentions refunds"


def test_chunks_per_doc_caps_each_document_and_chunks_are_ordered_by_score():
    system = _system(
        "hierarchical", ScriptedLLM(_summarising), chunker={"type": "word", "chunk_size": 6, "chunk_overlap": 0}, retrieval={"docs_k": 3, "chunks_per_doc": 2}
    )
    system.ingest(_topic_docs())

    chunks = system.fetch_context("laptops passwords parking bicycles receipts", top_k=10).chunks

    per_doc: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2 and len(per_doc) <= 3 and len(chunks) <= 6
    assert [c.score for c in chunks] == sorted((c.score for c in chunks), reverse=True) and [c.rank for c in chunks] == list(range(1, len(chunks) + 1))


def test_top_k_limits_the_returned_chunks_and_a_configured_top_k_sets_the_default_depth():
    system = _hier(docs_k=5, chunks_per_doc=2, top_k=4)
    system.ingest(_topic_docs())

    assert len(system.fetch_context("menus", top_k=3).chunks) == 3
    assert len(system.fetch_context("menus").chunks) == 4 and system.configured_context_k() == 4


def test_a_document_without_a_usable_summary_is_still_findable_from_its_lead_text():
    def partial(prompt: str):
        title = prompt.split("Document title:", 1)[1].splitlines()[0].strip()
        if title == "Returns handbook":
            return ""  # empty answer
        if title == "Facilities guide":
            return RuntimeError("timeout")  # error
        return SUMMARIES[title]

    system = _hier(ScriptedLLM(partial), docs_k=1, chunks_per_doc=2)

    ingestion = system.ingest(_topic_docs())

    assert ingestion.metadata["docs_without_summary"] == 2 and ingestion.metadata["summaries_generated"] == 3
    assert system.fetch_context("Customers may send items back within fourteen days", top_k=3).metadata["routed_docs"] == ["doc_refund"]
    assert system.fetch_context("Parking permits renew every January", top_k=3).metadata["routed_docs"] == ["doc_parking"]


def test_when_every_summary_call_fails_ingestion_raises():
    with pytest.raises(RuntimeError, match="bad key"):
        _hier(ScriptedLLM(lambda prompt: RuntimeError("bad key"))).ingest(_topic_docs())


def test_hierarchical_traces_two_retrieval_stages_and_pays_for_one_query_embedding():
    system = _hier(docs_k=2, chunks_per_doc=2)
    system.ingest(_topic_docs())

    answer = system.answer_question("refund for returned goods")

    assert [(s.kind, s.name) for s in answer.steps] == [("retrieve", "doc_search"), ("retrieve", "chunk_search"), ("generate", "answer")]
    assert answer.steps[1].cost.total_cost == 0.0 and answer.retrieval_result.metadata["retriever"] == "hierarchical"
    assert answer.steps[0].cost == answer.retrieval_result.cost, "the one query embedding is booked on the first stage"


def test_hierarchical_handles_an_empty_corpus_and_a_question_that_matches_nothing():
    empty = _hier()
    empty.ingest([])
    assert empty.fetch_context("anything").chunks == []

    system = _hier()
    system.ingest(_topic_docs())
    nothing = system.fetch_context("zzzz qqqq", top_k=3)
    assert len(nothing.metadata["routed_docs"]) == system.options.docs_k, "dense similarity still routes when no word matches"


def test_hierarchical_options_are_validated():
    for bad in ({"docs_k": 0}, {"chunks_per_doc": 0}, {"vector_store": "numpy"}):
        with pytest.raises(ValueError):
            SystemConfig(type="hierarchical", retrieval=bad)


# --- end to end -------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ctx_hier")
    docs = root / "docs"
    docs.mkdir()
    for document in _policy_docs() + _topic_docs():
        (docs / f"{document.doc_id}.md").write_text(document.text + "\n", encoding="utf-8")
    write_jsonl(
        root / "questions.jsonl",
        [
            {"id": "q_001", "question": "What is the Alpha escalation limit?", "reference_answer": "30 minutes.", "expected_keywords": ["30 minutes"], "relevant_doc_ids": ["doc_s1"], "category": "direct_fact"},
            {"id": "q_002", "question": "How do customers return goods?", "reference_answer": "Within fourteen days.", "expected_keywords": ["fourteen"], "relevant_doc_ids": ["doc_refund"], "category": "direct_fact"},
        ],
    )
    config = root / "config.yaml"
    config.write_text(
        f"""
run: {{name: ctx, output_dir: {root / "results"}}}
dataset: {{documents_path: {docs}, questions_path: {root / "questions.jsonl"}}}
systems:
  - {{type: contextual, name: ctx_x, retrieval: {{top_k: 3}}}}
  - {{type: hierarchical, name: hier_x, retrieval: {{docs_k: 2, top_k: 3}}}}
evaluation: {{k_values: [1, 3], max_workers: 1}}
""",
        encoding="utf-8",
    )
    return run_benchmark(config, force_mock=True, max_workers=1)


def test_both_systems_run_end_to_end_and_report_an_ingestion_row(run_dir):
    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    costs = pd.read_csv(run_dir / "cost_breakdown.csv")

    assert (summary["n_ok"] == 2).all() and (summary["n_error"] == 0).all()
    ingestion = costs[costs["stage"] == "ingestion"]
    assert set(ingestion["system"]) == {"ctx_x", "hier_x"} and {"llm_cost", "llm_prompt_tokens", "embedding_cost"} <= set(ingestion.columns)


def test_both_systems_are_listed():
    result = CliRunner().invoke(app, ["list-systems"])

    assert result.exit_code == 0 and "contextual" in result.output and "hierarchical" in result.output
