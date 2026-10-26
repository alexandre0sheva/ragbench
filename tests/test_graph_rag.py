"""`graph_rag` (GraphRAG-lite): LLM entity/relation extraction at ingestion, communities and their reports, then `local` or `global` search."""

from __future__ import annotations

import json
import threading
import time

import pytest
from pydantic import ValidationError

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import EXTRACT_GRAPH_MARKER, MAP_COMMUNITY_MARKER, SUMMARIZE_COMMUNITY_MARKER
from ragbench.rag_systems import create_rag_system

PER_CALL = 0.001
CHUNKER = {"type": "word", "chunk_size": 60, "chunk_overlap": 0}  # one chunk per document below


class RecordingLLM(MockLLM):
    """The mock's answers at a fixed price per call, with every call recorded; `override(prompt)` may answer instead (or raise)."""

    def __init__(self, override=None):
        super().__init__()
        self.override = override
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        with self._lock:
            self.calls.append({"prompt": prompt, "kwargs": kwargs})
        text = self.override(prompt) if self.override else None
        if isinstance(text, Exception):
            raise text
        if text is None:
            text = super().generate(messages, **kwargs).text
        cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
        return LLMResult(text=text, model="recording", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")

    def prompts(self, marker: str) -> list[dict]:
        return [call for call in self.calls if marker in call["prompt"]]


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, path=f"{doc_id}.md", title=doc_id, text=text)


def _docs() -> list[Document]:
    return [
        _doc("doc_a", "Alice Larsen founded Nordvik Labs in 2019. Alice Larsen studied physics at Bergen University."),
        _doc("doc_b", "Nordvik Labs is headquartered in Tromsø. Nordvik Labs builds sensors for fishing boats."),
        _doc("doc_c", "The city hosts a jazz festival every summer, and the city hosts a market on the lab square."),
        _doc("doc_d", "Harbor Freight Union represents dock workers. Harbor Freight Union bargains with Port Authority."),
    ]


def _system(llm=None, **retrieval):
    system = create_rag_system(SystemConfig(type="graph_rag", chunker=CHUNKER, retrieval=retrieval), force_mock=True)
    if llm is not None:
        system.llm = llm
    return system


# --- ingestion --------------------------------------------------------------------------------------------------------------


def test_ingestion_extracts_each_chunk_once_and_writes_one_report_per_community():
    llm = RecordingLLM()
    system = _system(llm)

    ingestion = system.ingest(_docs())

    extractions, reports = llm.prompts(EXTRACT_GRAPH_MARKER), llm.prompts(SUMMARIZE_COMMUNITY_MARKER)
    assert len(extractions) == ingestion.num_chunks == 4
    assert all(call["kwargs"].get("temperature") == 0 and call["kwargs"].get("json_mode") for call in extractions), "temperature 0, so the disk cache applies"
    assert all(call["kwargs"].get("temperature") == 0 for call in reports)
    meta = ingestion.metadata
    assert meta["entities"] == 6 and meta["relations"] == 4
    assert len(reports) == meta["communities"] >= 2, "one report per community; doc_c names no entity"
    assert ingestion.cost.llm_cost == pytest.approx((4 + meta["communities"]) * PER_CALL)
    assert meta["extraction_failures"] == meta["report_failures"] == 0 and meta["expensive"] is True


def test_a_failed_extraction_degrades_only_its_chunk_and_is_counted():
    llm = RecordingLLM(lambda prompt: RuntimeError("boom") if EXTRACT_GRAPH_MARKER in prompt and "Tromsø" in prompt else None)
    system = _system(llm)

    ingestion = system.ingest(_docs())

    assert ingestion.metadata["extraction_failures"] == 1
    assert "tromsø" not in system.graph.entities and "alice larsen" in system.graph.entities


def test_a_failed_report_falls_back_to_the_entities_and_relations_it_covers():
    llm = RecordingLLM(lambda prompt: RuntimeError("boom") if SUMMARIZE_COMMUNITY_MARKER in prompt and "Harbor" in prompt else None)
    system = _system(llm, mode="global")

    ingestion = system.ingest(_docs())

    assert ingestion.metadata["report_failures"] == 1
    harbor = next(report for report in system.reports if "Harbor Freight Union" in report.text)
    assert "Port Authority" in harbor.text and harbor.doc_ids == ["doc_d"]


def test_the_mock_extracts_capitalised_names_and_links_names_in_the_same_sentence():
    prompt = f"{EXTRACT_GRAPH_MARKER}. Name at most 2 entities.\n<text>\nThe pilot met Ada Byrne at Kestrel Air in Oslo.\n</text>"

    reply = json.loads(MockLLM().generate([{"role": "user", "content": prompt}], json_mode=True).text)

    assert [e["name"] for e in reply["entities"]] == ["Ada Byrne", "Kestrel Air"]
    assert [(r["source"], r["target"]) for r in reply["relations"]] == [("Ada Byrne", "Kestrel Air")]


def test_an_extraction_reply_that_is_not_json_counts_as_a_failed_chunk():
    llm = RecordingLLM(lambda prompt: "Sure! The entities are Nordvik Labs and Tromsø." if EXTRACT_GRAPH_MARKER in prompt and "Tromsø" in prompt else None)

    ingestion = _system(llm).ingest(_docs())

    assert ingestion.metadata["extraction_failures"] == 1


def test_when_no_extraction_reply_is_usable_ingestion_fails_instead_of_building_an_empty_graph():
    llm = RecordingLLM(lambda prompt: '{"entities": [{"name": "Al' if EXTRACT_GRAPH_MARKER in prompt else None)  # every reply cut off

    with pytest.raises(ValueError, match="extraction reply"):
        _system(llm).ingest(_docs())


def test_when_every_report_fails_the_fallback_reports_stand_in():
    llm = RecordingLLM(lambda prompt: RuntimeError("boom") if SUMMARIZE_COMMUNITY_MARKER in prompt else None)
    system = _system(llm, mode="global")

    ingestion = system.ingest(_docs())

    assert ingestion.metadata["report_failures"] == ingestion.metadata["communities"] >= 2
    assert all(report.text.startswith("Entities:") for report in system.reports)


def test_the_extraction_prompt_names_the_chunks_document():
    llm = RecordingLLM()
    document = Document(doc_id="doc_q", path="q.md", title="Quarterly report of Vela Systems", text="The company said its revenue rose.")

    _system(llm).ingest([document])

    assert "Document title: Quarterly report of Vela Systems" in llm.prompts(EXTRACT_GRAPH_MARKER)[0]["prompt"]


# --- local search -------------------------------------------------------------------------------------------------------------


def test_local_search_reaches_a_document_through_an_entity_neighbour():
    """Only the graph links the lab Alice Larsen founded to its city; doc_c shares the question's words but none of its entities."""
    question = "Which city hosts the lab Alice Larsen founded?"
    graph = _system(top_k=2)
    graph.ingest(_docs())

    result = graph.fetch_context(question, top_k=2)

    assert {chunk.doc_id for chunk in result.chunks} == {"doc_a", "doc_b"}
    reached = next(chunk for chunk in result.chunks if chunk.doc_id == "doc_b")
    assert "Nordvik Labs" in reached.metadata["graph_entities"]
    assert result.metadata["matched_entities"][0] == "Alice Larsen"


def test_local_search_shows_the_generator_graph_notes_and_the_judge_only_document_chunks():
    llm = RecordingLLM()
    system = _system(llm, top_k=2)
    system.ingest(_docs())
    llm.calls.clear()

    answer = system.answer_question("Which city hosts the lab Alice Larsen founded?")

    [generation] = llm.calls
    assert "Graph notes (not evidence by themselves)" in generation["prompt"]
    assert "Alice Larsen — Nordvik Labs: Alice Larsen founded Nordvik Labs in 2019." in generation["prompt"]
    chunk_ids = {chunk.chunk_id for chunk in system.chunks}
    assert set(answer.metadata["context_chunk_ids"]) <= chunk_ids, "the judge reads document chunks, not the notes"
    assert [step.name for step in answer.steps] == ["entity_search", "graph_expand", "answer"]


def test_a_hub_entity_spreads_weight_only_to_its_strongest_neighbours():
    partners = [f"Partner {letter}{letter.lower()}" for letter in "ABCDEFGHIJKLMNOPQRST"]
    hub = {
        "entities": [{"name": "Hubcorp Holdings", "type": "organization", "description": "The hub."}],
        "relations": [{"source": "Hubcorp Holdings", "target": name, "description": f"Hubcorp Holdings works with {name}."} for name in partners],
    }
    llm = RecordingLLM(lambda prompt: json.dumps(hub) if EXTRACT_GRAPH_MARKER in prompt and "Hubcorp" in prompt else None)
    docs = [_doc("doc_hub", "Hubcorp Holdings directory."), *(_doc(f"doc_{i}", f"{name} makes widgets.") for i, name in enumerate(partners))]
    system = _system(llm, entities_k=1, neighbours_per_entity=3)
    system.ingest(docs)

    result = system.fetch_context("Hubcorp Holdings directory", top_k=len(docs))

    assert result.metadata["matched_entities"] == ["Hubcorp Holdings"]
    linked = {chunk.doc_id for chunk in result.chunks if chunk.metadata["graph_score"] > 0}
    assert linked == {"doc_hub", "doc_0", "doc_1", "doc_2"}, "the hub's chunk and its three strongest (then first by name) neighbours'"


class _NoScan(dict):
    """A relation table that may be looked up by key but not walked: notes and reports must follow the graph's adjacency."""

    def __iter__(self):
        raise AssertionError("walked every relation of the graph")

    def items(self):
        raise AssertionError("walked every relation of the graph")

    def keys(self):
        raise AssertionError("walked every relation of the graph")

    def values(self):
        raise AssertionError("walked every relation of the graph")


def test_notes_and_reports_follow_the_adjacency_instead_of_walking_every_relation():
    system = _system()
    system.ingest(_docs())
    system.graph.relations = _NoScan(system.graph.relations)

    notes = system.fetch_context("Which city hosts the lab Alice Larsen founded?").metadata["graph_notes"]
    report = system._community_text(["alice larsen", "nordvik labs", "tromsø"])

    assert "Alice Larsen — Nordvik Labs" in notes
    assert "Nordvik Labs — Tromsø" in report and "Alice Larsen — Nordvik Labs" in report


def test_without_any_entity_local_search_is_plain_dense_retrieval():
    system = _system()
    system.ingest([_doc("doc_x", "lunch is served until two pm."), _doc("doc_y", "the gym opens at six.")])

    result = system.fetch_context("when is lunch served", top_k=2)

    assert [chunk.doc_id for chunk in result.chunks][0] == "doc_x" and result.metadata["matched_entities"] == []


# --- global search ------------------------------------------------------------------------------------------------------------


def test_global_search_maps_over_the_reports_it_hands_the_generator_then_reduces():
    llm = RecordingLLM()
    system = _system(llm, mode="global")
    system.ingest(_docs())
    llm.calls.clear()

    answer = system.answer_question("Who represents dock workers?", context_k=2)

    chunks = answer.retrieval_result.chunks
    assert chunks and all(chunk.chunk_id.startswith("community::") for chunk in chunks)
    assert all(chunk.doc_id == chunk.metadata["doc_ids"][0] for chunk in chunks)
    assert chunks[0].doc_id == "doc_d"
    assert len(llm.prompts(MAP_COMMUNITY_MARKER)) == min(2, len(chunks))
    assert [step.name for step in answer.steps] == ["community_search", "map_communities", "reduce"]
    assert answer.metadata["context_chunk_ids"] == [chunk.chunk_id for chunk in chunks[:2]], "the judge reads the reports the map step read"


def test_global_search_maps_its_reports_in_parallel_as_one_step():
    docs = [_doc(f"doc_{i}", f"{a} met {b} on Monday.") for i, (a, b) in enumerate([("Ana Holm", "Bo Lind"), ("Cy Moe", "Di Ness"), ("Ed Ost", "Fay Pik"), ("Gus Quist", "Hal Rud")])]

    def override(prompt: str):
        if MAP_COMMUNITY_MARKER in prompt:
            time.sleep(0.25)
        return None

    llm = RecordingLLM(override)
    system = _system(llm, mode="global")
    system.ingest(docs)
    llm.calls.clear()

    started = time.perf_counter()
    answer = system.answer_question("Who met on Monday?", context_k=4)
    elapsed = time.perf_counter() - started

    assert len(llm.prompts(MAP_COMMUNITY_MARKER)) == 4
    assert elapsed < 0.75, f"four 0.25 s map calls took {elapsed:.2f} s: they ran one after another"
    mapped = answer.steps[1]
    assert (mapped.kind, mapped.name) == ("generate", "map_communities")
    assert mapped.cost.llm_cost == pytest.approx(4 * PER_CALL) and mapped.latency_ms >= 250


def test_points_a_community_scores_zero_never_reach_the_reduce_step():
    def override(prompt: str):
        if MAP_COMMUNITY_MARKER not in prompt:
            return None
        if "Harbor" in prompt:
            return json.dumps({"points": [{"text": "Harbor Freight Union represents dock workers.", "score": 90}]})
        return json.dumps({"points": [{"text": "An unrelated remark.", "score": 0}]})

    llm = RecordingLLM(override)
    system = _system(llm, mode="global")
    system.ingest(_docs())
    llm.calls.clear()

    system.answer_question("Who represents dock workers?", context_k=5)

    reduce = llm.calls[-1]["prompt"]
    assert "Harbor Freight Union represents dock workers." in reduce and "An unrelated remark." not in reduce


def test_global_search_with_no_community_answers_without_any_context():
    system = _system(mode="global")
    system.ingest([_doc("doc_x", "lunch is served until two pm.")])

    answer = system.answer_question("when is lunch served")

    assert answer.retrieval_result.chunks == [] and "could not find" in answer.answer.lower()


# --- options ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("retrieval", [{"mode": "hybrid"}, {"entity_k": 3}, {"entities_k": 0}])
def test_bad_options_fail_when_the_system_is_built(retrieval):
    with pytest.raises(ValidationError):
        _system(**retrieval)
