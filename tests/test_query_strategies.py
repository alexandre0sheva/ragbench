"""`rag_fusion` (LLM query variants + RRF) and `decompose` (LLM sub-questions) against a scripted LLM, plus the shared planner helpers."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from typer.testing import CliRunner

from ragbench.cli import app
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import GENERATE_QUERY_VARIANTS_MARKER, PLAN_SUBQUESTIONS_MARKER
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.rag_systems.trace import STAGE_KEYS
from ragbench.utils.query_planning import parse_string_list, plan_query_variants_llm, plan_subquestions_llm

PER_CALL = 0.002

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="a", text="Zorblax turbines produce clean energy for coastal towns. Zorblax turbines are quiet."),
    Document(doc_id="doc_b", path="b.md", title="b", text="Quuxwell batteries store energy overnight for mountain villages. Quuxwell cells last long."),
    Document(doc_id="doc_c", path="c.md", title="c", text="Plinth harbours host fishing fleets in the northern bay. Plinth docks open at dawn."),
    Document(doc_id="doc_d", path="d.md", title="d", text="Marlowe orchards grow apples and pears in the southern valley. Marlowe harvest festivals draw crowds."),
]
QUESTION = "How is energy produced and stored?"


class ScriptedLLM(MockLLM):
    """Answers planner prompts with scripted text (a string, or an Exception to raise), records every call, and charges a fixed price per planner call.
    Prompts that are not planner prompts (question answering, sub-answers) go to the plain mock."""

    def __init__(self, variants: str | None = None, subquestions: str | None = None):
        super().__init__()
        self.scripts = {GENERATE_QUERY_VARIANTS_MARKER: variants, PLAN_SUBQUESTIONS_MARKER: subquestions}
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        for marker, text in self.scripts.items():
            if marker in prompt and text is not None:
                with self._lock:
                    self.calls.append({"marker": marker, "prompt": prompt, "kwargs": kwargs})
                if isinstance(text, Exception):
                    raise text
                cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
                return LLMResult(text=text, model="scripted", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")
        return super().generate(messages, **kwargs)


def _system(type_: str, llm: MockLLM | None = None, **retrieval):
    system = create_rag_system(SystemConfig(type=type_, retrieval={"top_k": 3, **retrieval}), force_mock=True)
    if llm is not None:
        system.llm = llm
    system.ingest(DOCS)
    return system


def _steps(result, kind=None, name=None):
    return [s for s in result.steps if (kind is None or s.kind == kind) and (name is None or s.name == name)]


# --- planner helpers ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"queries": ["a b", "c d"]}', ["a b", "c d"]),
        ('```json\n{"queries": ["a b", "c d"]}\n```', ["a b", "c d"]),
        ('Sure! Here you go: {"queries": ["a b"]} hope that helps', ["a b"]),
        ('["x y", "z w"]', ["x y", "z w"]),
        ('{"queries": [{"query": "from dict"}, {"question": "also dict"}, 7, null, "  ", "plain"]}', ["from dict", "also dict", "plain"]),
        ('{"queries": ["Same", "same", "SAME "]}', ["Same"]),
        ("not json at all", []),
        ('{"other": ["a"]}', []),
        ('{"queries": "not a list"}', []),
        ("", []),
    ],
)
def test_parse_string_list_is_forgiving(text, expected):
    assert parse_string_list(text, keys=("queries",)) == expected


def test_plan_query_variants_drops_the_original_and_caps():
    llm = ScriptedLLM(variants=json.dumps({"queries": [QUESTION, "wind turbines", "battery storage", "solar farms", "tidal power"]}))
    plan = plan_query_variants_llm(llm, QUESTION, 3)
    assert plan.items == ["wind turbines", "battery storage", "solar farms"]
    assert plan.fallback is False
    assert "3" in llm.calls[0]["prompt"] and QUESTION in llm.calls[0]["prompt"]
    assert llm.calls[0]["kwargs"]["json_mode"] is True


def test_plan_subquestions_caps_and_falls_back_on_garbage():
    llm = ScriptedLLM(subquestions='{"sub_questions": ["one?", "two?", "three?"]}')
    assert plan_subquestions_llm(llm, QUESTION, 2).items == ["one?", "two?"]
    broken = plan_subquestions_llm(ScriptedLLM(subquestions="¯\\_(ツ)_/¯"), QUESTION, 4)
    assert broken.items == [QUESTION] and broken.fallback is True


# --- rag_fusion -----------------------------------------------------------------------------------------------------------


def test_rag_fusion_merges_the_variant_rankings_with_rrf():
    # Each variant is an exact phrase of one document; the original question matches nothing in particular.
    llm = ScriptedLLM(variants=json.dumps({"queries": ["Zorblax turbines clean energy", "Quuxwell batteries store energy", "Zorblax turbines quiet"]}))
    system = _system("rag_fusion", llm, retriever="vector", num_queries=3)
    result = system.fetch_context(QUESTION, top_k=3)
    docs = [chunk.doc_id for chunk in result.chunks]
    assert docs[0] == "doc_a"  # two of the three variants rank doc_a first, so RRF puts it on top
    assert "doc_b" in docs[:2]
    assert result.metadata["queries"] == [QUESTION, "Zorblax turbines clean energy", "Quuxwell batteries store energy", "Zorblax turbines quiet"]
    assert [c.rank for c in result.chunks] == [1, 2, 3]
    assert all("rrf_source_ranks" in c.metadata for c in result.chunks)


def test_rag_fusion_can_leave_out_the_original_question():
    llm = ScriptedLLM(variants=json.dumps({"queries": ["Plinth harbours fishing"]}))
    result = _system("rag_fusion", llm, retriever="vector", include_original=False).fetch_context(QUESTION, top_k=2)
    assert result.metadata["queries"] == ["Plinth harbours fishing"]
    assert result.chunks[0].doc_id == "doc_c"


@pytest.mark.parametrize("retriever", ["hybrid", "vector"])
def test_rag_fusion_traces_one_search_per_query_and_costs_add_up(retriever):
    llm = ScriptedLLM(variants=json.dumps({"queries": ["Zorblax turbines", "Quuxwell batteries"]}))
    system = _system("rag_fusion", llm, retriever=retriever)
    answer = system.answer_question(QUESTION)
    planning = _steps(answer, "llm", "query_variants")
    assert len(planning) == 1 and planning[0].cost.total_cost == pytest.approx(PER_CALL)
    searches = _steps(answer, "retrieve", "variant_search")
    assert len(searches) == 3  # original + 2 variants
    assert [s.metadata["index"] for s in searches] == [0, 1, 2]
    assert searches[1].input_preview == "Zorblax turbines"
    assert _steps(answer, "generate", "answer")
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)
    assert answer.cost.total_cost >= PER_CALL
    assert not _steps(answer, name="untracked")


def test_rag_fusion_survives_invalid_json_by_searching_the_question_alone():
    system = _system("rag_fusion", ScriptedLLM(variants="I refuse to answer in JSON"), retriever="vector")
    answer = system.answer_question(QUESTION)
    planning = _steps(answer, "llm", "query_variants")[0]
    assert planning.metadata["fallback"] is True
    assert len(_steps(answer, "retrieve", "variant_search")) == 1
    assert answer.retrieval_result.metadata["queries"] == [QUESTION]
    assert answer.answer


def test_rag_fusion_with_hybrid_runs_both_searches_per_query():
    llm = ScriptedLLM(variants=json.dumps({"queries": ["Zorblax turbines"]}))
    system = _system("rag_fusion", llm, retriever="hybrid")
    result = system.fetch_context(QUESTION, top_k=3)
    assert result.chunks[0].doc_id == "doc_a"


# --- decompose ------------------------------------------------------------------------------------------------------------


def test_decompose_retrieves_for_every_subquestion_and_records_each_as_a_step():
    subs = ["Zorblax turbines clean energy", "Quuxwell batteries store energy", "Plinth harbours fishing fleets"]
    llm = ScriptedLLM(subquestions=json.dumps({"sub_questions": subs}))
    system = _system("decompose", llm, retriever="vector", max_subquestions=4)
    answer = system.answer_question(QUESTION)
    searches = _steps(answer, "retrieve", "subquestion_search")
    assert [s.input_preview for s in searches] == subs
    assert [s.metadata["index"] for s in searches] == [0, 1, 2]
    recorded = answer.retrieval_result.metadata["sub_questions"]
    assert [r["question"] for r in recorded] == subs
    assert [r["chunk_ids"][0].split("::")[0] for r in recorded] == ["doc_a", "doc_b", "doc_c"]  # each search found its own document first
    assert len(llm.calls) == 1  # parallel mode: one planning call, no per-sub-question LLM calls
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)
    assert not _steps(answer, name="untracked")


def test_decompose_caps_at_max_subquestions():
    llm = ScriptedLLM(subquestions=json.dumps({"sub_questions": [f"Zorblax question {i}" for i in range(9)]}))
    system = _system("decompose", llm, retriever="vector", max_subquestions=3)
    answer = system.answer_question(QUESTION)
    assert len(_steps(answer, "retrieve", "subquestion_search")) == 3
    assert "at most 3" in llm.calls[0]["prompt"]


def test_decompose_survives_invalid_json_and_flags_the_fallback():
    system = _system("decompose", ScriptedLLM(subquestions="{not valid json"), retriever="vector")
    answer = system.answer_question(QUESTION)
    planning = _steps(answer, "llm", "plan_subquestions")[0]
    assert planning.metadata["fallback"] is True
    searches = _steps(answer, "retrieve", "subquestion_search")
    assert len(searches) == 1 and searches[0].input_preview == QUESTION
    assert answer.retrieval_result.metadata["fallback"] is True
    assert answer.answer and answer.retrieval_result.chunks


def test_decompose_answer_prompt_lists_subquestions_with_their_evidence_and_doc_ids():
    seen: list[str] = []

    class Spy(ScriptedLLM):
        def generate(self, messages, **kwargs):
            if "Sub-question 1" in str(messages[-1]["content"]):
                seen.append(str(messages[-1]["content"]))
            return super().generate(messages, **kwargs)

    llm = Spy(subquestions=json.dumps({"sub_questions": ["Zorblax turbines clean energy", "Quuxwell batteries store energy"]}))
    system = _system("decompose", llm, retriever="vector", top_k=4)
    system.answer_question(QUESTION)
    assert len(seen) == 1
    prompt = seen[0]
    assert "Sub-question 1: Zorblax turbines clean energy" in prompt and "Sub-question 2: Quuxwell batteries store energy" in prompt
    assert prompt.index("Sub-question 1") < prompt.index("[doc_a") < prompt.index("Sub-question 2") < prompt.index("[doc_b")
    assert prompt.count("[doc_a") == 1  # a passage is listed under the one sub-question that ranked it highest
    assert prompt.rstrip().endswith(f"Question: {QUESTION}")


def test_decompose_sequential_feeds_earlier_answers_forward():
    llm = ScriptedLLM(subquestions=json.dumps({"sub_questions": ["Zorblax turbines clean energy", "Quuxwell batteries store energy"]}))
    system = _system("decompose", llm, retriever="vector", sequential=True)
    answer = system.answer_question(QUESTION)
    sub_answers = _steps(answer, "llm", "sub_answer")
    assert len(sub_answers) == 2
    second_search = _steps(answer, "retrieve", "subquestion_search")[1]
    first_finding = answer.retrieval_result.metadata["sub_questions"][0]["answer"]
    assert first_finding and first_finding[:30] in second_search.metadata["query"]
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)
    assert not _steps(answer, name="untracked")


def test_decompose_can_also_search_the_original_question():
    llm = ScriptedLLM(subquestions=json.dumps({"sub_questions": ["Zorblax turbines clean energy"]}))
    system = _system("decompose", llm, retriever="vector", include_original=True)
    answer = system.answer_question(QUESTION)
    searches = _steps(answer, "retrieve", "subquestion_search")
    assert [s.input_preview for s in searches] == ["Zorblax turbines clean energy", QUESTION]
    assert answer.retrieval_result.metadata["sub_questions"][-1]["original"] is True


# --- shared behaviour -----------------------------------------------------------------------------------------------------


def test_steps_are_isolated_between_concurrent_questions():
    llm = ScriptedLLM(variants=json.dumps({"queries": ["Zorblax turbines"]}))
    system = _system("rag_fusion", llm, retriever="vector")
    questions = [f"{QUESTION} variation {i}" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(system.answer_question, questions))
    for question, answer in zip(questions, answers, strict=True):
        assert len(_steps(answer, "retrieve", "variant_search")) == 2
        assert _steps(answer, "retrieve", "variant_search")[0].input_preview == question
        assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)


@pytest.mark.parametrize("type_", ["rag_fusion", "decompose"])
def test_mock_mode_runs_end_to_end_without_a_script(type_):
    answer = _system(type_, retriever="hybrid").answer_question("Compare Zorblax turbines and Quuxwell batteries by energy")
    assert answer.answer and answer.retrieval_result.chunks
    assert {s.kind for s in answer.steps} <= set(STAGE_KEYS)
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)


def test_both_systems_are_registered_with_specs_and_reject_typos():
    specs = {spec.type: spec for spec in all_specs()}
    for type_ in ("rag_fusion", "decompose"):
        assert specs[type_].requires_llm is True and specs[type_].retrieves is True
    with pytest.raises(ValueError, match="num_queries"):
        create_rag_system(SystemConfig(type="rag_fusion", retrieval={"num_querys": 3}), force_mock=True)
    with pytest.raises(ValueError, match="max_subquestions"):
        create_rag_system(SystemConfig(type="decompose", retrieval={"max_subquestion": 3}), force_mock=True)


def test_list_systems_shows_both():
    result = CliRunner().invoke(app, ["list-systems"])
    assert result.exit_code == 0
    assert "rag_fusion" in result.output and "decompose" in result.output
