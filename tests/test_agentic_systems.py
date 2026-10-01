"""`corrective` (grade, rewrite, retry) and `iterative` (multi-hop) against a scripted LLM, plus how the evaluator records them."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.models.prompts import CHECK_GROUNDED_MARKER, GRADE_CHUNKS_MARKER, NEXT_HOP_MARKER, REWRITE_QUERY_MARKER
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.utils.jsonl import read_jsonl, write_jsonl

PER_CALL = 0.01
QUESTION = "How do Zorblax turbines make energy?"

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="a", text="Zorblax turbines produce clean energy for coastal towns. Zorblax turbines are quiet."),
    Document(doc_id="doc_b", path="b.md", title="b", text="Quuxwell batteries store energy overnight for mountain villages. Quuxwell cells last long."),
    Document(doc_id="doc_c", path="c.md", title="c", text="Plinth harbours host fishing fleets in the northern bay. Plinth docks open at dawn."),
]


def _grades(*grades: str) -> str:
    return json.dumps({"grades": [{"id": i, "grade": g} for i, g in enumerate(grades, start=1)]})


ALL_IRRELEVANT = _grades(*["irrelevant"] * 20)
ALL_RELEVANT = _grades(*["relevant"] * 20)


class ScriptedLLM(MockLLM):
    """Replies to each kind of control prompt from a script: a list is consumed in order (the last entry repeats), a string always answers.
    Control calls cost `PER_CALL`; everything else (the answer itself) goes to the plain mock and is free."""

    def __init__(self, **scripts: str | list[str]):
        super().__init__()
        self.scripts = {
            GRADE_CHUNKS_MARKER: scripts.get("grade"),
            REWRITE_QUERY_MARKER: scripts.get("rewrite"),
            NEXT_HOP_MARKER: scripts.get("hop"),
            CHECK_GROUNDED_MARKER: scripts.get("check"),
        }
        self.calls: dict[str, list[str]] = {marker: [] for marker in self.scripts}
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def generate(self, messages, **kwargs) -> LLMResult:
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        with self._lock:
            self.prompts.append(prompt)
        for marker, script in self.scripts.items():
            if marker in prompt and script is not None:
                with self._lock:
                    self.calls[marker].append(prompt)
                    turn = len(self.calls[marker]) - 1
                text = script if isinstance(script, str) else script[min(turn, len(script) - 1)]
                cost = CostBreakdown(llm_prompt_tokens=10, llm_completion_tokens=5, llm_cost=PER_CALL)
                return LLMResult(text=text, model="scripted", prompt_tokens=10, completion_tokens=5, cost=cost, finish_reason="stop")
        return super().generate(messages, **kwargs)

    def count(self, marker: str) -> int:
        return len(self.calls[marker])


def _system(type_: str, llm: MockLLM | None = None, docs: list[Document] | None = None, chunker: dict | None = None, **retrieval):
    system = create_rag_system(
        SystemConfig(type=type_, retrieval={"top_k": 3, "retriever": "vector", **retrieval}, **({"chunker": chunker} if chunker else {})), force_mock=True
    )
    if llm is not None:
        system.llm = llm
    system.ingest(docs or DOCS)
    return system


def _steps(result, kind=None, name=None):
    return [s for s in result.steps if (kind is None or s.kind == kind) and (name is None or s.name == name)]


def _assert_costs_add_up(answer):
    assert sum(s.cost.total_cost for s in answer.steps) == pytest.approx(answer.cost.total_cost)
    assert not _steps(answer, name="untracked")


# --- corrective -----------------------------------------------------------------------------------------------------------


def test_corrective_irrelevant_first_round_triggers_exactly_one_rewrite_round_then_answers():
    llm = ScriptedLLM(grade=[ALL_IRRELEVANT, ALL_RELEVANT], rewrite=json.dumps({"query": "Zorblax turbines clean energy"}))
    answer = _system("corrective", llm).answer_question(QUESTION)
    assert llm.count(GRADE_CHUNKS_MARKER) == 2 and llm.count(REWRITE_QUERY_MARKER) == 1
    assert [(s.kind, s.name) for s in answer.steps if s.name != "agent_decision"] == [
        ("retrieve", "round_search"),
        ("grade", "grade_chunks"),
        ("llm", "rewrite_query"),
        ("retrieve", "round_search"),
        ("grade", "grade_chunks"),
        ("generate", "answer"),
    ]
    assert answer.retrieval_result.metadata["queries"] == [QUESTION, "Zorblax turbines clean energy"]
    agent = answer.metadata["agent"]
    assert (agent["steps"], agent["hops"], agent["termination"], agent["budget_exhausted"], agent["sufficient"]) == (2, 1, "finished", False, True)
    assert agent["llm_calls"] == 4  # two gradings, one rewrite, the answer
    assert answer.answer
    _assert_costs_add_up(answer)
    assert answer.cost.total_cost >= 3 * PER_CALL


def test_corrective_a_good_first_round_does_not_rewrite():
    llm = ScriptedLLM(grade=ALL_RELEVANT, rewrite='{"query": "never used"}')
    answer = _system("corrective", llm).answer_question(QUESTION)
    assert llm.count(REWRITE_QUERY_MARKER) == 0
    assert answer.metadata["agent"]["steps"] == 1 and answer.metadata["agent"]["hops"] == 0


def test_corrective_gives_up_after_max_rounds_and_says_the_evidence_was_insufficient():
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite=[json.dumps({"query": "alpha beta"}), json.dumps({"query": "gamma delta"})])
    answer = _system("corrective", llm, max_rounds=2).answer_question(QUESTION)
    assert llm.count(GRADE_CHUNKS_MARKER) == 3 and llm.count(REWRITE_QUERY_MARKER) == 2
    agent = answer.metadata["agent"]
    assert agent["steps"] == 3 and agent["hops"] == 2 and agent["sufficient"] is False and agent["termination"] == "finished"
    assert answer.answer and answer.retrieval_result.chunks  # best effort: it still answers from what it found
    _assert_costs_add_up(answer)


def test_corrective_with_zero_rounds_only_grades():
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite='{"query": "x y"}')
    answer = _system("corrective", llm, max_rounds=0).answer_question(QUESTION)
    assert llm.count(GRADE_CHUNKS_MARKER) == 1 and llm.count(REWRITE_QUERY_MARKER) == 0
    assert answer.metadata["agent"]["sufficient"] is False


def test_corrective_stops_when_a_rewrite_only_repeats_an_earlier_query():
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite=json.dumps({"query": QUESTION.upper()}))
    answer = _system("corrective", llm, max_rounds=3).answer_question(QUESTION)
    assert llm.count(GRADE_CHUNKS_MARKER) == 1
    assert answer.metadata["agent"]["rewrite_failed"] is True


def test_corrective_survives_an_unreadable_grade_reply_without_retrying():
    llm = ScriptedLLM(grade="I would rather not say", rewrite='{"query": "x y"}')
    answer = _system("corrective", llm).answer_question(QUESTION)
    assert llm.count(REWRITE_QUERY_MARKER) == 0
    assert answer.metadata["agent"]["grading_failed"] is True
    assert _steps(answer, "grade", "grade_chunks")[0].metadata["fallback"] is True
    assert answer.retrieval_result.chunks and answer.answer


def test_corrective_orders_chunks_by_grade_and_labels_them():
    llm = ScriptedLLM(grade=_grades("irrelevant", "ambiguous", "relevant"))
    system = _system("corrective", llm, max_rounds=0)
    ranking = system.fetch_context(QUESTION, top_k=3).chunks
    plain = _system("corrective", ScriptedLLM(grade=ALL_RELEVANT), max_rounds=0).fetch_context(QUESTION, top_k=3).chunks  # all equal: the search order
    assert [c.metadata["grade"] for c in ranking] == ["relevant", "ambiguous", "irrelevant"]
    assert [c.chunk_id for c in ranking] == [plain[2].chunk_id, plain[1].chunk_id, plain[0].chunk_id]
    assert [c.rank for c in ranking] == [1, 2, 3]


def test_corrective_cost_cap_stops_the_retry_before_the_next_llm_call():
    # The grading call brings the question to $0.01, the cap: the rewrite call must not be made.
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite='{"query": "alpha beta"}')
    answer = _system("corrective", llm, max_cost_usd=PER_CALL).answer_question(QUESTION)
    assert llm.count(GRADE_CHUNKS_MARKER) == 1 and llm.count(REWRITE_QUERY_MARKER) == 0
    agent = answer.metadata["agent"]
    assert agent["termination"] == "budget" and agent["budget_exhausted"] is True
    assert answer.answer  # budget exhaustion is never an error
    _assert_costs_add_up(answer)


def test_corrective_cost_cap_also_stops_the_loop_before_a_new_round():
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite=json.dumps({"query": "alpha beta"}))
    answer = _system("corrective", llm, max_cost_usd=PER_CALL * 1.5, max_rounds=5).answer_question(QUESTION)
    # grade ($0.01) + rewrite ($0.02) reach the cap, so the second round never starts
    assert llm.count(REWRITE_QUERY_MARKER) == 1 and llm.count(GRADE_CHUNKS_MARKER) == 1
    assert answer.metadata["agent"]["termination"] == "budget"


def test_corrective_token_cap_is_enforced_too():
    llm = ScriptedLLM(grade=ALL_IRRELEVANT, rewrite='{"query": "alpha beta"}')
    answer = _system("corrective", llm, max_tokens=15).answer_question(QUESTION)  # one grading call uses 15 tokens
    assert llm.count(REWRITE_QUERY_MARKER) == 0 and answer.metadata["agent"]["budget_exhausted"] is True


def test_corrective_retry_widens_the_search_to_hybrid_and_grades_more_chunks():
    llm = ScriptedLLM(grade=[ALL_IRRELEVANT, ALL_RELEVANT], rewrite=json.dumps({"query": "alpha beta"}))
    answer = _system("corrective", llm, expand_to="hybrid", grade_top_k=2).answer_question(QUESTION)
    first, second = _steps(answer, "retrieve", "round_search")
    assert first.metadata["retriever"] == "vector" and second.metadata["retriever"] == "hybrid"
    first_grade, second_grade = _steps(answer, "grade", "grade_chunks")
    assert first_grade.metadata["chunks"] == 2 and second_grade.metadata["chunks"] == 3  # twice as many wanted, only three chunks exist


def test_corrective_full_doc_retry_adds_every_chunk_of_the_best_documents():
    long_doc = Document(doc_id="doc_long", path="l.md", title="l", text=" ".join(f"filler{i}" for i in range(40)) + " Zorblax turbines")
    llm = ScriptedLLM(grade=[ALL_IRRELEVANT, ALL_RELEVANT], rewrite=json.dumps({"query": "turbines and filler words"}))
    system = _system("corrective", llm, docs=[long_doc, *DOCS], chunker={"type": "word", "chunk_size": 10, "chunk_overlap": 0}, expand_to="full_doc", grade_top_k=1)
    answer = system.answer_question("Zorblax turbines filler")
    expansion = _steps(answer, "retrieve", "expand_full_doc")
    assert len(expansion) == 1 and expansion[0].metadata["docs"] >= 1
    graded = _steps(answer, "grade", "grade_chunks")[1]
    assert graded.metadata["chunks"] > 1  # the expanded chunks were graded
    _assert_costs_add_up(answer)


def test_corrective_self_check_regenerates_once_when_the_answer_is_not_supported():
    llm = ScriptedLLM(grade=ALL_RELEVANT, check=json.dumps({"supported": False, "unsupported_claims": ["turbines are loud"]}))
    answer = _system("corrective", llm, self_check=True).answer_question(QUESTION)
    assert llm.count(CHECK_GROUNDED_MARKER) == 1
    assert _steps(answer, "grade", "groundedness")[0].metadata["supported"] is False
    assert len(_steps(answer, "generate", "answer_regenerated")) == 1 and answer.steps[-1].name == "answer_regenerated"
    assert any("turbines are loud" in prompt for prompt in llm.prompts)  # the regeneration was told what was wrong
    _assert_costs_add_up(answer)
    assert answer.cost.total_cost >= 2 * PER_CALL


def test_corrective_self_check_keeps_a_supported_answer():
    llm = ScriptedLLM(grade=ALL_RELEVANT, check=json.dumps({"supported": True, "unsupported_claims": []}))
    answer = _system("corrective", llm, self_check=True).answer_question(QUESTION)
    assert not _steps(answer, name="answer_regenerated")
    assert _steps(answer, "grade", "groundedness")[0].metadata["supported"] is True
    _assert_costs_add_up(answer)


def test_corrective_self_check_is_off_by_default_and_unreadable_verdicts_change_nothing():
    llm = ScriptedLLM(grade=ALL_RELEVANT, check="no idea")
    assert llm.count(CHECK_GROUNDED_MARKER) == 0
    _system("corrective", llm).answer_question(QUESTION)
    assert llm.count(CHECK_GROUNDED_MARKER) == 0
    answer = _system("corrective", llm, self_check=True).answer_question(QUESTION)
    assert _steps(answer, "grade", "groundedness")[0].metadata["fallback"] is True
    assert not _steps(answer, name="answer_regenerated")


# --- iterative ------------------------------------------------------------------------------------------------------------


def _hop(known: str, missing: str, next_query: str) -> str:
    return json.dumps({"known": known, "missing": missing, "next_query": next_query})


def test_iterative_stops_when_the_model_says_done():
    llm = ScriptedLLM(hop=_hop("Zorblax turbines make clean energy", "", "DONE"))
    answer = _system("iterative", llm).answer_question(QUESTION)
    assert llm.count(NEXT_HOP_MARKER) == 1 and len(_steps(answer, "retrieve", "hop_search")) == 1
    agent = answer.metadata["agent"]
    assert (agent["steps"], agent["hops"], agent["termination"], agent["sufficient"]) == (1, 0, "finished", True)
    _assert_costs_add_up(answer)


def test_iterative_follows_the_next_query_and_merges_the_rounds():
    llm = ScriptedLLM(hop=[_hop("Zorblax turbines exist", "how energy is stored", "Quuxwell batteries store energy"), _hop("batteries store it", "", "done.")])
    answer = _system("iterative", llm, top_k=4).answer_question(QUESTION)
    searches = _steps(answer, "retrieve", "hop_search")
    assert [s.input_preview for s in searches] == [QUESTION, "Quuxwell batteries store energy"]
    assert answer.retrieval_result.metadata["queries"] == [QUESTION, "Quuxwell batteries store energy"]
    assert answer.metadata["agent"]["hops"] == 1 and answer.metadata["agent"]["steps"] == 2
    assert "doc_b" in [c.doc_id for c in answer.retrieval_result.chunks]
    # the second round's prompt carries what the first round found
    assert "Zorblax turbines produce clean energy" in llm.calls[NEXT_HOP_MARKER][1]
    # and the final answer is told what the search learned
    final_prompt = llm.prompts[-1]
    assert "Notes from the search" in final_prompt and "Zorblax turbines exist" in final_prompt
    _assert_costs_add_up(answer)


def test_iterative_never_exceeds_max_hops():
    llm = ScriptedLLM(hop=[_hop("a", "b", f"unique query number {i}") for i in range(10)])
    answer = _system("iterative", llm, max_hops=2).answer_question(QUESTION)
    assert len(_steps(answer, "retrieve", "hop_search")) == 2 and llm.count(NEXT_HOP_MARKER) == 2
    agent = answer.metadata["agent"]
    assert agent["termination"] == "max_steps" and agent["sufficient"] is False and agent["hops"] == 2
    assert answer.answer


def test_iterative_stops_instead_of_repeating_a_query():
    llm = ScriptedLLM(hop=_hop("a", "b", QUESTION.lower()))
    answer = _system("iterative", llm, max_hops=5).answer_question(QUESTION)
    assert len(_steps(answer, "retrieve", "hop_search")) == 1
    assert answer.metadata["agent"]["repeated_query"] is True


def test_iterative_survives_an_unreadable_plan():
    llm = ScriptedLLM(hop="¯\\_(ツ)_/¯")
    answer = _system("iterative", llm).answer_question(QUESTION)
    assert answer.metadata["agent"]["fallback"] is True and _steps(answer, "llm", "next_hop")[0].metadata["fallback"] is True
    assert answer.answer and answer.retrieval_result.chunks


def test_iterative_cost_cap_stops_before_the_next_round():
    llm = ScriptedLLM(hop=[_hop("a", "b", "alpha beta gamma"), _hop("a", "b", "delta epsilon")])
    answer = _system("iterative", llm, max_hops=5, max_cost_usd=PER_CALL).answer_question(QUESTION)
    assert len(_steps(answer, "retrieve", "hop_search")) == 1 and llm.count(NEXT_HOP_MARKER) == 1
    agent = answer.metadata["agent"]
    assert agent["termination"] == "budget" and agent["budget_exhausted"] is True
    assert answer.answer
    _assert_costs_add_up(answer)


def test_iterative_a_cap_already_spent_before_the_llm_call_skips_the_call():
    llm = ScriptedLLM(hop=_hop("a", "b", "alpha beta"))
    system = _system("iterative", llm, max_tokens=1)
    # one token is "spent" by the time the first search is done only if a step used it; the search uses none, so the call happens once
    answer = system.answer_question(QUESTION)
    assert llm.count(NEXT_HOP_MARKER) == 1 and answer.metadata["agent"]["termination"] == "budget"


# --- shared ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("type_", ["corrective", "iterative"])
def test_fetch_context_alone_still_runs_the_loop_and_reports_the_agent(type_):
    llm = ScriptedLLM(grade=ALL_RELEVANT, hop=_hop("a", "", "DONE"))
    result = _system(type_, llm).fetch_context(QUESTION, top_k=2)
    assert len(result.chunks) == 2 and result.metadata["agent"]["termination"] == "finished"
    assert result.cost.total_cost == pytest.approx(PER_CALL)  # tracing is off, yet the agent's spend is counted


@pytest.mark.parametrize("type_", ["corrective", "iterative"])
def test_mock_mode_runs_end_to_end_without_a_script(type_):
    answer = _system(type_, retriever="hybrid").answer_question(QUESTION)
    assert answer.answer and answer.retrieval_result.chunks
    assert answer.metadata["agent"]["termination"] in {"finished", "max_steps", "budget", "abort"}
    _assert_costs_add_up(answer)


def test_both_systems_are_registered_as_agentic_and_reject_typos():
    specs = {spec.type: spec for spec in all_specs()}
    for type_ in ("corrective", "iterative"):
        assert specs[type_].agentic is True and specs[type_].requires_llm is True
    with pytest.raises(ValueError, match="max_rounds"):
        create_rag_system(SystemConfig(type="corrective", retrieval={"max_round": 2}), force_mock=True)
    with pytest.raises(ValueError, match="max_hops"):
        create_rag_system(SystemConfig(type="iterative", retrieval={"max_hop": 2}), force_mock=True)
    with pytest.raises(ValueError):
        create_rag_system(SystemConfig(type="iterative", retrieval={"max_cost_usd": 0}), force_mock=True)


# --- the evaluator ----------------------------------------------------------------------------------------------------------


def _dataset(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "doc_001.md").write_text("# Pricing\n\nHarborShield costs $200 per month for the marine module.\n")
    (docs / "doc_002.md").write_text("# Roadmap\n\nClaimPilot ships in Q3 with claims triage workflows.\n")
    write_jsonl(
        root / "questions.jsonl",
        [
            {"id": "q_001", "question": "How much does HarborShield cost?", "reference_answer": "$200 per month.", "relevant_doc_ids": ["doc_001"], "category": "direct_fact"},
            {"id": "q_002", "question": "When does ClaimPilot ship?", "reference_answer": "Q3.", "relevant_doc_ids": ["doc_002"], "category": "direct_fact"},
        ],
    )


def test_the_evaluator_records_the_agent_block_and_summarizes_agentic_systems_only(tmp_path):
    _dataset(tmp_path / "data")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
run: {{name: agents, output_dir: {tmp_path / "results"}}}
dataset: {{documents_path: {tmp_path / "data" / "docs"}, questions_path: {tmp_path / "data" / "questions.jsonl"}}}
systems:
  - {{type: bm25, name: plain, chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3}}}}
  - {{type: corrective, name: crag, chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3}}}}
  - {{type: iterative, name: hops, chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{top_k: 3, max_hops: 2}}}}
evaluation: {{k_values: [1, 3], judge_enabled: false}}
""",
        encoding="utf-8",
    )
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    by_system = {name: [r for r in rows if r["system"] == name] for name in ("plain", "crag", "hops")}
    assert all(r["agent"] is None for r in by_system["plain"])
    for name in ("crag", "hops"):
        for row in by_system[name]:
            agent = row["agent"]
            assert agent["termination"] in {"finished", "max_steps", "budget", "abort"}
            assert agent["steps"] >= 1 and agent["llm_calls"] >= 2 and agent["budget_exhausted"] is False
            assert any(step["name"] == "agent_decision" for step in row["steps"])
    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    assert summary.loc[["crag", "hops"], "avg_steps"].notna().all() and summary.loc[["crag", "hops"], "avg_llm_calls"].notna().all()
    assert (summary.loc[["crag", "hops"], "budget_exhausted_rate"] == 0).all()
    assert summary.loc["plain", ["avg_steps", "avg_llm_calls", "budget_exhausted_rate"]].isna().all()
