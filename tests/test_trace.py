from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fake_systems import question, register_fakes, write_experiment

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models.cost import CostBreakdown
from ragbench.models.embeddings import EmbeddingModel, EmbeddingResult, HashingEmbeddingModel
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.rag_systems import SYSTEMS, create_rag_system
from ragbench.rag_systems.trace import (
    STAGE_KEYS,
    Step,
    Tracer,
    activate,
    current_tracer,
    reconcile_steps,
    stage_costs,
    step_to_dict,
    sum_costs,
)
from ragbench.stores.vector_store import VectorStore

# --- Tracer basics ---------------------------------------------------------------------------


def test_tracer_records_ordered_steps_with_latency_cost_and_previews():
    tracer = Tracer()

    with tracer.step("retrieve", "vector_search", top_k=5) as step:
        time.sleep(0.002)
        step.set_input("what is x?")
        step.set_cost(CostBreakdown(embedding_cost=0.5, embedding_input_tokens=7))
        step.set_output("x" * 1000)
    with tracer.step("generate", "answer") as step:
        step.set_tokens(10, 4)

    assert [(s.kind, s.name) for s in tracer.steps] == [("retrieve", "vector_search"), ("generate", "answer")]
    first, second = tracer.steps
    assert first.latency_ms > 0 and first.cost.embedding_cost == 0.5 and first.metadata == {"top_k": 5}
    assert first.input_preview == "what is x?" and len(first.output_preview or "") <= 300
    assert (second.prompt_tokens, second.completion_tokens) == (10, 4)


def test_a_failing_step_is_still_recorded_and_the_error_propagates():
    tracer = Tracer()

    with pytest.raises(RuntimeError, match="boom"):
        with tracer.step("tool", "calculator"):
            raise RuntimeError("boom")

    assert tracer.steps[0].metadata["error"] == "RuntimeError: boom"


def test_no_active_tracer_means_a_silent_null_tracer():
    assert current_tracer().steps == []
    with current_tracer().step("retrieve", "x") as step:
        step.set_cost(CostBreakdown(llm_cost=1.0))
    assert current_tracer().steps == []  # nothing leaks into a shared global


def test_activate_nests_and_restores():
    outer, inner = Tracer(), Tracer()
    with activate(outer):
        with activate(inner):
            assert current_tracer() is inner
        assert current_tracer() is outer
    assert current_tracer() is not outer


def test_set_llm_copies_cost_tokens_and_output():
    result = LLMResult(text="hello world", model="m", prompt_tokens=3, completion_tokens=2, cost=CostBreakdown(llm_cost=0.25))
    tracer = Tracer()
    with tracer.step("llm", "rewrite") as step:
        step.set_llm(result, messages=[{"role": "user", "content": "rewrite this"}])

    recorded = tracer.steps[0]
    assert recorded.cost.llm_cost == 0.25 and (recorded.prompt_tokens, recorded.completion_tokens) == (3, 2)
    assert recorded.output_preview == "hello world" and recorded.input_preview == "rewrite this"


def test_step_serialization_includes_total_cost():
    step = Step(kind="rerank", name="r", cost=CostBreakdown(rerank_cost=0.1, llm_cost=0.2))
    assert step_to_dict(step)["cost"]["total_cost"] == pytest.approx(0.3)
    json.dumps(step_to_dict(step))  # JSON-safe


# --- cost reconciliation ---------------------------------------------------------------------


def test_reconcile_adds_a_visible_residual_step_when_steps_miss_cost():
    steps = [Step(kind="retrieve", name="a", cost=CostBreakdown(embedding_cost=1.0))]
    total = CostBreakdown(embedding_cost=1.0, llm_cost=0.5, llm_prompt_tokens=9)

    reconciled = reconcile_steps(steps, total)

    assert [s.name for s in reconciled] == ["a", "untracked"]
    assert sum_costs(s.cost for s in reconciled).total_cost == pytest.approx(total.total_cost)
    assert reconciled[1].cost.llm_prompt_tokens == 9


def test_reconcile_is_a_no_op_when_steps_add_up_and_flags_overcounting():
    steps = [Step(kind="llm", name="a", cost=CostBreakdown(llm_cost=0.5)), Step(kind="generate", name="b", cost=CostBreakdown(llm_cost=0.5))]
    assert reconcile_steps(steps, CostBreakdown(llm_cost=1.0)) == steps
    over = reconcile_steps(steps, CostBreakdown(llm_cost=0.75))
    assert over[-1].name == "untracked" and over[-1].cost.llm_cost == pytest.approx(-0.25)  # double counting stays visible


def test_stage_costs_roll_up_by_kind_and_isolate_untracked():
    steps = [
        Step(kind="retrieve", name="a", cost=CostBreakdown(embedding_cost=1.0)),
        Step(kind="retrieve", name="b", cost=CostBreakdown(embedding_cost=2.0)),
        Step(kind="generate", name="c", cost=CostBreakdown(llm_cost=4.0)),
        Step(kind="retrieve", name="untracked", cost=CostBreakdown(llm_cost=8.0)),
    ]
    rolled = stage_costs(steps)
    assert set(rolled) == set(STAGE_KEYS)
    assert rolled["retrieve"] == 3.0 and rolled["generate"] == 4.0 and rolled["untracked"] == 8.0 and rolled["tool"] == 0.0


# --- real systems ----------------------------------------------------------------------------

PRICE = 1e-6


class PricedEmbedding(EmbeddingModel):
    """Deterministic hashing embeddings that cost money, so cost accounting can be tested in mock mode."""

    model_name = "priced-hashing"

    def __init__(self) -> None:
        self.inner = HashingEmbeddingModel()

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        result = self.inner.embed_texts(texts)
        tokens = sum(len(t.split()) for t in texts)
        return EmbeddingResult(result.vectors, self.model_name, tokens, CostBreakdown(embedding_input_tokens=tokens, embedding_cost=tokens * PRICE))


class PricedLLM(MockLLM):
    def generate(self, messages, **kwargs) -> LLMResult:
        result = super().generate(messages, **kwargs)
        result.cost = CostBreakdown(
            llm_prompt_tokens=result.prompt_tokens,
            llm_completion_tokens=result.completion_tokens,
            llm_cost=(result.prompt_tokens + result.completion_tokens) * PRICE,
        )
        return result


def _make_paid(system) -> None:
    system.llm = PricedLLM()
    embedder = PricedEmbedding()
    for value in vars(system).values():
        if isinstance(value, VectorStore):
            value.embedding_model = embedder
    if hasattr(system, "embedding_model"):
        system.embedding_model = embedder
    reranker = getattr(system, "reranker", None)
    if reranker is not None and hasattr(reranker, "llm"):
        reranker.llm = system.llm


def _docs() -> list[Document]:
    return [
        Document(doc_id="doc_001", path="a.md", title="A", text="HarborShield AI reviews marine cargo submissions for underwriters."),
        Document(doc_id="doc_002", path="b.md", title="B", text="ClaimPilot summarizes loss notices and classifies claim severity."),
        Document(doc_id="doc_003", path="c.md", title="C", text="Aurora Risk Suite compares quote conversion across regions."),
    ]


SYSTEM_CONFIGS = {
    "bm25": {},
    "vector": {"retrieval": {"vector_store": "in_memory"}},
    "hybrid": {"retrieval": {"vector_store": "in_memory", "multi_query": True}},
    "hybrid_rerank": {"retrieval": {"vector_store": "in_memory"}},
    "rerank": {"retrieval": {"vector_store": "in_memory"}},
    "parent_doc": {"retrieval": {"vector_store": "in_memory"}},
    "hyde": {"retrieval": {"vector_store": "in_memory"}},
    "llm_heavy": {"retrieval": {"vector_store": "in_memory"}, "llm_features": {"enable_llm_rerank": True}},
}


def _system(system_type: str):
    cfg = {"chunker": {"type": "token", "chunk_size": 50, "chunk_overlap": 0}, **SYSTEM_CONFIGS[system_type]}
    if system_type == "parent_doc":
        cfg["chunker"] = {"parent_chunk_size": 50, "parent_chunk_overlap": 0, "child_chunk_size": 10, "child_chunk_overlap": 0}
    system = create_rag_system(SystemConfig(type=system_type, name=f"{system_type}_trace", **cfg), force_mock=True)
    _make_paid(system)
    system.ingest(_docs())
    return system


def test_every_builtin_system_is_covered_by_the_trace_tests():
    assert set(SYSTEM_CONFIGS) == set(SYSTEMS.names())


@pytest.mark.parametrize("system_type", sorted(SYSTEM_CONFIGS))
def test_steps_account_for_the_whole_answer_cost_with_no_untracked_residual(system_type):
    system = _system(system_type)

    result = system.answer_question("Which product reviews marine cargo submissions?")

    assert result.steps, "system recorded no steps"
    assert "untracked" not in [s.name for s in result.steps], f"{system_type} has cost that no step accounts for"
    total = sum_costs(s.cost for s in result.steps)
    assert total.total_cost == pytest.approx(result.cost.total_cost, rel=1e-9, abs=1e-15)
    assert total.total_cost > 0
    assert result.steps[-1].kind == "generate" and result.steps[-1].cost.llm_cost > 0
    assert any(s.kind == "retrieve" for s in result.steps)
    assert result.retrieval_result.steps == [s for s in result.steps if s.kind != "generate"]
    assert all(s.latency_ms >= 0 for s in result.steps)


def test_step_shapes_of_systems_with_extra_stages():
    hyde = _system("hyde").answer_question("Which product reviews marine cargo submissions?")
    assert [(s.kind, s.name) for s in hyde.steps] == [
        ("llm", "hypothetical_document"),
        ("retrieve", "vector_search:hypothesis"),
        ("retrieve", "vector_search:question"),
        ("generate", "answer"),
    ]
    assert hyde.steps[0].cost.query_rewrite_cost > 0  # rewrite-class cost stays classified as before

    hybrid = _system("hybrid_rerank").answer_question("Which product reviews marine cargo submissions?")
    assert [s.kind for s in hybrid.steps] == ["retrieve", "retrieve", "rerank", "generate"]

    heavy = _system("llm_heavy").answer_question("Which product reviews marine cargo submissions?")
    kinds = [s.kind for s in heavy.steps]
    assert kinds[0] == "llm" and "rerank" in kinds and kinds[-1] == "generate"
    assert next(s for s in heavy.steps if s.kind == "rerank").cost.total_cost > 0


def test_uninstrumented_custom_systems_get_a_coarse_trace_without_losing_cost(monkeypatch):
    register_fakes(monkeypatch)
    from fake_systems import RankedFakeSystem

    system = RankedFakeSystem(SystemConfig(type="fake_ranked", name="fake", retrieval={"top_k": 3}), force_mock=True)
    system.llm = PricedLLM()

    result = system.answer_question("What is topic 1?")

    assert [(s.kind, s.name) for s in result.steps] == [("retrieve", "fetch_context"), ("generate", "answer")]
    assert sum_costs(s.cost for s in result.steps).total_cost == pytest.approx(result.cost.total_cost)


def test_steps_are_isolated_between_concurrent_questions():
    system = _system("hybrid_rerank")
    questions = [f"Which product handles topic number {i} for underwriters?" for i in range(24)]

    def ask(text: str):
        return text, system.answer_question(text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(ask, questions))

    for text, result in results:
        assert [s.kind for s in result.steps] == ["retrieve", "retrieve", "rerank", "generate"], "another question's steps leaked in"
        first = next(s for s in result.steps if s.kind == "retrieve")
        assert first.input_preview == text
        assert sum_costs(s.cost for s in result.steps).total_cost == pytest.approx(result.cost.total_cost)


# --- evaluator persistence -------------------------------------------------------------------


def test_run_persists_steps_and_stage_cost_columns(tmp_path, monkeypatch):
    import pandas as pd

    register_fakes(monkeypatch)
    config = write_experiment(
        tmp_path,
        [question("q1", "What is topic 2?", ["doc_002"]), question("q2", "What is topic 3?", ["doc_003"])],
        [{"type": "fake_ranked", "name": "fake", "retrieval": {"top_k": 3}}, {"type": "bm25", "name": "bm25"}],
    )

    out = run_benchmark(config, force_mock=True)

    rows = [json.loads(line) for line in (out / "per_question_results.jsonl").read_text().splitlines()]
    for row in rows:
        assert [s["kind"] for s in row["steps"]][-1] == "generate"
        assert all({"kind", "name", "latency_ms", "cost", "prompt_tokens", "completion_tokens", "metadata"} <= set(s) for s in row["steps"])
        assert "total_cost" in row["steps"][0]["cost"]
    cost = pd.read_csv(out / "cost_breakdown.csv")
    questions = cost[cost["stage"] == "question"]
    stage_cols = [f"stage_{key}_cost" for key in STAGE_KEYS]
    assert set(stage_cols) <= set(cost.columns)
    # Answer-side stage costs plus the judge cost reproduce each question's total cost.
    assert (questions[stage_cols].sum(axis=1) + questions["judge_cost"] - questions["total_cost"]).abs().max() < 1e-12
    leaderboard = (out / "leaderboard.md").read_text()
    assert "## Latency by stage" in leaderboard and "retrieve" in leaderboard


def test_leaderboard_renders_cost_by_stage_only_for_stages_that_cost_something(tmp_path):
    from ragbench.reporting.markdown_report import write_leaderboard

    def zeros():
        return dict.fromkeys(STAGE_KEYS, 0.0)

    cost_a = {**zeros(), "retrieve": 0.000002, "generate": 0.0004}
    cost_b = {**zeros(), "retrieve": 0.000002, "llm": 0.0003, "rerank": 0.0001, "generate": 0.0004}
    rows = [
        {"system": "a", "cost": cost_a, "latency_ms": {**zeros(), "retrieve": 2.0, "generate": 5.0}},
        {"system": "b", "cost": cost_b, "latency_ms": {**zeros(), "retrieve": 3.0, "llm": 40.0, "generate": 5.0}},
    ]
    path = tmp_path / "lb.md"

    write_leaderboard(path, [{"system": "a", "system_type": "bm25"}, {"system": "b", "system_type": "bm25"}], stage_rows=rows)

    text = path.read_text()
    cost_table = text.split("## Cost by stage")[1].split("## Latency by stage")[0]
    header = next(line for line in cost_table.splitlines() if line.startswith("| System"))
    assert header == "| System | retrieve | rerank | llm | generate |"
    assert "$0.000400" in cost_table and "tool" not in header
    assert "## Latency by stage" in text


def test_leaderboard_has_no_stage_sections_without_data(tmp_path):
    from ragbench.reporting.markdown_report import write_leaderboard

    path = tmp_path / "lb.md"
    write_leaderboard(path, [{"system": "a", "system_type": "bm25"}])
    assert "by stage" not in path.read_text()
