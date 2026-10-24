"""`decision_gate`: the gating and abstain logic against a scripted decision model, the decision backends, their cache, and how the evaluator records a gate.

Nothing here touches a network: the decision model is scripted, the OpenAI backend talks to `httpx.MockTransport`, and runs use the mock models.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence

import httpx
import openai
import pandas as pd
import pytest

from ragbench.cache import CacheRuntime, DiskCache, activate_cache
from ragbench.config.schema import CacheConfig, SystemConfig
from ragbench.documents.schema import Document
from ragbench.evaluation.answer_metrics import is_refusal
from ragbench.evaluation.evaluator import run_benchmark
from ragbench.models import retry as retry_module
from ragbench.models.cost import CostBreakdown, estimate_decision_cost
from ragbench.models.decisions import (
    CachedDecisionModel,
    DecisionModel,
    DecisionResult,
    LLMJsonDecisionModel,
    MockDecisionModel,
    OpenAIDecisionModel,
    Predicate,
    create_decision_model,
    formats,
)
from ragbench.models.decisions.llm_json import parse_probabilities
from ragbench.models.errors import PermanentModelError
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.rag_systems import all_specs, create_rag_system
from ragbench.rag_systems.decision_gate_rag import ABSTAIN_ANSWER, INSUFFICIENT_EVIDENCE, NO_RELEVANT_CHUNKS
from ragbench.registry import UnknownComponentError
from ragbench.utils.jsonl import read_jsonl

PER_REQUEST = 0.01
QUESTION = "How do Zorblax turbines make energy?"

DOCS = [
    Document(doc_id="doc_a", path="a.md", title="a", text="Zorblax turbines produce clean energy for coastal towns. Zorblax turbines are quiet."),
    Document(doc_id="doc_b", path="b.md", title="b", text="Quuxwell batteries store energy overnight for mountain villages. Quuxwell cells last long."),
    Document(doc_id="doc_c", path="c.md", title="c", text="Plinth harbours host fishing fleets in the northern bay. Plinth docks open at dawn."),
]


def _words(text: str) -> str:
    return text.lower()


def by_topic(text: str) -> float:
    """The scripted answerability: Zorblax text answers, Quuxwell text half-does, the rest does not."""
    text = _words(text)
    return 0.9 if "zorblax" in text else 0.6 if "quuxwell" in text else 0.1


class ScriptedDecider(DecisionModel):
    """Answers each predicate from a script and charges `PER_REQUEST` per request.

    `passage(text)` scripts the answerability of the passage a predicate is about (one passage, or `passage_<i>` of a batch);
    `sufficiency` is the answer to the set-level question: a number, None (the model gave no answer), or a function of the evidence text.
    """

    provider = "scripted"
    model_name = "scripted-decisions"

    def __init__(self, passage: Callable[[str], float | None] = by_topic, sufficiency: float | None | Callable[[str], float | None] = 1.0):
        self.passage, self.sufficiency = passage, sufficiency
        self.requests: list[tuple[str, list[Predicate]]] = []
        self._lock = threading.Lock()

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        with self._lock:
            self.requests.append((text, list(predicates)))
        about = formats.parse_passages(text)
        probabilities: dict[str, float | None] = {}
        for predicate in predicates:
            if predicate.name == formats.SUFFICIENT:
                probabilities[predicate.name] = self.sufficiency(about[predicate.name]) if callable(self.sufficiency) else self.sufficiency
            else:
                probabilities[predicate.name] = self.passage(about[predicate.name])
        cost = CostBreakdown(decision_input_tokens=100, decision_cost=PER_REQUEST)
        return DecisionResult(probabilities=probabilities, model=self.model_name, input_tokens=100, cost=cost)

    def requests_named(self, prefix: str) -> list[tuple[str, list[Predicate]]]:
        return [r for r in self.requests if r[1][0].name.startswith(prefix)]


class SpyLLM(MockLLM):
    """The generator: records every prompt it is asked to answer."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def generate(self, messages, **kwargs) -> LLMResult:
        self.prompts.append("\n".join(str(m.get("content") or "") for m in messages))
        return super().generate(messages, **kwargs)


def make_gate(decider: DecisionModel | None = None, **retrieval):
    options = {"final_top_k": 3, "reranker": "simple_keyword_overlap", "vector_store": "numpy", **retrieval}
    system = create_rag_system(
        SystemConfig(type="decision_gate", retrieval=options, chunker={"type": "token", "chunk_size": 60, "chunk_overlap": 0}), force_mock=True
    )
    system.decider = decider or ScriptedDecider()
    system.llm = SpyLLM()
    system.ingest(DOCS)
    return system


def ask(system, context_k: int = 3):
    return system.answer_question(QUESTION, top_k=3, context_k=context_k)


# --- the gate: dropping chunks ---------------------------------------------------------------------------------------


def test_the_system_is_registered_with_a_spec():
    spec = next(s for s in all_specs() if s.type == "decision_gate")
    assert spec.requires_llm and not spec.agentic and spec.options.__name__ == "DecisionGateOptions"


def test_chunks_below_the_threshold_never_reach_the_generator():
    system = make_gate(answerability_threshold=0.5)
    result = ask(system)
    gate = result.metadata["gate"]
    assert gate["judged"] == 3 and gate["kept"] == 2 and not gate["abstained"]
    assert result.metadata["context_chunk_ids"] == gate["kept_chunk_ids"]
    prompt = system.llm.prompts[-1]
    assert "Zorblax" in prompt and "Quuxwell" in prompt and "Plinth" not in prompt


def test_a_higher_threshold_keeps_fewer_chunks_and_keeps_them_in_rank_order():
    system = make_gate(answerability_threshold=0.8)
    gate = ask(system).metadata["gate"]
    assert gate["kept"] == 1 and gate["kept_chunk_ids"][0].startswith("doc_a")
    ranked = [chunk.chunk_id for chunk in ask(make_gate(answerability_threshold=0.5)).retrieval_result.chunks]
    kept = ask(make_gate(answerability_threshold=0.5)).metadata["gate"]["kept_chunk_ids"]
    assert kept == [chunk_id for chunk_id in ranked if chunk_id in kept]


def test_the_ranking_is_the_ungated_one_and_only_the_context_changes():
    gated = ask(make_gate(answerability_threshold=0.8))
    plain = create_rag_system(
        SystemConfig(
            type="hybrid_rerank",
            retrieval={"final_top_k": 3, "reranker": "simple_keyword_overlap", "vector_store": "numpy"},
            chunker={"type": "token", "chunk_size": 60, "chunk_overlap": 0},
        ),
        force_mock=True,
    )
    plain.ingest(DOCS)
    assert [c.chunk_id for c in gated.retrieval_result.chunks] == [c.chunk_id for c in plain.fetch_context(QUESTION, top_k=3).chunks]
    assert len(gated.metadata["context_chunk_ids"]) == 1 < len(gated.retrieval_result.chunks)


def test_survivors_are_capped_at_the_generator_context_size_and_a_deeper_chunk_can_replace_a_dropped_one():
    system = make_gate(ScriptedDecider(passage=lambda text: 0.9), answerability_threshold=0.5)
    assert ask(system, context_k=2).metadata["gate"]["kept"] == 2
    # the first-ranked chunk is dropped, so the third-ranked one fills the second slot of a two-chunk context
    dropped_first = ScriptedDecider(passage=lambda text: 0.1 if "zorblax" in text.lower() else 0.9)
    gate = ask(make_gate(dropped_first), context_k=2).metadata["gate"]
    assert gate["kept"] == 2 and not any(chunk_id.startswith("doc_a") for chunk_id in gate["kept_chunk_ids"])


def test_max_candidates_to_judge_limits_what_is_asked_about():
    decider = ScriptedDecider()
    gate = ask(make_gate(decider, max_candidates_to_judge=2)).metadata["gate"]
    assert gate["judged"] == 2 and len(gate["answerability"]) == 2
    assert len(decider.requests_named("passage_")[0][1]) == 2


# --- the gate: abstaining --------------------------------------------------------------------------------------------


def test_no_surviving_chunk_abstains_without_a_generator_call_or_a_sufficiency_request():
    decider = ScriptedDecider(passage=lambda text: 0.1)
    system = make_gate(decider)
    result = ask(system)
    gate = result.metadata["gate"]
    assert result.answer == ABSTAIN_ANSWER and is_refusal(result.answer)
    assert gate["abstained"] and gate["abstain_reason"] == NO_RELEVANT_CHUNKS and gate["kept"] == 0 and gate["sufficiency"] is None
    assert system.llm.prompts == [] and result.metadata["context_chunk_ids"] == []
    assert decider.requests_named(formats.SUFFICIENT) == []
    assert result.cost.llm_cost == 0 and result.cost.decision_cost == pytest.approx(PER_REQUEST)


def test_insufficient_evidence_abstains_and_the_generator_never_runs():
    decider = ScriptedDecider(sufficiency=0.3)
    system = make_gate(decider, sufficiency_threshold=0.5)
    result = ask(system)
    gate = result.metadata["gate"]
    assert result.answer == ABSTAIN_ANSWER and is_refusal(result.answer)
    assert gate["abstain_reason"] == INSUFFICIENT_EVIDENCE and gate["sufficiency"] == 0.3 and gate["kept"] == 2
    assert system.llm.prompts == []
    assert result.cost.decision_cost == pytest.approx(2 * PER_REQUEST)


def test_sufficiency_at_the_threshold_still_answers():
    system = make_gate(ScriptedDecider(sufficiency=0.5), sufficiency_threshold=0.5)
    result = ask(system)
    assert not result.metadata["gate"]["abstained"] and result.answer != ABSTAIN_ANSWER and len(system.llm.prompts) == 1


def test_the_sufficiency_question_is_asked_about_the_kept_chunks_only():
    decider = ScriptedDecider()
    ask(make_gate(decider))
    evidence = decider.requests_named(formats.SUFFICIENT)[0][0]
    assert "Zorblax" in evidence and "Quuxwell" in evidence and "Plinth" not in evidence


def test_a_zero_sufficiency_threshold_turns_the_check_off_and_saves_its_request():
    decider = ScriptedDecider(sufficiency=0.0)
    result = ask(make_gate(decider, sufficiency_threshold=0))
    assert not result.metadata["gate"]["abstained"] and decider.requests_named(formats.SUFFICIENT) == []
    assert result.metadata["gate"]["decision_calls"] == 1


def test_a_zero_answerability_threshold_keeps_every_chunk():
    result = ask(make_gate(ScriptedDecider(passage=lambda text: 0.0), answerability_threshold=0, sufficiency_threshold=0))
    assert result.metadata["gate"]["kept"] == 3


# --- the gate: a model that gives no answer --------------------------------------------------------------------------


def test_an_answerability_refusal_keeps_the_chunk_and_is_counted():
    result = ask(make_gate(ScriptedDecider(passage=lambda text: None if "plinth" in text.lower() else 0.1), sufficiency_threshold=0))
    gate = result.metadata["gate"]
    assert gate["kept"] == 1 and gate["unreadable"] == 1 and not gate["abstained"]
    assert any(chunk_id.startswith("doc_c") for chunk_id in gate["kept_chunk_ids"])


def test_a_missing_sufficiency_answer_cannot_abstain():
    result = ask(make_gate(ScriptedDecider(sufficiency=None)))
    gate = result.metadata["gate"]
    assert not gate["abstained"] and gate["sufficiency"] is None and gate["unreadable"] == 1 and result.answer != ABSTAIN_ANSWER


# --- request shape and cost accounting -------------------------------------------------------------------------------


def test_batched_mode_judges_all_candidates_in_one_request():
    decider = ScriptedDecider()
    result = ask(make_gate(decider, judge_mode="batched"))
    batch = decider.requests_named("passage_")
    assert len(batch) == 1 and [p.name for p in batch[0][1]] == ["passage_1", "passage_2", "passage_3"]
    assert "passage [2]" in batch[0][1][1].instructions
    assert result.metadata["gate"]["decision_calls"] == 2  # the batch and the sufficiency question


def test_per_chunk_mode_sends_one_request_per_candidate_with_one_passage_each():
    decider = ScriptedDecider()
    result = ask(make_gate(decider, judge_mode="per_chunk"))
    single = decider.requests_named(formats.SINGLE)
    assert len(single) == 3 and all(len(r[1]) == 1 and "the passage" in r[1][0].instructions for r in single)
    assert result.metadata["gate"]["decision_calls"] == 4
    assert {r[0].split("Passage:\n")[1] for r in single} == {chunk.text for chunk in result.retrieval_result.chunks}


def test_both_modes_reach_the_same_decision_for_the_same_probabilities():
    batched = ask(make_gate(judge_mode="batched")).metadata["gate"]
    per_chunk = ask(make_gate(judge_mode="per_chunk")).metadata["gate"]
    assert batched["kept_chunk_ids"] == per_chunk["kept_chunk_ids"] and batched["answerability"] == per_chunk["answerability"]


def test_decision_spend_is_booked_as_decision_cost_and_the_steps_add_up():
    result = ask(make_gate(judge_mode="per_chunk"))
    assert result.cost.decision_cost == pytest.approx(4 * PER_REQUEST) and result.cost.decision_input_tokens == 400
    assert result.cost.total_cost == pytest.approx(sum(step.cost.total_cost for step in result.steps))
    assert [step.name for step in result.steps if step.kind == "grade"] == ["decide_answerability", "decide_sufficiency"]
    assert not any(step.name == "untracked" for step in result.steps)


def test_the_instruction_templates_can_be_overridden():
    decider = ScriptedDecider()
    ask(make_gate(decider, answerability_instructions="Is {passage} about turbines?", sufficiency_instructions="Enough?"))
    assert decider.requests_named("passage_")[0][1][0].instructions == "Is passage [1] about turbines?"
    assert decider.requests_named(formats.SUFFICIENT)[0][1][0].instructions == "Enough?"


def test_batched_instructions_must_say_which_passage_they_are_about():
    with pytest.raises(ValueError, match=r"\{passage\}"):
        create_rag_system(SystemConfig(type="decision_gate", retrieval={"answerability_instructions": "Is it relevant?"}), force_mock=True)
    create_rag_system(SystemConfig(type="decision_gate", retrieval={"answerability_instructions": "Is it relevant?", "judge_mode": "per_chunk"}), force_mock=True)


def test_options_reject_unknown_providers_with_a_hint_and_unknown_keys():
    with pytest.raises(ValueError, match="openai_decisions"):
        create_rag_system(SystemConfig(type="decision_gate", retrieval={"decision_model": {"provider": "openai_decision"}}), force_mock=True)
    with pytest.raises(ValueError):
        create_rag_system(SystemConfig(type="decision_gate", retrieval={"decision_threshold": 0.5}), force_mock=True)


# --- the backends -----------------------------------------------------------------------------------------------------


def test_the_mock_backend_is_deterministic_monotone_and_zero_for_unrelated_text():
    mock = MockDecisionModel()
    q = "When did HarborShield become generally available?"
    predicates = [Predicate(formats.SINGLE, "x")]
    full = mock.decide(formats.single_input(q, "When did HarborShield become generally available? In 2023."), predicates).probabilities["passage"]
    part = mock.decide(formats.single_input(q, "HarborShield is a product."), predicates).probabilities["passage"]
    none = mock.decide(formats.single_input(q, "The Lisbon office opened."), predicates).probabilities["passage"]
    assert full == 1.0 and 0 < part < full and none == 0.0
    assert mock.decide(formats.single_input(q, "HarborShield is a product."), predicates).probabilities["passage"] == part


def test_the_mock_backend_reads_all_three_request_formats():
    mock = MockDecisionModel()
    q = "Who leads the Zurich security team?"
    batch = mock.decide(formats.batch_input(q, ["The Zurich security team is led by Maxine.", "Unrelated note."]), [Predicate("passage_1", ""), Predicate("passage_2", "")])
    assert batch.probabilities["passage_1"] > 0 == batch.probabilities["passage_2"]
    evidence = mock.decide(formats.evidence_input(q, ["The Zurich security team is led by Maxine."]), [Predicate(formats.SUFFICIENT, "")])
    assert evidence.probabilities[formats.SUFFICIENT] > 0 and evidence.cost.decision_cost == 0 and evidence.cost.decision_input_tokens > 0


class Server:
    """A scripted Decisions endpoint behind `httpx.MockTransport`: records requests, answers from a list of responses or callables."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, json.loads(request.content or b"{}")))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return step(request) if callable(step) else step

    def model(self) -> OpenAIDecisionModel:
        client = openai.OpenAI(api_key="sk-test-not-real", base_url="https://api.example.test/v1", http_client=httpx.Client(transport=httpx.MockTransport(self)), max_retries=0)
        return OpenAIDecisionModel("gpt-6-luna", client=client)


def decision_reply(answers: list[dict], input_tokens: int = 1000) -> httpx.Response:
    return httpx.Response(200, json={"model": "gpt-6-luna", "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0, "total_tokens": input_tokens}})


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(retry_module.time, "sleep", lambda s: None)


def test_the_openai_backend_posts_typed_predicate_questions_and_reads_answers_by_name():
    server = Server(
        decision_reply(
            [{"type": "predicate", "name": "b", "probability": 0.25}, {"type": "predicate", "name": "a", "probability": 1.0}, {"type": "refusal", "name": "c"}]
        )
    )
    result = server.model().decide("some text", [Predicate("a", "is a?"), Predicate("b", "is b?"), Predicate("c", "is c?")])
    path, body = server.requests[0]
    assert path.endswith("/decisions")
    assert body == {
        "model": "gpt-6-luna",
        "input": "some text",
        "questions": [{"type": "predicate", "name": n, "instructions": f"is {n}?"} for n in "abc"],
    }
    assert result.probabilities == {"a": 1.0, "b": 0.25, "c": None}  # a refusal is "unknown", never 0


def test_the_openai_backend_bills_input_tokens_only_at_the_models_input_rate():
    result = Server(decision_reply([{"type": "predicate", "name": "a", "probability": 0.5}], input_tokens=2_000_000)).model().decide("t", [Predicate("a", "?")])
    assert result.input_tokens == 2_000_000
    assert result.cost.decision_cost == pytest.approx(0.20) == pytest.approx(estimate_decision_cost("gpt-6-luna", 2_000_000))
    assert result.cost.llm_cost == 0 and result.cost.decision_input_tokens == 2_000_000
    assert result.cost.total_cost == pytest.approx(0.20)


def test_the_openai_backend_retries_rate_limits_but_not_bad_requests():
    limited = httpx.Response(429, json={"error": {"message": "slow down", "type": "x", "code": None}})
    ok = decision_reply([{"type": "predicate", "name": "a", "probability": 0.7}])
    server = Server(limited, ok)
    assert server.model().decide("t", [Predicate("a", "?")]).probabilities == {"a": 0.7} and len(server.requests) == 2
    bad = Server(httpx.Response(400, json={"error": {"message": "invalid", "type": "x", "code": None}}))
    with pytest.raises(PermanentModelError):
        bad.model().decide("t", [Predicate("a", "?")])
    assert len(bad.requests) == 1


def test_the_openai_backend_rejects_unexpected_replies_and_duplicate_names():
    with pytest.raises(PermanentModelError, match="unexpected reply"):
        Server(httpx.Response(200, json={"oops": 1})).model().decide("t", [Predicate("a", "?")])
    with pytest.raises(ValueError, match="unique"):
        Server(decision_reply([])).model().decide("t", [Predicate("a", "?"), Predicate("a", "?")])


def test_probabilities_are_clamped_and_a_non_number_is_unknown():
    reply = decision_reply([{"type": "predicate", "name": "a", "probability": 1.4}, {"type": "predicate", "name": "b", "probability": "high"}])
    assert Server(reply).model().decide("t", [Predicate("a", ""), Predicate("b", "")]).probabilities == {"a": 1.0, "b": None}


class JsonLLM(MockLLM):
    model_name = "json-model"

    def __init__(self, reply: str) -> None:
        super().__init__()
        self.reply, self.calls = reply, []

    def generate(self, messages, **kwargs) -> LLMResult:
        self.calls.append((messages, kwargs))
        return LLMResult(text=self.reply, model=self.model_name, prompt_tokens=300, completion_tokens=20, cost=CostBreakdown(llm_cost=0.05), finish_reason="stop")


def test_the_llm_json_backend_asks_for_json_and_books_everything_as_decision_cost():
    llm = JsonLLM('{"probabilities": {"a": 0.9, "b": 0.1}}')
    result = LLMJsonDecisionModel(llm).decide("text", [Predicate("a", "is a?"), Predicate("b", "is b?")])
    messages, kwargs = llm.calls[0]
    assert kwargs["json_mode"] is True and "- a: is a?" in messages[-1]["content"] and "- b: is b?" in messages[-1]["content"]
    assert result.probabilities == {"a": 0.9, "b": 0.1}
    assert result.cost.decision_cost == 0.05 and result.cost.llm_cost == 0 and result.cost.decision_input_tokens == 300


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"probabilities": {"a": 0.4}}', {"a": 0.4, "b": None}),
        ('{"a": 0.4, "b": 2}', {"a": 0.4, "b": 1.0}),
        ('Sure! {"probabilities": {"a": 0.4, "b": "n/a"}} Hope that helps.', {"a": 0.4, "b": None}),
        ("not json at all", {"a": None, "b": None}),
        ('["a", "b"]', {"a": None, "b": None}),
    ],
)
def test_the_llm_json_reply_is_read_leniently_and_what_is_missing_is_unknown(reply, expected):
    assert parse_probabilities(reply, ["a", "b"]) == expected


def test_the_factory_builds_by_name_and_falls_back_to_the_mock_like_chat_models_do(monkeypatch):
    assert isinstance(create_decision_model("mock"), MockDecisionModel)
    assert isinstance(create_decision_model("openai_decisions", force_mock=True), MockDecisionModel)
    assert isinstance(create_decision_model("openai_decisions"), MockDecisionModel)  # no key in the test environment
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    live = create_decision_model("openai_decisions")
    assert isinstance(live, CachedDecisionModel) and live.model_name == "gpt-6-luna" and live.provider == "openai_decisions"
    with pytest.raises(UnknownComponentError, match="llm_json"):
        create_decision_model("llm_jsn")


# --- the cache --------------------------------------------------------------------------------------------------------


class CountingDecider(ScriptedDecider):
    provider = "counting"
    model_name = "gpt-6-luna"


@pytest.fixture()
def runtime(tmp_path):
    rt = CacheRuntime(disk=DiskCache(tmp_path / "cache.sqlite3"), config=CacheConfig())
    with activate_cache(rt):
        yield rt


def test_a_repeated_request_is_served_from_the_cache_at_the_current_price(runtime):
    inner = CountingDecider(passage=lambda text: 0.8)
    cached = CachedDecisionModel(inner)
    predicates = [Predicate(formats.SINGLE, "is it?")]
    text = formats.single_input("q?", "a passage")
    first = cached.decide(text, predicates)
    second = cached.decide(text, predicates)
    assert len(inner.requests) == 1
    assert not first.cached and second.cached and second.probabilities == first.probabilities == {"passage": 0.8}
    assert second.latency_ms == first.latency_ms
    assert second.cost.decision_cost == pytest.approx(estimate_decision_cost("gpt-6-luna", 100))  # charged as if called, at today's price
    assert runtime.disk.stats()["saved_cost_usd"] == pytest.approx(second.cost.total_cost)


def test_any_change_to_the_text_or_the_instructions_is_a_cache_miss(runtime):
    inner = CountingDecider()
    cached = CachedDecisionModel(inner)
    text = formats.single_input("q?", "a passage")
    cached.decide(text, [Predicate(formats.SINGLE, "is it?")])
    cached.decide(text, [Predicate(formats.SINGLE, "is it really?")])
    cached.decide(formats.single_input("q?", "another passage"), [Predicate(formats.SINGLE, "is it?")])
    assert len(inner.requests) == 3


def test_without_an_active_cache_the_wrapper_is_transparent():
    inner = CountingDecider()
    cached = CachedDecisionModel(inner)
    text, predicates = formats.single_input("q?", "p"), [Predicate(formats.SINGLE, "?")]
    cached.decide(text, predicates)
    cached.decide(text, predicates)
    assert len(inner.requests) == 2


def test_a_cached_gate_step_reports_the_replayed_latency(runtime):
    system = make_gate(CachedDecisionModel(CountingDecider()))
    ask(system)
    second = ask(system)
    steps = [step for step in second.steps if step.kind == "grade"]
    assert all(step.metadata.get("cached") for step in steps)


# --- end to end through the evaluator ---------------------------------------------------------------------------------


def test_a_mock_run_records_the_gate_and_scores_abstains_on_unanswerable_questions(tmp_path, tiny_dataset):
    config = tmp_path / "gate.yaml"
    config.write_text(
        f"""
run: {{name: gate, output_dir: {tmp_path / "results"}}}
dataset: {{documents_path: {tiny_dataset["documents_path"]}, questions_path: {tiny_dataset["questions_path"]}}}
systems:
  - {{type: hybrid_rerank, name: plain, chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}, retrieval: {{final_top_k: 3}}}}
  - type: decision_gate
    name: gate
    chunker: {{type: token, chunk_size: 60, chunk_overlap: 0}}
    retrieval: {{final_top_k: 3, decision_model: {{provider: mock}}, answerability_threshold: 0.5, sufficiency_threshold: 0.5}}
evaluation: {{k_values: [1, 3], judge_enabled: false}}
""",
        encoding="utf-8",
    )
    run_dir = run_benchmark(config, force_mock=True, max_workers=1)
    rows = read_jsonl(run_dir / "per_question_results.jsonl")
    plain, gated = ([r for r in rows if r["system"] == name] for name in ("plain", "gate"))
    assert all(r["gate"] is None for r in plain) and all(r["gate"] is not None for r in gated)
    unanswerable = next(r for r in gated if r["question_id"] == "q8")
    assert unanswerable["gate"]["abstained"] and unanswerable["refused"] and unanswerable["answer"] == ABSTAIN_ANSWER
    for row in gated:
        assert any(step["name"] == "decide_answerability" for step in row["steps"])
        assert row["cost"]["decision_cost"] == pytest.approx(sum(step["cost"]["decision_cost"] for step in row["steps"]))
        assert {c["chunk_id"] for c in row["retrieved_contexts"] if c["in_context"]} == set(row["gate"]["kept_chunk_ids"])
    summary = pd.read_csv(run_dir / "metrics_summary.csv").set_index("system")
    assert summary.loc["gate", "gate_abstain_rate"] > 0 and summary.loc["gate", "avg_decision_calls"] > 0 and summary.loc["gate", "avg_chunks_kept"] > 0
    assert summary.loc["gate", "abstain_recall"] == 1.0  # the one unanswerable question was refused
    assert summary.loc["plain", ["gate_abstain_rate", "avg_decision_calls", "avg_chunks_kept"]].isna().all()
