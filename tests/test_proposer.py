"""The harness proposer: reads an evidence tree, returns an INV or DOM harness patch; an invalid patch gets one retry with the error, then
the current harness is inherited."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from evidence_support import copy_run, make_run, question_texts
from scripted_llm import ScriptedLLM

from ragbench.config.schema import SystemConfig
from ragbench.datasets.splits import Splits
from ragbench.harness.apply import effective_harness
from ragbench.models.llms import MockLLM
from ragbench.optimize.evidence import FORBIDDEN_WORDS, build_evidence, run_config
from ragbench.optimize.proposer import PROPOSER_SYSTEM_PROMPT, Proposal, Proposer


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, Splits]:
    return make_run(tmp_path_factory.mktemp("proposer_run"))


@pytest.fixture(scope="module")
def evidence(run, tmp_path_factory) -> Path:
    return build_evidence(run[0], system="hybrid", out_dir=tmp_path_factory.mktemp("ev") / "hybrid")


@pytest.fixture
def system(run) -> SystemConfig:
    return next(system for system in run_config(run[0]).systems if system.resolved_name == "hybrid")


def reply(harness: dict, rationale: str = "because") -> str:
    return json.dumps({"rationale": rationale, "harness": harness})


def scripted(*replies: str, price: float = 0.0) -> ScriptedLLM:
    queue = list(replies)
    return ScriptedLLM(lambda prompt: queue.pop(0) if len(queue) > 1 else queue[0], price_per_call=price)


# -- the mock round trip ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["INV", "DOM"])
def test_the_mock_proposer_returns_a_valid_patch_that_changes_the_harness(evidence, system, kind):
    proposer = Proposer.from_ref("gpt-6-luna", force_mock=True)
    assert isinstance(proposer.llm, MockLLM)

    proposal = proposer.propose(evidence, system, kind)

    assert proposal.kind == kind and proposal.patch and not proposal.inherited and proposal.attempts == 1
    candidate = proposal.apply(system)
    assert effective_harness(candidate) != effective_harness(system)
    assert {name: effective_harness(candidate)[name] for name in proposal.patch} == proposal.patch


def test_the_mock_proposer_varies_with_the_kind_and_the_variant(evidence, system):
    proposer = Proposer.from_ref("gpt-6-luna", force_mock=True)

    patches = [proposer.propose(evidence, system, kind, variant=variant).patch for kind in ("INV", "DOM") for variant in (0, 1)]

    assert len({json.dumps(patch, sort_keys=True) for patch in patches}) == 4


def test_the_mock_proposer_is_deterministic(evidence, system):
    first = Proposer.from_ref(None, force_mock=True).propose(evidence, system, "DOM", variant=2)
    again = Proposer.from_ref(None, force_mock=True).propose(evidence, system, "DOM", variant=2)

    assert first.patch == again.patch and first.rationale == again.rationale


# -- what the model is sent -------------------------------------------------------------------------------------------------------------------


def test_the_request_holds_the_evidence_the_kind_and_the_fields(evidence, system):
    llm = scripted(reply({"answer_instructions": "Answer in one sentence."}))

    Proposer(llm).propose(evidence, system, "DOM", variant=1)

    prompt = llm.prompts[0]
    assert "# Mission" in prompt and "# The task" in prompt and "rollouts/" in prompt
    assert "Kind: DOM" in prompt and "Proposal number: 2" in prompt
    assert "answer_instructions" in prompt and "top_k" in prompt


def test_the_proposer_prompt_never_mentions_the_measurement(evidence, system):
    llm = scripted(reply({"answer_instructions": "Answer in one sentence."}))
    Proposer(llm).propose(evidence, system, "INV")

    assert not re.search("|".join(FORBIDDEN_WORDS), PROPOSER_SYSTEM_PROMPT, flags=re.I)
    request = llm.prompts[0].split("=== request ===", 1)[1]
    assert not re.search("|".join(FORBIDDEN_WORDS), request, flags=re.I)


def test_a_small_evidence_budget_keeps_the_framing_and_says_what_it_left_out(evidence, system):
    llm = scripted(reply({"answer_instructions": "Answer in one sentence."}))

    Proposer(llm, max_evidence_chars=3000).propose(evidence, system, "INV")

    prompt = llm.prompts[0]
    assert "# Mission" in prompt and "=== request ===" in prompt
    assert re.search(r"\d+ more files? not shown", prompt)


def test_held_out_text_in_the_rendered_evidence_is_refused_before_any_model_call(run, system, tmp_path):
    run_dir, splits = copy_run(run[0], tmp_path / "run"), run[1]
    evidence = build_evidence(run_dir, system="hybrid", out_dir=tmp_path / "ev")
    planted = question_texts(run[0].parent.parent)[splits.test[0]]["question"]
    (evidence / "library" / "note.md").write_text(f"Seen before: {planted}\n", encoding="utf-8")
    llm = scripted(reply({"answer_instructions": "x"}))

    from ragbench.datasets.leakage import LeakageError

    with pytest.raises(LeakageError):
        Proposer(llm).propose(evidence, system, "INV")
    assert llm.calls == 0


# -- invalid patches: one retry, then inherit -------------------------------------------------------------------------------------------------


def test_an_invalid_patch_is_retried_once_with_the_error(evidence, system):
    llm = scripted(reply({"top_kk": 8}), reply({"top_k": 8}), price=0.01)

    proposal = Proposer(llm).propose(evidence, system, "INV")

    assert proposal.patch == {"top_k": 8} and proposal.attempts == 2 and not proposal.inherited
    assert "Did you mean 'top_k'" in llm.prompts[1]
    assert proposal.cost.llm_cost == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        ("not json at all", "not one JSON object"),
        (json.dumps({"rationale": "x"}), "`harness`"),
        (reply({}), "changes nothing"),
        (reply({"top_k": 500}), "top_k"),
        (reply({"max_hops": 3}), "no knob `max_hops`"),
        (reply({"tool_guidance": "Search twice."}), "no `tool_guidance`"),
    ],
)
def test_two_invalid_replies_inherit_the_current_harness(evidence, system, bad, error):
    llm = scripted(bad)

    proposal = Proposer(llm).propose(evidence, system, "DOM")

    assert proposal.inherited and proposal.patch == {} and proposal.attempts == 2 and llm.calls == 2
    assert proposal.harness == system.harness and proposal.apply(system) == system
    assert error in (proposal.error or "")


def test_a_patch_that_repeats_the_current_values_changes_nothing(evidence, system):
    current = effective_harness(system)
    llm = scripted(reply({"answer_instructions": current["answer_instructions"]}))

    proposal = Proposer(llm).propose(evidence, system, "INV")

    assert proposal.inherited and "changes nothing" in (proposal.error or "")


def test_a_few_shot_example_holding_a_held_out_question_is_refused_without_naming_it(run, evidence, system):
    splits = run[1]
    held_out = question_texts(run[0].parent.parent)[splits.test[0]]
    llm = scripted(reply({"few_shot": [{"question": held_out["question"], "answer": "something"}]}))

    proposal = Proposer(llm).propose(evidence, system, "DOM")

    assert proposal.inherited and "questions you were shown" in (proposal.error or "")
    assert held_out["id"] not in llm.prompts[1].split("=== request ===", 1)[1].split("Proposal number")[1]


def test_a_valid_patch_keeps_the_fields_the_system_already_patched(evidence, system):
    patched = system.model_copy(update={"harness": {"citation_style": "Cite [doc]."}})
    llm = scripted(reply({"answer_instructions": "Answer in one sentence."}))

    proposal = Proposer(llm).propose(evidence, patched, "INV")

    assert proposal.harness == {"citation_style": "Cite [doc].", "answer_instructions": "Answer in one sentence."}
    assert isinstance(proposal, Proposal) and proposal.rationale == "because"


# -- composition ------------------------------------------------------------------------------------------------------------------------------


def test_a_composition_request_lists_the_candidate_changes(evidence, system):
    llm = scripted(reply({"answer_instructions": "Answer in one sentence.", "top_k": 6}))
    candidates = {"p0-inv": {"answer_instructions": "Answer in one sentence."}, "p0-dom": {"top_k": 6}}

    proposal = Proposer(llm).propose(evidence, system, "COMPOSE", candidates=candidates)

    request = llm.prompts[0].split("=== request ===", 1)[1]
    assert "Kind: COMPOSE" in request and 'Candidate changes (JSON): {"p0-inv"' in request
    assert proposal.kind == "COMPOSE" and proposal.patch == {"answer_instructions": "Answer in one sentence.", "top_k": 6}


def test_a_composition_needs_candidates(evidence, system):
    with pytest.raises(ValueError, match="candidates"):
        Proposer(scripted(reply({}))).propose(evidence, system, "COMPOSE")


def test_the_mock_composition_merges_the_candidates_and_varies(evidence, system):
    candidates = {"a": {"answer_instructions": "Answer in one sentence."}, "b": {"top_k": 7}, "c": {}}
    proposer = Proposer.from_ref(None, force_mock=True)

    merged = proposer.propose(evidence, system, "COMPOSE", candidates=candidates, variant=0)
    other = proposer.propose(evidence, system, "COMPOSE", candidates=candidates, variant=1)

    assert merged.patch == {"answer_instructions": "Answer in one sentence.", "top_k": 7}
    assert other.patch and other.patch != merged.patch
