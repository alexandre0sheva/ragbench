"""`AgentLoop`: every termination reason, budget enforcement, and the guarantee that a misbehaving policy cannot loop forever."""

from __future__ import annotations

import pytest

from ragbench.agents import BUDGET, Abort, AgentBudget, AgentLoop, Continue, Finish, llm_calls
from ragbench.agents.budget import ABSOLUTE_MAX_STEPS
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.trace import Tracer


def _spend(tracer: Tracer, usd: float = 0.0, tokens: int = 0, kind="llm", name="work") -> None:
    with tracer.step(kind, name) as step:
        step.set_cost(CostBreakdown(llm_cost=usd))
        step.set_tokens(tokens, 0)


def _run(policy, budget: AgentBudget | None = None, tracer: Tracer | None = None):
    tracer = tracer or Tracer()
    return AgentLoop(budget or AgentBudget(), tracer).run(policy, question="q"), tracer


def test_finish_ends_the_loop_and_counts_steps_and_hops():
    def policy(state):
        return Finish("done") if state.iteration == 3 else Continue(f"query {state.iteration}")

    state, _ = _run(policy)
    assert state.termination == "finished" and state.answer == "done"
    assert state.iteration == 3 and state.hops == 2
    assert state.query == "query 2"
    assert state.metadata() == {"steps": 3, "hops": 2, "budget_exhausted": False, "termination": "finished"}


def test_a_policy_that_never_stops_ends_at_max_steps():
    calls = []

    def policy(state):
        calls.append(state.iteration)
        return Continue("again")

    state, _ = _run(policy, AgentBudget(max_steps=4))
    assert state.termination == "max_steps" and calls == [1, 2, 3, 4]
    assert state.metadata()["budget_exhausted"] is False


def test_the_hard_cap_holds_even_for_an_absurd_budget():
    calls = []

    def policy(state):
        calls.append(1)
        return Continue("again")

    state, _ = _run(policy, AgentBudget(max_steps=10**9))
    assert state.termination == "max_steps" and len(calls) == ABSOLUTE_MAX_STEPS


def test_abort_records_the_reason():
    state, _ = _run(lambda s: Abort("nothing to search"))
    assert state.termination == "abort" and state.abort_reason == "nothing to search"
    assert state.metadata()["budget_exhausted"] is False


def test_abort_with_the_budget_reason_counts_as_a_budget_stop():
    state, _ = _run(lambda s: Abort(BUDGET))
    assert state.termination == "budget" and state.metadata()["budget_exhausted"] is True


def test_a_policy_returning_garbage_aborts_instead_of_looping():
    state, _ = _run(lambda s: "keep going")  # type: ignore[arg-type,return-value]
    assert state.termination == "abort" and "str" in state.abort_reason


def test_a_policy_exception_is_not_swallowed():
    def policy(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run(policy)


def test_cost_cap_stops_the_loop_before_the_next_iteration():
    calls = []

    def policy(state):
        calls.append(state.iteration)
        _spend(state.tracer, usd=0.5)
        return Continue("more")

    state, _ = _run(policy, AgentBudget(max_steps=10, max_cost_usd=1.0))
    assert calls == [1, 2]  # $1.00 reached after two iterations: the third never starts
    assert state.termination == "budget" and state.metadata()["budget_exhausted"] is True
    assert state.cost_usd == pytest.approx(1.0)


def test_cost_cap_counts_spending_made_before_the_loop_started():
    tracer = Tracer()
    _spend(tracer, usd=2.0)
    calls = []
    state, _ = _run(lambda s: calls.append(1) or Finish(), AgentBudget(max_cost_usd=1.0), tracer)
    assert calls == [] and state.termination == "budget" and state.iteration == 0


def test_token_cap_stops_the_loop():
    def policy(state):
        _spend(state.tracer, tokens=400)
        return Continue("more")

    state, _ = _run(policy, AgentBudget(max_steps=10, max_tokens=1000))
    assert state.iteration == 3 and state.termination == "budget"
    assert state.tokens == 1200


def test_a_policy_can_see_the_budget_and_stop_before_an_llm_call():
    llm_calls_made = []

    def policy(state):
        _spend(state.tracer, usd=0.6)  # e.g. retrieval plus grading
        if state.over_budget():
            return Abort(BUDGET)
        llm_calls_made.append(state.iteration)  # the extra call a policy makes only while it can afford it
        return Continue("more")

    state, _ = _run(policy, AgentBudget(max_steps=10, max_cost_usd=1.0))
    assert llm_calls_made == [1] and state.termination == "budget" and state.iteration == 2


def test_no_cap_means_unlimited_spending():
    def policy(state):
        _spend(state.tracer, usd=100.0, tokens=10**6)
        return Finish() if state.iteration == 3 else Continue("more")

    state, _ = _run(policy)
    assert state.termination == "finished"


def test_every_decision_is_a_free_route_step_in_the_trace():
    def policy(state):
        return Finish() if state.iteration == 2 else Continue("second query")

    _, tracer = _run(policy)
    decisions = [s for s in tracer.steps if s.name == "agent_decision"]
    assert [(s.kind, s.metadata["iteration"], s.metadata["action"]) for s in decisions] == [("route", 1, "continue"), ("route", 2, "finish")]
    assert decisions[0].input_preview == "second query"
    assert all(s.cost.total_cost == 0 for s in tracer.steps)


def test_a_loop_stop_is_recorded_too():
    _, tracer = _run(lambda s: Continue("x"), AgentBudget(max_steps=2))
    stop = [s for s in tracer.steps if s.name == "agent_stop"]
    assert len(stop) == 1 and stop[0].metadata["termination"] == "max_steps"


def test_policy_info_is_reported_with_the_loop_metadata():
    def policy(state):
        state.info["sufficient"] = False
        return Finish()

    state, _ = _run(policy)
    assert state.metadata()["sufficient"] is False


@pytest.mark.parametrize("kwargs", [{"max_steps": 0}, {"max_cost_usd": 0}, {"max_cost_usd": -1.0}, {"max_tokens": 0}])
def test_invalid_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AgentBudget(**kwargs)


def test_llm_calls_counts_the_steps_that_used_tokens():
    tracer = Tracer()
    _spend(tracer, tokens=10)
    _spend(tracer, tokens=0, kind="retrieve")
    _spend(tracer, tokens=5)
    assert llm_calls(tracer.steps) == 2
