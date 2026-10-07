"""OpenAI's Decisions API (`POST /v1/decisions`): typed questions in, probabilities out, billed on input tokens only."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

from ragbench.models.cost import CostBreakdown, estimate_decision_cost
from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate, clamp_probability
from ragbench.models.decisions.cached import CachedDecisionModel
from ragbench.models.defaults import DEFAULT_DECISION_MODEL
from ragbench.models.errors import PermanentModelError, translate_openai_exception
from ragbench.models.retry import call_with_retry
from ragbench.runtime.context import limiter_for
from ragbench.utils.text import estimate_tokens

DEFAULT_TIMEOUT_S = 120.0


class OpenAIDecisionModel(DecisionModel):
    """Sends one `predicate` question per name and reads the answers back by name (the API also returns them in question order).

    A question the model declines to answer comes back as a `refusal`; it is reported as a None probability, never as 0. Retries and
    timeouts are owned here (the SDK's own retries are off), like the chat provider. The request goes through the SDK's generic
    `post`, so it works on every `openai` release that is new enough to be a ragbench dependency, whether or not it has a `decisions` resource.
    """

    provider = "openai_decisions"

    def __init__(self, model_name: str = DEFAULT_DECISION_MODEL, client: Any | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        self.model_name = model_name
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=timeout, max_retries=0)
        self.client = client

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        names = [p.name for p in predicates]
        if len(set(names)) != len(names):
            raise ValueError(f"predicate names must be unique, got {names}")
        body = {
            "model": self.model_name,
            "input": text,
            "questions": [{"type": "predicate", "name": p.name, "instructions": p.instructions} for p in predicates],
        }
        limiter = limiter_for("openai")
        estimated = estimate_tokens(text, self.model_name) if limiter is not None and limiter.limits_tokens else 0

        def attempt() -> dict[str, Any]:
            with limiter.acquire(estimated) if limiter is not None else nullcontext():
                try:
                    reply = self.client.post("/decisions", cast_to=object, body=body)
                except Exception as exc:
                    translated = translate_openai_exception(exc)
                    if translated is exc:
                        raise
                    raise translated from exc
            if not isinstance(reply, dict) or not isinstance(reply.get("answers"), list):
                raise PermanentModelError(f"the Decisions API returned an unexpected reply: {str(reply)[:200]}")
            return reply

        reply = call_with_retry(attempt)
        answers = reply["answers"]
        by_name = {a.get("name"): a for a in answers if isinstance(a, dict)}
        probabilities: dict[str, float | None] = {}
        for index, name in enumerate(names):
            answer = by_name.get(name) or (answers[index] if index < len(answers) and isinstance(answers[index], dict) else {})
            probabilities[name] = clamp_probability(answer.get("probability")) if answer.get("type") == "predicate" else None
        usage = reply.get("usage") or {}
        tokens = int(usage.get("input_tokens") or 0) or estimate_tokens(text + " ".join(p.instructions for p in predicates), self.model_name)
        cost = CostBreakdown(decision_input_tokens=tokens, decision_cost=estimate_decision_cost(self.model_name, tokens))
        return DecisionResult(probabilities=probabilities, model=self.model_name, input_tokens=tokens, cost=cost)


def create_openai_decisions(model: str | None) -> DecisionModel:
    """The registered factory: the Decisions API behind the run's disk cache."""
    return CachedDecisionModel(OpenAIDecisionModel(model or DEFAULT_DECISION_MODEL))
