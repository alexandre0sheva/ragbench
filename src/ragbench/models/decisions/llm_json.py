"""A prompted chat model as the decision model: the same questions, answered as JSON by an ordinary LLM.

It exists to make a fair comparison possible. With `openai_decisions` and `llm_json` on the same model, the only thing that differs is
whether the probabilities come from a purpose-built typed endpoint or from a model asked to write them down.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate, clamp_probability
from ragbench.models.llms import LLM

SYSTEM_PROMPT = (
    "You answer yes/no questions about the text you are given. For each question give the probability, between 0 and 1, that the "
    'answer is yes. Reply with JSON only, in the form {"probabilities": {"<question name>": <number>, ...}} with every question name listed.'
)


def parse_probabilities(text: str, names: Sequence[str]) -> dict[str, float | None]:
    """The probabilities in a reply; a name that is missing or not a number maps to None."""
    try:
        data = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        try:
            data = json.loads(match.group(0)) if match else None
        except ValueError:
            data = None
    found = data.get("probabilities", data) if isinstance(data, dict) else {}
    found = found if isinstance(found, dict) else {}
    return {name: clamp_probability(found.get(name)) for name in names}


class LLMJsonDecisionModel(DecisionModel):
    provider = "llm_json"

    def __init__(self, llm: LLM):
        self.llm = llm
        self.model_name = llm.model_name

    def decide(self, text: str, predicates: Sequence[Predicate]) -> DecisionResult:
        questions = "\n".join(f"- {p.name}: {p.instructions}" for p in predicates)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{text}\n\nQuestions:\n{questions}"}]
        result = self.llm.generate(messages, temperature=0, json_mode=True, max_tokens=40 + 16 * len(predicates))
        cost = CostBreakdown(decision_input_tokens=result.prompt_tokens, decision_cost=result.cost.total_cost)
        return DecisionResult(
            probabilities=parse_probabilities(result.text, [p.name for p in predicates]),
            model=result.model,
            input_tokens=result.prompt_tokens,
            cost=cost,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )
