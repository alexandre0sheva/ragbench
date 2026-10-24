"""Typed decision models: `create_decision_model(provider, model)` builds one of the registered backends.

Importing this package registers the built-ins in `DECISION_MODELS`: `openai_decisions` (OpenAI's Decisions API), `llm_json` (a prompted
chat model) and `mock`. Third-party backends register through the `ragbench.decision_models` entry-point group or
`@DECISION_MODELS.register("name")`; see docs/extending.md.
"""

from __future__ import annotations

from ragbench.models.decisions.base import DecisionModel, DecisionResult, Predicate
from ragbench.models.decisions.cached import CachedDecisionModel
from ragbench.models.decisions.llm_json import LLMJsonDecisionModel
from ragbench.models.decisions.mock import MockDecisionModel
from ragbench.models.decisions.openai_decisions import OpenAIDecisionModel, create_openai_decisions
from ragbench.models.defaults import DEFAULT_DECISION_MODEL
from ragbench.models.llms import MockLLM, create_llm
from ragbench.models.refs import provider_reachable
from ragbench.registry import DECISION_MODELS

MOCK_PROVIDER = "mock"


def _llm_json(model: str | None) -> DecisionModel:
    llm = create_llm(model or DEFAULT_DECISION_MODEL)  # already cache-wrapped; a missing key gives the mock LLM
    return MockDecisionModel() if isinstance(llm, MockLLM) else LLMJsonDecisionModel(llm)


DECISION_MODELS.add(MOCK_PROVIDER, lambda model: MockDecisionModel())
DECISION_MODELS.add("openai_decisions", create_openai_decisions)
DECISION_MODELS.add("llm_json", _llm_json)
DECISION_MODELS.load_entry_points("ragbench.decision_models")

# Providers that call a hosted API and what they need; with `force_mock` or without the key the mock stands in, as for chat models.
_REQUIRES_OPENAI = {"openai_decisions"}


def create_decision_model(provider: str, model: str | None = None, force_mock: bool = False) -> DecisionModel:
    """The backend `provider` names (did-you-mean error for an unknown one). `force_mock`, or `openai_decisions` without an OpenAI key, selects
    the mock; `llm_json` follows its chat model the same way. A backend that cannot be built raises `ModelInitError`."""
    factory = DECISION_MODELS.get(provider)
    if force_mock or (provider in _REQUIRES_OPENAI and not provider_reachable("openai")):
        return MockDecisionModel()
    return factory(model)


__all__ = [
    "CachedDecisionModel",
    "DecisionModel",
    "DecisionResult",
    "LLMJsonDecisionModel",
    "MockDecisionModel",
    "OpenAIDecisionModel",
    "Predicate",
    "create_decision_model",
]
