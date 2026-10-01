from __future__ import annotations

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ragbench.models import mock_agent
from ragbench.models.cost import CostBreakdown
from ragbench.models.defaults import DEFAULT_GENERATOR_MODEL
from ragbench.models.errors import ModelInitError
from ragbench.models.prompts import (
    CHECK_GROUNDED_MARKER,
    GENERATE_QUERY_VARIANTS_MARKER,
    GRADE_CHUNKS_MARKER,
    NEXT_HOP_MARKER,
    PLAN_SUBQUESTIONS_MARKER,
    REACT_JSON_MARKER,
    REWRITE_QUERY_MARKER,
    ROUTE_QUESTION_MARKER,
    SITUATE_CHUNK_MARKER,
    SUMMARIZE_DOCUMENT_MARKER,
    TOOL_AGENT_MARKER,
)
from ragbench.models.refs import parse_model_ref, provider_reachable
from ragbench.models.usage import note_llm
from ragbench.utils.query_planning import generate_query_variants, split_subquestions_locally
from ragbench.utils.text import estimate_tokens, tokenize

logger = logging.getLogger(__name__)

__all__ = ["LLM", "LLMResult", "MockLLM", "ToolCall", "create_llm", "parse_model_ref"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: CostBreakdown
    # Tools the model asked to call. Feed the turn back with `assistant_message()` plus one `role: "tool"` message per call.
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None  # "stop" | "length" | "tool_calls" | provider-specific
    raw: Any | None = None
    # Wall-clock time of the underlying call; for a cache hit, the time the original call took (replayed).
    latency_ms: float | None = None
    cached: bool = False

    def assistant_message(self) -> dict[str, Any]:
        """This turn as a chat message (OpenAI shape: tool-call arguments are a JSON string), ready to append to `messages`."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in self.tool_calls
            ]
        return message


class LLM(ABC):
    model_name: str
    # Identifies the backing service in persistent cache keys (two providers may serve the same model name).
    provider: str = "openai"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        """One chat completion.

        `messages` are OpenAI-style chat messages; they may include `{"role": "assistant", "tool_calls": [...]}` and
        `{"role": "tool", "tool_call_id": ..., "content": ...}`. `tools` is a list of OpenAI function schemas
        (`{"type": "function", "function": {"name", "description", "parameters"}}`); providers translate as needed.
        """
        raise NotImplementedError


Predicate = Callable[[str, bool], bool]
Responder = Callable[[str], str]


class MockLLM(LLM):
    """A deterministic offline model. Prompts are matched against a table of responders, newest registration first;
    a prompt nothing matches is answered from its `Context:` block (see `_answer_from_prompt`).

    Add a mock-safe behaviour for a new kind of prompt with `MockLLM.register_responder(predicate, respond)`:
    `predicate(prompt, json_mode)` says whether it applies and `respond(prompt)` returns the text.
    """

    _responders: ClassVar[list[tuple[str, Predicate, Responder]]] = []
    _responders_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        self.model_name = "mock-llm"

    @classmethod
    def register_responder(cls, predicate: Predicate, respond: Responder, *, name: str | None = None) -> str:
        """Answer prompts for which `predicate(prompt, json_mode)` is true with `respond(prompt)`; returns the responder's name.

        The newest registration wins over older ones, including the built-ins; registering a name again replaces it.
        """
        key = name or getattr(respond, "__name__", None) or "responder"
        with cls._responders_lock:
            cls._responders = [entry for entry in cls._responders if entry[0] != key]
            cls._responders.insert(0, (key, predicate, respond))
        return key

    @classmethod
    def unregister_responder(cls, name: str) -> None:
        with cls._responders_lock:
            cls._responders = [entry for entry in cls._responders if entry[0] != name]

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResult:
        result = self._generate(messages, json_mode, tools)
        note_llm(result.prompt_tokens, result.completion_tokens)  # a no-op unless `ragbench estimate` is measuring
        return result

    def _generate(self, messages: list[dict[str, Any]], json_mode: bool, tools: list[dict[str, Any]] | None) -> LLMResult:
        prompt = "\n".join(m.get("content") or "" for m in messages)
        if TOOL_AGENT_MARKER in prompt and (tools or REACT_JSON_MARKER in prompt):
            return self._tool_agent_turn(messages, prompt, tools)  # the tool-using agents' turns; any other prompt ignores `tools`
        text = next((respond(prompt) for _, predicate, respond in self._responders if predicate(prompt, json_mode)), None)
        if text is None:
            text = self._answer_from_prompt(prompt)
        prompt_tokens = estimate_tokens(prompt, self.model_name)
        completion_tokens = estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text, model=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=CostBreakdown(), finish_reason="stop"
        )

    def _tool_agent_turn(self, messages: list[dict[str, Any]], prompt: str, tools: list[dict[str, Any]] | None) -> LLMResult:
        """A turn of a tool-using agent: call a tool (native `tool_calls`, or a JSON action in ReAct mode) or give the final answer."""
        available = [t["function"]["name"] for t in tools] if tools else mock_agent.available_react_tools(prompt)
        question = mock_agent.question_of(messages)
        observations = mock_agent.observations_of(messages)
        turn = mock_agent.decide(question, observations, available, force_answer=mock_agent.forced_to_answer(messages))
        answer = "" if turn.tool else self._agent_answer(question, observations)
        react = not tools
        tool_calls = [ToolCall(mock_agent.native_call_id(observations), turn.tool, turn.args)] if turn.tool and not react else []
        text = mock_agent.react_reply(turn, answer) if react else answer
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=estimate_tokens(prompt, self.model_name),
            completion_tokens=estimate_tokens(text + json.dumps(turn.args), self.model_name),
            cost=CostBreakdown(),
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    def _agent_answer(self, question: str, observations: list[tuple[str, str]]) -> str:
        """Answer from the tool observations through the usual mock path; an exact result from `calculator` / `date_calc` leads the answer."""
        computed = [text for name, text in observations if name in ("calculator", "date_calc") and not text.startswith("Error")]
        context = "\n".join(text for name, text in observations if name not in ("calculator", "date_calc"))
        answer = self._answer_from_prompt(f"Context:\n{context}\n\nQuestion: {question}")
        if not computed:
            return answer
        return computed[-1] if answer.startswith("I could not find") else f"{computed[-1]} {answer}"

    @staticmethod
    def _hypothetical_from_prompt(prompt: str) -> str:
        # Echo the question's content words as a pseudo-passage so HyDE's
        # search probe stays on-topic for the hashing embedding model.
        question_match = re.search(r"Question:\s*(.*?)\s*$", prompt, flags=re.S | re.I)
        question = question_match.group(1) if question_match else prompt
        words = [t for t in tokenize(question) if len(t) > 2]
        topic = " ".join(words) or question
        return f"{topic}. This passage describes {topic} in detail. Reference information about {topic}."

    @staticmethod
    def _summarize(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])[:500]

    def _answer_from_prompt(self, prompt: str) -> str:
        context_match = re.search(r"Context:\s*(.*?)\n\nQuestion:", prompt, flags=re.S | re.I)
        question_match = re.search(r"Question:\s*(.*?)\s*$", prompt, flags=re.S | re.I)
        context = context_match.group(1) if context_match else prompt
        question = question_match.group(1) if question_match else ""
        question_tokens = {t for t in tokenize(question) if len(t) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        scored: list[tuple[int, str]] = []
        for sentence in sentences:
            overlap = len(question_tokens.intersection(tokenize(sentence)))
            if overlap:
                scored.append((overlap, sentence))
        if not scored:
            return "I could not find the answer in the provided documents."
        scored.sort(key=lambda item: item[0], reverse=True)
        answer = " ".join(sentence for _, sentence in scored[:2])
        citations = sorted(set(re.findall(r"\[(doc_[A-Za-z0-9_-]+)\s*\|", answer + "\n" + context)))
        if citations and not re.search(r"\[doc_[A-Za-z0-9_-]+\]", answer):
            answer = f"{answer} " + " ".join(f"[{doc_id}]" for doc_id in citations[:3])
        return answer[:1200]


def _mock_json(prompt: str) -> str:
    return json.dumps({"score": 3, "reasoning": "Mock JSON response."})


def _mock_rewrite(prompt: str) -> str:
    return json.dumps({"queries": [prompt.strip().splitlines()[-1]]})


def _mock_enrichment(prompt: str) -> str:
    return json.dumps({"summary": MockLLM._summarize(prompt), "key_entities": [], "hypothetical_questions": []})


def _mock_situate(prompt: str) -> str:
    title = re.search(r"Document title:[ \t]*(.*)", prompt)
    return f'This passage is from the document "{title.group(1).strip()}".' if title and title.group(1).strip() else "This passage is from the document."


def _mock_document_summary(prompt: str) -> str:
    body = prompt.split("Document text:", 1)[-1]
    body = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    return " ".join(re.split(r"(?<=[.!?])\s+", body.strip())[:2])[:500]


def _mock_question(prompt: str) -> str:
    match = re.search(r"Question:\s*(.*?)\s*$", prompt, flags=re.S | re.I)
    return match.group(1) if match else prompt.strip()


def _mock_query_variants(prompt: str) -> str:
    # The local rule-based variants plus the question's content words: deterministic and on-topic for the hashing embeddings.
    question = _mock_question(prompt)
    keywords = " ".join(t for t in tokenize(question) if len(t) > 3)
    variants = [v for v in generate_query_variants(question, max_queries=6)[1:] if v.lower() != question.lower()]
    return json.dumps({"queries": [*variants, *([keywords] if keywords else [])]})


def _mock_subquestions(prompt: str) -> str:
    question = _mock_question(prompt)
    return json.dumps({"sub_questions": split_subquestions_locally(question) or [question]})


def _group(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.S)
    return match.group(1) if match else default


def _content_words(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 3}


def _search_words(text: str) -> str:
    return " ".join(dict.fromkeys(token for token in tokenize(text) if len(token) > 3)) or text.strip()


def _mock_grades(prompt: str) -> str:
    # Keyword overlap between the question and each numbered passage: two shared content words (or half of them) is relevant, one is ambiguous.
    question = _content_words(_group(r"Question:\s*(.*?)\n\nPassages:", prompt))
    grades = []
    for number, passage in re.findall(r"^\[(\d+)\] (.*)$", prompt.split("Passages:", 1)[-1], flags=re.M):
        shared = len(question & _content_words(passage))
        grade = "relevant" if shared >= 2 or (question and shared / len(question) >= 0.5) else "ambiguous" if shared == 1 else "irrelevant"
        grades.append({"id": int(number), "grade": grade})
    return json.dumps({"grades": grades})


def _mock_rewrite_query(prompt: str) -> str:
    question = _group(r"Question:\s*(.*?)\nPrevious query:", prompt, prompt)
    return json.dumps({"query": _search_words(question)})


def _mock_next_hop(prompt: str) -> str:
    # Done as soon as the evidence shares a content word with the question; otherwise search again for just the question's content words.
    question = _group(r"Question:\s*(.*?)\n\nEvidence so far:", prompt)
    evidence = prompt.split("Evidence so far:", 1)[-1]
    known = MockLLM._summarize(re.sub(r"\[[^\]]*\]", "", evidence).strip())[:200]
    found = bool(_content_words(question) & _content_words(evidence))
    return json.dumps({"known": known, "missing": "" if found else "more detail", "next_query": "DONE" if found else _search_words(question)})


def _mock_route(prompt: str) -> str:
    # The mock router is the rule-based one: it picks among the routes listed in the prompt.
    from ragbench.agents.router import heuristic_route

    routes = re.findall(r"^- ([A-Za-z0-9_-]+): ", prompt.split("Routes:", 1)[-1].split("\n\n", 1)[0], flags=re.M)
    decision = heuristic_route(_group(r"Question:\s*(.*)$", prompt), routes)
    return json.dumps({"route": decision.route, "reason": decision.reason})


def _mock_groundedness(prompt: str) -> str:
    match = re.search(r"Context:\s*(.*?)\n\nAnswer:\s*(.*?)\n\nReturn JSON", prompt, flags=re.S)
    context, answer = (match[1], match[2]) if match else ("", "")
    supported = "could not find" in answer.lower() or bool(_content_words(answer) & _content_words(context))
    return json.dumps({"supported": supported, "unsupported_claims": [] if supported else [answer[:80]]})


# Registered lowest precedence first: JSON mode wins over everything, then the specific prompt kinds, then the answer fallback.
for _name, _predicate, _respond in (
    ("enrichment", lambda prompt, json_mode: "summary" in prompt.lower() and "key entities" in prompt.lower(), _mock_enrichment),
    ("rewrite", lambda prompt, json_mode: "Rewrite the question" in prompt or "search queries" in prompt, _mock_rewrite),
    ("hypothetical", lambda prompt, json_mode: "hypothetical passage" in prompt.lower(), MockLLM._hypothetical_from_prompt),
    # The `no_retrieval` baseline: a mock model knows nothing.
    ("no_documents", lambda prompt, json_mode: "no documents are provided" in prompt.lower(), lambda prompt: "I could not find the answer from my own knowledge."),
    ("json", lambda prompt, json_mode: json_mode, _mock_json),
    # Ingestion-time prompts of `contextual` and `hierarchical` (plain text, never JSON mode).
    ("situate_chunk", lambda prompt, json_mode: not json_mode and SITUATE_CHUNK_MARKER in prompt, _mock_situate),
    ("summarize_document", lambda prompt, json_mode: not json_mode and SUMMARIZE_DOCUMENT_MARKER in prompt, _mock_document_summary),
    # Query-time planners of `rag_fusion` and `decompose` (JSON mode, so they must outrank the generic JSON responder).
    ("query_variants", lambda prompt, json_mode: GENERATE_QUERY_VARIANTS_MARKER in prompt, _mock_query_variants),
    ("plan_subquestions", lambda prompt, json_mode: PLAN_SUBQUESTIONS_MARKER in prompt, _mock_subquestions),
    # Control prompts of the agentic systems (`corrective`, `iterative`).
    ("grade_chunks", lambda prompt, json_mode: GRADE_CHUNKS_MARKER in prompt, _mock_grades),
    ("rewrite_query", lambda prompt, json_mode: REWRITE_QUERY_MARKER in prompt, _mock_rewrite_query),
    ("next_hop", lambda prompt, json_mode: NEXT_HOP_MARKER in prompt, _mock_next_hop),
    ("check_grounded", lambda prompt, json_mode: CHECK_GROUNDED_MARKER in prompt, _mock_groundedness),
    ("route_question", lambda prompt, json_mode: ROUTE_QUESTION_MARKER in prompt, _mock_route),
):
    MockLLM.register_responder(_predicate, _respond, name=_name)


def create_llm(
    model_name: str | None = None, force_mock: bool = False, strict: bool = True, providers: dict[str, Any] | None = None
) -> LLM:
    """Build the (cache-wrapped) chat model for a model ref such as `gpt-6-luna` or `anthropic:claude-haiku-4-5`.

    `force_mock`, or a hosted provider (OpenAI, Anthropic) whose API key is not set, selects the mock. With credentials
    present a failing client constructor raises `ModelInitError` (`strict=True`, default) instead of silently producing
    mock scores; pass `strict=False` to restore the old fall-back-to-mock behaviour. A missing optional SDK raises
    `ImportError` naming the extra to install. `providers` maps `providers:` endpoint names to their config; when omitted
    the active run's endpoints are used.
    """
    import ragbench.models.providers  # noqa: F401  (registers the built-in providers)
    from ragbench.registry import LLM_PROVIDERS
    from ragbench.runtime.context import current_runtime

    ref = model_name or DEFAULT_GENERATOR_MODEL
    provider, model = parse_model_ref(ref)
    if force_mock or not provider_reachable(provider):
        return MockLLM()
    factory = LLM_PROVIDERS.get(provider)
    endpoints = providers if providers is not None else current_runtime().providers
    try:
        from ragbench.models.cached import CachedLLM

        return CachedLLM(factory(model, providers=endpoints))
    except ImportError:
        raise  # the message already names the extra to install
    except Exception as exc:
        if strict:
            raise ModelInitError(f"Could not create the {provider} client for {ref!r}: {exc}") from exc
        logger.warning("%s client init failed (%s); falling back to the mock LLM.", provider, exc)
        return MockLLM()
