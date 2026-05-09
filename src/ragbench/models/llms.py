from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.utils.env import has_openai_key
from ragbench.utils.text import estimate_tokens, tokenize


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: CostBreakdown
    raw: Any | None = None


class LLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        raise NotImplementedError


class MockLLM(LLM):
    def __init__(self):
        self.model_name = "mock-llm"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        prompt = "\n".join(m.get("content", "") for m in messages)
        if kwargs.get("json_mode"):
            text = json.dumps({"score": 3, "reasoning": "Mock JSON response."})
        elif "Rewrite the question" in prompt or "search queries" in prompt:
            question = prompt.strip().splitlines()[-1]
            text = json.dumps({"queries": [question]})
        elif "summary" in prompt.lower() and "key entities" in prompt.lower():
            text = json.dumps({"summary": self._summarize(prompt), "key_entities": [], "hypothetical_questions": []})
        else:
            text = self._answer_from_prompt(prompt)
        prompt_tokens = estimate_tokens(prompt, self.model_name)
        completion_tokens = estimate_tokens(text, self.model_name)
        return LLMResult(text=text, model=self.model_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=CostBreakdown())

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


class OpenAILLM(LLM):
    def __init__(self, model_name: str = "gpt-5.4-nano"):
        self.model_name = model_name
        from openai import OpenAI

        self.client = OpenAI()

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        try:
            return self._generate_chat_completions(messages, **kwargs)
        except Exception:
            if hasattr(self.client, "responses"):
                return self._generate_responses(messages, **kwargs)
            raise

    def _generate_chat_completions(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        response_format = {"type": "json_object"} if kwargs.get("json_mode") else None
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
        }
        if response_format:
            params["response_format"] = response_format
        if kwargs.get("max_tokens"):
            params["max_tokens"] = kwargs["max_tokens"]
        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""
        prompt_tokens = int(response.usage.prompt_tokens) if response.usage else sum(estimate_tokens(m["content"], self.model_name) for m in messages)
        completion_tokens = int(response.usage.completion_tokens) if response.usage else estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=CostBreakdown(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
            ),
            raw=response,
        )

    def _generate_responses(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResult:
        system_messages = [m.get("content", "") for m in messages if m.get("role") == "system"]
        input_messages = [m for m in messages if m.get("role") != "system"]
        if kwargs.get("json_mode"):
            system_messages.append("Return JSON only.")
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": input_messages,
            "instructions": "\n\n".join(system_messages) or None,
            "temperature": kwargs.get("temperature", 0),
        }
        if kwargs.get("max_tokens"):
            params["max_output_tokens"] = kwargs["max_tokens"]
        response = self.client.responses.create(**params)
        text = getattr(response, "output_text", "") or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        if not prompt_tokens:
            prompt_tokens = sum(estimate_tokens(m.get("content", ""), self.model_name) for m in messages)
        if not completion_tokens:
            completion_tokens = estimate_tokens(text, self.model_name)
        return LLMResult(
            text=text,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=CostBreakdown(
                llm_prompt_tokens=prompt_tokens,
                llm_completion_tokens=completion_tokens,
                llm_cost=estimate_model_cost(self.model_name, prompt_tokens, completion_tokens),
            ),
            raw=response,
        )


def create_llm(model_name: str | None = None, force_mock: bool = False) -> LLM:
    if force_mock or not has_openai_key():
        return MockLLM()
    try:
        return OpenAILLM(model_name or "gpt-5.4-nano")
    except Exception:
        return MockLLM()
