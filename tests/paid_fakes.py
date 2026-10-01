"""Mock models that charge like paid ones: a "live-style" run whose real cost is known exactly, without any network.

`install(monkeypatch)` swaps the OpenAI clients for these, so a run with a (fake) key goes down the live code path, is priced from
the config's `pricing:` table, and `ragbench estimate`'s projection can be compared with what the run was actually charged.
"""

from __future__ import annotations

from typing import Any

from fake_openai_server import JUDGE_REPLY

from ragbench.models.cost import CostBreakdown, estimate_model_cost
from ragbench.models.embeddings import EmbeddingResult, HashingEmbeddingModel
from ragbench.models.llms import LLMResult, MockLLM
from ragbench.utils.text import estimate_tokens

GENERATOR = "fake-paid-model"
JUDGE = "fake-judge"
EMBEDDING = "fake-embedding"
PRICING = {GENERATOR: {"input": 1000.0, "output": 2000.0}, JUDGE: {"input": 500.0, "output": 1500.0}, EMBEDDING: {"input": 400.0}}


class PaidMockLLM(MockLLM):
    """Mock content at a real-looking price. The judge model answers judge prompts with a valid verdict, like the fake OpenAI server does."""

    def __init__(self, model_name: str = GENERATOR) -> None:
        super().__init__()
        self.model_name = model_name

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResult:
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        if kwargs.get("json_mode") and "evaluation judge" in system:
            prompt = "\n".join(m.get("content") or "" for m in messages)
            result = LLMResult(
                text=JUDGE_REPLY,
                model=self.model_name,
                prompt_tokens=estimate_tokens(prompt, self.model_name),
                completion_tokens=estimate_tokens(JUDGE_REPLY, self.model_name),
                cost=CostBreakdown(),
                finish_reason="stop",
            )
        else:
            result = super().generate(messages, **kwargs)
        result.cost = CostBreakdown(
            llm_prompt_tokens=result.prompt_tokens,
            llm_completion_tokens=result.completion_tokens,
            llm_cost=estimate_model_cost(self.model_name, result.prompt_tokens, result.completion_tokens),
        )
        return result


class PaidHashingEmbedding(HashingEmbeddingModel):
    cacheable = True

    def __init__(self, *_args: Any) -> None:
        super().__init__()
        self.model_name = EMBEDDING

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        result = super().embed_texts(texts)
        tokens = sum(estimate_tokens(text, self.model_name) for text in texts)
        cost = CostBreakdown(embedding_input_tokens=tokens, embedding_cost=estimate_model_cost(self.model_name, input_tokens=tokens))
        return EmbeddingResult(result.vectors, self.model_name, tokens, cost)


def install(monkeypatch: Any) -> None:
    """Make every OpenAI client in the process one of the paid fakes, and give the process a (fake) key so runs are live."""
    import ragbench.models.providers.openai as openai_provider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(openai_provider, "OpenAILLM", lambda model_name: PaidMockLLM(model_name))
    monkeypatch.setattr(openai_provider, "OpenAIEmbeddingModel", PaidHashingEmbedding)
