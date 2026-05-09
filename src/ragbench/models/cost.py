from __future__ import annotations

from pydantic import BaseModel

# Approximate USD prices per 1M tokens. Pricing changes; update this registry
# before using RAGBench for budgeting or procurement decisions.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "mock-llm": {"input": 0.0, "output": 0.0},
    "hashing-embedding": {"input": 0.0, "output": 0.0},
}


class CostBreakdown(BaseModel):
    embedding_input_tokens: int = 0
    embedding_cost: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_cost: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_cost: float = 0.0
    rerank_cost: float = 0.0
    query_rewrite_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.embedding_cost + self.llm_cost + self.judge_cost + self.rerank_cost + self.query_rewrite_cost

    def plus(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            embedding_input_tokens=self.embedding_input_tokens + other.embedding_input_tokens,
            embedding_cost=self.embedding_cost + other.embedding_cost,
            llm_prompt_tokens=self.llm_prompt_tokens + other.llm_prompt_tokens,
            llm_completion_tokens=self.llm_completion_tokens + other.llm_completion_tokens,
            llm_cost=self.llm_cost + other.llm_cost,
            judge_prompt_tokens=self.judge_prompt_tokens + other.judge_prompt_tokens,
            judge_completion_tokens=self.judge_completion_tokens + other.judge_completion_tokens,
            judge_cost=self.judge_cost + other.judge_cost,
            rerank_cost=self.rerank_cost + other.rerank_cost,
            query_rewrite_cost=self.query_rewrite_cost + other.query_rewrite_cost,
        )

    def as_dict(self) -> dict[str, float | int]:
        data = self.model_dump()
        data["total_cost"] = self.total_cost
        return data


def estimate_model_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    pricing = MODEL_PRICING_USD_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
