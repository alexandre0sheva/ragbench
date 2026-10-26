"""The `harness:` block of a system: a patch over the system's default prompts and a few bounded knobs. Every field it omits is inherited."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PROMPT_FIELDS = ("system_prompt", "grounding", "refusal_policy", "answer_instructions", "citation_style", "query_rewrite_prompt", "tool_guidance")

# Each knob sets an option of the system's `retrieval:` (or `chunker:`) section; the first target field the system's option model has is the
# one written (`top_k` is `final_top_k` where a system has both). The ranges are on `HarnessPatch`, tighter than the options' own.
KNOBS: dict[str, tuple[str, tuple[str, ...]]] = {
    "top_k": ("retrieval", ("final_top_k", "top_k", "top_k_parents")),
    "chunk_size": ("chunker", ("chunk_size",)),
    "reranker": ("retrieval", ("reranker",)),
    "vector_weight": ("retrieval", ("vector_weight",)),
    "max_hops": ("retrieval", ("max_hops",)),
    "max_rounds": ("retrieval", ("max_rounds",)),
    "max_steps": ("retrieval", ("max_steps",)),
}


class FewShotExample(BaseModel):
    """A worked example shown to the generator. It must come from the train split (checked when a run starts)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class HarnessPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = Field(default=None, description="Opening of the generator's system message: who it is and what it answers from.")
    grounding: str | None = Field(default=None, description="Rule: what the answer may draw on (the context, the tools' results, the model's own knowledge).")
    refusal_policy: str | None = Field(default=None, description="Rule: when and how to say the answer is not available.")
    answer_instructions: str | None = Field(default=None, description="Rule: the length, form and style of the answer.")
    citation_style: str | None = Field(default=None, description="Rule: how to cite the source documents.")
    query_rewrite_prompt: str | None = Field(
        default=None, description="System message of the call that writes search queries (`hyde`, `llm_heavy`, `rag_fusion`, `decompose`, `corrective`, `iterative`); the reply format stays fixed."
    )
    tool_guidance: str | None = Field(default=None, description="Guidance on using the tools, after the rules of a tool-using agent.")
    few_shot: list[FewShotExample] | None = Field(default=None, description="Worked `{question, answer}` examples at the end of the system message; train-split questions only.")
    top_k: int | None = Field(default=None, ge=1, le=50, description="1 to 50. Chunks the generator reads (`final_top_k`, `top_k` or `top_k_parents`, whichever the system has).")
    chunk_size: int | None = Field(default=None, ge=64, le=4096, description="64 to 4096. `chunker.chunk_size`.")
    reranker: str | None = Field(default=None, description="`retrieval.reranker` (one of the system's rerankers).")
    vector_weight: float | None = Field(default=None, ge=0, le=4, description="0 to 4. `retrieval.vector_weight`: weight of the vector ranking in hybrid fusion.")
    max_hops: int | None = Field(default=None, ge=1, le=10, description="1 to 10. `retrieval.max_hops` (`iterative`).")
    max_rounds: int | None = Field(default=None, ge=0, le=10, description="0 to 10. `retrieval.max_rounds` (`corrective`).")
    max_steps: int | None = Field(default=None, ge=1, le=50, description="1 to 50. `retrieval.max_steps` (tool-using agents).")

    def prompts(self) -> dict[str, str]:
        """The prompt fields this patch sets."""
        return {name: value for name in PROMPT_FIELDS if (value := getattr(self, name)) is not None}

    def knobs(self) -> dict[str, object]:
        """The knobs this patch sets."""
        return {name: value for name in KNOBS if (value := getattr(self, name)) is not None}
