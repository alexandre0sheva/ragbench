"""The default prompt texts a harness patch edits, and how a resolved harness renders the generator's system message.

An answer prompt is an opening (`system_prompt`) followed by one rule per field (`grounding`, `refusal_policy`, `answer_instructions`,
`citation_style`); an empty field drops its rule. The defaults render exactly the prompts the systems sent before the harness existed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ragbench.harness.schema import PROMPT_FIELDS, FewShotExample, HarnessPatch
from ragbench.models.prompts import TOOL_AGENT_MARKER

ANSWER_PROMPTS = {
    "system_prompt": "You are answering questions using only the provided context.",
    "grounding": "Use only the provided context.",
    "refusal_policy": 'If the answer is not available in the context, say: "I could not find the answer in the provided documents."',
    "answer_instructions": "Be concise but complete.",
    "citation_style": "Cite source document IDs in square brackets when possible, for example [doc_014].",
}
NO_RETRIEVAL_PROMPTS = {
    "system_prompt": "You are answering questions from your own knowledge only. No documents are provided.",
    "grounding": "Answer only what you actually know; do not invent names, numbers, dates, or identifiers.",
    "refusal_policy": 'If you do not know, say: "I could not find the answer from my own knowledge."',
    "answer_instructions": "Be concise but complete.",
    "citation_style": "",
}
# `graph_rag`'s `global` mode answers from scored points drawn from community reports.
REDUCE_PROMPTS = {
    **ANSWER_PROMPTS,
    "system_prompt": "You are answering questions about a document collection from points that were drawn from reports on groups of related entities.",
    "grounding": "Use only the points provided; points with a higher score matter more.",
    "refusal_policy": 'If the points do not answer the question, say: "I could not find the answer in the provided documents."',
}
# The tool-using agents; their protocol lines (`agents/prompts.py`) are not part of the harness.
TOOL_AGENT_PROMPTS = {
    "system_prompt": f"{TOOL_AGENT_MARKER}, answering a question about a collection of documents.",
    "grounding": "Base the answer only on what the tools returned; never guess or use outside knowledge.",
    "refusal_policy": 'If the documents do not contain the answer, say: "I could not find the answer in the provided documents."',
    "answer_instructions": "Be concise but complete.",
    "citation_style": "Cite the source document ID in square brackets, for example [doc_014], after each fact taken from a document.",
    "tool_guidance": "",
}
EXAMPLES_HEADING = "Examples of good answers (they show the form of an answer; they are not evidence):"


@dataclass(frozen=True)
class PromptHarness:
    """A system's prompt texts after its `harness:` patch: a field is None when the system has no such prompt."""

    system_prompt: str | None = None
    grounding: str | None = None
    refusal_policy: str | None = None
    answer_instructions: str | None = None
    citation_style: str | None = None
    query_rewrite_prompt: str | None = None
    tool_guidance: str | None = None
    few_shot: tuple[FewShotExample, ...] = ()

    @classmethod
    def resolve(cls, defaults: Mapping[str, str], patch: HarnessPatch | Mapping[str, Any] | None = None) -> PromptHarness:
        """`defaults` (the fields a system uses) with `patch` applied; a patched field the system does not use is an error."""
        patch = patch if isinstance(patch, HarnessPatch) else HarnessPatch.model_validate(patch or {})
        texts = dict(defaults)
        unused = [name for name in patch.prompts() if name not in texts]
        if patch.few_shot is not None and "system_prompt" not in texts:
            unused.append("few_shot")
        if unused:
            raise ValueError(f"this system has no {', '.join(f'`{name}`' for name in unused)} (its harness prompts: {', '.join(texts) or 'none'})")
        texts.update(patch.prompts())
        return cls(**texts, few_shot=tuple(patch.few_shot or ()))

    @property
    def rewrite_prompt(self) -> str:
        """`query_rewrite_prompt`, for a system that writes search queries (it always has one)."""
        if self.query_rewrite_prompt is None:
            raise RuntimeError("this system's harness has no query_rewrite_prompt")
        return self.query_rewrite_prompt

    def fields(self) -> dict[str, str]:
        """The prompt fields the system uses, in their resolved form."""
        return {name: value for name in PROMPT_FIELDS if (value := getattr(self, name)) is not None}

    def answer_system(self) -> str:
        """The generator's system message: the opening, one `- rule` line per non-empty rule field, then any examples."""
        rules = "".join(f"- {rule}\n" for rule in (self.grounding, self.refusal_policy, self.answer_instructions, self.citation_style) if rule)
        text = f"{self.system_prompt}\n\nRules:\n{rules}" if rules else f"{self.system_prompt}\n"
        return text + self.examples()

    def examples(self) -> str:
        """The few-shot block (empty without examples). `Q:`/`A:`, not `Question:`, so nothing mistakes an example for the question asked."""
        if not self.few_shot:
            return ""
        return f"\n{EXAMPLES_HEADING}\n" + "\n\n".join(f"Q: {example.question}\nA: {example.answer}" for example in self.few_shot) + "\n"
