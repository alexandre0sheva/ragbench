from __future__ import annotations

from typing import Any

from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import ANSWER_SYSTEM_PROMPT, RetrievalResult
from ragbench.rag_systems.llm_query_base import LLMQueryRAG
from ragbench.rag_systems.options import DecomposeOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.registry import SYSTEMS
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.query_planning import plan_subquestions_llm
from ragbench.utils.text import normalize_text
from ragbench.utils.timing import timer

SUB_ANSWER_MAX_TOKENS = 150
CARRIED_FINDING_CHARS = 200
DECOMPOSE_SYSTEM_PROMPT = (
    ANSWER_SYSTEM_PROMPT
    + "\nThe question was broken into sub-questions, and the evidence is listed under each one. Combine the evidence across sub-questions "
    "to answer the full question, and cite the document IDs of the passages you rely on."
)


@SYSTEMS.register("decompose")
class DecomposeRAG(LLMQueryRAG):
    """Question decomposition: the LLM splits a complex question into ordered sub-questions, each is searched, and one answer is synthesized.

    Every sub-question is a traced `retrieve` step. The merged ranking (RRF over the sub-questions' rankings) is what retrieval is
    scored on; the answer prompt lists each sub-question with the passages that came from its search, doc IDs included. With
    `sequential: true` each sub-question is answered before the next is searched, and its finding is added to that search.
    """

    spec = SystemSpec(
        type="decompose",
        title="Question decomposition",
        summary="The LLM splits a complex question into sub-questions; each is searched (optionally in sequence, feeding earlier answers forward), then one answer is synthesized from all the evidence",
        best_for="Multi-part, comparison and multi-hop questions",
        cost_profile="medium",
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=DecomposeOptions,
    )
    options: DecomposeOptions

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        opts = self.options
        final_top_k = opts.resolve_top_k(top_k)
        per_query_k = max(opts.per_query_top_k, final_top_k)
        with timer() as t:
            plan, cost = self._plan("plan_subquestions", plan_subquestions_llm, question, opts.max_subquestions)
            asked = [(sub, False) for sub in plan.items]
            if opts.include_original and not (plan.fallback or len(plan.items) == 1 and _same(plan.items[0], question)):
                asked.append((question, True))
            sub_questions: list[dict[str, Any]] = []
            rankings: list[list[RetrievedChunk]] = []
            findings: list[str] = []
            for index, (sub, is_original) in enumerate(asked):
                query = sub if is_original or not (opts.sequential and findings) else f"{sub}\n{' '.join(findings)}"
                chunks, search_cost = self._search(query, per_query_k, "subquestion_search", index=index, query=query, original=is_original)
                cost = cost.plus(search_cost)
                rankings.append(chunks)
                entry: dict[str, Any] = {"question": sub, "chunk_ids": [chunk.chunk_id for chunk in chunks], "original": is_original}
                if opts.sequential and not is_original:
                    finding, answer_cost = self._answer_sub_question(sub, chunks[: opts.chunks_per_subquestion], findings, index)
                    cost = cost.plus(answer_cost)
                    entry["answer"] = finding
                    findings.append(finding[:CARRIED_FINDING_CHARS])
                sub_questions.append(entry)
            merged = reciprocal_rank_fusion(rankings, top_k=final_top_k, rrf_k=opts.rrf_k)
        return RetrievalResult(
            question=question,
            chunks=merged,
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={
                "retriever": "decompose",
                "sub_questions": sub_questions,
                "fallback": plan.fallback,
                "sequential": opts.sequential,
                "search": opts.retriever,
            },
        )

    def _answer_sub_question(self, sub_question: str, chunks: list[RetrievedChunk], findings: list[str], index: int) -> tuple[str, CostBreakdown]:
        earlier = "Findings so far:\n" + "\n".join(f"- {finding}" for finding in findings) + "\n\n" if findings else ""
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{earlier}Context:\n{self._format_context(chunks)}\n\nQuestion: {sub_question}"},
        ]
        with self.trace.step("llm", "sub_answer", index=index) as step:
            result = self.llm.generate(messages, temperature=0, max_tokens=SUB_ANSWER_MAX_TOKENS)
            step.set_llm(result)
            step.set_input(sub_question)
        return " ".join(result.text.split()), result.cost

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]):
        subs = retrieval.metadata.get("sub_questions") or []
        if len(subs) < 2:  # nothing was split (a simple question, or the plan fell back): the plain prompt says it all
            return super()._generate_from_retrieval(retrieval, question, chunks)
        context = self._format_evidence(subs, chunks)
        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        with self.trace.step("generate", "answer", context_chunks=len(chunks), sub_questions=len(subs)) as step:
            result = self.llm.generate(messages, temperature=0)
            step.set_llm(result)
            step.set_input(question)
        return result

    def _format_evidence(self, subs: list[dict[str, Any]], chunks: list[RetrievedChunk]) -> str:
        """Each sub-question followed by the context passages it ranked highest (a passage is listed under one sub-question only)."""
        owner: dict[str, int] = {}
        for chunk in chunks:
            ranks = [(sub["chunk_ids"].index(chunk.chunk_id), number) for number, sub in enumerate(subs) if chunk.chunk_id in sub["chunk_ids"]]
            owner[chunk.chunk_id] = min(ranks)[1] if ranks else 0  # the best rank wins, the earlier sub-question breaks ties
        blocks: list[str] = []
        for number, sub in enumerate(subs):
            label = "Original question" if sub.get("original") else f"Sub-question {number + 1}"
            lines = [f"{label}: {sub['question']}"]
            if sub.get("answer"):
                lines.append(f"Preliminary finding: {sub['answer']}")
            found = [chunk for chunk in chunks if owner[chunk.chunk_id] == number]
            lines.extend(f"[{chunk.doc_id} | {chunk.chunk_id.split('::chunk::')[-1]}]\n{chunk.text}" for chunk in found)
            if not found:
                lines.append("(no passage matched this sub-question best)")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _same(a: str, b: str) -> bool:
    return normalize_text(a).strip(" ?").lower() == normalize_text(b).strip(" ?").lower()
