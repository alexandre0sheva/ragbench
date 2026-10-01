from __future__ import annotations

from typing import Any

from ragbench.agents import BUDGET, Abort, AgentAction, AgentLoop, AgentState, Continue, Finish, prompts
from ragbench.agents.replies import parse_next_hop
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.llms import LLMResult
from ragbench.rag_systems.agentic_base import AgenticRAG
from ragbench.rag_systems.base import ANSWER_SYSTEM_PROMPT, RetrievalResult
from ragbench.rag_systems.options import IterativeOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import Tracer
from ragbench.registry import SYSTEMS
from ragbench.stores.hybrid_store import reciprocal_rank_fusion
from ragbench.utils.text import normalize_text

NEXT_HOP_MAX_TOKENS = 300
EVIDENCE_CHARS = 600


@SYSTEMS.register("iterative")
class IterativeRAG(AgenticRAG):
    """Iterative multi-hop retrieval (IRCoT-style): search, let the LLM say what is known and what is missing, search for the missing part, repeat.

    Each round searches the current query and shows the LLM the best chunks found so far. The LLM answers with notes and either `DONE` or
    the next query. The loop ends on `DONE`, after `max_hops` rounds, when the next query repeats an earlier one, or at the cost / token cap.
    Retrieval is scored on the rounds' rankings merged with RRF, and the answer is written from that evidence plus the LLM's notes.
    """

    spec = SystemSpec(
        type="iterative",
        title="Iterative multi-hop",
        summary="Search, ask the LLM what is known and what is missing, search for the missing part, and repeat until it says it is done or the hop limit is reached",
        best_for="Multi-hop questions whose second search depends on what the first one found",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=True,
        agentic=True,
        options=IterativeOptions,
    )
    options: IterativeOptions

    def _agent(self, question: str, final_top_k: int, tracer: Tracer) -> tuple[AgentState, list[RetrievedChunk], dict]:
        opts = self.options
        per_query_k = max(opts.per_query_top_k, final_top_k)
        rankings: list[list[RetrievedChunk]] = []
        evidence: dict[str, RetrievedChunk] = {}
        queries: list[str] = []
        notes: list[dict[str, Any]] = []

        def policy(state: AgentState) -> AgentAction:
            queries.append(state.query)
            chunks, _ = self._search(state.query, per_query_k, "hop_search", hop=state.iteration)
            rankings.append(chunks)
            for chunk in chunks[: opts.chunks_per_hop]:
                evidence.setdefault(chunk.chunk_id, chunk)
            if state.over_budget():
                return Abort(BUDGET)
            reply, _ = self._ask(
                "llm", "next_hop", prompts.next_hop(question, self._evidence_text(evidence), state.iteration, opts.max_hops), parse_next_hop,
                max_tokens=NEXT_HOP_MAX_TOKENS, hop=state.iteration,
            )
            if reply is None:  # an unreadable plan: answer from what has been found
                state.info["fallback"] = True
                return Finish()
            notes.append({"hop": state.iteration, "known": reply.known, "missing": reply.missing})
            if reply.next_query is None:
                state.info["sufficient"] = True
                return Finish()
            if normalize_text(reply.next_query).lower() in {normalize_text(q).lower() for q in queries}:
                state.info["repeated_query"] = True
                return Finish()
            return Continue(reply.next_query)

        state = AgentLoop(self._budget(opts.max_hops), tracer).run(policy, question=question)
        state.info.setdefault("sufficient", False)
        if not rankings:  # the budget was spent before the first round: still search for the question once
            rankings.append(self._search(question, per_query_k, "hop_search", hop=0)[0])
        merged = reciprocal_rank_fusion(rankings, top_k=final_top_k, rrf_k=opts.rrf_k)
        return state, merged, {"retriever": "iterative", "queries": queries, "notes": notes}

    @staticmethod
    def _evidence_text(evidence: dict[str, RetrievedChunk]) -> str:
        return "\n".join(
            f"[{chunk.doc_id} | {chunk.chunk_id.split('::chunk::')[-1]}] {' '.join(chunk.text.split())[:EVIDENCE_CHARS]}" for chunk in evidence.values()
        )

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        notes = [note for note in retrieval.metadata.get("notes", []) if note.get("known")]
        if not notes:
            return super()._generate_from_retrieval(retrieval, question, chunks)
        found = "\n".join(f"- Round {note['hop']}: {note['known']}" for note in notes)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Notes from the search (not evidence by themselves):\n{found}\n\nContext:\n{self._format_context(chunks)}\n\nQuestion: {question}"},
        ]
        with self.trace.step("generate", "answer", context_chunks=len(chunks), rounds=len(notes)) as step:
            result = self.llm.generate(messages, temperature=0)
            step.set_llm(result)
            step.set_input(question)
        return result
