from __future__ import annotations

from ragbench.agents import BUDGET, Abort, AgentAction, AgentLoop, AgentState, Continue, Finish, prompts
from ragbench.agents.replies import AMBIGUOUS, IRRELEVANT, RELEVANT, parse_grades, parse_groundedness, parse_query
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.llms import LLMResult
from ragbench.rag_systems.agentic_base import AgenticRAG
from ragbench.rag_systems.base import ANSWER_SYSTEM_PROMPT, RetrievalResult
from ragbench.rag_systems.options import CorrectiveOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import Tracer
from ragbench.registry import SYSTEMS
from ragbench.utils.text import normalize_text, unique_preserve_order

GRADE_MAX_TOKENS = 200
REWRITE_MAX_TOKENS = 120
CHECK_MAX_TOKENS = 200
FULL_DOC_COUNT = 2  # documents a `full_doc` retry expands
# Retries grade more chunks than the first round: twice as many when widening to hybrid, three times with whole documents added.
WIDEN_GRADING = {"none": 1, "hybrid": 2, "full_doc": 3}
_GRADE_ORDER = {RELEVANT: 0, AMBIGUOUS: 1, None: 2, IRRELEVANT: 3}  # ungraded chunks rank between ambiguous and irrelevant ones


@SYSTEMS.register("corrective")
class CorrectiveRAG(AgenticRAG):
    """Corrective RAG (CRAG-style): retrieve, have the LLM grade the chunks, and if too little is relevant rewrite the query and widen the search.

    Retrieval repeats for at most `max_rounds` retries; graded chunks are ordered relevant, ambiguous, ungraded, irrelevant, so the
    grades also act as a reranker. An optional self-check asks whether the final answer is supported by its context and regenerates once if
    not. The agent stops early when the grader cannot be parsed, a rewrite repeats an earlier query, or `max_cost_usd` / `max_tokens` is reached.
    """

    spec = SystemSpec(
        type="corrective",
        title="Corrective RAG",
        summary="Retrieve, grade each chunk with the LLM, and when too little is relevant rewrite the query and widen the search; optional answer self-check",
        best_for="Corpora where the first search often misses and a bad context should be noticed rather than answered from",
        cost_profile="high",
        latency_profile="slow",
        requires_llm=True,
        agentic=True,
        options=CorrectiveOptions,
    )
    options: CorrectiveOptions

    def _agent(self, question: str, final_top_k: int, tracer: Tracer) -> tuple[AgentState, list[RetrievedChunk], dict]:
        opts = self.options
        per_query_k = max(opts.per_query_top_k, final_top_k)
        encountered: dict[str, RetrievedChunk] = {}  # every chunk seen, in the order it was first retrieved
        grades: dict[str, str] = {}  # the best grade each graded chunk received
        queries: list[str] = []

        def policy(state: AgentState) -> AgentAction:
            retry = state.iteration > 1
            queries.append(state.query)
            candidates = self._candidates(state.query, per_query_k, retry, state.iteration)
            for chunk in candidates:
                encountered.setdefault(chunk.chunk_id, chunk)
            batch = candidates[: opts.grade_top_k * (WIDEN_GRADING[opts.expand_to] if retry else 1)]
            if not batch or state.over_budget():
                return Abort(BUDGET) if batch else Finish()  # nothing retrieved (empty corpus): there is nothing to grade or retry
            batch_grades, _ = self._ask(
                "grade", "grade_chunks", prompts.grade_chunks(question, [chunk.text for chunk in batch]), lambda text: parse_grades(text, len(batch)),
                max_tokens=GRADE_MAX_TOKENS, round=state.iteration, chunks=len(batch),
            )
            if batch_grades is None:  # an unreadable grade is not a reason to spend more: keep the chunks as they ranked
                state.info["grading_failed"] = True
                return Finish()
            for chunk, grade in zip(batch, batch_grades, strict=True):
                previous = grades.get(chunk.chunk_id)
                if previous is None or _GRADE_ORDER[grade] < _GRADE_ORDER[previous]:  # a chunk graded again keeps its best grade
                    grades[chunk.chunk_id] = grade
            relevant = sum(1 for grade in grades.values() if grade == RELEVANT)
            state.info["sufficient"] = relevant >= opts.min_relevant
            if state.info["sufficient"] or state.iteration > opts.max_rounds:
                return Finish()
            if state.over_budget():
                return Abort(BUDGET)
            new_query, _ = self._ask(
                "llm", "rewrite_query", prompts.rewrite_query(question, state.query, self._found(batch)), parse_query, max_tokens=REWRITE_MAX_TOKENS, round=state.iteration
            )
            if new_query is None or new_query.lower() in {normalize_text(q).lower() for q in queries}:  # no new query: another round would repeat itself
                state.info["rewrite_failed"] = True
                return Finish()
            return Continue(new_query)

        state = AgentLoop(self._budget(opts.max_rounds + 1), tracer).run(policy, question=question)
        state.info.setdefault("sufficient", False)
        chunks = self._ranked(encountered, grades, final_top_k)
        metadata = {
            "retriever": "corrective",
            "queries": queries,
            "grades": {grade: sum(1 for g in grades.values() if g == grade) for grade in (RELEVANT, AMBIGUOUS, IRRELEVANT)},
            "expand_to": opts.expand_to,
        }
        return state, chunks, metadata

    def _candidates(self, query: str, k: int, retry: bool, round_: int) -> list[RetrievedChunk]:
        """The chunks to grade this round. A retry widens the search as `expand_to` says."""
        opts = self.options
        widen_hybrid = retry and opts.expand_to == "hybrid"
        ranking, _ = self._search(query, k * 2 if widen_hybrid else k, "round_search", retriever="hybrid" if widen_hybrid else None, round=round_)
        if not (retry and opts.expand_to == "full_doc"):
            return ranking
        head, tail = ranking[: opts.grade_top_k], ranking[opts.grade_top_k :]
        docs = unique_preserve_order(chunk.doc_id for chunk in head)[:FULL_DOC_COUNT]
        with self.trace.step("retrieve", "expand_full_doc", docs=len(docs)) as step:
            seen = {chunk.chunk_id for chunk in head}
            whole = [
                RetrievedChunk(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, text=chunk.text, score=0.0, rank=0, metadata=dict(chunk.metadata))
                for chunk in self.bm25_store.chunks
                if chunk.doc_id in docs and chunk.chunk_id not in seen
            ]
            step.set_chunks(whole)
        extra = {chunk.chunk_id for chunk in whole}
        return [*head, *whole, *(chunk for chunk in tail if chunk.chunk_id not in extra)]

    @staticmethod
    def _found(batch: list[RetrievedChunk]) -> str:
        return " | ".join(" ".join(chunk.text.split())[:160] for chunk in batch[:3]) or "nothing"

    @staticmethod
    def _ranked(encountered: dict[str, RetrievedChunk], grades: dict[str, str], k: int) -> list[RetrievedChunk]:
        """Relevant chunks first, then ambiguous, ungraded, irrelevant; first-retrieved order within each group."""
        chosen = sorted(encountered.values(), key=lambda chunk: _GRADE_ORDER[grades.get(chunk.chunk_id)])[:k]  # a stable sort keeps retrieval order
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=chunk.score,
                rank=rank,
                metadata={**chunk.metadata, "grade": grades.get(chunk.chunk_id, "ungraded")},
            )
            for rank, chunk in enumerate(chosen, start=1)
        ]

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        draft = super()._generate_from_retrieval(retrieval, question, chunks)
        if not self.options.self_check or not chunks or self._over_budget():
            return draft
        context = self._format_context(chunks)
        verdict, check_cost = self._ask(
            "grade", "groundedness", prompts.check_grounded(question, context, draft.text), parse_groundedness, max_tokens=CHECK_MAX_TOKENS
        )
        total_cost, prompt_tokens, completion_tokens = draft.cost.plus(check_cost), draft.prompt_tokens, draft.completion_tokens
        final = draft
        if verdict is not None:
            self.trace.steps[-1].metadata["supported"] = verdict[0]
        if verdict is not None and not verdict[0] and not self._over_budget():
            messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT + prompts.regenerate_note(verdict[1])},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
            with self.trace.step("generate", "answer_regenerated", context_chunks=len(chunks)) as step:
                final = self.llm.generate(messages, temperature=0)
                step.set_llm(final)
                step.set_input(question)
            total_cost = total_cost.plus(final.cost)
            prompt_tokens, completion_tokens = prompt_tokens + final.prompt_tokens, completion_tokens + final.completion_tokens
        return LLMResult(
            text=final.text, model=final.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=total_cost, finish_reason=final.finish_reason
        )
