from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import DecisionResult, Predicate, create_decision_model, formats
from ragbench.models.defaults import DEFAULT_DECISION_MODEL
from ragbench.models.llms import LLMResult
from ragbench.rag_systems.base import RetrievalResult
from ragbench.rag_systems.hybrid_rerank_rag import HybridRerankRAG
from ragbench.rag_systems.options import DecisionGateOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import StepHandle
from ragbench.registry import SYSTEMS
from ragbench.runtime.parallel import ordered_parallel_map

# The sentence the generation prompt tells a model to say when the context has no answer; `is_refusal` recognizes it, so an abstain counts as a refusal.
ABSTAIN_ANSWER = "I could not find the answer in the provided documents."
PER_CHUNK_PARALLEL_CALLS = 5
NO_RELEVANT_CHUNKS = "no_relevant_chunks"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class Gate:
    """What the gate decided for one question; `as_metadata` is what the evaluator stores as the question's `gate` block."""

    judged: list[RetrievedChunk]
    answerability: list[float | None]
    kept: list[RetrievedChunk] = field(default_factory=list)
    sufficiency: float | None = None
    sufficiency_asked: bool = False
    abstain_reason: str | None = None
    decision_calls: int = 0
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def abstained(self) -> bool:
        return self.abstain_reason is not None

    @property
    def unreadable(self) -> int:
        """Probabilities the decision model did not give (a refusal, an unparseable reply). Those chunks are kept and a missing sufficiency does not abstain."""
        return sum(p is None for p in self.answerability) + (self.sufficiency_asked and self.sufficiency is None)

    def as_metadata(self, opts: DecisionGateOptions) -> dict[str, Any]:
        return {
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "judged": len(self.judged),
            "kept": len(self.kept),
            "kept_chunk_ids": [chunk.chunk_id for chunk in self.kept],
            "answerability": {chunk.chunk_id: p for chunk, p in zip(self.judged, self.answerability, strict=True)},
            "sufficiency": self.sufficiency,
            "decision_calls": self.decision_calls,
            "unreadable": self.unreadable,
            "judge_mode": opts.judge_mode,
            "answerability_threshold": opts.answerability_threshold,
            "sufficiency_threshold": opts.sufficiency_threshold,
            "decision_model": f"{opts.decision_model.provider}:{opts.decision_model.model or 'default'}",
        }


@SYSTEMS.register("decision_gate")
class DecisionGateRAG(HybridRerankRAG):
    """Decision-gated RAG: retrieve like `hybrid_rerank`, then ask a typed decision model two yes/no questions before generating.

    1. *Answerability*, per retrieved chunk: does it help answer the question? Chunks below `answerability_threshold` are dropped.
    2. *Sufficiency*, once, on what is left: is it enough to answer the question from the documents alone? Below `sufficiency_threshold` the system
       abstains instead of generating (no generator call; the answer is the usual "could not find" sentence).

    The ranking handed back (and scored for retrieval) is `hybrid_rerank`'s own, so retrieval metrics are directly comparable; what the generator
    saw is the surviving chunks, which is what the context metrics and the judge use. Every decision is a traced `grade` step, its spend is booked
    as `decision_cost`, and the probabilities are stored per question (`gate` in `per_question_results.jsonl`) so thresholds can be studied offline.
    A probability the model refuses to give counts as "unknown": the chunk is kept and the sufficiency check cannot abstain.
    """

    spec = SystemSpec(
        type="decision_gate",
        title="Decision-gated RAG",
        summary="Hybrid + rerank retrieval, then a typed decision model drops chunks that do not help answer the question and abstains when the rest is not enough",
        best_for="Corpora where many questions have no answer in the documents, and a confident wrong answer costs more than a refusal",
        cost_profile="high",  # measured: about 3x hybrid_rerank per answer, because every request carries the candidate chunks
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=DecisionGateOptions,
    )
    options: DecisionGateOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        choice = self.options.decision_model
        self.decider = create_decision_model(choice.provider, choice.model, force_mock=force_mock)
        # The model whose price applies to the decision requests (`ragbench estimate` prices the measured tokens with it, even when the measurement ran on the mock).
        self.decision_model_ref = choice.model or DEFAULT_DECISION_MODEL

    # -- the gate -----------------------------------------------------------------------------------------------------

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        gate = self._gate(question, retrieval.chunks[: self.options.max_candidates_to_judge], limit=len(chunks))
        retrieval.metadata["gate"] = gate.as_metadata(self.options)
        if gate.abstained:
            return LLMResult(text=ABSTAIN_ANSWER, model=self.llm.model_name, prompt_tokens=0, completion_tokens=0, cost=gate.cost, finish_reason="abstain")
        answer = super()._generate_from_retrieval(retrieval, question, gate.kept)
        answer.cost = answer.cost.plus(gate.cost)
        return answer

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None):
        result = super().answer_question(question, top_k=top_k, context_k=context_k)
        gate = result.retrieval_result.metadata["gate"]
        result.metadata["context_chunk_ids"] = gate["kept_chunk_ids"]  # what the generator saw, not the first `context_k` of the ranking
        result.metadata["gate"] = gate
        # The gate is part of the evidence phase of the trace: everything before the final `generate` step belongs to the retrieval result.
        result.retrieval_result.steps = [step for step in result.steps if step.kind != "generate"]
        return result

    def _gate(self, question: str, candidates: list[RetrievedChunk], limit: int) -> Gate:
        opts = self.options
        gate = Gate(judged=candidates, answerability=[])
        if candidates:
            gate.answerability = self._answerability(question, candidates, gate)
        threshold = opts.answerability_threshold
        gate.kept = [chunk for chunk, p in zip(candidates, gate.answerability, strict=True) if p is None or p >= threshold][:limit]
        if not gate.kept:
            gate.abstain_reason = NO_RELEVANT_CHUNKS
        elif opts.sufficiency_threshold > 0:
            gate.sufficiency_asked = True
            gate.sufficiency = self._sufficiency(question, gate.kept, gate)
            if gate.sufficiency is not None and gate.sufficiency < opts.sufficiency_threshold:
                gate.abstain_reason = INSUFFICIENT_EVIDENCE
        return gate

    def _answerability(self, question: str, candidates: list[RetrievedChunk], gate: Gate) -> list[float | None]:
        template = self.options.answerability_instructions or formats.ANSWERABILITY_INSTRUCTIONS
        with self.trace.step("grade", "decide_answerability", mode=self.options.judge_mode, candidates=len(candidates)) as step:
            step.set_input(question)
            if self.options.judge_mode == "batched":
                predicates = [
                    Predicate(f"{formats.BATCH_PREFIX}{i}", template.format(passage=formats.target(i))) for i in range(1, len(candidates) + 1)
                ]
                results = [self.decider.decide(formats.batch_input(question, [c.text for c in candidates]), predicates)]
                probabilities = [results[0].probabilities.get(p.name) for p in predicates]
            else:
                predicate = Predicate(formats.SINGLE, template.format(passage=formats.target(None)))
                results = ordered_parallel_map(
                    lambda chunk: self.decider.decide(formats.single_input(question, chunk.text), [predicate]),
                    candidates,
                    workers=PER_CHUNK_PARALLEL_CALLS,
                    thread_name_prefix="ragbench-gate",
                )
                probabilities = [result.probabilities.get(predicate.name) for result in results]
            self._book(step, gate, results)
            step.set_meta(probabilities=probabilities)
            step.set_output(f"answerability of {len(candidates)} chunks: " + ", ".join("?" if p is None else f"{p:.2f}" for p in probabilities))
        return probabilities

    def _sufficiency(self, question: str, kept: list[RetrievedChunk], gate: Gate) -> float | None:
        instructions = self.options.sufficiency_instructions or formats.SUFFICIENCY_INSTRUCTIONS
        with self.trace.step("grade", "decide_sufficiency", evidence=len(kept)) as step:
            step.set_input(question)
            result = self.decider.decide(formats.evidence_input(question, [c.text for c in kept]), [Predicate(formats.SUFFICIENT, instructions)])
            self._book(step, gate, [result])
            probability = result.probabilities.get(formats.SUFFICIENT)
            step.set_meta(probability=probability)
            step.set_output("sufficiency: " + ("?" if probability is None else f"{probability:.2f}"))
        return probability

    @staticmethod
    def _book(step: StepHandle, gate: Gate, results: list[DecisionResult]) -> None:
        """Charge the requests to the gate and the step; a step that replays cached requests reports the slowest one's original latency."""
        cost = CostBreakdown()
        for result in results:
            cost = cost.plus(result.cost)
        gate.cost = gate.cost.plus(cost)
        gate.decision_calls += len(results)
        step.set_cost(cost)
        step.set_tokens(cost.decision_input_tokens, 0)
        step.set_meta(requests=len(results), model=results[0].model if results else None)
        replayed = [r.latency_ms for r in results if r.cached and r.latency_ms is not None]
        if replayed:
            step.set_meta(cached=len(replayed))
            step.latency_override_ms = max(replayed)
