from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import RetrievedChunk
from ragbench.models.cost import CostBreakdown
from ragbench.models.decisions import Predicate, create_decision_model, formats
from ragbench.models.defaults import DEFAULT_DECISION_MODEL
from ragbench.models.llms import LLMResult
from ragbench.models.prompts import COMPRESS_PASSAGES_MARKER
from ragbench.rag_systems.base import RetrievalResult
from ragbench.rag_systems.hybrid_rag import HybridRAG
from ragbench.rag_systems.options import CompressionOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import StepHandle, sum_costs
from ragbench.registry import SYSTEMS
from ragbench.runtime.parallel import ordered_parallel_map
from ragbench.utils.query_planning import loads_lenient
from ragbench.utils.text import estimate_tokens, normalize_text, sentence_pieces

EXTRACT_MAX_TOKENS = 2000
DECISION_PARALLEL_CALLS = 5
JOIN = " … "  # between the kept sentences of one chunk


@dataclass
class Compression:
    """What compression did for one question; `as_metadata` is the `compression` block of `per_question_results.jsonl`."""

    compressor: str
    candidates: int
    tokens_before: int
    kept: list[tuple[RetrievedChunk, str]] = field(default_factory=list)  # (chunk, the text the generator reads), rank order, within the budget
    unverified_quotes: int = 0
    fallback: bool = False  # the extractor's reply was unusable: the candidates were packed uncompressed
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    @property
    def context_texts(self) -> dict[str, str]:
        return {chunk.chunk_id: text for chunk, text in self.kept}

    def as_metadata(self) -> dict[str, Any]:
        return {
            "compressor": self.compressor,
            "candidates": self.candidates,
            "kept": len(self.kept),
            "tokens_before": self.tokens_before,
            "tokens_after": sum(estimate_tokens(text) for _, text in self.kept),
            "unverified_quotes": self.unverified_quotes,
            "fallback": self.fallback,
            "context_texts": self.context_texts,
        }


@SYSTEMS.register("compression")
class CompressionRAG(HybridRAG):
    """Contextual compression: retrieve `candidate_k` hybrid candidates, cut each down to what bears on the question, pack to a token budget.

    The compressor keeps whole sentences of the chunk (an `llm_extract` quote that is not in its chunk word for word is dropped and counted), so
    the generator never reads text the documents do not contain. Like `decision_gate`, the ranking handed back (and scored for retrieval) is the
    uncompressed hybrid one; the context metrics and the judge read the compressed text the generator read (`context_texts`).
    """

    spec = SystemSpec(
        type="compression",
        title="Contextual compression",
        summary=(
            "Hybrid retrieval of `candidate_k` candidates, each cut down to the sentences that bear on the question (LLM quotes, an embedding filter or "
            "a decision model), then packed to a token budget"
        ),
        best_for="Long or noisy chunks where most of each passage is beside the point; smaller prompts on long contexts",
        cost_profile="medium",
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=CompressionOptions,
    )
    options: CompressionOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        if self.options.compressor == "decision":
            choice = self.options.decision_model
            self.decider = create_decision_model(choice.provider, choice.model, force_mock=force_mock)
            self.decision_model_ref = choice.model or DEFAULT_DECISION_MODEL  # priced by `ragbench estimate`

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        result = super().fetch_context(question, top_k=top_k)
        result.metadata["retriever"] = "hybrid_rrf+compression"
        return result

    def _generate_from_retrieval(self, retrieval: RetrievalResult, question: str, chunks: list[RetrievedChunk]) -> LLMResult:
        compression = self._compress(question, chunks)
        retrieval.metadata["compression"] = compression.as_metadata()
        given = [chunk.model_copy(update={"text": text}) for chunk, text in compression.kept]
        answer = super()._generate_from_retrieval(retrieval, question, given)
        answer.cost = answer.cost.plus(compression.cost)
        return answer

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None):
        result = super().answer_question(question, top_k=top_k, context_k=context_k)
        block = result.retrieval_result.metadata["compression"]
        result.metadata["compression"] = block
        result.metadata["context_texts"] = block["context_texts"]  # what the judge and the context metrics read
        result.metadata["context_chunk_ids"] = list(block["context_texts"])
        # Compression belongs to the evidence phase of the trace: everything before the final `generate` step is the retrieval result's.
        result.retrieval_result.steps = [step for step in result.steps if step.kind != "generate"]
        return result

    # --- compressing ----------------------------------------------------------------------------------------------------------

    def _compress(self, question: str, chunks: list[RetrievedChunk]) -> Compression:
        compression = Compression(compressor=self.options.compressor, candidates=len(chunks), tokens_before=sum(estimate_tokens(c.text) for c in chunks))
        if not chunks:
            return compression
        if self.options.compressor == "llm_extract":
            with self.trace.step("llm", "compress", candidates=len(chunks)) as step:
                texts = self._extract(question, chunks, compression, step)
        elif self.options.compressor == "embedding_filter":
            with self.trace.step("embed", "compress", candidates=len(chunks)) as step:
                texts = self._embedding_filter(question, chunks, compression, step)
        else:
            with self.trace.step("grade", "compress", candidates=len(chunks)) as step:
                texts = self._decide(question, chunks, compression, step)
        budget, used = self.options.context_token_budget, 0
        for chunk, text in zip(chunks, texts, strict=True):
            tokens = estimate_tokens(text) if text else 0
            if text and used + tokens <= budget:
                compression.kept.append((chunk, text))
                used += tokens
        return compression

    def _extract(self, question: str, chunks: list[RetrievedChunk], compression: Compression, step: StepHandle) -> list[str]:
        passages = "\n".join(f"[{number}] {normalize_text(chunk.text)}" for number, chunk in enumerate(chunks, start=1))
        messages = [
            {"role": "system", "content": "You select evidence for a question-answering system. Answer with JSON only."},
            {
                "role": "user",
                "content": (
                    f"{COMPRESS_PASSAGES_MARKER}. Copy each sentence exactly as it is written; do not rephrase, shorten or combine. Give a passage with "
                    'nothing relevant an empty list. Return JSON like {"passages": [{"id": 1, "quotes": ["..."]}]}.\n'
                    f"Question: {question}\n\nPassages:\n{passages}"
                ),
            },
        ]
        result = self.llm.generate(messages, temperature=0, json_mode=True, max_tokens=EXTRACT_MAX_TOKENS)
        step.set_llm(result)
        step.set_input(question)
        compression.cost = result.cost
        reply = loads_lenient(result.text)
        if not isinstance(reply, dict) or not isinstance(reply.get("passages"), list):
            compression.fallback = True  # an unusable reply must not leave the generator with nothing: the candidates go in uncompressed
            return [chunk.text for chunk in chunks]
        quotes: dict[int, list[str]] = {}
        for item in reply["passages"]:
            number = _passage_number(item.get("id")) if isinstance(item, dict) else None
            if number is not None and isinstance(item.get("quotes"), list):
                quotes.setdefault(number, []).extend(quote for quote in item["quotes"] if isinstance(quote, str))
        texts: list[str] = []
        for number, chunk in enumerate(chunks, start=1):
            source = normalize_text(chunk.text)
            found: list[tuple[int, str]] = []
            for quote in quotes.get(number, []):
                clean = normalize_text(quote)
                position = source.find(clean) if clean else -1
                if position < 0:
                    compression.unverified_quotes += 1
                elif clean not in [text for _, text in found]:
                    found.append((position, clean))
            texts.append(JOIN.join(text for _, text in sorted(found)))
        step.set_meta(unverified_quotes=compression.unverified_quotes)
        return texts

    def _embedding_filter(self, question: str, chunks: list[RetrievedChunk], compression: Compression, step: StepHandle) -> list[str]:
        pieces = [sentence_pieces(chunk.text) for chunk in chunks]
        flat = [piece for chunk_pieces in pieces for piece in chunk_pieces]
        query = self.embedding_model.embed_query(question)
        embedded = self.embedding_model.embed_texts(flat) if flat else None
        compression.cost = query.cost.plus(embedded.cost) if embedded is not None else query.cost
        step.set_cost(compression.cost)
        step.set_input(question)
        similarity = embedded.vectors @ query.vectors[0] if embedded is not None else np.zeros(0)
        texts: list[str] = []
        row = 0
        for chunk_pieces in pieces:
            kept = [piece for offset, piece in enumerate(chunk_pieces) if similarity[row + offset] >= self.options.similarity_threshold]
            row += len(chunk_pieces)
            texts.append(JOIN.join(normalize_text(piece) for piece in kept))
        return texts

    def _decide(self, question: str, chunks: list[RetrievedChunk], compression: Compression, step: StepHandle) -> list[str]:
        pieces = [sentence_pieces(chunk.text) for chunk in chunks]
        template = formats.ANSWERABILITY_INSTRUCTIONS

        def judge(chunk_pieces: list[str]):
            predicates = [Predicate(f"{formats.BATCH_PREFIX}{i}", template.format(passage=formats.target(i))) for i in range(1, len(chunk_pieces) + 1)]
            return self.decider.decide(formats.batch_input(question, chunk_pieces), predicates), predicates

        decided = ordered_parallel_map(judge, [p for p in pieces if p], workers=DECISION_PARALLEL_CALLS, thread_name_prefix="ragbench-compress")
        compression.cost = sum_costs(result.cost for result, _ in decided)
        step.set_cost(compression.cost)
        step.set_tokens(compression.cost.decision_input_tokens, 0)
        step.set_input(question)
        step.set_meta(requests=len(decided))
        replayed = [result.latency_ms for result, _ in decided if result.cached and result.latency_ms is not None]
        if replayed:
            step.latency_override_ms = max(replayed)
        outcomes = iter(decided)
        texts: list[str] = []
        for chunk_pieces in pieces:
            if not chunk_pieces:
                texts.append("")
                continue
            result, predicates = next(outcomes)
            probabilities = [result.probabilities.get(predicate.name) for predicate in predicates]
            kept = [piece for piece, p in zip(chunk_pieces, probabilities, strict=True) if p is None or p >= self.options.relevance_threshold]
            texts.append(JOIN.join(normalize_text(piece) for piece in kept))
        return texts


def _passage_number(value: object) -> int | None:
    """A passage id as the model wrote it: a number, or a string of digits (`"2"`); anything else names no passage."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None
