"""Shared by the tests of the external systems: the example fake RAG service (examples/fake_rag_server.py) and callables for `python_callable`."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SIMPLE_RAG = ROOT / "examples" / "simple_rag.py"


def _example(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def start_fake_rag():
    server, rag = _example("fake_rag_server").make_server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, rag


INGESTED: list[str] = []


def ingest(documents: list[Any]) -> None:
    INGESTED[:] = [document.doc_id for document in documents]


def answer(question: str, top_k: int) -> dict[str, Any]:
    return {"answer": f"echo: {question}", "contexts": [{"doc_id": "doc_001", "text": "HarborShield costs $200 per month.", "score": 1.0}][:top_k],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}, "cost_usd": 0.002}


def answer_result(question: str, top_k: int):
    from ragbench.models.cost import CostBreakdown
    from ragbench.rag_systems.base import AnswerResult, RetrievalResult, RetrievedChunk

    chunk = RetrievedChunk(chunk_id="doc_002::chunk::0", doc_id="doc_002", text="ClaimPilot ships in Q3.", score=1.0, rank=1)
    return AnswerResult(question=question, answer="Q3.", retrieval_result=RetrievalResult(question=question, chunks=[chunk]), model_name="my-model",
                        latency_ms=1.0, cost=CostBreakdown(llm_cost=0.004, llm_prompt_tokens=20, llm_completion_tokens=2))


def not_a_result(question: str, top_k: int) -> int:
    return 42


def answer_result_with_steps(question: str, top_k: int):
    """What a framework adapter returns: an AnswerResult whose steps split retrieval from generation."""
    from ragbench.models.cost import CostBreakdown
    from ragbench.rag_systems.base import AnswerResult, RetrievalResult, RetrievedChunk
    from ragbench.rag_systems.trace import Step

    chunk = RetrievedChunk(chunk_id="doc_002::chunk::0", doc_id="doc_002", text="ClaimPilot ships in Q3.", score=1.0, rank=1)
    retrieve = Step(kind="retrieve", name="framework_retriever", latency_ms=3.0, cost=CostBreakdown(embedding_input_tokens=8, embedding_cost=0.0001))
    generate = Step(kind="generate", name="framework_llm", latency_ms=9.0, prompt_tokens=20, completion_tokens=2,
                    cost=CostBreakdown(llm_prompt_tokens=20, llm_completion_tokens=2, llm_cost=0.004))
    return AnswerResult(question=question, answer="Q3.", retrieval_result=RetrievalResult(question=question, chunks=[chunk]), model_name="my-model",
                        latency_ms=12.0, cost=retrieve.cost.plus(generate.cost), steps=[retrieve, generate])


def answer_traced(question: str, top_k: int) -> dict[str, Any]:
    """Code that records its own steps on RAGBench's active tracer while it runs (as an adapter's callbacks would)."""
    from ragbench.models.cost import CostBreakdown
    from ragbench.rag_systems.trace import current_tracer

    tracer = current_tracer()
    with tracer.step("retrieve", "my_search") as step:
        step.set_cost(CostBreakdown(embedding_input_tokens=5, embedding_cost=0.0002))
    with tracer.step("generate", "my_answer") as step:
        step.set_cost(CostBreakdown(llm_prompt_tokens=30, llm_completion_tokens=4, llm_cost=0.003))
        step.set_tokens(30, 4)
    return {"answer": "traced", "contexts": [{"doc_id": "doc_001", "text": "HarborShield costs $200 per month."}]}


def answer_with_step_dicts(question: str, top_k: int) -> dict[str, Any]:
    """Plain data, no RAGBench import: steps as dicts in the reply."""
    return {"answer": "plain", "contexts": [], "steps": [
        {"kind": "retrieve", "name": "lookup", "latency_ms": 1.0},
        {"kind": "generate", "name": "llm", "latency_ms": 2.0, "prompt_tokens": 12, "completion_tokens": 3, "cost": {"llm_prompt_tokens": 12, "llm_completion_tokens": 3, "llm_cost": 0.001}},
    ]}
