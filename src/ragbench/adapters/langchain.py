"""LangChain pipelines as `python_callable` targets. Needs `pip install 'ragbench[langchain]'`.

    # my_rag.py
    from ragbench.adapters.langchain import from_chain, from_retriever
    answer = from_retriever(retriever, generate, model="gpt-6-luna")          # a retriever and a runnable that writes the answer
    answer = from_chain(retrieval_chain, model="gpt-6-luna")                  # or one runnable that does both

    # config
    retrieval: {target: my_rag.py:answer, model: gpt-6-luna}

A callback handler follows each question through the pipeline: every retriever call becomes a `retrieve` step and every model call a
`generate` step, with the tokens the model reported priced at its model (or at `model`, when it reports none or one without a price).
Passages are the pipeline's own retrieved documents; a document's `metadata` names the corpus file it came from (`source`, `file_path`, ...).
LangChain decides how many documents come back, so set `k` on your retriever; `top_k` from RAGBench is not forwarded.
"""

from __future__ import annotations

import time
from functools import cache
from typing import Any
from uuid import UUID

from ragbench.adapters._common import Call, Recorder, Target, first_int, passage, recording, reply, source_ref
from ragbench.utils.extras import require_extra


def _require() -> None:
    require_extra("langchain_core", "The LangChain adapter", "langchain", package="langchain-core")


def _text(output: Any) -> str:
    """The text of a runnable's output: a string, a message, or the first string field of a dict."""
    if isinstance(output, str):
        return output
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(block if isinstance(block, str) else str(block.get("text", "")) for block in content)
    return str(output)


def passages_of(documents: Any) -> list[dict[str, Any]]:
    """LangChain `Document`s as passages."""
    return [passage(doc.page_content, source_ref(getattr(doc, "metadata", None)), (getattr(doc, "metadata", None) or {}).get("score")) for doc in documents or []]


@cache
def _handler_class() -> type:
    from langchain_core.callbacks import BaseCallbackHandler

    class RagbenchHandler(BaseCallbackHandler):
        """Records retriever and model calls of the question it is passed to."""

        raise_error = False  # a failing recorder must not fail the question

        def __init__(self, recorder: Recorder) -> None:
            self.recorder = recorder
            self._started: dict[UUID, tuple[float, str]] = {}

        # retrievers
        def on_retriever_start(self, serialized: dict[str, Any], query: str, *, run_id: UUID, **kwargs: Any) -> None:
            self._started[run_id] = (time.perf_counter(), query)

        def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
            found = passages_of(documents)
            self.recorder.passages = found
            self._finish(run_id, "retrieve", "langchain_retriever", output=f"{len(found)} documents")

        def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            self._finish(run_id, "retrieve", "langchain_retriever", error=error)

        # models
        def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
            self._started[run_id] = (time.perf_counter(), "\n".join(prompts))

        def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: UUID, **kwargs: Any) -> None:
            self._started[run_id] = (time.perf_counter(), "\n".join(_text(message) for batch in messages for message in batch))

        def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
            llm_output = response.llm_output or {}
            usage = llm_output.get("token_usage") or {}
            generation = response.generations[0][0] if response.generations and response.generations[0] else None
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            self._finish(
                run_id,
                "generate",
                "langchain_llm",
                model=llm_output.get("model_name") or (getattr(message, "response_metadata", None) or {}).get("model_name"),
                prompt_tokens=first_int(usage.get("prompt_tokens"), metadata.get("input_tokens")),
                completion_tokens=first_int(usage.get("completion_tokens"), metadata.get("output_tokens")),
                output=getattr(generation, "text", None),
            )

        def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
            self._finish(run_id, "generate", "langchain_llm", error=error)

        def _finish(self, run_id: UUID, kind: Any, name: str, *, error: BaseException | None = None, output: str | None = None, **fields: Any) -> None:
            started, prompt = self._started.pop(run_id, (time.perf_counter(), ""))
            self.recorder.add(
                Call(kind, name, (time.perf_counter() - started) * 1000, input_preview=prompt, output_preview=output, error=None if error is None else f"{type(error).__name__}: {error}", **fields)
            )

    return RagbenchHandler


def from_chain(
    chain: Any, *, input_key: str | None = "input", answer_key: str = "answer", context_key: str = "context", model: str | None = None
) -> Target:
    """A runnable that retrieves and answers, e.g. `create_retrieval_chain(...)` or `retriever | prompt | llm | parser`.

    It is invoked with `{input_key: question}` (the bare question when `input_key` is None). It may return a string, a message, or a dict
    with the answer under `answer_key` and the documents it used under `context_key`; without that key, the passages are what its retrievers returned.
    """
    _require()

    def answer(question: str, top_k: int) -> dict[str, Any]:
        with recording(model) as recorder:
            output = chain.invoke(question if input_key is None else {input_key: question}, config={"callbacks": [_handler_class()(recorder)]})
        if isinstance(output, dict):
            contexts = passages_of(output[context_key]) if output.get(context_key) is not None else recorder.passages
            return reply(_text(output[answer_key]), contexts, recorder)
        return reply(_text(output), recorder.passages, recorder)

    return answer


def from_retriever(retriever: Any, generate: Any, *, model: str | None = None) -> Target:
    """A retriever and a runnable that writes the answer from `{"question": str, "context": list[Document]}` (format the documents inside it)."""
    _require()

    def answer(question: str, top_k: int) -> dict[str, Any]:
        with recording(model) as recorder:
            config = {"callbacks": [_handler_class()(recorder)]}
            documents = retriever.invoke(question, config=config)
            output = generate.invoke({"question": question, "context": documents}, config=config)
        return reply(_text(output), passages_of(documents), recorder)

    return answer


__all__ = ["from_chain", "from_retriever", "passages_of"]
