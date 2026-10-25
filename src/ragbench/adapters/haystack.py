"""Haystack pipelines as `python_callable` targets. Needs `pip install 'ragbench[haystack]'`.

    # my_rag.py
    from ragbench.adapters.haystack import from_pipeline
    answer = from_pipeline(
        pipeline,
        inputs=lambda question, top_k: {"embedder": {"text": question}, "retriever": {"top_k": top_k}, "prompt_builder": {"question": question}},
        retriever="retriever",
        generator="llm",
        model="gpt-6-luna",
    )

    # config
    retrieval: {target: my_rag.py:answer, model: gpt-6-luna}

`inputs(question, top_k)` is the `data` argument of `Pipeline.run`. The retriever component's run becomes a `retrieve` step and the generator's a
`generate` step, with the tokens the generator reported (`meta["usage"]`) priced at its model (or at `model`, when it reports none or one
without a price); the components of the pipeline you pass are wrapped in place to time them. Passages are the retriever's `documents`; a
document's `meta` names the corpus file it came from (`file_path`, `source_id`, ...). The answer is the generator's first reply, or pass
`answer(outputs)` to read it from `outputs[generator]` yourself. Only the synchronous `Pipeline` is supported.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Mapping
from typing import Any

from ragbench.adapters._common import Call, Target, current_recorder, first_int, passage, recording, reply, source_ref
from ragbench.utils.extras import require_extra


def _require() -> None:
    require_extra("haystack", "The Haystack adapter", "haystack", package="haystack-ai")


def passages_of(documents: Any) -> list[dict[str, Any]]:
    """Haystack `Document`s as passages."""
    return [passage(doc.content or "", source_ref(doc.meta), doc.score) for doc in documents or []]


def _reply_text(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str):
        return text
    data = getattr(item, "data", None)  # a GeneratedAnswer
    return data if isinstance(data, str) else str(item)


def default_answer(outputs: Mapping[str, Any]) -> str:
    """The first reply (or `GeneratedAnswer`) of a generator's outputs."""
    replies = outputs.get("replies") or outputs.get("answers") or []
    if not replies:
        raise ValueError(f"the generator returned no `replies` or `answers`; its outputs are {sorted(outputs)}: pass `answer=` to read them")
    return _reply_text(replies[0])


def _usage(outputs: Mapping[str, Any]) -> tuple[int, int, str | None]:
    """`(prompt_tokens, completion_tokens, model)` from a generator's `meta` list, or the `meta` of its first reply (a chat message)."""
    replies = outputs.get("replies") or []
    metas = list(outputs.get("meta") or []) or [getattr(reply_, "meta", None) or {} for reply_ in replies[:1]]
    prompt = completion = 0
    model = None
    for meta in metas:
        usage = meta.get("usage") or {}
        prompt += first_int(usage.get("prompt_tokens"), usage.get("input_tokens"))
        completion += first_int(usage.get("completion_tokens"), usage.get("output_tokens"))
        model = model or meta.get("model")
    return prompt, completion, model if isinstance(model, str) else None


def _timed(instance: Any, name: str, observe: Callable[[dict[str, Any], float], Call]) -> None:
    """Wrap `instance.run` so that, while a question is being recorded, each call is filed as a step."""
    run = instance.run
    if getattr(run, "_ragbench_wrapped", False):
        return

    @functools.wraps(run)
    def timed(*args: Any, **kwargs: Any) -> Any:
        recorder = current_recorder()
        started = time.perf_counter()
        outputs = run(*args, **kwargs)
        if recorder is not None:
            try:
                recorder.add(observe(outputs, (time.perf_counter() - started) * 1000))
            except Exception:  # noqa: BLE001 - a recorder bug must not fail the question
                pass
            if name == "retrieve":
                recorder.passages = passages_of(outputs.get("documents"))
        return outputs

    timed._ragbench_wrapped = True  # type: ignore[attr-defined]
    instance.run = timed


def from_pipeline(
    pipeline: Any,
    inputs: Callable[[str, int], dict[str, Any]],
    *,
    retriever: str = "retriever",
    generator: str = "llm",
    answer: Callable[[Mapping[str, Any]], str] = default_answer,
    model: str | None = None,
) -> Target:
    """A `Pipeline` with a retriever component and a generator component, run on `inputs(question, top_k)`."""
    _require()
    _timed(pipeline.get_component(retriever), "retrieve", lambda outputs, ms: Call("retrieve", "haystack_retriever", ms, output_preview=f"{len(outputs.get('documents') or [])} documents"))

    def generated(outputs: dict[str, Any], ms: float) -> Call:
        prompt, completion, reported = _usage(outputs)
        return Call("generate", "haystack_generator", ms, model=reported, prompt_tokens=prompt, completion_tokens=completion, output_preview=default_text(outputs))

    _timed(pipeline.get_component(generator), "generate", generated)

    def run(question: str, top_k: int) -> dict[str, Any]:
        with recording(model) as recorder:
            outputs = pipeline.run(inputs(question, top_k), include_outputs_from={retriever, generator})
        if generator not in outputs:
            raise ValueError(f"the pipeline did not run the generator component {generator!r}; its components are {sorted(outputs)}")
        documents = outputs.get(retriever, {}).get("documents")
        return reply(answer(outputs[generator]), passages_of(documents) if documents is not None else recorder.passages, recorder)

    return run


def default_text(outputs: Mapping[str, Any]) -> str | None:
    try:
        return default_answer(outputs)
    except ValueError:
        return None


__all__ = ["default_answer", "from_pipeline", "passages_of"]
