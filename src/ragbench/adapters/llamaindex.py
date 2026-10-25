"""LlamaIndex query engines as `python_callable` targets. Needs `pip install 'ragbench[llamaindex]'`.

    # my_rag.py
    from ragbench.adapters.llamaindex import from_query_engine
    answer = from_query_engine(index.as_query_engine(similarity_top_k=5), model="gpt-6-luna")

    # config
    retrieval: {target: my_rag.py:answer, model: gpt-6-luna}

LlamaIndex reports what it does as instrumentation events. The adapter listens once, process-wide, and files each event under the question
being answered in that thread: a retrieval is a `retrieve` step and each model call a `generate` step, with the tokens the model reported
priced at its model (or at `model`, when it reports none or one without a price). Passages are the response's `source_nodes`; a node's
`metadata` names the corpus file it came from (`file_path`, `file_name`, ...). Set `similarity_top_k` on your engine: `top_k` from RAGBench is not forwarded.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ragbench.adapters._common import Call, Target, current_recorder, first_int, passage, recording, reply, source_ref
from ragbench.utils.extras import require_extra

_registered = False
_lock = threading.Lock()


def _private(path: Path) -> bool:
    """NLTK's own test for data it will trust: owned by us (or root) and not writable by group or others."""
    info = path.stat()
    return info.st_uid in (os.geteuid(), 0) and not info.st_mode & 0o022


def _default_nltk_dir() -> Path:
    """Where LlamaIndex keeps NLTK data when `NLTK_DATA` is unset: its bundled cache if that has files, else a folder in the user cache."""
    spec = importlib.util.find_spec("llama_index.core")
    bundled = Path(spec.origin).parent / "_static" / "nltk_cache" if spec and spec.origin else None
    if bundled is not None and bundled.is_dir() and any(bundled.iterdir()):
        return bundled
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("llama_index")) / "_static" / "nltk_cache"


def private_nltk_dir(default: Path, env: Mapping[str, str], cache_root: Path, umask: int | None = None) -> Path | None:
    """A private directory for NLTK data, when `default` (where LlamaIndex would put it) is one NLTK refuses to use; else None.

    LlamaIndex keeps its sentence-splitter data in site-packages. NLTK reads and downloads only into directories no other account can write,
    and a shared Python install (a CI runner whose umask is 002 creates it group-writable) fails that, so every question dies with a
    LookupError. A directory that does not exist yet is judged by the umask it will be created under; an existing one by its mode and entries.
    """
    if "NLTK_DATA" in env:
        return None  # the user's own choice
    if default.exists():
        if _private(default) and all(_private(Path(top, name)) for top, dirs, files in os.walk(default) for name in [*dirs, *files]):
            return None
    else:
        if umask is None:
            umask = os.umask(0)
            os.umask(umask)
        if umask & 0o022 == 0o022:
            return None
    chosen = cache_root / "nltk_data"
    chosen.mkdir(parents=True, mode=0o700, exist_ok=True)
    chosen.chmod(0o700)
    return chosen


def _keep_nltk_data_private() -> None:
    try:
        from platformdirs import user_cache_dir

        chosen = private_nltk_dir(_default_nltk_dir(), os.environ, Path(user_cache_dir("ragbench")))
    except (ImportError, OSError):
        return  # LlamaIndex or the cache folder is not usable: leave NLTK alone, the pipeline reports what it needs
    if chosen is not None:
        os.environ["NLTK_DATA"] = str(chosen)


_keep_nltk_data_private()


def _require() -> None:
    require_extra("llama_index.core", "The LlamaIndex adapter", "llamaindex", package="llama-index-core")


def passages_of(nodes: Any) -> list[dict[str, Any]]:
    """LlamaIndex `NodeWithScore`s as passages."""
    found = []
    for item in nodes or []:
        node = getattr(item, "node", item)
        found.append(passage(node.get_content(), source_ref(getattr(node, "metadata", None), getattr(node, "ref_doc_id", None)), getattr(item, "score", None)))
    return found


def _field(source: Any, name: str) -> Any:
    return source.get(name) if isinstance(source, dict) else getattr(source, name, None)


def _usage(response: Any) -> tuple[int, int, str | None]:
    """`(prompt_tokens, completion_tokens, model)` of a chat or completion response, from the provider's raw payload or the response's extras."""
    raw = _field(response, "raw")
    usage = _field(raw, "usage")
    extras = _field(response, "additional_kwargs") or {}
    prompt = first_int(_field(usage, "prompt_tokens"), _field(usage, "input_tokens"), extras.get("prompt_tokens"), extras.get("input_tokens"))
    completion = first_int(_field(usage, "completion_tokens"), _field(usage, "output_tokens"), extras.get("completion_tokens"), extras.get("output_tokens"))
    model = _field(raw, "model")
    return prompt, completion, model if isinstance(model, str) else None


def _handler() -> Any:
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler
    from llama_index.core.instrumentation.events.llm import (
        LLMChatEndEvent,
        LLMChatStartEvent,
        LLMCompletionEndEvent,
        LLMCompletionStartEvent,
    )
    from llama_index.core.instrumentation.events.retrieval import RetrievalEndEvent, RetrievalStartEvent

    class RagbenchHandler(BaseEventHandler):
        """Files retrieval and model events under the recorder of the question being answered in this thread."""

        started: dict[str, tuple[datetime, str]] = {}

        @classmethod
        def class_name(cls) -> str:
            return "RagbenchHandler"

        def handle(self, event: Any, **kwargs: Any) -> None:
            recorder = current_recorder()
            if recorder is None:
                return
            try:
                span = event.span_id or ""
                if isinstance(event, RetrievalStartEvent):
                    self.started[span] = (event.timestamp, str(event.str_or_query_bundle))
                elif isinstance(event, LLMChatStartEvent):
                    self.started[span] = (event.timestamp, "\n".join(str(message.content) for message in event.messages))
                elif isinstance(event, LLMCompletionStartEvent):
                    self.started[span] = (event.timestamp, str(event.prompt))
                elif isinstance(event, RetrievalEndEvent):
                    found = passages_of(event.nodes)
                    recorder.passages = found
                    recorder.add(self._call(event, "retrieve", "llamaindex_retriever", output=f"{len(found)} nodes"))
                elif isinstance(event, LLMChatEndEvent | LLMCompletionEndEvent):
                    prompt, completion, model = _usage(event.response)
                    text = getattr(event.response, "text", None) or ""
                    recorder.add(self._call(event, "generate", "llamaindex_llm", model=model, prompt_tokens=prompt, completion_tokens=completion, output=text))
            except Exception:  # noqa: BLE001 - a recorder bug must not fail the question
                return

        def _call(self, event: Any, kind: Any, name: str, *, output: str | None = None, **fields: Any) -> Call:
            started, prompt = self.started.pop(event.span_id or "", (event.timestamp, ""))
            return Call(kind, name, (event.timestamp - started).total_seconds() * 1000, input_preview=prompt, output_preview=output, **fields)

    return RagbenchHandler()


def _listen() -> None:
    global _registered
    with _lock:
        if not _registered:
            from llama_index.core.instrumentation import get_dispatcher

            get_dispatcher().add_event_handler(_handler())
            _registered = True


def from_query_engine(engine: Any, *, model: str | None = None) -> Target:
    """Any object with `.query(question)` that returns a response with `.response` and `.source_nodes` (a query engine, or a chat engine's `.query`)."""
    _require()
    _listen()

    def answer(question: str, top_k: int) -> dict[str, Any]:
        with recording(model) as recorder:
            response = engine.query(question)
        nodes = getattr(response, "source_nodes", None)
        text = getattr(response, "response", None)
        return reply(str(text if text is not None else response), passages_of(nodes) if nodes is not None else recorder.passages, recorder)

    return answer


__all__ = ["from_query_engine", "passages_of"]
